import os
import torch
import argparse
import numpy as np
import torch.nn.functional as F
from GT import TransformerModel
from scoring import DistanceGatedScoringFunction
from torch.utils.data import DataLoader
from GFM_utils import reset_gnn_weights, set_random_seed
from data_buddy import get_link_prediction_data_buddy
from pretrain_model_buddy import BuddyPretrainModule
from gate_models import LinearRegressionModel
from evaluation_buddy import evaluate_mrr_merge_all
from copy import deepcopy
from argparse import Namespace
import yaml


def set_model_state(ckpt_dir, ckpt_name, epoch, pretrain_model):
    """
    Loads a pre-trained model from a checkpoint or initializes weights if no checkpoint is provided.
    
    Args:
        ckpt_dir (str): Directory containing checkpoint files
        ckpt_name (str): Base name of the checkpoint file
        epoch (int): Epoch number to load
        pretrain_model: Model to load weights into
        
    Returns:
        None: Updates the model in-place
    """
    if ckpt_name != 'none':
        ckpt_path = os.path.join(ckpt_dir, f'{ckpt_name}-{epoch}.ckpt')
        pretrain_model.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=True)
        print(f'pretrain model loaded from {ckpt_path}.')
    else:
        # backbone = pretrain_model.model
        reset_gnn_weights(pretrain_model)
        print('pretrain model init.')

    return 

@torch.no_grad()
def get_all_logits(data, model_1, model_2, args):
    """
    Computes logits from both models for all training examples.
    
    Args:
        data (dict): Dictionary containing graph data and edge information
        model_1: First pre-trained model
        model_2: Second pre-trained model
        args: Arguments containing configuration parameters
        
    Returns:
        tuple: (logits, labels) where logits are concatenated outputs from both models
               and labels are binary indicators (1 for positive edges, 0 for negative)
    """
    model_1.eval()
    model_2.eval()

    # fetch data
    x = data['x']
    train_pos = data['train_pos']
    train_neg = data['train_neg']
    buddy_train_pos = data['buddy_train_pos']
    buddy_train_neg = data['buddy_train_neg']
    mp_link = torch.cat((train_pos, train_pos.flip(1)),dim=0)

    batch_data = (x, mp_link, train_pos, train_neg, buddy_train_pos, buddy_train_neg)
    batch_data = [t.to(train_pos.device) for t in batch_data]
    logits_1 = model_1.get_all_output(*batch_data) # (P+N, num_experts)
    logits_2 = model_2.get_all_output(*batch_data) # (P+N, num_experts)
    logits = torch.cat([logits_1, logits_2], dim=1) # (P+N, 2*num_experts)
    labels = torch.cat([torch.ones(train_pos.shape[0]), torch.zeros(train_neg.shape[0])]).to(train_pos.device) # (P+N)

    return logits, labels


def train(feats, labels, gate_model, optimizer, batch_size, args):
    """
    Trains the gate model for one epoch.
    
    Args:
        feats (torch.Tensor): Feature tensor containing expert logits from both models
        labels (torch.Tensor): Binary labels (1 for positive edges, 0 for negative)
        gate_model: Linear regression model that combines expert outputs
        optimizer: Optimizer for training the gate model
        batch_size (int): Batch size for training
        args: Arguments containing configuration parameters
        
    Returns:
        float: Mean loss for the epoch
    """
    gate_model.train()
    total_loss = n_batches = 0
    for perm in DataLoader(range(feats.size(0)), batch_size,
                        shuffle=True):
        optimizer.zero_grad()
        feats_batch = feats[perm]
        labels_batch = labels[perm]
        logits = gate_model(feats_batch, return_prob=False).squeeze() 
        loss = F.binary_cross_entropy_with_logits(logits, labels_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gate_model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    mean_loss = total_loss / n_batches

    return mean_loss


def get_model(input_channel, hidden_channels, gnn_layers, dropout, head, device, args):
    """
    Creates a pre-train model with specified parameters.
    
    Args:
        input_channel (int): Dimension of input node features
        hidden_channels (int): Dimension of hidden representations
        gnn_layers (int): Number of GNN layers
        dropout (float): Dropout rate
        head (int): Number of attention heads
        device: Device to place the model on
        args: Arguments containing configuration parameters
        
    Returns:
        BuddyPretrainModule: Initialized model
    """
    
    node_model = TransformerModel(args.hops, input_channel, gnn_layers, head, hidden_channels, dropout, agg=args.nag_agg)
    
    gate_in_dim = hidden_channels
    if 'node' in args.score_input:
        score_in_dim = hidden_channels
    if 'edge' in args.score_input:
        score_in_dim = 8

    score_func = DistanceGatedScoringFunction(score_in_dim, gate_in_dim, gate_in_dim, hidden_channels, args.num_experts, dropout, args)
    score_func.cluster_centers = torch.nn.Parameter(torch.randn(args.num_experts, input_channel))

    pretrain_model = BuddyPretrainModule(node_model, None, score_func, device, args)
    
    return pretrain_model

def prepare_models(args, device):
    """
    Prepares a model with specified parameters and loads pre-trained weights.
    
    Args:
        args: Arguments containing model configuration
        device: Device to place the model on
        
    Returns:
        BuddyPretrainModule: Prepared model with loaded weights
    """
    pretrain_model = get_model(args.in_dim, args.hidden_dim, args.gnn_layers, 
                    args.dropout, args.head, device, args).to(device)
    set_model_state(args.ckpt_dir, args.ckpt_name, args.pretrain_epoch, pretrain_model)

    return pretrain_model
    

def train_and_eval(args, args_1, args_2):
    """
    Main function for training and evaluating the model.
    
    Args:
        args: General arguments for training
        args_1: Arguments for the first model
        args_2: Arguments for the second model
        
    Returns:
        None: Prints evaluation results
    """
    if torch.cuda.is_available():
        print("CUDA is available")
        print("CUDA version:", torch.version.cuda)
        device = torch.device(f"cuda:{args.device}")
    else:
        print("CUDA is not available")
        device = torch.device("cpu")
    print(device)

    seed = args.seed
    set_random_seed(seed)

    # Load dataset
    data_dir = args.downstream_root
    data_name = args.data_name
    try:
        data_ori = torch.load(os.path.join(data_dir, f'{data_name}_fixed_sbert.pt'))
    except:
        try:
            data_ori = torch.load(os.path.join(data_dir, f'{data_name}.pt'))
        except:
            raise ValueError(f'file does not exist: {data_name}')

    # Generate link prediction data
    data = get_link_prediction_data_buddy(data_name, train_ratio=args.train_ratio, valid_ratio=args.val_ratio, K=args.K, seed=args.data_seed,
                                          save_dir=args.link_root)
    data['x'] = data_ori.x

    # Move data to device
    for k, v in data.items():
        if torch.is_tensor(v):
            data[k] = v.to(device)

    # Load pre-trained models
    print('-----------------------model_1------------------------')
    print(args_1)
    print('-----------------------model_2------------------------')
    print(args_2)
    model_1 = prepare_models(args_1, device)
    model_2 = prepare_models(args_2, device)

    # Get logits from both models for training
    feats, labels = get_all_logits(data, model_1, model_2, args)

    # Initialize linear regression model for combining expert outputs
    gate_model = LinearRegressionModel(args_1.num_experts+args_2.num_experts).to(device)

    mrr_list = []
    for run in range(args.runs):
        print('#################################          ', run, '          #################################')

        model_1 = prepare_models(args_1, device)
        model_2 = prepare_models(args_2, device)
        gate_model.reset_parameters(run)
        
        # Freeze pre-trained models
        for param in model_1.parameters():
            param.requires_grad = False
        for param in model_2.parameters():
            param.requires_grad = False
            
        optimizer = torch.optim.Adam(gate_model.parameters(), lr=args.lr, weight_decay=args.l2)

        best_valid = 0
        best_epoch = -1
        count = 0
        
        # Training loop
        for epoch in range(1, 1 + args.epochs):
            loss = train(feats, labels, gate_model, optimizer, args.batch_size, args)
            
            # Evaluate on validation set
            if epoch % args.eval_steps == 0:
                mrr = evaluate_mrr_merge_all(device, data, gate_model, model_1, model_2, split='valid', batch_size=20000, 
                                                          args_1=args_1, args_2=args_2, args=args)
                print('Train Loss: {:.4f}, Valid MRR: {:.4f}'.format(loss, mrr))

                # Save best model
                if best_valid < mrr:
                    best_valid = mrr
                    best_epoch = epoch
                    count = 0
                    model_weights = deepcopy(gate_model.state_dict())
                else:
                    count += 1
                    if count >= args.patience:
                        break 

        print('RUN: {}, Training Stop! Best Valid MRR: {:.4f} at Epoch {}'.format(run, best_valid, best_epoch))
        gate_model.load_state_dict(model_weights)
        print('Best ckpt loaded.')
        
        # Evaluate on test set
        test_mrr = evaluate_mrr_merge_all(device, data, gate_model, model_1, model_2, split='test', batch_size=20000, 
                                                          args_1=args_1, args_2=args_2, args=args)
        mrr_list.append(test_mrr)
        print('RUN-{} | TEST MRR: {:.4f}'.format(run, test_mrr))

    # Print final results
    print('TEST MRR LIST: {}'.format(mrr_list))
    print('TEST MRR | MEAN : {:.4f}, STD: {:.4f}'.format(np.mean(mrr_list), np.std(mrr_list)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, required=True)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--l2', type=float, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--K', type=int, default=100)
    parser.add_argument('--align', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--data_seed', type=int, default=0)
    parser.add_argument('--train_ratio', type=float, default=0.4)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--emb_type', type=str, default='sbert')
    parser.add_argument('--downstream_root', type=str, default='./node_data')
    parser.add_argument('--link_root', type=str, default='./link_data')
    parser.add_argument('--device', type=int, default=0)

    args = parser.parse_args()

    # Read model configs from YAML files
    with open('model_1_config.yaml', 'r') as f:
        config_1 = yaml.safe_load(f)
    args_1 = Namespace(**config_1)

    with open('model_2_config.yaml', 'r') as f:
        config_2 = yaml.safe_load(f)
    args_2 = Namespace(**config_2)

    train_and_eval(args, args_1, args_2)