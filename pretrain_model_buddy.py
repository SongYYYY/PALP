import dgl
import torch
import torch.nn.functional as F
import math
import copy
from torch import Tensor 
import numpy as np
import torch.nn as nn
from dgl import GCNNorm
from torch.cuda.amp import GradScaler, autocast
import torch_sparse
from torch_sparse import SparseTensor

class BuddyPretrainModule(torch.nn.Module):
    def __init__(self, node_model, edge_model, score_model, device, args):
        super(BuddyPretrainModule, self).__init__()
        self.node_model = node_model
        self.edge_model = edge_model
        self.score_func = score_model
        self.device = device
        self.temperature = args.temperature
        self.score_input = args.score_input
        self.align = args.align
    
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):

        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)


    def forward(self, data):
        device = self.device
        if len(data) == 6:
            x, mp_link, pos_link, neg_link, pos_feats, neg_feats  = data
            graph_emb = None
        elif len(data) == 7:
            x, mp_link, pos_link, neg_link, pos_feats, neg_feats, graph_emb  = data
        else:
            raise ValueError(f'batch data length mismatch.')
        x, mp_link, pos_link, neg_link = x.to(device), mp_link.to(device), pos_link.to(device), neg_link.to(device)
        pos_feats, neg_feats = pos_feats.to(device), neg_feats.to(device)
        mp_link = mp_link.t() # (2, E)
        adj = SparseTensor.from_edge_index(mp_link, torch.ones(mp_link.shape[1]).to(device), [x.shape[0], x.shape[0]])
        if graph_emb is not None:
            graph_emb = graph_emb.to(device)
        loss = self.p_link(x, adj, pos_link, neg_link, pos_feats, neg_feats, graph_emb)

        return loss


    def p_link(self, x, adj, pos_link, neg_link, pos_feats, neg_feats, graph_emb=None):
        x_ori = x.clone()
        if self.align:
            miu = x.mean(dim=0, keepdim=True)
            x = x - miu
            
        h = self.node_model(x, adj) # (N, d)  

        pos_u, pos_v = pos_link[:, 0], pos_link[:, 1]
        neg_u, neg_v = neg_link[:, 0], neg_link[:, 1]
        
        # Node embeddings for positive and negative links
        h_pos_u, h_pos_v = h[pos_u], h[pos_v]
        h_neg_u, h_neg_v = h[neg_u], h[neg_v]
        h_pos = h_pos_u * h_pos_v
        h_neg = h_neg_u * h_neg_v

        h_all = torch.cat([h_pos, h_neg], dim=0)  # (P+N, d)
        e_all = torch.cat([pos_feats, neg_feats], dim=0) # (P+N, d)

        score_input = []
        if 'node' in self.score_input:
            score_input.append(h_all)
        if 'edge' in self.score_input:
            score_input.append(e_all)
        h_score = torch.concat(score_input, dim=1)

        
        gate_input = []
        x_pos_u, x_pos_v = x_ori[pos_u], x_ori[pos_v]
        x_neg_u, x_neg_v = x_ori[neg_u], x_ori[neg_v]
        if 'node_diff' in self.gate_input:
            # Element-wise difference of node embeddings
            x_pos = torch.abs(x_pos_u - x_pos_v)  # (P, d)
            x_neg = torch.abs(x_neg_u - x_neg_v)  # (N, d)
            x_gate = torch.cat([x_pos, x_neg], dim=0)     # (P+N, d)
            gate_input.append(x_gate)
        if 'node_sum' in self.gate_input:
            # Element-wise sum of node embeddings
            x_pos = x_pos_u + x_pos_v  # (P, d)
            x_neg = x_neg_u + x_neg_v  # (N, d)
            x_gate = torch.cat([x_pos, x_neg], dim=0)     # (P+N, d)
            gate_input.append(x_gate)
        if 'node_product' in self.gate_input:
            # Element-wise product of node embeddings
            x_pos = x_pos_u * x_pos_v  # (P, d)
            x_neg = x_neg_u * x_neg_v  # (N, d)
            x_gate = torch.cat([x_pos, x_neg], dim=0)     # (P+N, d)
            gate_input.append(x_gate)
        
        h_gate = torch.concat(gate_input, dim=1) # (P+N, d1+d2+...)
        
        logits = self.score_func(score_input=h_score, gate_input=h_gate, return_prob=False, temperature=self.temperature)
        logits = logits.squeeze()
        target = torch.cat([torch.ones(pos_link.shape[0]), torch.zeros(neg_link.shape[0])]).to(self.device) # (P+N)
        loss = F.binary_cross_entropy_with_logits(logits, target)

        return loss

    
    def get_all_output(self, x, mp_link, pos_link, neg_link, pos_feats, neg_feats, graph_emb=None):
        mp_link = mp_link.t() # (2, E)
        adj = SparseTensor.from_edge_index(mp_link, torch.ones(mp_link.shape[1]).to(self.device), [x.shape[0], x.shape[0]])

        if self.align:
            miu = x.mean(dim=0, keepdim=True)
            x = x - miu

        h = self.node_model(x, adj) # (N, d)  

        pos_u, pos_v = pos_link[:, 0], pos_link[:, 1]
        neg_u, neg_v = neg_link[:, 0], neg_link[:, 1]
        
        # Node embeddings for positive and negative links
        h_pos_u, h_pos_v = h[pos_u], h[pos_v]
        h_neg_u, h_neg_v = h[neg_u], h[neg_v]
        h_pos = h_pos_u * h_pos_v
        h_neg = h_neg_u * h_neg_v

        h_all = torch.cat([h_pos, h_neg], dim=0)  # (P+N, d)
        e_all = torch.cat([pos_feats, neg_feats], dim=0) # (P+N, d)

        score_input = []
        if 'node' in self.score_input:
            score_input.append(h_all)
        if 'edge' in self.score_input:
            score_input.append(e_all)
        h_score = torch.concat(score_input, dim=1)

        logits = self.score_func.get_expert_outputs(score_input=h_score, return_prob=False) # (P+N, num_experts)

        return logits

