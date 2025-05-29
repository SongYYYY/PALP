import torch
from ogb.linkproppred import Evaluator

@torch.no_grad()
def compute_edge_scores(
    score_func, node_emb, edge_emb, src, dst, neg_dst, device, batch_size=4096, 
    args=None,
):
    """Compute Mean Reciprocal Rank (MRR) in batches."""
    all_scores = []
    for start in range(0, src.shape[0], batch_size):
        end = min(start + batch_size, src.shape[0])
        all_dst = torch.cat([dst[start:end, None], neg_dst[start:end]], 1) # (N, K+1)
        h_src = node_emb[src[start:end]][:, None, :].to(device) # (N, 1, d)
        h_dst = node_emb[all_dst.view(-1)].view(*all_dst.shape, -1).to(device) # (N, K+1, d)
        h_src = h_src.expand(h_dst.shape[0], h_dst.shape[1], h_src.shape[2]) # (N, K+1, d)
        h_all = h_src * h_dst # (N, K+1, d)
        e_all = edge_emb[start:end] # (N, K+1, d)

        score_input = []
        if 'node' in args.score_input:
            score_input.append(h_all)
        if 'edge' in args.score_input:
            score_input.append(e_all)
        h_score = torch.concat(score_input, dim=-1)
        
        if args.merge_method == 'logit':
            pred = score_func.get_expert_outputs(score_input=h_score, return_prob=False)
        elif args.merge_method == 'prob':
            pred = score_func.get_expert_outputs(score_input=h_score, return_prob=True)
        else:
            raise NotImplementedError(f'unrecognized merge_method: {args.merge_method}.')

        all_scores.append(pred) # (N, K+1, num_experts)
        continue

    all_scores = torch.cat(all_scores, dim=0) # (N, K+1, num_experts)
    return all_scores


def evaluate_mrr_merge_all(device, data, gate_model, model_1, model_2, split='test', batch_size=4096, args_1=None, args_2=None, args=None):
    model_1.eval()
    model_2.eval()
    gate_model.eval()
    evaluator = Evaluator(name="ogbl-citation2")
    x = data['x'].to(device)
    if args.align:
        miu = x.mean(dim=0, keepdim=True)
        x = x - miu

    adj = data['adj'].to(device)
    src = data[f'{split}_pos'].t()[0].to(device) # test_pos: [n, 2] -> src: [n]
    dst = data[f'{split}_pos'].t()[1].to(device) # dst: [n]
    neg_dst = data[f'{split}_neg'].to(device) # test_neg: [n, K]
    pos_feat = data[f'buddy_{split}_pos'].unsqueeze(1).to(device) # (n, 1, d)
    neg_feat = data[f'buddy_{split}_neg'].to(device) # (n, K, d)
    edge_emb = torch.cat([pos_feat, neg_feat], dim=1) # (n, K+1, d)

    # model_1
    node_model, edge_model, score_func = model_1.node_model, model_1.edge_model, model_1.score_func
    with torch.no_grad():
        node_emb = node_model(x, adj)
        scores_1 = compute_edge_scores(
                score_func, node_emb, edge_emb, src, dst, neg_dst, device, batch_size, args_1
            ) # scores: (N, K+1, num_experts)
        
    # model_2
    node_model, edge_model, score_func = model_2.node_model, model_2.edge_model, model_2.score_func
    with torch.no_grad():
        node_emb = node_model(x, adj)
        scores_2 = compute_edge_scores(
                score_func, node_emb, edge_emb, src, dst, neg_dst, device, batch_size, args_2
            ) # scores: (N, K+1, num_experts)
        
    # merge logits from experts
    scores = torch.cat([scores_1, scores_2], dim=-1) # (N, K+1, 2*num_experts)
    scores = gate_model(scores, return_prob=True) 
    scores = scores.squeeze() # (N, K+1)

    input_dict = {"y_pred_pos": scores[:, 0], "y_pred_neg": scores[:, 1:]}
    mrr = evaluator.eval(input_dict)["mrr_list"]

    return mrr.mean().cpu()


