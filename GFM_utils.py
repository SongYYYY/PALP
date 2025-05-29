import numpy as np
import random
from torch_sparse import SparseTensor
import torch

def reset_gnn_weights(model):
    for m in model.modules():
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()


def set_random_seed(seed):
    # Set the random seed for NumPy
    np.random.seed(seed)

    # Set the random seed for Python
    random.seed(seed)

    # Set the random seed for PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

from torch_sparse import SparseTensor, matmul, sum as sparse_sum, fill_diag, mul

def calculate_gcn_laplacian(adj: SparseTensor) -> SparseTensor:
    """
    Calculate the normalized Laplacian matrix used in GCN.
    L = D^(-1/2) * (A + I) * D^(-1/2)
    where A is the adjacency matrix and D is the degree matrix.

    Args:
    adj (SparseTensor): Adjacency matrix as a SparseTensor.

    Returns:
    SparseTensor: Normalized Laplacian matrix as a SparseTensor.
    """
    # Get the number of nodes
    num_nodes = adj.size(0)

    # Add self-loops to the adjacency matrix
    adj_with_self_loops = fill_diag(adj, 1.0)

    # Calculate the degree matrix
    deg = sparse_sum(adj_with_self_loops, dim=1)

    # Calculate D^(-1/2)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt.isinf()] = 0

    # Calculate L = D^(-1/2) * (A + I) * D^(-1/2)
    laplacian = adj_with_self_loops
    laplacian = mul(laplacian, deg_inv_sqrt.view(-1, 1))
    laplacian = mul(laplacian, deg_inv_sqrt.view(1, -1))

    return laplacian

def construct_features_sparse(adj_matrix: SparseTensor, X: torch.Tensor, K: int) -> torch.Tensor:
    """
    Construct features by iteratively propagating through the graph.

    Args:
    adj_matrix (SparseTensor): Adjacency matrix as a SparseTensor (normalized Laplacian).
    X (torch.Tensor): Initial node features of shape [N, d].
    K (int): Number of hops to propagate.

    Returns:
    torch.Tensor: Propagated features of shape [N, K+1, d].
    """
    # Initialize X_new with self-features
    X_new = [X]

    # Current features for propagation
    current_features = X

    # Iteratively propagate features
    for _ in range(K):
        # SparseTensor matrix multiplication for feature propagation
        current_features = matmul(adj_matrix, current_features)
        X_new.append(current_features)

    # Concatenate along a new dimension to form [N, K+1, d]
    X_new = torch.stack(X_new, dim=1)

    return X_new