import torch
import torch.nn.functional as F
import torch.nn as nn

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.input_dim = input_dim
        self.linear = nn.Linear(input_dim, 1, bias=False) 

    def reset_parameters(self, k=0):
        with torch.no_grad():
            one_hot_matrix = torch.zeros_like(self.linear.weight)  
            one_hot_matrix[:, k] = 1 
            self.linear.weight.copy_(one_hot_matrix)

        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)  

    def forward(self, expert_outputs, return_prob=True):
        if return_prob:
            out =  torch.sigmoid(self.linear(expert_outputs))
        else:
            out =  self.linear(expert_outputs)
        
        return out
        


