import torch
import torch.nn.functional as F
import torch.nn as nn

class DistanceGatedScoringFunction(torch.nn.Module):
    def __init__(self, score_in_channels, gate_in_channels, latent_dim, hidden_channels, num_experts, dropout, args):
        super().__init__()

        # Expert Networks (each is a 3-layer MLP)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(score_in_channels, hidden_channels),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_channels, 1)  
            ) for _ in range(num_experts)
        ])

        self.gate_transform = nn.Sequential(
            nn.Linear(gate_in_channels, latent_dim),
        )

        # Learnable Cluster Centers in latent space
        if args.use_gate_proj:
            self.cluster_centers = nn.Parameter(torch.randn(num_experts, latent_dim))
        else:
            self.cluster_centers = nn.Parameter(torch.randn(num_experts, gate_in_channels))

        self.gating_mlp = nn.Sequential(
            nn.Linear(num_experts, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, num_experts)
        )

        self.dropout = dropout
        self.use_gate_mlp = args.use_gate_mlp
        self.use_hard_sample = args.use_hard_sample
        self.cluster_init_method = args.cluster_init_method
        self.use_gate_proj = args.use_gate_proj

        self.reset_parameters()

    def reset_parameters(self):
        for expert in self.experts:
            for layer in expert:
                if isinstance(layer, nn.Linear):
                    layer.reset_parameters()
        for layer in self.gate_transform:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
        if self.cluster_init_method == 'normal':
            nn.init.normal_(self.cluster_centers)
        elif self.cluster_init_method == 'ortho':
            nn.init.orthogonal_(self.cluster_centers)
        else:
            raise NotImplementedError(f'Unrecognized clutser_init_method: {self.cluster_init_method}.')

        for layer in self.gating_mlp:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

    def forward(self, score_input, gate_input=None, return_prob=True, temperature=1.0):
        if gate_input is None:
            gate_input = score_input

        if self.use_gate_proj:
            gate_latent = self.gate_transform(gate_input)  # Shape: (*, latent_dim)
        else:
            gate_latent = gate_input

        gate_latent_exp = gate_latent.unsqueeze(-2)  # Shape: (*, 1, latent_dim)
        cluster_centers_exp = self.cluster_centers.unsqueeze(0)  # Shape: (1, num_experts, latent_dim)

        # Compute squared Euclidean distances in latent space
        distances = ((gate_latent_exp - cluster_centers_exp) ** 2).sum(-1)  # Shape: (*, num_experts)
        gating_logits = -distances  # Shape: (*, num_experts)

        if self.use_gate_mlp:
            gating_logits = self.gating_mlp(gating_logits)  # Shape: (*, num_experts)

        gating_probs = F.gumbel_softmax(
            gating_logits,
            tau=temperature,
            hard=self.use_hard_sample,
            dim=-1
        )  # Shape: (*, num_experts)

        expert_outputs = []
        for expert in self.experts:
            expert_output = expert(score_input)  # Shape: (*, 1)
            expert_outputs.append(expert_output)
        expert_outputs = torch.cat(expert_outputs, dim=-1)  # Shape: (*, num_experts)

        # Weighted sum of expert outputs using gating probabilities
        output = (expert_outputs * gating_probs).sum(dim=-1, keepdim=True)  # Shape: (*, 1)

        if return_prob:
            output = torch.sigmoid(output)

        return output
        
    def get_expert_outputs(self, score_input, return_prob=True):
        # Compute outputs from each expert
        expert_outputs = []
        for expert in self.experts:
            expert_output = expert(score_input)  # Shape: (batch_size, 1)
            expert_outputs.append(expert_output)
        expert_outputs = torch.cat(expert_outputs, dim=-1)  # Shape: (batch_size, num_experts)

        if return_prob:
            return torch.sigmoid(expert_outputs)
        else:
            return expert_outputs
