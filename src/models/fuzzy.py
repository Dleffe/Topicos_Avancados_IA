import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianMembership(nn.Module):
    def __init__(self, num_features, num_rules):
        super(GaussianMembership, self).__init__()
        self.mu        = nn.Parameter(torch.randn(num_rules, num_features))
        self.log_sigma = nn.Parameter(torch.full((num_rules, num_features), -1.0))

    def forward(self, x):
        x     = x.unsqueeze(1)
        sigma = F.softplus(self.log_sigma) + 0.05
        return torch.exp(-0.5 * ((x - self.mu) / sigma) ** 2)

class NeuroFuzzyLayer(nn.Module):
    def __init__(self, input_size, num_rules, n_proj=4):
        super(NeuroFuzzyLayer, self).__init__()
        self.proj         = nn.Linear(input_size, n_proj)
        self.norm         = nn.LayerNorm(n_proj)
        self.membership   = GaussianMembership(n_proj, num_rules)
        self.rule_weights = nn.Parameter(torch.randn(num_rules, 1))

    def forward(self, x):
        x               = self.norm(self.proj(x))
        mu_x            = self.membership(x)
        firing_strength = torch.prod(mu_x, dim=2)
        normalized      = firing_strength / (firing_strength.sum(dim=1, keepdim=True) + 1e-9)
        return torch.matmul(normalized, self.rule_weights)