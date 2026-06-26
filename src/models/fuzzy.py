import torch
import torch.nn as nn

class GaussianMembership(nn.Module):
    def __init__(self, num_features, num_rules):
        super(GaussianMembership, self).__init__()
        self.mu = nn.Parameter(torch.randn(num_rules, num_features))
        self.sigma = nn.Parameter(torch.ones(num_rules, num_features))

    def forward(self, x):
        x = x.unsqueeze(1)
        return torch.exp(-0.5 * ((x - self.mu) / (self.sigma ** 2 + 1e-6)) ** 2)

class NeuroFuzzyLayer(nn.Module):
    def __init__(self, input_size, num_rules):
        super(NeuroFuzzyLayer, self).__init__()
        self.membership = GaussianMembership(input_size, num_rules)
        self.rule_weights = nn.Parameter(torch.randn(num_rules, 1))

    def forward(self, x):
        mu_x = self.membership(x)
        firing_strength = torch.prod(mu_x, dim=2)
        normalized_firing = firing_strength / (torch.sum(firing_strength, dim=1, keepdim=True) + 1e-6)
        return torch.matmul(normalized_firing, self.rule_weights)