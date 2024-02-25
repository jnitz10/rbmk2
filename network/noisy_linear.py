import torch
from torch import nn
import torch.nn.functional as F
from math import sqrt


class NoisyLinear(nn.Module):
    """Noisy linear module.  Allows network to utilize noisy linear
    layers for exploration instead of epsilon-greedy exploration.

    :param in_features: int, input size of module
    :param out_features: int, output size of module
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            std_init: float = 0.5
    ):
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer(
            'weight_epsilon', torch.Tensor(out_features, in_features)
        )

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer(
            'bias_epsilon', torch.Tensor(out_features)
        )

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / sqrt(self.in_features)
        )
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.std_init / sqrt(self.out_features)
        )

    def reset_noise(self):
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        self.weight_epsilon.copy_(torch.outer(epsilon_out, epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon
        )

    @staticmethod
    def scale_noise(size):
        x = torch.randn(size)

        return x.sign().mul(x.abs().sqrt())
