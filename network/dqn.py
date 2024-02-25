import torch
from torch import nn
import torch.nn.functional as F
from .noisy_linear import NoisyLinear


class DQN(nn.Module):
    def __init__(
            self,
            out_dim: int,
            n_atoms: int,
            features_dim: int = 512
    ):
        super(DQN, self).__init__()

        self.out_dim = out_dim
        self.n_atoms = n_atoms

        self.convs = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.advantage_hidden = NoisyLinear(3136, features_dim)
        self.advantage_out = NoisyLinear(features_dim, out_dim * n_atoms)

        self.value_hidden = NoisyLinear(3136, features_dim)
        self.value_out = NoisyLinear(features_dim, n_atoms)

    def forward(self, x, log=False) -> torch.Tensor:
        """
        Forward pass of the model.

        Computes Q-value distributions for each action.

        Parameters:
        - x (torch.Tensor): The input tensor of shape (batch_size, channels, height, width).
        - log (bool, optional): If True, returns the log-softmax of the Q-value distributions;
          otherwise, returns the softmax. Defaults to False.

        Returns:
        - torch.Tensor: The Q-value distributions of shape (batch_size, n_actions, n_atoms)
          as a softmax or log-softmax tensor, depending on the `log` parameter.
        """
        conv_out = self.convs(x / 255.0)

        adv_hid = F.relu(self.advantage_hidden(conv_out.view(-1, 3136)))
        val_hid = F.relu((self.value_hidden(conv_out.view(-1, 3136))))

        # advantage of each action, shape: batch_size, n_actions, n_atoms
        advantage = self.advantage_out(adv_hid).view(
            -1, self.out_dim, self.n_atoms
        )
        # value of given state, shape: batch_size, 1, n_atoms
        value = self.value_out(val_hid).view(
            -1, 1, self.n_atoms
        )
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)

        if log:
            q = F.log_softmax(q_atoms, dim=2)
        else:
            q = F.softmax(q_atoms, dim=2)

        return q

    def reset_noise(self):
        self.advantage_out.reset_noise()
        self.advantage_hidden.reset_noise()
        self.value_out.reset_noise()
        self.value_hidden.reset_noise()
