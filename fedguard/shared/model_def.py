"""
Shared model architecture for FedGuard.
This file is mounted into both server and node containers.
"""
import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    """
    2-layer MLP for MNIST classification (784 → 128 → 10).
    Used identically on server (for evaluation) and all nodes (for training).
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def get_model() -> SimpleMLP:
    """Return a fresh, randomly-initialised model instance."""
    return SimpleMLP()
