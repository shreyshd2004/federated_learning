"""
Shared model architectures for FedGuard.
Mounted into both server and node containers.

Models
------
SimpleMLP    — MNIST classification (784 → 128 → 10)
NSLKDDMLP   — NSL-KDD intrusion detection (dynamic_dim → 256 → 128 → 5)
"""
import torch
import torch.nn as nn

# NSL-KDD feature dimension after one-hot encoding categorical columns.
# protocol_type(3) + service(70) + flag(11) + 38 numeric = 122
NSLKDD_INPUT_DIM = 122
NSLKDD_NUM_CLASSES = 5   # normal, dos, probe, r2l, u2r


class SimpleMLP(nn.Module):
    """2-layer MLP for MNIST (784 → 128 → 10)."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class NSLKDDMLP(nn.Module):
    """
    3-layer MLP for NSL-KDD network intrusion detection.
    Deeper than SimpleMLP to handle the richer, mixed-type feature space.
    Dropout regularisation counters the class imbalance typical in IDS datasets.
    """

    def __init__(self, input_dim: int = NSLKDD_INPUT_DIM, num_classes: int = NSLKDD_NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def get_model(dataset: str = "mnist", **kwargs) -> nn.Module:
    """
    Factory: return a fresh, randomly-initialised model for the given dataset.

    Args:
        dataset: "mnist" or "nslkdd"
        **kwargs: forwarded to NSLKDDMLP (e.g. input_dim, num_classes)
    """
    dataset = dataset.lower()
    if dataset == "mnist":
        return SimpleMLP()
    if dataset == "nslkdd":
        return NSLKDDMLP(**kwargs)
    raise ValueError(f"Unknown dataset '{dataset}'. Choose 'mnist' or 'nslkdd'.")
