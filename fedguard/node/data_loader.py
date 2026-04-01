"""
Data loading for FedGuard edge nodes.

Supports two datasets
---------------------
mnist   : classic digit classification; clean, balanced, easy to debug
nslkdd  : NSL-KDD network intrusion dataset; tabular, class-imbalanced,
           realistic for anomaly detection

Partitioning strategy
---------------------
IID  (alpha → ∞): equal random splits; trivial convergence baseline
Non-IID (Dirichlet α):
  Each node's label distribution is drawn from Dir(α).
  α = 0.1 → extreme skew (each node sees ~1 class)
  α = 1.0 → moderate skew
  α = 100  → near-IID

Set env var DIRICHLET_ALPHA to control skew.
"""
import io
import logging
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).parent.parent))

log = logging.getLogger(__name__)

DATA_DIR = os.environ.get("DATA_DIR", "/tmp/fedguard_data")


# ---------------------------------------------------------------------------
# Dirichlet partitioning (works for any labelled dataset)
# ---------------------------------------------------------------------------

def dirichlet_partition(
    labels: np.ndarray,
    node_id: int,
    total_nodes: int,
    alpha: float,
    seed: int = 42,
) -> np.ndarray:
    """
    Return sample indices assigned to *node_id* via Dirichlet partitioning.

    Each class's samples are split across nodes according to proportions
    drawn from Dir(alpha).  Small alpha → high label skew.
    """
    rng = np.random.default_rng(seed)
    num_classes = int(labels.max()) + 1
    node_indices = []

    for c in range(num_classes):
        class_indices = np.where(labels == c)[0]
        rng.shuffle(class_indices)

        # Draw proportions for all nodes from Dirichlet
        proportions = rng.dirichlet(np.repeat(alpha, total_nodes))
        # Convert to split points
        splits = (np.cumsum(proportions) * len(class_indices)).astype(int)
        splits = np.clip(splits, 0, len(class_indices))

        # Assign this node's slice
        start = splits[node_id - 2] if node_id > 1 else 0
        end = splits[node_id - 1]
        node_indices.extend(class_indices[start:end].tolist())

    return np.array(node_indices)


# ---------------------------------------------------------------------------
# MNIST
# ---------------------------------------------------------------------------

def _mnist_loader(
    node_id: int,
    total_nodes: int,
    alpha: float,
    batch_size: int,
) -> DataLoader:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    full_dataset = datasets.MNIST(
        root=DATA_DIR, train=True, download=True, transform=transform
    )
    labels = np.array(full_dataset.targets)

    if alpha <= 0:
        # Strict IID uniform split
        n = len(full_dataset)
        shard = n // total_nodes
        start = (node_id - 1) * shard
        end = start + shard if node_id < total_nodes else n
        indices = list(range(start, end))
    else:
        indices = dirichlet_partition(labels, node_id, total_nodes, alpha)

    log.info(
        "MNIST shard: node=%d  samples=%d  alpha=%.2f  "
        "class_dist=%s",
        node_id,
        len(indices),
        alpha,
        dict(zip(*np.unique(labels[indices], return_counts=True))),
    )
    return DataLoader(Subset(full_dataset, indices), batch_size=batch_size, shuffle=True)


# ---------------------------------------------------------------------------
# NSL-KDD
# ---------------------------------------------------------------------------

_NSLKDD_TRAIN_URL = (
    "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt"
)
_NSLKDD_TEST_URL = (
    "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt"
)

_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty",
]

_CATEGORICAL = ["protocol_type", "service", "flag"]

_LABEL_MAP = {
    # normal
    "normal": 0,
    # DoS attacks
    "back": 1, "land": 1, "neptune": 1, "pod": 1, "smurf": 1,
    "teardrop": 1, "apache2": 1, "udpstorm": 1, "processtable": 1, "worm": 1,
    # Probe attacks
    "ipsweep": 2, "nmap": 2, "portsweep": 2, "satan": 2, "mscan": 2, "saint": 2,
    # R2L attacks
    "ftp_write": 3, "guess_passwd": 3, "imap": 3, "multihop": 3, "phf": 3,
    "spy": 3, "warezclient": 3, "warezmaster": 3, "sendmail": 3, "named": 3,
    "snmpgetattack": 3, "snmpguess": 3, "xlock": 3, "xsnoop": 3, "httptunnel": 3,
    # U2R attacks
    "buffer_overflow": 4, "loadmodule": 4, "perl": 4, "rootkit": 4,
    "mailbomb": 4, "ps": 4, "sqlattack": 4, "xterm": 4,
}


def _download_nslkdd(cache_dir: str) -> Tuple[str, str]:
    """Download NSL-KDD train/test files if not cached."""
    import urllib.request
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    train_path = os.path.join(cache_dir, "KDDTrain+.txt")
    test_path = os.path.join(cache_dir, "KDDTest+.txt")
    for url, path in [(_NSLKDD_TRAIN_URL, train_path), (_NSLKDD_TEST_URL, test_path)]:
        if not os.path.exists(path):
            log.info("Downloading %s → %s", url, path)
            urllib.request.urlretrieve(url, path)
    return train_path, test_path


def _preprocess_nslkdd(path: str):
    """Load and preprocess NSL-KDD CSV → (X: np.float32, y: np.int64)."""
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler

    df = pd.read_csv(path, header=None, names=_COLUMNS)
    df = df.drop(columns=["difficulty"])

    # Map labels to integers (unknown labels → normal)
    df["label"] = df["label"].str.strip(".").map(_LABEL_MAP).fillna(0).astype(int)

    # One-hot encode categorical features
    df = pd.get_dummies(df, columns=_CATEGORICAL)

    y = df["label"].values.astype(np.int64)
    X = df.drop(columns=["label"]).values.astype(np.float32)

    # Min-max scale numeric features
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    return X, y


def _nslkdd_loader(
    node_id: int,
    total_nodes: int,
    alpha: float,
    batch_size: int,
) -> Tuple[DataLoader, int]:
    """Returns (DataLoader, feature_dim) for this node's NSL-KDD shard."""
    cache_dir = os.path.join(DATA_DIR, "nslkdd")
    train_path, _ = _download_nslkdd(cache_dir)

    X, y = _preprocess_nslkdd(train_path)
    feature_dim = X.shape[1]

    if alpha <= 0:
        n = len(y)
        shard = n // total_nodes
        start = (node_id - 1) * shard
        end = start + shard if node_id < total_nodes else n
        indices = np.arange(start, end)
    else:
        indices = dirichlet_partition(y, node_id, total_nodes, alpha)

    X_node = torch.tensor(X[indices])
    y_node = torch.tensor(y[indices])
    dataset = TensorDataset(X_node, y_node)

    log.info(
        "NSL-KDD shard: node=%d  samples=%d  features=%d  alpha=%.2f",
        node_id, len(indices), feature_dim, alpha,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True), feature_dim


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_data_loader(
    node_id: int,
    total_nodes: int = 3,
    dataset: str = "mnist",
    alpha: float = 0.5,
    batch_size: int = 64,
) -> Tuple[DataLoader, int]:
    """
    Return (DataLoader, feature_dim) for this node's private shard.

    Args:
        node_id:     1-indexed node identifier.
        total_nodes: Number of nodes in the federation.
        dataset:     "mnist" or "nslkdd".
        alpha:       Dirichlet concentration.  ≤0 → IID uniform split.
        batch_size:  Mini-batch size.

    Returns:
        (DataLoader, feature_dim); feature_dim=784 for MNIST, ~122 for NSL-KDD.
    """
    dataset = dataset.lower()
    if dataset == "mnist":
        loader = _mnist_loader(node_id, total_nodes, alpha, batch_size)
        return loader, 784
    if dataset == "nslkdd":
        return _nslkdd_loader(node_id, total_nodes, alpha, batch_size)
    raise ValueError(f"Unknown dataset '{dataset}'")
