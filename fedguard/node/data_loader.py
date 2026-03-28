"""
Data loading for FedGuard nodes.

MNIST is partitioned into non-overlapping shards.
Each node receives exactly one shard, simulating private data.

Partition strategy
------------------
- 3 nodes → dataset split into 3 equal chunks by sample index
- NODE_ID (1-indexed) selects which chunk this node uses
- Partitioning is deterministic (same seed everywhere)
"""
import os
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_data_loader(
    node_id: int,
    total_nodes: int = 3,
    batch_size: int = 64,
    data_dir: str = "/tmp/mnist_data",
) -> DataLoader:
    """
    Return a DataLoader for this node's private data partition.

    Args:
        node_id:     1-indexed node identifier.
        total_nodes: Total number of nodes in the federation.
        batch_size:  Mini-batch size for local training.
        data_dir:    Directory where MNIST will be cached.

    Returns:
        DataLoader over this node's shard.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    n = len(dataset)
    shard_size = n // total_nodes
    start = (node_id - 1) * shard_size
    # Last node gets any remainder samples
    end = start + shard_size if node_id < total_nodes else n
    indices = list(range(start, end))

    shard = Subset(dataset, indices)
    return DataLoader(shard, batch_size=batch_size, shuffle=True)
