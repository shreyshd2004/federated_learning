"""
Local training loop for a FedGuard edge node.
"""
import logging
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)


def train_local(
    model: nn.Module,
    dataloader: DataLoader,
    epochs: int = 2,
    lr: float = 0.01,
) -> Dict[str, torch.Tensor]:
    """
    Train *model* on *dataloader* for *epochs* passes.

    Args:
        model:      The model to fine-tune (weights already loaded from server).
        dataloader: Node's private data partition.
        epochs:     Number of local epochs per federated round.
        lr:         Learning rate for SGD.

    Returns:
        Updated state_dict (weights only — raw data never leaves the node).
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for X, y in dataloader:
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(y)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += len(y)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        log.info(
            "Epoch %d/%d — loss=%.4f acc=%.4f",
            epoch + 1,
            epochs,
            epoch_loss,
            epoch_acc,
        )

    return model.state_dict()
