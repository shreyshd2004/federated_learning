"""
Server-side model management: holds and evaluates the global model.
"""
import sys
import io
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Allow importing shared module whether running inside Docker or locally
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.model_def import SimpleMLP, get_model


class GlobalModel:
    """Wraps the global model with thread-safe weight access and evaluation."""

    def __init__(self):
        self.model: SimpleMLP = get_model()
        self._eval_loader: Optional[DataLoader] = None

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def get_weights_bytes(self) -> bytes:
        """Serialise current state_dict to bytes (for HTTP response)."""
        buf = io.BytesIO()
        torch.save(self.model.state_dict(), buf)
        return buf.getvalue()

    def load_weights_bytes(self, data: bytes) -> None:
        """Deserialise bytes into the global model."""
        buf = io.BytesIO(data)
        state_dict = torch.load(buf, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state_dict)

    def get_state_dict(self):
        return self.model.state_dict()

    def set_state_dict(self, state_dict) -> None:
        self.model.load_state_dict(state_dict)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _get_eval_loader(self) -> DataLoader:
        if self._eval_loader is None:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])
            dataset = datasets.MNIST(
                root="/tmp/mnist_data",
                train=False,
                download=True,
                transform=transform,
            )
            self._eval_loader = DataLoader(dataset, batch_size=256, shuffle=False)
        return self._eval_loader

    def evaluate(self) -> float:
        """Run evaluation on the MNIST test set; return accuracy (0–1)."""
        self.model.eval()
        correct = total = 0
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0

        loader = self._get_eval_loader()
        with torch.no_grad():
            for X, y in loader:
                logits = self.model(X)
                total_loss += criterion(logits, y).item() * len(y)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += len(y)

        return correct / total if total > 0 else 0.0
