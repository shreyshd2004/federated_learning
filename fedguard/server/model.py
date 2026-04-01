"""
Server-side global model management.

Handles:
- Weight serialisation / deserialisation
- MNIST and NSL-KDD test-set evaluation
- Delta reconstruction (when nodes send compressed deltas)
"""
import io
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.model_def import get_model, NSLKDD_INPUT_DIM

log = logging.getLogger(__name__)

DATA_DIR = os.environ.get("DATA_DIR", "/tmp/fedguard_data")
DATASET  = os.environ.get("DATASET", "mnist").lower()

StateDict = Dict[str, torch.Tensor]


class GlobalModel:
    """Thread-safe global model with evaluation capability."""

    def __init__(self, dataset: str = "mnist"):
        self.dataset = dataset.lower()
        self._feature_dim: Optional[int] = None
        self.model: nn.Module = self._build_model()
        self._eval_loader: Optional[DataLoader] = None

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def _build_model(self, feature_dim: Optional[int] = None) -> nn.Module:
        if self.dataset == "nslkdd":
            dim = feature_dim or self._feature_dim or NSLKDD_INPUT_DIM
            return get_model("nslkdd", input_dim=dim)
        return get_model("mnist")

    def reinitialise(self, feature_dim: Optional[int] = None) -> None:
        if feature_dim:
            self._feature_dim = feature_dim
        self.model = self._build_model(feature_dim)
        self._eval_loader = None
        log.info("Global model re-initialised (dataset=%s)", self.dataset)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def get_weights_bytes(self) -> bytes:
        buf = io.BytesIO()
        torch.save(self.model.state_dict(), buf)
        return buf.getvalue()

    def get_state_dict(self) -> StateDict:
        return {k: v.clone() for k, v in self.model.state_dict().items()}

    def set_state_dict(self, state_dict: StateDict) -> None:
        self.model.load_state_dict(state_dict)

    # ------------------------------------------------------------------
    # Delta reconstruction
    # ------------------------------------------------------------------

    def apply_delta(self, avg_delta: StateDict) -> None:
        """w_global ← w_global + avg_delta  (used with compressed uploads)."""
        new_sd = {
            k: self.model.state_dict()[k].float() + avg_delta[k].float()
            for k in avg_delta
        }
        self.model.load_state_dict(new_sd)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _get_eval_loader(self) -> DataLoader:
        if self._eval_loader is not None:
            return self._eval_loader

        if self.dataset == "mnist":
            from torchvision import datasets, transforms
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])
            ds = datasets.MNIST(
                root=DATA_DIR, train=False, download=True, transform=transform
            )
            self._eval_loader = DataLoader(ds, batch_size=512, shuffle=False)

        elif self.dataset == "nslkdd":
            import numpy as np
            from torch.utils.data import TensorDataset
            # Reuse the node-side preprocessing for the test split
            sys.path.insert(0, str(Path(__file__).parent.parent / "node"))
            try:
                from data_loader import _download_nslkdd, _preprocess_nslkdd
                cache_dir = os.path.join(DATA_DIR, "nslkdd")
                _, test_path = _download_nslkdd(cache_dir)
                X, y = _preprocess_nslkdd(test_path)
                self._feature_dim = X.shape[1]
                ds = TensorDataset(
                    torch.tensor(X, dtype=torch.float32),
                    torch.tensor(y, dtype=torch.long),
                )
                self._eval_loader = DataLoader(ds, batch_size=512, shuffle=False)
            except Exception as exc:
                log.warning("NSL-KDD eval loader failed: %s; accuracy will be 0", exc)
                return None

        return self._eval_loader

    def evaluate(self) -> float:
        """Return accuracy (0–1) on the held-out test set."""
        loader = self._get_eval_loader()
        if loader is None:
            return 0.0

        self.model.eval()
        correct = total = 0
        with torch.no_grad():
            for X, y in loader:
                preds = self.model(X).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += len(y)

        acc = correct / total if total > 0 else 0.0
        log.info("Global model accuracy: %.4f (%d/%d)", acc, correct, total)
        return acc
