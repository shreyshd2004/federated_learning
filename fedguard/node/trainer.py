"""
Local training for FedGuard edge nodes.

Supports two algorithms
-----------------------
FedAvg  (mu=0)   : standard local SGD
FedProx (mu>0)   : adds a proximal penalty ½μ‖w − w_global‖² that limits
                   how far local weights drift from the global model.
                   Critical for convergence on non-IID data.

Optionally wraps training in Differential Privacy via Opacus (DP-SGD):
- Per-sample gradient clipping (max_grad_norm)
- Gaussian noise injection calibrated to (ε, δ) privacy budget
- Tracks epsilon spent after each epoch

References
----------
Li et al., "Federated Optimization in Heterogeneous Networks" (FedProx, 2020)
Abadi et al., "Deep Learning with Differential Privacy" (DP-SGD, 2016)
Opacus: https://opacus.ai
"""
import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core training loop
# ---------------------------------------------------------------------------

def train_local(
    model: nn.Module,
    dataloader: DataLoader,
    global_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    epochs: int = 2,
    lr: float = 0.01,
    mu: float = 0.0,
    enable_dp: bool = False,
    dp_epsilon: float = 10.0,
    dp_delta: float = 1e-5,
    dp_max_grad_norm: float = 1.0,
) -> Tuple[Dict[str, torch.Tensor], dict]:
    """
    Train *model* on *dataloader* and return (updated_state_dict, metadata).

    Args:
        model:            Model initialised with global weights.
        dataloader:       Node's private data shard.
        global_state_dict: Global weights (needed for FedProx proximal term).
                           If None, proximal term is skipped even if mu > 0.
        epochs:           Local training epochs per federated round.
        lr:               SGD learning rate.
        mu:               FedProx proximal coefficient. 0.0 = vanilla FedAvg.
        enable_dp:        Whether to enable DP-SGD via Opacus.
        dp_epsilon:       Target privacy budget ε.
        dp_delta:         Target δ (typically 1/|dataset|).
        dp_max_grad_norm: Per-sample gradient clip norm.

    Returns:
        state_dict:  Trained weights (raw data never included).
        metadata:    Dict with training stats and privacy accounting.
    """
    if enable_dp:
        return _train_with_dp(
            model, dataloader, global_state_dict,
            epochs, lr, mu, dp_epsilon, dp_delta, dp_max_grad_norm,
        )
    return _train_standard(model, dataloader, global_state_dict, epochs, lr, mu)


# ---------------------------------------------------------------------------
# Standard (non-DP) training
# ---------------------------------------------------------------------------

def _train_standard(
    model: nn.Module,
    dataloader: DataLoader,
    global_state_dict: Optional[Dict[str, torch.Tensor]],
    epochs: int,
    lr: float,
    mu: float,
) -> Tuple[Dict[str, torch.Tensor], dict]:
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # Freeze a snapshot of global weights for the proximal term
    global_params: Dict[str, torch.Tensor] = {}
    if mu > 0 and global_state_dict is not None:
        global_params = {k: v.detach().float() for k, v in global_state_dict.items()}

    model.train()
    total_loss = correct = total = 0

    for epoch in range(epochs):
        epoch_loss = epoch_correct = epoch_total = 0

        for X, y in dataloader:
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)

            # FedProx proximal term
            if global_params:
                prox = sum(
                    ((p.float() - global_params[n]) ** 2).sum()
                    for n, p in model.named_parameters()
                    if n in global_params
                )
                loss = loss + (mu / 2.0) * prox

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(y)
            preds = logits.argmax(dim=1)
            epoch_correct += (preds == y).sum().item()
            epoch_total += len(y)

        epoch_acc = epoch_correct / max(epoch_total, 1)
        log.info("Epoch %d/%d | loss=%.4f acc=%.4f (mu=%.3f)",
                 epoch + 1, epochs, epoch_loss / max(epoch_total, 1), epoch_acc, mu)
        total_loss += epoch_loss
        correct += epoch_correct
        total += epoch_total

    metadata = {
        "algorithm": "fedprox" if mu > 0 else "fedavg",
        "mu": mu,
        "epochs": epochs,
        "local_loss": round(total_loss / max(total, 1), 6),
        "local_accuracy": round(correct / max(total, 1), 4),
        "samples": total // epochs,
        "dp_enabled": False,
        "epsilon_spent": None,
        "delta": None,
    }
    return model.state_dict(), metadata


# ---------------------------------------------------------------------------
# DP-SGD training (Opacus)
# ---------------------------------------------------------------------------

def _train_with_dp(
    model: nn.Module,
    dataloader: DataLoader,
    global_state_dict: Optional[Dict[str, torch.Tensor]],
    epochs: int,
    lr: float,
    mu: float,
    target_epsilon: float,
    target_delta: float,
    max_grad_norm: float,
) -> Tuple[Dict[str, torch.Tensor], dict]:
    try:
        from opacus import PrivacyEngine
        from opacus.validators import ModuleValidator
    except ImportError:
        log.warning("Opacus not installed; falling back to non-DP training")
        return _train_standard(model, dataloader, global_state_dict, epochs, lr, mu)

    # Opacus requires certain module properties (no BatchNorm in-place issues)
    if not ModuleValidator.is_valid(model):
        model = ModuleValidator.fix(model)
        log.info("Model patched by Opacus ModuleValidator")

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    privacy_engine = PrivacyEngine()
    model, optimizer, dp_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=dataloader,
        epochs=epochs,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        max_grad_norm=max_grad_norm,
    )

    # Global params for proximal term (parameter names gain _module. prefix)
    global_params: Dict[str, torch.Tensor] = {}
    if mu > 0 and global_state_dict is not None:
        global_params = {f"_module.{k}": v.detach().float()
                         for k, v in global_state_dict.items()}

    model.train()
    total_loss = correct = total = 0

    for epoch in range(epochs):
        epoch_loss = epoch_correct = epoch_total = 0

        for X, y in dp_loader:
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)

            if global_params:
                prox = sum(
                    ((p.float() - global_params[n]) ** 2).sum()
                    for n, p in model.named_parameters()
                    if n in global_params
                )
                loss = loss + (mu / 2.0) * prox

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(y)
            preds = logits.argmax(dim=1)
            epoch_correct += (preds == y).sum().item()
            epoch_total += len(y)

        eps_spent = privacy_engine.get_epsilon(target_delta)
        epoch_acc = epoch_correct / max(epoch_total, 1)
        log.info(
            "Epoch %d/%d | loss=%.4f acc=%.4f ε=%.4f δ=%.2e",
            epoch + 1, epochs,
            epoch_loss / max(epoch_total, 1),
            epoch_acc, eps_spent, target_delta,
        )
        total_loss += epoch_loss
        correct += epoch_correct
        total += epoch_total

    epsilon_spent = privacy_engine.get_epsilon(target_delta)

    # Strip _module. prefix so state_dict matches the global model
    raw_sd = model.state_dict()
    clean_sd = {k.replace("_module.", "", 1): v for k, v in raw_sd.items()}

    metadata = {
        "algorithm": "dp-fedprox" if mu > 0 else "dp-fedavg",
        "mu": mu,
        "epochs": epochs,
        "local_loss": round(total_loss / max(total, 1), 6),
        "local_accuracy": round(correct / max(total, 1), 4),
        "samples": total // epochs,
        "dp_enabled": True,
        "epsilon_spent": round(epsilon_spent, 4),
        "delta": target_delta,
        "max_grad_norm": max_grad_norm,
    }
    return clean_sd, metadata
