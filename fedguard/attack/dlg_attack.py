"""
Deep Leakage from Gradients (DLG) Attack
=========================================
Given only the gradient tensor a node uploads to the server, this attack
reconstructs the node's *private training images* without any access to
the original data.

Two variants
------------
DLG  (Zhu et al., NeurIPS 2019)
  Simultaneously optimises dummy images AND dummy labels via L-BFGS to
  minimise the distance between dummy and target gradients.

iDLG (Zhao et al., 2020)
  Analytically extracts the ground-truth label from the gradient of the
  last FC layer BEFORE running reconstruction — faster and more reliable.

Why this works
--------------
For a single sample (x, y) and a model f_θ:

    ∇_θ L(f_θ(x), y)  is uniquely determined by x and y.

So inverting ∇ → (x, y) is well-posed when batch_size = 1.
At larger batch sizes the system becomes under-determined, but partial
reconstruction is still possible and alarming.

DP-SGD defence
--------------
Adding calibrated Gaussian noise η ~ N(0, σ²) to the gradient before
upload breaks the reconstruction: the optimiser chases noise rather than
signal.  At ε = 1 (strong privacy) the reconstructed image is indistinguishable
from random noise.

Reference
---------
Zhu, Ligeng, Zhijian Liu, and Song Han.
"Deep leakage from gradients." NeurIPS 2019.
https://arxiv.org/abs/1906.08935
"""
import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import LBFGS

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Label extraction (iDLG)
# ---------------------------------------------------------------------------

def extract_label_idlg(gradients: List[torch.Tensor], num_classes: int = 10) -> int:
    """
    Analytically extract the ground-truth label from uploaded gradients.

    For a single-sample cross-entropy loss, the gradient of the last FC
    weight matrix w.r.t. the pre-softmax logits equals:
        ∂L/∂z_c = p_c - 1[c == y]

    This is negative only at the true class index c = y.
    Averaging over the input features, the most-negative column of the
    last weight gradient identifies the true class.

    Args:
        gradients:   Ordered list of parameter gradients (same order as
                     model.parameters()).
        num_classes: Number of output classes.

    Returns:
        Predicted true label index.
    """
    # Last FC weight is second-to-last gradient (last is bias)
    fc_weight_grad = None
    for g in reversed(gradients):
        if g.dim() == 2:          # weight tensor of a Linear layer
            fc_weight_grad = g
            break

    if fc_weight_grad is None:
        log.warning("Could not find FC weight grad — defaulting to label 0")
        return 0

    # Shape: (num_classes, hidden_dim) — mean over hidden dim, find min
    label = int(fc_weight_grad.mean(dim=1).argmin().item())
    return label


# ---------------------------------------------------------------------------
# Core DLG reconstruction
# ---------------------------------------------------------------------------

def dlg_reconstruct(
    model: nn.Module,
    target_gradients: List[torch.Tensor],
    image_shape: Tuple[int, ...] = (1, 1, 28, 28),
    num_classes: int = 10,
    iterations: int = 300,
    lr: float = 1.0,
    use_idlg: bool = True,
    tv_weight: float = 1e-4,
) -> Tuple[torch.Tensor, Optional[int], List[float]]:
    """
    Reconstruct private training images from uploaded gradients.

    Args:
        model:            Shared model architecture (state loaded from server).
        target_gradients: Gradients the victim node uploaded, as a list of
                          tensors matching model.parameters().
        image_shape:      (batch, C, H, W) of images to reconstruct.
        num_classes:      Output classes (10 for MNIST, 5 for NSL-KDD).
        iterations:       L-BFGS steps (300 is usually enough for MNIST).
        lr:               L-BFGS step size.
        use_idlg:         Extract label analytically (faster) vs jointly optimise.
        tv_weight:        Total-variation regularisation — smooths reconstruction.

    Returns:
        reconstructed:   Tensor of shape image_shape.
        predicted_label: Extracted label (or None if not using iDLG).
        grad_diffs:      ‖∇_dummy − ∇_target‖² per iteration (convergence curve).
    """
    device = next(model.parameters()).device
    target_grads = [g.to(device).detach() for g in target_gradients]

    # Detach model — we only differentiate through it, not update it
    model.eval()

    # ── Dummy inputs ────────────────────────────────────────────────────────
    dummy_data = torch.randn(image_shape, requires_grad=True, device=device)

    predicted_label: Optional[int] = None
    if use_idlg:
        predicted_label = extract_label_idlg(target_grads, num_classes)
        dummy_labels = torch.tensor(
            [predicted_label] * image_shape[0], dtype=torch.long, device=device
        )
        opt_params = [dummy_data]
        label_mode = "hard"
        log.info("iDLG extracted label: %d", predicted_label)
    else:
        dummy_labels = torch.randn(
            image_shape[0], num_classes, requires_grad=True, device=device
        )
        opt_params = [dummy_data, dummy_labels]
        label_mode = "soft"

    optimizer = LBFGS(opt_params, lr=lr, max_iter=20, line_search_fn="strong_wolfe")
    grad_diffs: List[float] = []

    # ── Optimisation loop ────────────────────────────────────────────────────
    def closure():
        optimizer.zero_grad()

        dummy_out = model(dummy_data)

        if label_mode == "hard":
            dummy_loss = F.cross_entropy(dummy_out, dummy_labels)
        else:
            dummy_loss = F.cross_entropy(dummy_out, F.softmax(dummy_labels, dim=-1))

        dummy_grads = torch.autograd.grad(
            dummy_loss, model.parameters(), create_graph=True, allow_unused=True
        )

        # Gradient matching loss
        grad_diff = sum(
            ((dg - tg) ** 2).sum()
            for dg, tg in zip(dummy_grads, target_grads)
            if dg is not None
        )

        # Total-variation regularisation (smooths pixel-level noise)
        if tv_weight > 0 and dummy_data.dim() == 4:
            tv = (
                (dummy_data[:, :, :, 1:] - dummy_data[:, :, :, :-1]).abs().sum()
                + (dummy_data[:, :, 1:, :] - dummy_data[:, :, :-1, :]).abs().sum()
            )
            grad_diff = grad_diff + tv_weight * tv

        grad_diff.backward()
        return grad_diff

    for i in range(iterations):
        loss = optimizer.step(closure)
        val = loss.item() if loss is not None else float("inf")
        grad_diffs.append(round(val, 6))

        if i % 50 == 0:
            log.info("DLG iter %d/%d  grad_diff=%.6f", i, iterations, val)

        if val < 1e-7:
            log.info("DLG converged at iteration %d", i)
            break

    return dummy_data.detach().cpu(), predicted_label, grad_diffs


# ---------------------------------------------------------------------------
# DP noise simulation
# ---------------------------------------------------------------------------

def apply_dp_noise(
    gradients: List[torch.Tensor],
    noise_multiplier: float,
    max_grad_norm: float = 1.0,
) -> List[torch.Tensor]:
    """
    Simulate DP-SGD noise on a gradient list.

    Clips each gradient tensor to max_grad_norm, then adds
    N(0, (noise_multiplier * max_grad_norm)²) Gaussian noise.

    Args:
        gradients:       Original gradient tensors.
        noise_multiplier: σ / max_grad_norm.  Higher = more noise = smaller ε.
        max_grad_norm:   Gradient clipping threshold.

    Returns:
        Noisy gradient list (same structure).
    """
    noisy = []
    for g in gradients:
        # Per-tensor clipping (approximate per-sample clipping for demo)
        norm = g.norm(2)
        scale = min(1.0, max_grad_norm / (norm.item() + 1e-8))
        g_clipped = g * scale
        noise = torch.randn_like(g_clipped) * noise_multiplier * max_grad_norm
        noisy.append(g_clipped + noise)
    return noisy


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------

def psnr(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """Peak Signal-to-Noise Ratio (dB). Higher = better reconstruction."""
    orig = original.float().clamp(0, 1)
    recon = reconstructed.float().clamp(0, 1)
    mse = F.mse_loss(recon, orig).item()
    if mse < 1e-10:
        return float("inf")
    return round(10 * torch.log10(torch.tensor(1.0 / mse)).item(), 2)


def mse(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    return round(
        F.mse_loss(original.float(), reconstructed.float()).item(), 6
    )
