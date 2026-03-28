"""
FedGuard Attack Server — FastAPI service

Exposes the DLG attack as REST endpoints so the dashboard can trigger
reconstructions and display results without Docker exec.

Workflow
--------
1. Load current global model from FedGuard server (GET /get_model)
2. Sample a target image from MNIST test set
3. Compute the gradients the victim node would have uploaded
4. Optionally add simulated DP noise (various noise_multiplier levels)
5. Run DLG / iDLG reconstruction
6. Return original + reconstructed images (base64 PNG) + metrics

The three noise levels below map roughly to DP budgets:
  noise_multiplier = 0.0  →  ε = ∞   (no privacy)
  noise_multiplier = 0.3  →  ε ≈ 10  (weak privacy)
  noise_multiplier = 1.1  →  ε ≈ 1   (strong privacy)
"""
import base64
import io
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import requests
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from torchvision import datasets, transforms

sys.path.insert(0, "/app/shared")
from model_def import get_model

from dlg_attack import apply_dp_noise, dlg_reconstruct, mse, psnr

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SERVER_URL = os.environ.get("SERVER_URL", "http://server:8000").rstrip("/")
DATA_DIR   = os.environ.get("DATA_DIR", "/tmp/fedguard_data")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ATTACK] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

app = FastAPI(title="FedGuard Attack Server", version="1.0.0")

# Cache latest results so dashboard can poll without re-running
_cached_results: dict = {}

# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class AttackRequest(BaseModel):
    image_index: int = 0          # MNIST test-set index to use as victim sample
    iterations: int = 300         # DLG L-BFGS iterations
    use_idlg: bool = True         # Use analytical label extraction
    tv_weight: float = 1e-4       # Total-variation regularisation weight
    noise_multiplier: float = 0.0 # DP noise (0 = no privacy)
    max_grad_norm: float = 1.0    # DP clipping norm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

_mnist_test: Optional[datasets.MNIST] = None


def _get_mnist() -> datasets.MNIST:
    global _mnist_test
    if _mnist_test is None:
        _mnist_test = datasets.MNIST(
            root=DATA_DIR, train=False, download=True, transform=_transform
        )
    return _mnist_test


def _load_global_model() -> nn.Module:
    """Fetch current global weights from the FedGuard server."""
    r = requests.get(f"{SERVER_URL}/get_model", timeout=30)
    r.raise_for_status()
    buf = io.BytesIO(r.content)
    state_dict = torch.load(buf, map_location="cpu", weights_only=True)
    model = get_model("mnist")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _tensor_to_b64_png(tensor: torch.Tensor) -> str:
    """Convert a (1, H, W) or (H, W) float tensor to base64-encoded PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = tensor.squeeze().numpy()

    # Denormalise MNIST: reverse Normalize((0.1307,), (0.3081,))
    img = img * 0.3081 + 0.1307
    img = img.clip(0, 1)

    fig, ax = plt.subplots(figsize=(2, 2))
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
    fig.tight_layout(pad=0)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=80)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _convergence_b64(grad_diffs: list) -> str:
    """Plot convergence curve and return as base64 PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 2.5))
    ax.semilogy(grad_diffs, color="#e74c3c", linewidth=1.5)
    ax.set_xlabel("Iteration", fontsize=9)
    ax.set_ylabel("‖∇_dummy − ∇_target‖²", fontsize=9)
    ax.set_title("DLG Convergence", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=80)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/results")
def get_results():
    """Return cached results from the last attack run."""
    if not _cached_results:
        return {"status": "no_results", "message": "No attack has been run yet."}
    return _cached_results


@app.post("/run")
def run_attack(req: AttackRequest):
    """
    Run a full DLG attack experiment for one noise level.

    Returns original image, reconstructed image, metrics, and convergence plot
    all as base64-encoded PNGs in a JSON payload.
    """
    global _cached_results

    log.info(
        "Running DLG attack: image_idx=%d  iters=%d  noise_mult=%.2f",
        req.image_index, req.iterations, req.noise_multiplier,
    )

    # ── Load victim image ────────────────────────────────────────────────────
    dataset = _get_mnist()
    if req.image_index >= len(dataset):
        raise HTTPException(
            status_code=400,
            detail=f"image_index {req.image_index} out of range (max {len(dataset)-1})"
        )

    original_tensor, true_label = dataset[req.image_index]
    original_tensor = original_tensor.unsqueeze(0)   # (1, 1, 28, 28)
    label_tensor = torch.tensor([true_label])

    # ── Load model ───────────────────────────────────────────────────────────
    try:
        model = _load_global_model()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Cannot reach FedGuard server: {exc}")

    # ── Compute victim gradients ─────────────────────────────────────────────
    model.zero_grad()
    loss = torch.nn.CrossEntropyLoss()(model(original_tensor), label_tensor)
    loss.backward()
    raw_gradients = [p.grad.detach().clone() for p in model.parameters()]

    # ── Apply DP noise (if requested) ────────────────────────────────────────
    if req.noise_multiplier > 0:
        attack_gradients = apply_dp_noise(
            raw_gradients, req.noise_multiplier, req.max_grad_norm
        )
        log.info("Applied DP noise: σ=%.2f", req.noise_multiplier)
    else:
        attack_gradients = raw_gradients

    # Reset model gradients before reconstruction
    model.zero_grad()

    # ── DLG reconstruction ───────────────────────────────────────────────────
    reconstructed, pred_label, grad_diffs = dlg_reconstruct(
        model=model,
        target_gradients=attack_gradients,
        image_shape=(1, 1, 28, 28),
        num_classes=10,
        iterations=req.iterations,
        use_idlg=req.use_idlg,
        tv_weight=req.tv_weight,
    )

    # ── Metrics ──────────────────────────────────────────────────────────────
    psnr_val = psnr(original_tensor, reconstructed)
    mse_val  = mse(original_tensor, reconstructed)
    final_grad_diff = grad_diffs[-1] if grad_diffs else None

    # ── Encode images ────────────────────────────────────────────────────────
    original_b64      = _tensor_to_b64_png(original_tensor.squeeze(0))
    reconstructed_b64 = _tensor_to_b64_png(reconstructed.squeeze(0))
    convergence_b64   = _convergence_b64(grad_diffs)

    result = {
        "status": "ok",
        "params": {
            "image_index":     req.image_index,
            "true_label":      int(true_label),
            "predicted_label": pred_label,
            "label_correct":   pred_label == int(true_label) if pred_label is not None else None,
            "iterations":      len(grad_diffs),
            "noise_multiplier": req.noise_multiplier,
        },
        "metrics": {
            "psnr_db":        psnr_val,
            "mse":            mse_val,
            "final_grad_diff": final_grad_diff,
        },
        "images": {
            "original":      original_b64,
            "reconstructed": reconstructed_b64,
            "convergence":   convergence_b64,
        },
    }

    _cached_results = result
    log.info(
        "Attack complete: PSNR=%.2f dB  MSE=%.6f  label_correct=%s",
        psnr_val, mse_val, pred_label == int(true_label),
    )
    return result


@app.post("/run_comparison")
def run_comparison(image_index: int = 0, iterations: int = 300):
    """
    Run the full privacy demonstration: attack at three noise levels and
    return all three reconstructions in one payload.

    Noise levels:
        0.0  → no DP   (ε = ∞)
        0.3  → weak DP (ε ≈ 10)
        1.1  → strong DP (ε ≈ 1)
    """
    noise_levels = [
        {"label": "No DP (ε=∞)",    "noise": 0.0},
        {"label": "Weak DP (ε≈10)", "noise": 0.3},
        {"label": "Strong DP (ε≈1)","noise": 1.1},
    ]

    dataset = _get_mnist()
    if image_index >= len(dataset):
        raise HTTPException(status_code=400, detail="image_index out of range")

    original_tensor, true_label = dataset[image_index]
    original_tensor = original_tensor.unsqueeze(0)
    label_tensor = torch.tensor([int(true_label)])

    try:
        model = _load_global_model()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    # Compute clean gradients once
    model.zero_grad()
    loss = torch.nn.CrossEntropyLoss()(model(original_tensor), label_tensor)
    loss.backward()
    clean_grads = [p.grad.detach().clone() for p in model.parameters()]

    results = []
    for nl in noise_levels:
        model.zero_grad()
        attack_grads = apply_dp_noise(clean_grads, nl["noise"]) \
            if nl["noise"] > 0 else clean_grads

        reconstructed, pred_label, grad_diffs = dlg_reconstruct(
            model=model,
            target_gradients=attack_grads,
            image_shape=(1, 1, 28, 28),
            iterations=iterations,
            tv_weight=1e-4,
        )
        model.zero_grad()

        results.append({
            "label":            nl["label"],
            "noise_multiplier": nl["noise"],
            "psnr_db":          psnr(original_tensor, reconstructed),
            "mse":              mse(original_tensor, reconstructed),
            "predicted_label":  pred_label,
            "label_correct":    pred_label == int(true_label),
            "reconstructed_b64": _tensor_to_b64_png(reconstructed.squeeze(0)),
            "convergence_b64":   _convergence_b64(grad_diffs),
        })
        log.info("Comparison %s: PSNR=%.1f dB", nl["label"], results[-1]["psnr_db"])

    payload = {
        "status":        "ok",
        "true_label":    int(true_label),
        "image_index":   image_index,
        "original_b64":  _tensor_to_b64_png(original_tensor.squeeze(0)),
        "comparisons":   results,
    }
    _cached_results = payload
    return payload
