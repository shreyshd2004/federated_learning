"""
FedGuard Edge Node — federated training client.

Extended capabilities vs. baseline
------------------------------------
- Non-IID Dirichlet data partitioning
- FedProx local training (configurable μ)
- DP-SGD via Opacus (configurable ε, δ)
- Top-K gradient compression with error feedback
- Model-poisoning simulation (set NODE_POISONED=true on one node)
- Rich metadata sent alongside weights for server-side auditing

Environment variables
---------------------
NODE_ID              1-indexed node identifier
SERVER_URL           http://server:8000
TOTAL_NODES          3
DATASET              mnist | nslkdd
DIRICHLET_ALPHA      0.5  (non-IID skew; ≤0 → IID)
LOCAL_EPOCHS         2
MAX_ROUNDS           10
ROUND_POLL_INTERVAL  5    (seconds between rounds)
FEDPROX_MU           0.01 (0 = standard FedAvg)
ENABLE_DP            false
DP_EPSILON           10.0
DP_DELTA             1e-5
DP_MAX_GRAD_NORM     1.0
ENABLE_COMPRESSION   false
COMPRESSION_TOP_K    0.1
NODE_POISONED        false  (gradient inversion attack simulation)
RETRY_LIMIT          5
"""
import io
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import requests
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.model_def import get_model

from compressor import TopKCompressor, compute_delta
from data_loader import get_data_loader
from trainer import train_local

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NODE_ID              = int(os.environ.get("NODE_ID", "1"))
SERVER_URL           = os.environ.get("SERVER_URL", "http://localhost:8000").rstrip("/")
TOTAL_NODES          = int(os.environ.get("TOTAL_NODES", "3"))
DATASET              = os.environ.get("DATASET", "mnist").lower()
DIRICHLET_ALPHA      = float(os.environ.get("DIRICHLET_ALPHA", "0.5"))
LOCAL_EPOCHS         = int(os.environ.get("LOCAL_EPOCHS", "2"))
MAX_ROUNDS           = int(os.environ.get("MAX_ROUNDS", "10"))
ROUND_POLL_INTERVAL  = float(os.environ.get("ROUND_POLL_INTERVAL", "5"))
FEDPROX_MU           = float(os.environ.get("FEDPROX_MU", "0.01"))
ENABLE_DP            = os.environ.get("ENABLE_DP", "false").lower() == "true"
DP_EPSILON           = float(os.environ.get("DP_EPSILON", "10.0"))
DP_DELTA             = float(os.environ.get("DP_DELTA", "1e-5"))
DP_MAX_GRAD_NORM     = float(os.environ.get("DP_MAX_GRAD_NORM", "1.0"))
ENABLE_COMPRESSION   = os.environ.get("ENABLE_COMPRESSION", "false").lower() == "true"
COMPRESSION_TOP_K    = float(os.environ.get("COMPRESSION_TOP_K", "0.1"))
NODE_POISONED        = os.environ.get("NODE_POISONED", "false").lower() == "true"
RETRY_LIMIT          = int(os.environ.get("RETRY_LIMIT", "5"))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format=f"%(asctime)s [NODE-{NODE_ID}] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wait_for_server(timeout: int = 120) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{SERVER_URL}/health", timeout=5)
            if r.status_code == 200:
                log.info("Server ready.")
                return
        except requests.exceptions.RequestException:
            pass
        log.info("Waiting for server…")
        time.sleep(3)
    raise RuntimeError(f"Server unreachable after {timeout}s")


def _download_model(feature_dim: int) -> torch.nn.Module:
    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            r = requests.get(f"{SERVER_URL}/get_model", timeout=30)
            r.raise_for_status()
            buf = io.BytesIO(r.content)
            state_dict = torch.load(buf, map_location="cpu", weights_only=True)
            model = get_model(DATASET, input_dim=feature_dim) if DATASET == "nslkdd" \
                else get_model(DATASET)
            model.load_state_dict(state_dict)
            log.info("Downloaded global model (%d bytes)", len(r.content))
            return model
        except Exception as exc:
            wait = 2 ** attempt
            log.warning("Download attempt %d failed: %s — retry in %ds", attempt, exc, wait)
            time.sleep(wait)
    raise RuntimeError("Model download failed after retries")


def _poison_weights(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Gradient inversion attack: negate all weight tensors.
    Simulates a malicious node trying to push the global model in the
    opposite direction — Byzantine detection should flag this node.
    """
    log.warning("NODE IS POISONED — injecting gradient inversion attack")
    return {k: -v.clone() for k, v in state_dict.items()}


def _upload(
    state_dict: Dict[str, torch.Tensor],
    metadata: dict,
    is_delta: bool = False,
) -> int:
    """Serialise weights/delta and POST to server. Returns new server round."""
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    buf.seek(0)

    payload_meta = {**metadata, "is_delta": is_delta}

    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            r = requests.post(
                f"{SERVER_URL}/submit_weights",
                data={
                    "node_id": str(NODE_ID),
                    "metadata": json.dumps(payload_meta),
                },
                files={"weights": ("weights.pt", buf, "application/octet-stream")},
                timeout=60,
            )
            r.raise_for_status()
            resp = r.json()
            log.info("Uploaded weights → server round %s", resp.get("round", "?"))
            return resp.get("round", 0)
        except Exception as exc:
            wait = 2 ** attempt
            log.warning("Upload attempt %d failed: %s — retry in %ds", attempt, exc, wait)
            buf.seek(0)
            time.sleep(wait)
    raise RuntimeError("Weight upload failed after retries")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    log.info(
        "Starting FedGuard node | id=%d  dataset=%s  alpha=%.2f  "
        "fedprox_mu=%.3f  dp=%s  compression=%s  poisoned=%s",
        NODE_ID, DATASET, DIRICHLET_ALPHA, FEDPROX_MU,
        ENABLE_DP, ENABLE_COMPRESSION, NODE_POISONED,
    )
    _wait_for_server()

    dataloader, feature_dim = get_data_loader(
        node_id=NODE_ID,
        total_nodes=TOTAL_NODES,
        dataset=DATASET,
        alpha=DIRICHLET_ALPHA,
    )
    log.info("Data shard ready: %d batches  feature_dim=%d", len(dataloader), feature_dim)

    compressor = TopKCompressor(COMPRESSION_TOP_K) if ENABLE_COMPRESSION else None
    participated = 0

    while participated < MAX_ROUNDS:
        log.info("=== Federated Round %d / %d ===", participated + 1, MAX_ROUNDS)

        # 1. Download current global model
        model = _download_model(feature_dim)
        global_sd = {k: v.clone() for k, v in model.state_dict().items()}

        # 2. Local training
        local_sd, train_meta = train_local(
            model=model,
            dataloader=dataloader,
            global_state_dict=global_sd,
            epochs=LOCAL_EPOCHS,
            lr=0.01,
            mu=FEDPROX_MU,
            enable_dp=ENABLE_DP,
            dp_epsilon=DP_EPSILON,
            dp_delta=DP_DELTA,
            dp_max_grad_norm=DP_MAX_GRAD_NORM,
        )

        # 3. Poisoning simulation (one node sends inverted weights)
        if NODE_POISONED:
            local_sd = _poison_weights(local_sd)

        # 4. Optional: Top-K compression on weight delta
        compress_meta = {}
        is_delta = False
        if compressor is not None and not NODE_POISONED:
            delta = compute_delta(local_sd, global_sd)
            local_sd, compress_meta = compressor.compress(delta)
            is_delta = True
            log.info(
                "Compressed delta: sparsity=%.1f%%  kept=%d/%d params",
                compress_meta["sparsity"] * 100,
                compress_meta["kept_params"],
                compress_meta["total_params"],
            )

        # 5. Upload
        metadata = {**train_meta, "compression": compress_meta}
        server_round = _upload(local_sd, metadata, is_delta=is_delta)
        participated += 1

        log.info(
            "Round %d/%d done  server_round=%d  local_acc=%.4f%s",
            participated, MAX_ROUNDS, server_round,
            train_meta["local_accuracy"],
            f"  ε={train_meta['epsilon_spent']:.3f}" if train_meta.get("epsilon_spent") else "",
        )

        time.sleep(ROUND_POLL_INTERVAL)

    log.info("Node %d finished %d rounds.", NODE_ID, MAX_ROUNDS)


if __name__ == "__main__":
    main()
