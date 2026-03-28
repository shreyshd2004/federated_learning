"""
FedGuard Edge Node — polling client.

Flow per round
--------------
1. GET  /get_model   → download global weights
2. Train locally on private data shard
3. POST /submit_weights → upload updated weights
4. Sleep briefly, then repeat until max rounds reached
"""
import io
import logging
import os
import sys
import time
from pathlib import Path

import requests
import torch

# Allow importing shared module whether running inside Docker or locally
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.model_def import get_model

from data_loader import get_data_loader
from trainer import train_local

# ---------------------------------------------------------------------------
# Configuration (from environment variables)
# ---------------------------------------------------------------------------
NODE_ID = int(os.environ.get("NODE_ID", "1"))
SERVER_URL = os.environ.get("SERVER_URL", "http://localhost:8000").rstrip("/")
TOTAL_NODES = int(os.environ.get("TOTAL_NODES", "3"))
LOCAL_EPOCHS = int(os.environ.get("LOCAL_EPOCHS", "2"))
MAX_ROUNDS = int(os.environ.get("MAX_ROUNDS", "10"))
ROUND_POLL_INTERVAL = float(os.environ.get("ROUND_POLL_INTERVAL", "5"))  # seconds
RETRY_LIMIT = int(os.environ.get("RETRY_LIMIT", "5"))

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
    """Block until the server health endpoint responds."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{SERVER_URL}/health", timeout=5)
            if r.status_code == 200:
                log.info("Server is ready.")
                return
        except requests.exceptions.RequestException:
            pass
        log.info("Waiting for server...")
        time.sleep(3)
    raise RuntimeError(f"Server did not become ready within {timeout}s")


def _download_model() -> torch.nn.Module:
    """Download global weights and load into a fresh model instance."""
    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            r = requests.get(f"{SERVER_URL}/get_model", timeout=30)
            r.raise_for_status()
            buf = io.BytesIO(r.content)
            state_dict = torch.load(buf, map_location="cpu", weights_only=True)
            model = get_model()
            model.load_state_dict(state_dict)
            log.info("Downloaded global model (%d bytes)", len(r.content))
            return model
        except Exception as exc:
            wait = 2 ** attempt
            log.warning("Download attempt %d failed: %s — retrying in %ds", attempt, exc, wait)
            time.sleep(wait)
    raise RuntimeError("Failed to download model after multiple attempts")


def _upload_weights(state_dict: dict) -> int:
    """Serialise and upload state_dict; return the new round number from server."""
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    buf.seek(0)

    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            r = requests.post(
                f"{SERVER_URL}/submit_weights",
                data={"node_id": str(NODE_ID)},
                files={"weights": ("weights.pt", buf, "application/octet-stream")},
                timeout=60,
            )
            r.raise_for_status()
            resp = r.json()
            log.info("Weights uploaded. Server round is now %d", resp.get("round", "?"))
            return resp.get("round", 0)
        except Exception as exc:
            wait = 2 ** attempt
            log.warning("Upload attempt %d failed: %s — retrying in %ds", attempt, exc, wait)
            buf.seek(0)
            time.sleep(wait)
    raise RuntimeError("Failed to upload weights after multiple attempts")


def _get_server_round() -> int:
    """Poll /status and return the current server round."""
    r = requests.get(f"{SERVER_URL}/status", timeout=10)
    r.raise_for_status()
    return r.json().get("round", 0)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    log.info("Starting FedGuard node (NODE_ID=%d, SERVER=%s)", NODE_ID, SERVER_URL)
    _wait_for_server()

    dataloader = get_data_loader(node_id=NODE_ID, total_nodes=TOTAL_NODES)
    log.info("Data shard loaded: %d batches", len(dataloader))

    participated_rounds = 0

    while participated_rounds < MAX_ROUNDS:
        log.info("--- Federated Round %d (local count) ---", participated_rounds + 1)

        # 1. Download current global model
        model = _download_model()

        # 2. Train locally (weights stay on this node)
        log.info("Training locally for %d epoch(s)...", LOCAL_EPOCHS)
        updated_weights = train_local(model, dataloader, epochs=LOCAL_EPOCHS)

        # 3. Upload weights (no raw data sent)
        server_round = _upload_weights(updated_weights)
        participated_rounds += 1

        log.info(
            "Completed local round %d / %d (server round=%d)",
            participated_rounds,
            MAX_ROUNDS,
            server_round,
        )

        # 4. Small pause before next round so all nodes stay roughly in sync
        time.sleep(ROUND_POLL_INTERVAL)

    log.info("Node %d finished %d federated rounds. Exiting.", NODE_ID, MAX_ROUNDS)


if __name__ == "__main__":
    main()
