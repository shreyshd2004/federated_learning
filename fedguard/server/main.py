"""
FedGuard Central Server — FastAPI application.

Endpoints
---------
GET  /status           – current round, nodes, accuracy history
GET  /get_model        – download current global model weights
POST /submit_weights   – node uploads its local weights
POST /reset            – restart training from round 0 (dev utility)
"""
import io
import logging
import threading
import time
from typing import Dict, List, Optional

import torch
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import Response, JSONResponse

from aggregator import fed_avg
from model import GlobalModel

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SERVER] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App & state
# ---------------------------------------------------------------------------
app = FastAPI(title="FedGuard Server", version="1.0.0")

global_model = GlobalModel()
lock = threading.Lock()

# Training state
current_round: int = 0
# node_id -> state_dict bytes (raw upload)
pending_weights: Dict[str, bytes] = {}
# Round history: list of {round, accuracy, num_nodes, timestamp}
round_history: List[dict] = []
# Nodes that have been seen at least once
known_nodes: set = set()

# Minimum nodes required to trigger aggregation (fault-tolerance: 2 of 3)
MIN_NODES_FOR_AGGREGATION = 2
TOTAL_EXPECTED_NODES = 3


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _try_aggregate() -> None:
    """Check if enough weights have arrived; if so run FedAvg and advance round."""
    global current_round

    if len(pending_weights) < MIN_NODES_FOR_AGGREGATION:
        log.info(
            "Waiting for more nodes (%d/%d received)",
            len(pending_weights),
            MIN_NODES_FOR_AGGREGATION,
        )
        return

    # Deserialise all weight blobs
    weight_list = []
    for node_id, raw in pending_weights.items():
        buf = io.BytesIO(raw)
        state_dict = torch.load(buf, map_location="cpu", weights_only=True)
        weight_list.append(state_dict)
        log.info("Aggregating weights from node %s", node_id)

    # FedAvg
    avg_weights = fed_avg(weight_list)
    global_model.set_state_dict(avg_weights)

    # Evaluate
    accuracy = global_model.evaluate()
    current_round += 1

    round_history.append({
        "round": current_round,
        "accuracy": round(accuracy, 4),
        "num_nodes": len(pending_weights),
        "timestamp": time.time(),
    })

    log.info(
        "Round %d complete | nodes=%d | accuracy=%.4f",
        current_round,
        len(pending_weights),
        accuracy,
    )

    # Clear for next round
    pending_weights.clear()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/status")
def status():
    with lock:
        return {
            "round": current_round,
            "pending_nodes": list(pending_weights.keys()),
            "known_nodes": list(known_nodes),
            "history": round_history,
            "min_nodes_for_aggregation": MIN_NODES_FOR_AGGREGATION,
            "total_expected_nodes": TOTAL_EXPECTED_NODES,
        }


@app.get("/get_model")
def get_model_weights():
    """Returns the current global model weights as a binary blob."""
    with lock:
        data = global_model.get_weights_bytes()
    return Response(content=data, media_type="application/octet-stream")


@app.post("/submit_weights")
async def submit_weights(
    node_id: str = Form(...),
    weights: UploadFile = File(...),
):
    """
    Node uploads its locally-trained weights.
    Triggers aggregation once MIN_NODES_FOR_AGGREGATION nodes have reported.
    """
    raw = await weights.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty weights payload")

    with lock:
        known_nodes.add(node_id)
        pending_weights[node_id] = raw
        log.info(
            "Received weights from node %s (%d bytes) | round %d",
            node_id,
            len(raw),
            current_round,
        )
        _try_aggregate()

    return {"status": "ok", "round": current_round}


@app.post("/reset")
def reset():
    """Reset training state (useful for development / demo re-runs)."""
    global current_round
    with lock:
        current_round = 0
        pending_weights.clear()
        round_history.clear()
        known_nodes.clear()
        # Re-initialise model weights
        from shared.model_def import get_model
        global_model.model = get_model()
    log.info("Training state reset")
    return {"status": "reset"}


@app.get("/health")
def health():
    return {"status": "healthy"}
