"""
FedGuard Central Server — FastAPI application.

Endpoints
---------
GET  /health            Liveness probe
GET  /status            Full training state for dashboard
GET  /get_model         Download current global model weights
POST /submit_weights    Node uploads weights + metadata
POST /reset             Restart training (dev utility)

Advanced features vs baseline
------------------------------
- Byzantine detection (cosine similarity + norm screening)
- Pluggable aggregation: fedavg | median | trimmed_mean | krum
- Delta reconstruction (nodes may send compressed weight deltas)
- Per-round rich metadata: privacy budgets, cosine scores, flags,
  compression ratios, local accuracies

Environment variables
---------------------
DATASET                 mnist | nslkdd
AGGREGATION_STRATEGY    fedavg | median | trimmed_mean | krum
BYZANTINE_DETECTION     true | false
BYZANTINE_COS_THRESHOLD 0.0   (flag nodes with cosine_sim < this)
BYZANTINE_NORM_SIGMA    2.0   (flag nodes with norm > mean + k*std)
MIN_NODES               2     (trigger aggregation when this many nodes report)
TOTAL_NODES             3
"""
import io
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

sys.path.insert(0, str(Path(__file__).parent.parent))

from aggregator import aggregate
from defender import run_defence
from model import GlobalModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATASET               = os.environ.get("DATASET", "mnist").lower()
AGGREGATION_STRATEGY  = os.environ.get("AGGREGATION_STRATEGY", "fedavg").lower()
BYZANTINE_DETECTION   = os.environ.get("BYZANTINE_DETECTION", "true").lower() == "true"
BYZ_COS_THRESHOLD     = float(os.environ.get("BYZANTINE_COS_THRESHOLD", "0.0"))
BYZ_NORM_SIGMA        = float(os.environ.get("BYZANTINE_NORM_SIGMA", "2.0"))
MIN_NODES             = int(os.environ.get("MIN_NODES", "2"))
TOTAL_NODES           = int(os.environ.get("TOTAL_NODES", "3"))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SERVER] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------
app = FastAPI(title="FedGuard Server", version="2.0.0")
lock = threading.Lock()

global_model = GlobalModel(dataset=DATASET)
current_round: int = 0

# node_id → {"weights": bytes, "metadata": dict, "is_delta": bool}
pending: Dict[str, dict] = {}

# Round history entries
round_history: List[dict] = []
known_nodes: set = set()

StateDict = Dict[str, torch.Tensor]


# ---------------------------------------------------------------------------
# Aggregation helper
# ---------------------------------------------------------------------------

def _deserialise(raw: bytes) -> StateDict:
    buf = io.BytesIO(raw)
    return torch.load(buf, map_location="cpu", weights_only=True)


def _try_aggregate() -> None:
    global current_round

    if len(pending) < MIN_NODES:
        log.info("Waiting (%d/%d nodes received)", len(pending), MIN_NODES)
        return

    node_ids   = list(pending.keys())
    weight_raw = [pending[nid]["weights"]   for nid in node_ids]
    meta_list  = [pending[nid]["metadata"]  for nid in node_ids]
    is_delta   = [pending[nid]["is_delta"]  for nid in node_ids]

    weight_list: List[StateDict] = [_deserialise(raw) for raw in weight_raw]

    # --- Byzantine detection -------------------------------------------
    defence_report: dict = {}
    flagged_ids: List[str] = []
    if BYZANTINE_DETECTION and len(weight_list) > 1:
        weight_list, node_ids, flagged_ids, defence_report = run_defence(
            weight_list, node_ids,
            cosine_threshold=BYZ_COS_THRESHOLD,
            norm_k_sigma=BYZ_NORM_SIGMA,
        )
        if flagged_ids:
            log.warning("Byzantine nodes flagged: %s", flagged_ids)

    if not weight_list:
        log.error("No clean updates after defence screening — skipping aggregation")
        return

    # --- Aggregation ---------------------------------------------------
    # If any submission is a delta, reconstruct full weights first
    global_sd = global_model.get_state_dict()
    reconstructed = []
    for w, delta_flag in zip(weight_list, [pending.get(nid, {}).get("is_delta", False) for nid in node_ids]):
        if delta_flag:
            reconstructed.append(
                {k: global_sd[k].float() + w[k].float() for k in global_sd}
            )
        else:
            reconstructed.append(w)

    avg_weights = aggregate(reconstructed, strategy=AGGREGATION_STRATEGY)
    global_model.set_state_dict(avg_weights)

    # --- Evaluate ------------------------------------------------------
    accuracy = global_model.evaluate()
    current_round += 1

    # Collect per-node metadata for history
    avg_epsilon = None
    epsilon_vals = [m.get("epsilon_spent") for m in meta_list if m.get("epsilon_spent")]
    if epsilon_vals:
        avg_epsilon = round(sum(epsilon_vals) / len(epsilon_vals), 4)

    avg_compression = None
    comp_vals = [m.get("compression", {}).get("compression_ratio") for m in meta_list
                 if m.get("compression", {}).get("compression_ratio") is not None]
    if comp_vals:
        avg_compression = round(sum(comp_vals) / len(comp_vals), 4)

    local_accs = {nid: m.get("local_accuracy") for nid, m in zip(list(pending.keys()), meta_list)}

    round_history.append({
        "round":               current_round,
        "accuracy":            round(accuracy, 4),
        "num_nodes":           len(pending),
        "clean_nodes":         node_ids,
        "flagged_nodes":       flagged_ids,
        "aggregation":         AGGREGATION_STRATEGY,
        "byzantine_detection": BYZANTINE_DETECTION,
        "cosine_similarities": defence_report.get("cosine_similarities", {}),
        "local_accuracies":    local_accs,
        "avg_epsilon":         avg_epsilon,
        "avg_compression":     avg_compression,
        "timestamp":           time.time(),
    })

    log.info(
        "Round %d complete | nodes=%d (flagged=%d) | accuracy=%.4f | strategy=%s",
        current_round, len(pending), len(flagged_ids), accuracy, AGGREGATION_STRATEGY,
    )
    pending.clear()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/status")
def status():
    with lock:
        return {
            "round":           current_round,
            "pending_nodes":   list(pending.keys()),
            "known_nodes":     list(known_nodes),
            "history":         round_history,
            "config": {
                "dataset":               DATASET,
                "aggregation_strategy":  AGGREGATION_STRATEGY,
                "byzantine_detection":   BYZANTINE_DETECTION,
                "byzantine_cos_threshold": BYZ_COS_THRESHOLD,
                "min_nodes":             MIN_NODES,
                "total_nodes":           TOTAL_NODES,
            },
        }


@app.get("/get_model")
def get_model_weights():
    with lock:
        data = global_model.get_weights_bytes()
    return Response(content=data, media_type="application/octet-stream")


@app.post("/submit_weights")
async def submit_weights(
    node_id:  str        = Form(...),
    metadata: str        = Form("{}"),
    weights:  UploadFile = File(...),
):
    raw = await weights.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty weights payload")

    try:
        meta = json.loads(metadata)
    except json.JSONDecodeError:
        meta = {}

    is_delta = bool(meta.pop("is_delta", False))

    with lock:
        known_nodes.add(node_id)
        pending[node_id] = {"weights": raw, "metadata": meta, "is_delta": is_delta}
        log.info(
            "Received weights from node %s (%d bytes, is_delta=%s) | round %d",
            node_id, len(raw), is_delta, current_round,
        )
        _try_aggregate()

    return {"status": "ok", "round": current_round}


@app.post("/reset")
def reset():
    global current_round
    with lock:
        current_round = 0
        pending.clear()
        round_history.clear()
        known_nodes.clear()
        global_model.reinitialise()
    log.info("Training state reset")
    return {"status": "reset"}
