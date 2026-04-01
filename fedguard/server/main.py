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
AGGREGATION_NOISE_STD   0     optional Gaussian noise on fedavg output (demo, not formal DP)

FL cycle gating
---------------
GET /get_model sends X-FL-Cycle. POST /submit_weights must echo cycle_id; stale
uploads get HTTP 409 (avoids aggregating weights from an outdated global model).
"""
import io
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

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
AGGREGATION_NOISE_STD = float(os.environ.get("AGGREGATION_NOISE_STD", "0"))

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
accepting_cycle: int = 0

# node_id → {"weights": bytes, "metadata": dict, "is_delta": bool, "sample_count": int}
pending: Dict[str, dict] = {}

# Round history entries
round_history: List[dict] = []
known_nodes: set = set()

StateDict = Dict[str, torch.Tensor]

# Runtime-mutable config (can be changed via POST /config without restart)
runtime_config: dict = {
    "aggregation_strategy":    AGGREGATION_STRATEGY,
    "byzantine_detection":     BYZANTINE_DETECTION,
    "byzantine_cos_threshold": BYZ_COS_THRESHOLD,
    "byzantine_norm_sigma":    BYZ_NORM_SIGMA,
    "aggregation_noise_std":   AGGREGATION_NOISE_STD,
    "simulate_byzantine":      False,   # inject a fake poisoned node for demo
}


# ---------------------------------------------------------------------------
# Aggregation helper
# ---------------------------------------------------------------------------

def _deserialise(raw: bytes) -> StateDict:
    buf = io.BytesIO(raw)
    return torch.load(buf, map_location="cpu", weights_only=True)


def _try_aggregate() -> None:
    global current_round, accepting_cycle

    if len(pending) < MIN_NODES:
        log.info("Waiting (%d/%d nodes received)", len(pending), MIN_NODES)
        return

    submitted_ids = list(pending.keys())
    weight_raw = [pending[nid]["weights"] for nid in submitted_ids]
    weight_list: List[StateDict] = [_deserialise(raw) for raw in weight_raw]
    node_ids = submitted_ids.copy()

    # --- Simulate Byzantine node (demo mode) ---------------------------
    if runtime_config["simulate_byzantine"]:
        # Inject a fake poisoned update: negate the first clean update
        fake_weights = {k: -v.clone() for k, v in weight_list[0].items()}
        weight_list.append(fake_weights)
        node_ids.append("sim_byzantine")
        log.info("Injected simulated Byzantine update for demo")

    # --- Byzantine detection -------------------------------------------
    defence_report: dict = {}
    flagged_ids: List[str] = []
    if runtime_config["byzantine_detection"] and len(weight_list) > 1:
        weight_list, clean_ids, flagged_ids, defence_report = run_defence(
            weight_list,
            node_ids,
            cosine_threshold=runtime_config["byzantine_cos_threshold"],
            norm_k_sigma=runtime_config["byzantine_norm_sigma"],
        )
        node_ids = clean_ids
        if flagged_ids:
            log.warning("Byzantine nodes flagged: %s", flagged_ids)

    if not weight_list:
        log.error("No clean updates after defence screening — skipping aggregation")
        return

    # --- Aggregation ---------------------------------------------------
    global_sd = global_model.get_state_dict()
    reconstructed = []
    sample_weights = []
    for w, nid in zip(weight_list, node_ids):
        if pending[nid]["is_delta"]:
            reconstructed.append(
                {k: global_sd[k].float() + w[k].float() for k in global_sd}
            )
        else:
            reconstructed.append(w)
        sample_weights.append(float(pending[nid].get("sample_count", 1)))

    avg_weights = aggregate(
        reconstructed,
        strategy=runtime_config["aggregation_strategy"],
        sample_weights=sample_weights,
        noise_std=runtime_config["aggregation_noise_std"],
    )
    global_model.set_state_dict(avg_weights)

    # --- Evaluate ------------------------------------------------------
    accuracy = global_model.evaluate()
    current_round += 1

    # Metadata aligned to *all* submitters; ε / compression from clean nodes only
    meta_clean = [pending[nid]["metadata"] for nid in node_ids]
    avg_epsilon = None
    epsilon_vals = [m.get("epsilon_spent") for m in meta_clean if m.get("epsilon_spent")]
    if epsilon_vals:
        avg_epsilon = round(sum(epsilon_vals) / len(epsilon_vals), 4)

    avg_compression = None
    comp_vals = [
        m.get("compression", {}).get("compression_ratio")
        for m in meta_clean
        if m.get("compression", {}).get("compression_ratio") is not None
    ]
    if comp_vals:
        avg_compression = round(sum(comp_vals) / len(comp_vals), 4)

    local_accs = {
        nid: pending[nid]["metadata"].get("local_accuracy") for nid in submitted_ids
    }

    round_history.append({
        "round":               current_round,
        "accuracy":            round(accuracy, 4),
        "num_nodes":           len(submitted_ids),
        "fl_cycle":            accepting_cycle,
        "clean_nodes":         node_ids,
        "flagged_nodes":       flagged_ids,
        "aggregation":         runtime_config["aggregation_strategy"],
        "byzantine_detection": runtime_config["byzantine_detection"],
        "cosine_similarities": defence_report.get("cosine_similarities", {}),
        "local_accuracies":    local_accs,
        "avg_epsilon":         avg_epsilon,
        "avg_compression":     avg_compression,
        "timestamp":           time.time(),
    })

    log.info(
        "Round %d complete | fl_cycle=%d | submitters=%d clean=%d (flagged=%d) | acc=%.4f | strat=%s",
        current_round,
        accepting_cycle,
        len(submitted_ids),
        len(node_ids),
        len(flagged_ids),
        accuracy,
        AGGREGATION_STRATEGY,
    )
    pending.clear()
    accepting_cycle += 1


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
            "accepting_cycle": accepting_cycle,
            "config": {
                "dataset":               DATASET,
                "min_nodes":             MIN_NODES,
                "total_nodes":           TOTAL_NODES,
                **runtime_config,
            },
        }


@app.post("/config")
def update_config(body: dict):
    """Update runtime configuration without restarting the server."""
    allowed = {
        "aggregation_strategy", "byzantine_detection",
        "byzantine_cos_threshold", "byzantine_norm_sigma",
        "aggregation_noise_std", "simulate_byzantine",
    }
    with lock:
        for key, val in body.items():
            if key in allowed:
                runtime_config[key] = val
                log.info("Config updated: %s = %s", key, val)
            else:
                raise HTTPException(status_code=400, detail=f"Unknown config key: {key}")
    return {"status": "ok", "config": runtime_config}


@app.get("/get_model")
def get_model_weights():
    with lock:
        data = global_model.get_weights_bytes()
        cycle = accepting_cycle
    return Response(
        content=data,
        media_type="application/octet-stream",
        headers={"X-FL-Cycle": str(cycle)},
    )


@app.post("/submit_weights")
async def submit_weights(
    node_id: str = Form(...),
    cycle_id: int = Form(...),
    sample_count: int = Form(1),
    metadata: str = Form("{}"),
    weights: UploadFile = File(...),
):
    raw = await weights.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty weights payload")

    if sample_count < 1:
        raise HTTPException(status_code=400, detail="sample_count must be >= 1")

    try:
        meta = json.loads(metadata)
    except json.JSONDecodeError:
        meta = {}

    is_delta = bool(meta.pop("is_delta", False))

    with lock:
        if cycle_id != accepting_cycle:
            log.warning(
                "Stale upload from node %s: cycle_id=%s open=%s",
                node_id,
                cycle_id,
                accepting_cycle,
            )
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "stale_or_mismatched_cycle",
                    "submitted_cycle": cycle_id,
                    "accepting_cycle": accepting_cycle,
                },
            )

        known_nodes.add(node_id)
        pending[node_id] = {
            "weights": raw,
            "metadata": meta,
            "is_delta": is_delta,
            "sample_count": sample_count,
        }
        log.info(
            "Received weights from node %s | cycle=%d | samples=%d | %d bytes | is_delta=%s",
            node_id,
            cycle_id,
            sample_count,
            len(raw),
            is_delta,
        )
        _try_aggregate()

    return {"status": "ok", "round": current_round, "accepting_cycle": accepting_cycle}


@app.post("/reset")
def reset():
    global current_round, accepting_cycle
    with lock:
        current_round = 0
        accepting_cycle = 0
        pending.clear()
        round_history.clear()
        known_nodes.clear()
        global_model.reinitialise()
    log.info("Training state reset")
    return {"status": "reset"}
