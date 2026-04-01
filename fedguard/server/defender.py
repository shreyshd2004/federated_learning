"""
Byzantine fault detection for FedGuard.

Threat model
------------
Up to f < n/2 nodes may be Byzantine; they can send arbitrary weight
updates (gradient inversion, sign-flip, large-norm noise, label-flipping).

Detection methods
-----------------
1. Cosine similarity screening
   Compute cosine similarity between each node's flattened update and the
   coordinate-wise mean.  Nodes far from the mean (sim < threshold) are
   flagged as potential adversaries.

2. Euclidean norm outlier detection
   Flag nodes whose update L2-norm is > mean + k·std of the group.
   Catches large-norm (scaling) attacks.

3. Krum (Byzantine-robust selection)
   For each node i, compute the sum of squared distances to its n-f-1
   nearest neighbours.  Select the node with the minimum score.
   Tolerates up to f Byzantine nodes.

References
----------
Blanchard et al., "Machine Learning with Adversaries: Byzantine Tolerant
Gradient Descent" (Krum, NeurIPS 2017)
"""
import logging
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)

StateDict = Dict[str, torch.Tensor]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _flatten(weight_list: List[StateDict]) -> List[torch.Tensor]:
    """Flatten each state_dict to a single 1-D float tensor."""
    return [
        torch.cat([v.float().flatten() for v in sd.values()])
        for sd in weight_list
    ]


# ---------------------------------------------------------------------------
# Cosine similarity screening
# ---------------------------------------------------------------------------

def cosine_similarities(weight_list: List[StateDict]) -> List[float]:
    """
    Return cosine similarity of each node's update against the group mean.

    Range: -1 (perfectly opposite) → +1 (perfectly aligned).
    Values close to -1 indicate gradient inversion / sign-flip attacks.
    """
    flat = _flatten(weight_list)
    mean = torch.stack(flat).mean(dim=0)
    return [
        round(F.cosine_similarity(f.unsqueeze(0), mean.unsqueeze(0)).item(), 4)
        for f in flat
    ]


def screen_by_cosine(
    weight_list: List[StateDict],
    node_ids: List[str],
    threshold: float = 0.0,
) -> Tuple[List[StateDict], List[str], List[str], List[float]]:
    """
    Remove updates whose cosine similarity to the mean falls below *threshold*.

    Returns:
        clean_weights:  filtered list of state_dicts
        clean_ids:      node IDs considered trustworthy
        flagged_ids:    node IDs flagged as potentially Byzantine
        similarities:   per-node cosine similarity score
    """
    sims = cosine_similarities(weight_list)
    clean_weights, clean_ids, flagged_ids = [], [], []

    for w, nid, sim in zip(weight_list, node_ids, sims):
        if sim >= threshold:
            clean_weights.append(w)
            clean_ids.append(nid)
        else:
            flagged_ids.append(nid)
            log.warning(
                "Byzantine alert: node %s flagged (cosine_sim=%.4f < threshold=%.4f)",
                nid, sim, threshold,
            )

    if not clean_weights:
        log.error("All nodes flagged; falling back to full set to avoid empty aggregation")
        return weight_list, node_ids, [], sims

    return clean_weights, clean_ids, flagged_ids, sims


# ---------------------------------------------------------------------------
# L2-norm outlier detection
# ---------------------------------------------------------------------------

def screen_by_norm(
    weight_list: List[StateDict],
    node_ids: List[str],
    k_sigma: float = 2.0,
) -> Tuple[List[StateDict], List[str], List[str], List[float]]:
    """
    Remove updates whose L2 norm is an outlier (> mean + k·std).

    Catches scaling attacks where an adversary amplifies their update.
    """
    flat = _flatten(weight_list)
    norms = torch.tensor([f.norm().item() for f in flat])
    mean_n, std_n = norms.mean().item(), norms.std().item()
    threshold = mean_n + k_sigma * std_n

    clean_weights, clean_ids, flagged_ids = [], [], []
    norm_list = norms.tolist()

    for w, nid, n in zip(weight_list, node_ids, norm_list):
        if n <= threshold:
            clean_weights.append(w)
            clean_ids.append(nid)
        else:
            flagged_ids.append(nid)
            log.warning(
                "Byzantine alert: node %s flagged (norm=%.2f > threshold=%.2f)",
                nid, n, threshold,
            )

    if not clean_weights:
        return weight_list, node_ids, [], norm_list

    return clean_weights, clean_ids, flagged_ids, norm_list


# ---------------------------------------------------------------------------
# Krum selection
# ---------------------------------------------------------------------------

def krum_select(
    weight_list: List[StateDict],
    node_ids: List[str],
    f: int = 1,
) -> Tuple[StateDict, str, List[float]]:
    """
    Return the single most-trustworthy update according to the Krum score.

    Each node i is scored as the sum of distances to its (n-f-1) nearest
    neighbours.  The node with the minimum score is selected.

    Args:
        weight_list: Per-node state_dicts.
        node_ids:    Corresponding node identifiers.
        f:           Maximum number of Byzantine nodes tolerated.

    Returns:
        best_weights: The selected state_dict.
        best_id:      The selected node's ID.
        scores:       Per-node Krum scores (lower = more trusted).
    """
    n = len(weight_list)
    flat = _flatten(weight_list)
    neighbours = max(1, n - f - 1)

    scores = []
    for i in range(n):
        dists = sorted(
            torch.norm(flat[i] - flat[j]).item()
            for j in range(n) if j != i
        )
        scores.append(round(sum(dists[:neighbours]), 4))

    best_idx = int(torch.tensor(scores).argmin().item())
    log.info("Krum selected node %s (score=%.4f)", node_ids[best_idx], scores[best_idx])
    return weight_list[best_idx], node_ids[best_idx], scores


# ---------------------------------------------------------------------------
# Combined defence pipeline
# ---------------------------------------------------------------------------

def run_defence(
    weight_list: List[StateDict],
    node_ids: List[str],
    cosine_threshold: float = 0.0,
    norm_k_sigma: float = 2.0,
) -> Tuple[List[StateDict], List[str], List[str], dict]:
    """
    Apply cosine screening then norm screening sequentially.

    Returns:
        clean_weights:  Filtered state_dicts ready for aggregation.
        clean_ids:      Trusted node IDs.
        flagged_ids:    All flagged node IDs (union of both screens).
        report:         Dict with per-method scores for dashboard.
    """
    cos_sims = cosine_similarities(weight_list)

    w1, cids, flagged_cos, _ = screen_by_cosine(
        weight_list, node_ids, threshold=cosine_threshold
    )
    ids_after_cos = [nid for nid in node_ids if nid in cids]

    w2, cids2, flagged_norm, norms = screen_by_norm(w1, ids_after_cos, k_sigma=norm_k_sigma)

    all_flagged = list(set(flagged_cos + flagged_norm))

    report = {
        "cosine_similarities": dict(zip(node_ids, cos_sims)),
        "flagged_cosine": flagged_cos,
        "flagged_norm": flagged_norm,
        "all_flagged": all_flagged,
        "clean_count": len(w2),
    }
    return w2, cids2, all_flagged, report
