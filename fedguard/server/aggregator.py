"""
Federated aggregation strategies for FedGuard.

Strategies
----------
fedavg        — coordinate-wise mean (McMahan et al., 2017)
median        — coordinate-wise median; robust to sign-flip attacks
trimmed_mean  — trim top/bottom k% then average; robust to outlier values
krum          — select the single most-central update (Blanchard et al., 2017)

All functions accept List[state_dict] and return a single aggregated state_dict.
The caller is responsible for Byzantine screening before passing to these
functions (or use krum which has built-in robustness).
"""
import logging
from typing import Dict, List

import torch

log = logging.getLogger(__name__)

StateDict = Dict[str, torch.Tensor]


# ---------------------------------------------------------------------------
# FedAvg
# ---------------------------------------------------------------------------

def fed_avg(weight_list: List[StateDict]) -> StateDict:
    """
    Standard Federated Averaging: coordinate-wise mean across all nodes.
    Fast, simple, optimal on IID data.  Diverges on highly skewed non-IID.
    """
    if not weight_list:
        raise ValueError("Cannot aggregate empty list")
    avg: StateDict = {}
    for key in weight_list[0]:
        avg[key] = torch.stack([w[key].float() for w in weight_list]).mean(dim=0)
    return avg


# ---------------------------------------------------------------------------
# Coordinate-wise median
# ---------------------------------------------------------------------------

def coordinate_median(weight_list: List[StateDict]) -> StateDict:
    """
    Coordinate-wise median.

    For each parameter element, take the median across nodes instead of
    the mean.  Tolerates up to ⌊(n-1)/2⌋ Byzantine nodes in theory.
    Robust to sign-flip and large-norm attacks.
    """
    if not weight_list:
        raise ValueError("Cannot aggregate empty list")
    med: StateDict = {}
    for key in weight_list[0]:
        stacked = torch.stack([w[key].float() for w in weight_list])
        med[key] = stacked.median(dim=0).values
    return med


# ---------------------------------------------------------------------------
# Trimmed mean
# ---------------------------------------------------------------------------

def trimmed_mean(weight_list: List[StateDict], trim_ratio: float = 0.1) -> StateDict:
    """
    Trim the top and bottom *trim_ratio* fraction of values per coordinate,
    then average the remaining values.

    With trim_ratio=0.1 and n=3 nodes, k=0 (no trimming) — use n≥5 to see
    effect.  Tolerates up to ⌊n·trim_ratio⌋ adversaries at each coordinate.
    """
    if not weight_list:
        raise ValueError("Cannot aggregate empty list")
    n = len(weight_list)
    k = max(0, int(n * trim_ratio))
    result: StateDict = {}

    for key in weight_list[0]:
        stacked = torch.stack([w[key].float() for w in weight_list])
        sorted_vals, _ = stacked.sort(dim=0)
        if 2 * k < n:
            trimmed = sorted_vals[k: n - k]
        else:
            trimmed = sorted_vals   # nothing to trim with few nodes
        result[key] = trimmed.mean(dim=0)

    return result


# ---------------------------------------------------------------------------
# Krum  (built-in Byzantine tolerance)
# ---------------------------------------------------------------------------

def krum_aggregate(weight_list: List[StateDict], f: int = 1) -> StateDict:
    """
    Krum: select the update closest to its n-f-1 nearest neighbours.

    Unlike the other strategies this SELECTS one update rather than
    averaging — information from non-selected nodes is discarded.
    Use when Byzantine fraction may be high (f close to n/2).
    """
    from defender import krum_select
    best_weights, best_id, scores = krum_select(weight_list, [str(i) for i in range(len(weight_list))], f=f)
    log.info("Krum selected update index %s (scores=%s)", best_id, scores)
    return best_weights


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_STRATEGIES = {
    "fedavg":       fed_avg,
    "median":       coordinate_median,
    "trimmed_mean": trimmed_mean,
    "krum":         krum_aggregate,
}


def aggregate(
    weight_list: List[StateDict],
    strategy: str = "fedavg",
    **kwargs,
) -> StateDict:
    """
    Aggregate *weight_list* using the named *strategy*.

    Args:
        weight_list: List of state_dicts (already Byzantine-screened if desired).
        strategy:    One of fedavg | median | trimmed_mean | krum.
        **kwargs:    Forwarded to the strategy function (e.g. trim_ratio, f).

    Returns:
        Aggregated state_dict.
    """
    if not weight_list:
        raise ValueError("weight_list is empty")
    strategy = strategy.lower()
    if strategy not in _STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from {list(_STRATEGIES)}")
    fn = _STRATEGIES[strategy]
    log.info("Aggregating %d updates with strategy='%s'", len(weight_list), strategy)
    return fn(weight_list, **kwargs)
