"""
Federated Averaging (FedAvg) aggregation logic.
"""
from typing import List, Dict
import torch


def fed_avg(weight_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Average model weights across all participating nodes.

    Args:
        weight_list: List of state_dicts, one per contributing node.

    Returns:
        A single averaged state_dict.
    """
    if not weight_list:
        raise ValueError("Cannot average an empty list of weights")

    avg_weights: Dict[str, torch.Tensor] = {}
    for key in weight_list[0].keys():
        stacked = torch.stack([w[key].float() for w in weight_list])
        avg_weights[key] = stacked.mean(dim=0)

    return avg_weights
