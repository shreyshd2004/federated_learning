"""
Top-K gradient / weight-delta compression with error feedback.

How it works
------------
Instead of uploading full model weights after local training, nodes transmit
only weight *deltas* (Δ = w_local − w_global) compressed to keep the top-K%
of values by absolute magnitude.  The rest are zeroed.

Error feedback
--------------
Compression discards low-magnitude values.  Without correction these losses
accumulate round over round.  Error feedback stores the discarded residual
and adds it back to the next round's delta before re-compressing, ensuring
convergence properties similar to uncompressed SGD.

Communication savings
---------------------
top_k_ratio = 0.10  →  ~90 % bandwidth reduction
top_k_ratio = 0.01  →  ~99 % bandwidth reduction (at accuracy cost)

Reference: Alistarh et al., "QSGD" (NeurIPS 2017);
           Stich et al., "Sparsified SGD with Memory" (NeurIPS 2018).
"""
import logging
from typing import Dict, Tuple

import torch

log = logging.getLogger(__name__)


class TopKCompressor:
    """
    Stateful compressor: keeps per-layer error feedback across rounds.

    Usage (node side)
    -----------------
    compressor = TopKCompressor(top_k_ratio=0.1)

    # After local training:
    delta = {k: local_w[k] - global_w[k] for k in local_w}
    compressed_delta, stats = compressor.compress(delta)
    # Ship compressed_delta to server (sparse but same dict structure)

    # Server side:
    reconstructed = decompress(compressed_delta)  # no-op; already full shape
    # Aggregate reconstructed deltas, apply to global model.
    """

    def __init__(self, top_k_ratio: float = 0.1):
        if not 0 < top_k_ratio <= 1.0:
            raise ValueError("top_k_ratio must be in (0, 1]")
        self.top_k_ratio = top_k_ratio
        self._error: Dict[str, torch.Tensor] = {}   # per-layer residual

    # ------------------------------------------------------------------
    # Core ops
    # ------------------------------------------------------------------

    def compress(
        self,
        delta: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], dict]:
        """
        Compress weight delta in-place with Top-K sparsification + error feedback.

        Args:
            delta: {layer_name: Δ_tensor}; difference between local and global weights.

        Returns:
            compressed_delta: same structure, non-top-K values zeroed.
            stats: dict with sparsity, bandwidth metrics.
        """
        compressed: Dict[str, torch.Tensor] = {}
        total_params = kept_params = 0

        for key, tensor in delta.items():
            t = tensor.float()

            # Add accumulated error from previous round
            if key in self._error:
                t = t + self._error[key]

            flat = t.flatten()
            n = flat.numel()
            k = max(1, int(n * self.top_k_ratio))

            # Keep top-K by magnitude; zero the rest
            topk_vals, topk_idx = torch.topk(flat.abs(), k)
            mask = torch.zeros_like(flat)
            mask[topk_idx] = 1.0
            compressed_flat = flat * mask

            # Store residual for next round
            self._error[key] = (t.flatten() - compressed_flat).reshape(tensor.shape).detach()
            compressed[key] = compressed_flat.reshape(tensor.shape)

            total_params += n
            kept_params += k

        sparsity = 1.0 - kept_params / max(total_params, 1)
        stats = {
            "top_k_ratio": self.top_k_ratio,
            "sparsity": round(sparsity, 4),
            "total_params": total_params,
            "kept_params": kept_params,
            "compression_ratio": round(self.top_k_ratio, 4),
        }
        log.debug("Compressed delta: sparsity=%.2f%%  kept=%d/%d params",
                  sparsity * 100, kept_params, total_params)
        return compressed, stats

    def reset(self) -> None:
        """Clear error feedback (call when global model is re-initialised)."""
        self._error.clear()


# ------------------------------------------------------------------
# Server-side helper (stateless; compression is lossless in shape)
# ------------------------------------------------------------------

def compute_delta(
    local_weights: Dict[str, torch.Tensor],
    global_weights: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Δ = w_local − w_global for every layer."""
    return {k: local_weights[k].float() - global_weights[k].float() for k in local_weights}


def apply_delta(
    global_weights: Dict[str, torch.Tensor],
    avg_delta: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """w_global_new = w_global + avg_Δ (used when nodes send deltas)."""
    return {k: global_weights[k].float() + avg_delta[k].float() for k in global_weights}
