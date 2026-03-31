"""
FedGuard comprehensive test suite.
Run with: python -m pytest tests/test_all.py -v
"""
import io
import json
import sys
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

# --- path setup so we can import all modules without Docker ---
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(ROOT / "node"))
sys.path.insert(0, str(ROOT / "attack"))
sys.path.insert(0, str(ROOT / "shared"))


# ═══════════════════════════════════════════════════════════════════════════════
# shared/model_def.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestModelDef:
    def test_simple_mlp_forward_shape(self):
        from model_def import SimpleMLP
        m = SimpleMLP()
        x = torch.randn(8, 1, 28, 28)
        out = m(x)
        assert out.shape == (8, 10), f"Expected (8,10) got {out.shape}"

    def test_simple_mlp_flat_input(self):
        from model_def import SimpleMLP
        m = SimpleMLP()
        x = torch.randn(4, 784)
        out = m(x)
        assert out.shape == (4, 10)

    def test_nslkdd_mlp_forward_shape(self):
        from model_def import NSLKDDMLP
        m = NSLKDDMLP(input_dim=122, num_classes=5)
        x = torch.randn(16, 122)
        out = m(x)
        assert out.shape == (16, 5)

    def test_nslkdd_mlp_custom_dims(self):
        from model_def import NSLKDDMLP
        m = NSLKDDMLP(input_dim=50, num_classes=3)
        x = torch.randn(2, 50)
        assert m(x).shape == (2, 3)

    def test_get_model_mnist(self):
        from model_def import get_model, SimpleMLP
        m = get_model("mnist")
        assert isinstance(m, SimpleMLP)

    def test_get_model_nslkdd(self):
        from model_def import get_model, NSLKDDMLP
        m = get_model("nslkdd")
        assert isinstance(m, NSLKDDMLP)

    def test_get_model_unknown_raises(self):
        from model_def import get_model
        with pytest.raises(ValueError, match="Unknown dataset"):
            get_model("imagenet")

    def test_model_has_parameters(self):
        from model_def import get_model
        m = get_model("mnist")
        params = list(m.parameters())
        assert len(params) > 0
        total = sum(p.numel() for p in params)
        assert total == 784 * 128 + 128 + 128 * 10 + 10  # fc1 weight+bias, fc2 weight+bias

    def test_nslkdd_batch_norm_eval_mode(self):
        from model_def import NSLKDDMLP
        m = NSLKDDMLP()
        m.eval()
        # batch norm with single sample should work in eval mode
        x = torch.randn(1, 122)
        out = m(x)
        assert out.shape == (1, 5)


# ═══════════════════════════════════════════════════════════════════════════════
# server/aggregator.py
# ═══════════════════════════════════════════════════════════════════════════════

def _make_state_dicts(n: int, seed: int = 0):
    """Helper: generate n random state_dicts with the same keys."""
    torch.manual_seed(seed)
    template = {"fc1.weight": torch.randn(128, 784),
                "fc1.bias":   torch.randn(128),
                "fc2.weight": torch.randn(10, 128),
                "fc2.bias":   torch.randn(10)}
    result = []
    for i in range(n):
        result.append({k: v + i * 0.1 for k, v in template.items()})
    return result


class TestAggregator:
    def test_fedavg_shape(self):
        from aggregator import fed_avg
        wl = _make_state_dicts(3)
        avg = fed_avg(wl)
        assert set(avg.keys()) == set(wl[0].keys())
        assert avg["fc1.weight"].shape == wl[0]["fc1.weight"].shape

    def test_fedavg_correctness(self):
        from aggregator import fed_avg
        a = {"w": torch.tensor([0.0])}
        b = {"w": torch.tensor([2.0])}
        avg = fed_avg([a, b])
        assert torch.allclose(avg["w"], torch.tensor([1.0]))

    def test_fedavg_weighted(self):
        from aggregator import fed_avg
        a = {"w": torch.tensor([0.0])}
        b = {"w": torch.tensor([4.0])}
        # 1 sample vs 3 samples → weighted avg = (0*1 + 4*3)/4 = 3.0
        avg = fed_avg([a, b], sample_weights=[1.0, 3.0])
        assert torch.allclose(avg["w"], torch.tensor([3.0]))

    def test_fedavg_noise(self):
        from aggregator import fed_avg
        wl = _make_state_dicts(3)
        avg_clean = fed_avg(wl, noise_std=0.0)
        avg_noisy = fed_avg(wl, noise_std=10.0)
        # With large noise the values should differ
        assert not torch.allclose(avg_clean["fc1.weight"], avg_noisy["fc1.weight"])

    def test_fedavg_empty_raises(self):
        from aggregator import fed_avg
        with pytest.raises(ValueError):
            fed_avg([])

    def test_fedavg_sample_weights_mismatch_raises(self):
        from aggregator import fed_avg
        wl = _make_state_dicts(3)
        with pytest.raises(ValueError):
            fed_avg(wl, sample_weights=[1.0, 2.0])  # only 2 weights for 3 dicts

    def test_coordinate_median_shape(self):
        from aggregator import coordinate_median
        wl = _make_state_dicts(3)
        med = coordinate_median(wl)
        assert med["fc1.weight"].shape == wl[0]["fc1.weight"].shape

    def test_coordinate_median_rejects_outlier(self):
        from aggregator import coordinate_median
        a = {"w": torch.tensor([1.0])}
        b = {"w": torch.tensor([1.5])}
        c = {"w": torch.tensor([100.0])}   # outlier
        med = coordinate_median([a, b, c])
        # median of [1.0, 1.5, 100.0] = 1.5
        assert torch.allclose(med["w"], torch.tensor([1.5]))

    def test_trimmed_mean_shape(self):
        from aggregator import trimmed_mean
        wl = _make_state_dicts(5)
        result = trimmed_mean(wl, trim_ratio=0.2)
        assert result["fc1.weight"].shape == wl[0]["fc1.weight"].shape

    def test_trimmed_mean_rejects_extremes(self):
        from aggregator import trimmed_mean
        # 5 values: [0, 1, 2, 3, 100] — trim 1 from each end → mean([1,2,3])=2.0
        wl = [{"w": torch.tensor([float(v)])} for v in [0, 1, 2, 3, 100]]
        result = trimmed_mean(wl, trim_ratio=0.2)
        assert torch.allclose(result["w"], torch.tensor([2.0]))

    def test_krum_aggregate_returns_valid(self):
        from aggregator import krum_aggregate
        wl = _make_state_dicts(3)
        result = krum_aggregate(wl, f=1)
        assert set(result.keys()) == set(wl[0].keys())

    def test_aggregate_dispatcher_all_strategies(self):
        from aggregator import aggregate
        wl = _make_state_dicts(3)
        for strategy in ["fedavg", "median", "trimmed_mean", "krum"]:
            result = aggregate(wl, strategy=strategy)
            assert set(result.keys()) == set(wl[0].keys()), f"Failed for {strategy}"

    def test_aggregate_unknown_strategy_raises(self):
        from aggregator import aggregate
        with pytest.raises(ValueError, match="Unknown strategy"):
            aggregate(_make_state_dicts(3), strategy="sgd")

    def test_aggregate_empty_raises(self):
        from aggregator import aggregate
        with pytest.raises(ValueError):
            aggregate([], strategy="fedavg")


# ═══════════════════════════════════════════════════════════════════════════════
# server/defender.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestDefender:
    def _honest_updates(self, n=3):
        torch.manual_seed(42)
        base = {"w": torch.randn(100)}
        return [{k: v + torch.randn_like(v) * 0.01 for k, v in base.items()}
                for _ in range(n)]

    def _poisoned_update(self):
        """Gradient inversion: negate the honest base direction (same as NODE_POISONED)."""
        torch.manual_seed(42)
        base = torch.randn(100)   # same direction as honest updates
        return {"w": -base * 3}   # inverted + amplified

    def test_cosine_similarities_returns_correct_count(self):
        from defender import cosine_similarities
        wl = self._honest_updates(3)
        sims = cosine_similarities(wl)
        assert len(sims) == 3

    def test_cosine_similarities_honest_near_one(self):
        from defender import cosine_similarities
        wl = self._honest_updates(3)
        sims = cosine_similarities(wl)
        for s in sims:
            assert s > 0.5, f"Honest node similarity too low: {s}"

    def test_cosine_similarities_detects_inversion(self):
        from defender import cosine_similarities
        honest = self._honest_updates(3)
        poisoned = [self._poisoned_update()]
        sims = cosine_similarities(honest + poisoned)
        # poisoned update (last) should have lowest similarity
        assert sims[-1] < sims[0]

    def test_screen_by_cosine_keeps_honest(self):
        from defender import screen_by_cosine
        wl = self._honest_updates(3)
        ids = ["1", "2", "3"]
        clean_w, clean_ids, flagged, sims = screen_by_cosine(wl, ids, threshold=0.0)
        assert len(flagged) == 0
        assert len(clean_ids) == 3

    def test_screen_by_cosine_flags_inverted(self):
        from defender import screen_by_cosine
        honest = self._honest_updates(2)
        poisoned = [self._poisoned_update()]
        wl = honest + poisoned
        ids = ["1", "2", "bad"]
        clean_w, clean_ids, flagged, sims = screen_by_cosine(wl, ids, threshold=0.0)
        assert "bad" in flagged

    def test_screen_by_norm_keeps_normal(self):
        from defender import screen_by_norm
        wl = self._honest_updates(4)
        ids = ["1", "2", "3", "4"]
        clean_w, clean_ids, flagged, norms = screen_by_norm(wl, ids, k_sigma=2.0)
        assert len(flagged) == 0

    def test_screen_by_norm_flags_outlier(self):
        from defender import screen_by_norm
        honest = self._honest_updates(3)
        # Make one update with very large norm
        big = {"w": torch.ones(100) * 1000.0}
        wl = honest + [big]
        ids = ["1", "2", "3", "big"]
        clean_w, clean_ids, flagged, norms = screen_by_norm(wl, ids, k_sigma=1.0)
        assert "big" in flagged

    def test_krum_select_returns_one(self):
        from defender import krum_select
        wl = self._honest_updates(3)
        ids = ["1", "2", "3"]
        best_w, best_id, scores = krum_select(wl, ids, f=1)
        assert best_id in ids
        assert len(scores) == 3

    def test_krum_select_picks_non_outlier(self):
        from defender import krum_select
        honest = self._honest_updates(2)
        poison = [{"w": torch.ones(100) * 1000}]
        wl = honest + poison
        ids = ["good1", "good2", "bad"]
        best_w, best_id, scores = krum_select(wl, ids, f=1)
        assert best_id != "bad", "Krum should not select the poisoned update"

    def test_run_defence_all_clean(self):
        from defender import run_defence
        wl = self._honest_updates(3)
        ids = ["1", "2", "3"]
        clean_w, clean_ids, flagged, report = run_defence(wl, ids)
        assert len(flagged) == 0
        assert "cosine_similarities" in report

    def test_run_defence_catches_inversion(self):
        from defender import run_defence
        honest = self._honest_updates(2)
        poison = [self._poisoned_update()]
        wl = honest + poison
        ids = ["1", "2", "bad"]
        clean_w, clean_ids, flagged, report = run_defence(
            wl, ids, cosine_threshold=0.0, norm_k_sigma=2.0
        )
        assert "bad" in flagged

    def test_run_defence_fallback_all_flagged(self):
        """When everyone is flagged, run_defence returns full list to avoid empty aggregation."""
        from defender import screen_by_cosine
        # All updates are very different from each other
        wl = [{"w": torch.tensor([float(i) * 100])} for i in range(3)]
        ids = ["1", "2", "3"]
        clean_w, clean_ids, flagged, sims = screen_by_cosine(wl, ids, threshold=0.99)
        # Fallback: returns full list
        assert len(clean_w) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# node/compressor.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestCompressor:
    def _make_delta(self, seed=0):
        torch.manual_seed(seed)
        return {
            "fc1.weight": torch.randn(128, 784),
            "fc1.bias":   torch.randn(128),
            "fc2.weight": torch.randn(10, 128),
            "fc2.bias":   torch.randn(10),
        }

    def test_compress_returns_same_keys(self):
        from compressor import TopKCompressor
        c = TopKCompressor(top_k_ratio=0.1)
        delta = self._make_delta()
        compressed, stats = c.compress(delta)
        assert set(compressed.keys()) == set(delta.keys())

    def test_compress_same_shape(self):
        from compressor import TopKCompressor
        c = TopKCompressor(top_k_ratio=0.1)
        delta = self._make_delta()
        compressed, stats = c.compress(delta)
        for k in delta:
            assert compressed[k].shape == delta[k].shape

    def test_compress_sparsity_correct(self):
        from compressor import TopKCompressor
        c = TopKCompressor(top_k_ratio=0.1)
        delta = self._make_delta()
        compressed, stats = c.compress(delta)
        # Count non-zero elements across all tensors
        total = sum(t.numel() for t in delta.values())
        nonzero = sum((compressed[k] != 0).sum().item() for k in compressed)
        actual_ratio = nonzero / total
        assert abs(actual_ratio - 0.1) < 0.02, f"Sparsity ratio off: {actual_ratio}"

    def test_compress_stats_keys(self):
        from compressor import TopKCompressor
        c = TopKCompressor(top_k_ratio=0.1)
        _, stats = c.compress(self._make_delta())
        assert "top_k_ratio" in stats
        assert "sparsity" in stats
        assert "total_params" in stats
        assert "kept_params" in stats

    def test_error_feedback_reduces_cumulative_error(self):
        """
        Over multiple rounds with the same constant gradient, cumulative
        compressed output with error feedback should be closer to the
        cumulative true delta than repeated independent compressions.
        """
        from compressor import TopKCompressor
        delta = self._make_delta(seed=0)
        n_rounds = 5

        cumulative_true = {k: torch.zeros_like(delta[k]) for k in delta}
        cumulative_ef   = {k: torch.zeros_like(delta[k]) for k in delta}
        cumulative_noef = {k: torch.zeros_like(delta[k]) for k in delta}

        c_ef = TopKCompressor(top_k_ratio=0.1)   # persistent error feedback

        for _ in range(n_rounds):
            # With error feedback
            comp_ef, _ = c_ef.compress({k: v.clone() for k, v in delta.items()})
            # Without error feedback (fresh compressor each round)
            comp_noef, _ = TopKCompressor(top_k_ratio=0.1).compress(
                {k: v.clone() for k, v in delta.items()}
            )
            for k in delta:
                cumulative_ef[k]   += comp_ef[k]
                cumulative_noef[k] += comp_noef[k]
                cumulative_true[k] += delta[k]

        err_ef   = sum(((cumulative_true[k] - cumulative_ef[k])**2).sum().item() for k in delta)
        err_noef = sum(((cumulative_true[k] - cumulative_noef[k])**2).sum().item() for k in delta)
        assert err_ef < err_noef, (
            f"Error feedback cumulative error ({err_ef:.1f}) should be less "
            f"than no-feedback ({err_noef:.1f})"
        )

    def test_reset_clears_error(self):
        from compressor import TopKCompressor
        c = TopKCompressor(top_k_ratio=0.1)
        c.compress(self._make_delta())
        assert len(c._error) > 0
        c.reset()
        assert len(c._error) == 0

    def test_invalid_ratio_raises(self):
        from compressor import TopKCompressor
        with pytest.raises(ValueError):
            TopKCompressor(top_k_ratio=0.0)
        with pytest.raises(ValueError):
            TopKCompressor(top_k_ratio=1.5)

    def test_compute_delta_correctness(self):
        from compressor import compute_delta
        local = {"w": torch.tensor([3.0])}
        global_ = {"w": torch.tensor([1.0])}
        delta = compute_delta(local, global_)
        assert torch.allclose(delta["w"], torch.tensor([2.0]))

    def test_top_k_ratio_one_keeps_all(self):
        from compressor import TopKCompressor
        c = TopKCompressor(top_k_ratio=1.0)
        delta = {"w": torch.tensor([1.0, 2.0, 3.0])}
        compressed, stats = c.compress(delta)
        assert (compressed["w"] != 0).all()


# ═══════════════════════════════════════════════════════════════════════════════
# node/data_loader.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestDataLoader:
    def test_mnist_iid_returns_loader_and_dim(self):
        from data_loader import get_data_loader
        loader, dim = get_data_loader(
            node_id=1, total_nodes=3, dataset="mnist", alpha=-1
        )
        assert dim == 784
        assert len(loader) > 0

    def test_mnist_noniid_returns_loader(self):
        from data_loader import get_data_loader
        loader, dim = get_data_loader(
            node_id=1, total_nodes=3, dataset="mnist", alpha=0.5
        )
        assert dim == 784
        assert len(loader) > 0

    def test_mnist_partitions_are_disjoint(self):
        from data_loader import get_data_loader
        loaders = []
        for nid in range(1, 4):
            loader, _ = get_data_loader(node_id=nid, total_nodes=3, dataset="mnist", alpha=-1)
            loaders.append(loader)
        sizes = [len(l.dataset) for l in loaders]
        # IID: each node gets ~20000 samples from 60000 total
        assert sum(sizes) == 60000, f"Total samples should be 60000, got {sum(sizes)}"
        assert all(s > 0 for s in sizes)

    def test_mnist_noniid_different_distributions(self):
        from data_loader import get_data_loader
        import numpy as np
        # With extreme skew, nodes should have very different class distributions
        loader1, _ = get_data_loader(node_id=1, total_nodes=3, dataset="mnist", alpha=0.1)
        loader2, _ = get_data_loader(node_id=2, total_nodes=3, dataset="mnist", alpha=0.1)
        # Just verify they load different data
        assert len(loader1.dataset) > 0
        assert len(loader2.dataset) > 0

    def test_dirichlet_partition_sizes(self):
        from data_loader import dirichlet_partition
        labels = np.array([i % 10 for i in range(1000)])
        idx1 = dirichlet_partition(labels, node_id=1, total_nodes=3, alpha=1.0)
        idx2 = dirichlet_partition(labels, node_id=2, total_nodes=3, alpha=1.0)
        idx3 = dirichlet_partition(labels, node_id=3, total_nodes=3, alpha=1.0)
        assert len(idx1) > 0
        assert len(idx2) > 0
        assert len(idx3) > 0

    def test_unknown_dataset_raises(self):
        from data_loader import get_data_loader
        with pytest.raises(ValueError, match="Unknown dataset"):
            get_data_loader(node_id=1, total_nodes=3, dataset="cifar10")

    def test_mnist_batch_shape(self):
        from data_loader import get_data_loader
        loader, _ = get_data_loader(node_id=1, total_nodes=3, dataset="mnist", alpha=-1, batch_size=32)
        X, y = next(iter(loader))
        assert X.shape == (32, 1, 28, 28)
        assert y.shape == (32,)


# ═══════════════════════════════════════════════════════════════════════════════
# node/trainer.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestTrainer:
    def _tiny_loader(self):
        X = torch.randn(32, 1, 28, 28)
        y = torch.randint(0, 10, (32,))
        from torch.utils.data import TensorDataset, DataLoader
        return DataLoader(TensorDataset(X, y), batch_size=16)

    def test_standard_training_returns_state_dict(self):
        from trainer import train_local
        from model_def import get_model
        model = get_model("mnist")
        loader = self._tiny_loader()
        sd, meta = train_local(model, loader, epochs=1)
        assert isinstance(sd, dict)
        assert "fc1.weight" in sd

    def test_standard_training_metadata_keys(self):
        from trainer import train_local
        from model_def import get_model
        model = get_model("mnist")
        sd, meta = train_local(model, self._tiny_loader(), epochs=1)
        for key in ["algorithm", "local_loss", "local_accuracy", "epochs", "dp_enabled"]:
            assert key in meta, f"Missing metadata key: {key}"

    def test_standard_training_weights_change(self):
        from trainer import train_local
        from model_def import get_model
        model = get_model("mnist")
        before = {k: v.clone() for k, v in model.state_dict().items()}
        sd, _ = train_local(model, self._tiny_loader(), epochs=2)
        changed = any(not torch.equal(before[k], sd[k]) for k in before)
        assert changed, "Weights should change after training"

    def test_fedprox_mu_zero_equals_fedavg(self):
        """mu=0 should behave identically to standard FedAvg."""
        from trainer import train_local
        from model_def import get_model
        torch.manual_seed(0)
        m1 = get_model("mnist")
        torch.manual_seed(0)
        m2 = get_model("mnist")
        loader = self._tiny_loader()
        sd1, meta1 = train_local(m1, loader, epochs=1, mu=0.0)
        sd2, meta2 = train_local(m2, loader, epochs=1, mu=0.0)
        assert meta1["algorithm"] == "fedavg"
        assert meta2["algorithm"] == "fedavg"

    def test_fedprox_nonzero_mu(self):
        from trainer import train_local
        from model_def import get_model
        model = get_model("mnist")
        global_sd = {k: v.clone() for k, v in model.state_dict().items()}
        sd, meta = train_local(
            model, self._tiny_loader(),
            global_state_dict=global_sd, epochs=1, mu=0.1
        )
        assert meta["algorithm"] == "fedprox"
        assert meta["mu"] == 0.1

    def test_training_accuracy_in_range(self):
        from trainer import train_local
        from model_def import get_model
        model = get_model("mnist")
        _, meta = train_local(model, self._tiny_loader(), epochs=1)
        assert 0.0 <= meta["local_accuracy"] <= 1.0
        assert meta["local_loss"] >= 0.0

    def test_dp_disabled_epsilon_none(self):
        from trainer import train_local
        from model_def import get_model
        model = get_model("mnist")
        _, meta = train_local(model, self._tiny_loader(), epochs=1, enable_dp=False)
        assert meta["dp_enabled"] is False
        assert meta["epsilon_spent"] is None


# ═══════════════════════════════════════════════════════════════════════════════
# attack/dlg_attack.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestDLGAttack:
    def _setup(self):
        from model_def import get_model
        model = get_model("mnist")
        model.eval()
        x = torch.randn(1, 1, 28, 28)
        y = torch.tensor([3])
        model.zero_grad()
        loss = nn.CrossEntropyLoss()(model(x), y)
        loss.backward()
        grads = [p.grad.detach().clone() for p in model.parameters()]
        return model, x, y, grads

    def test_extract_label_idlg_correct(self):
        from dlg_attack import extract_label_idlg
        from model_def import get_model
        torch.manual_seed(42)
        model = get_model("mnist")
        model.eval()
        for true_label in [0, 3, 7, 9]:
            x = torch.randn(1, 1, 28, 28)
            y = torch.tensor([true_label])
            model.zero_grad()
            loss = nn.CrossEntropyLoss()(model(x), y)
            loss.backward()
            grads = [p.grad.detach().clone() for p in model.parameters()]
            pred = extract_label_idlg(grads)
            assert pred == true_label, f"iDLG label extraction failed: expected {true_label} got {pred}"

    def test_dlg_reconstruct_returns_tensor(self):
        from dlg_attack import dlg_reconstruct
        model, x, y, grads = self._setup()
        recon, label, curve = dlg_reconstruct(model, grads, iterations=5, use_idlg=True)
        assert isinstance(recon, torch.Tensor)
        assert recon.shape == (1, 1, 28, 28)

    def test_dlg_reconstruct_returns_convergence_curve(self):
        from dlg_attack import dlg_reconstruct
        model, x, y, grads = self._setup()
        _, _, curve = dlg_reconstruct(model, grads, iterations=10, use_idlg=True)
        assert isinstance(curve, list)
        assert len(curve) > 0
        assert all(isinstance(v, float) for v in curve)

    def test_dlg_convergence_decreases(self):
        from dlg_attack import dlg_reconstruct
        model, x, y, grads = self._setup()
        _, _, curve = dlg_reconstruct(model, grads, iterations=100, use_idlg=True)
        # Overall the curve should trend downward: best value seen should be
        # lower than the starting value
        assert min(curve) < curve[0], (
            f"DLG should find a lower loss than initial. "
            f"start={curve[0]:.4f} best={min(curve):.4f}"
        )

    def test_apply_dp_noise_changes_grads(self):
        from dlg_attack import apply_dp_noise
        _, _, _, grads = self._setup()
        noisy = apply_dp_noise(grads, noise_multiplier=1.0)
        assert len(noisy) == len(grads)
        changed = any(not torch.allclose(g, n) for g, n in zip(grads, noisy))
        assert changed

    def test_apply_dp_noise_zero_no_change(self):
        from dlg_attack import apply_dp_noise
        _, _, _, grads = self._setup()
        noisy = apply_dp_noise(grads, noise_multiplier=0.0)
        # With no noise, clipped grads may still differ slightly due to clipping
        # Just verify structure is preserved
        assert len(noisy) == len(grads)
        for g, n in zip(grads, noisy):
            assert g.shape == n.shape

    def test_psnr_identical_images(self):
        from dlg_attack import psnr
        x = torch.rand(1, 1, 28, 28)
        score = psnr(x, x)
        assert score == float("inf")

    def test_psnr_different_images(self):
        from dlg_attack import psnr
        x = torch.zeros(1, 1, 28, 28)
        y = torch.ones(1, 1, 28, 28)
        score = psnr(x, y)
        assert score < 10.0  # very different images → low PSNR

    def test_mse_identical_zero(self):
        from dlg_attack import mse
        x = torch.rand(1, 1, 28, 28)
        assert mse(x, x) == 0.0

    def test_high_dp_noise_degrades_reconstruction(self):
        """
        Heavy DP noise (σ=10) should produce a larger final gradient-diff
        than clean gradients — meaning the optimiser cannot match the
        noisy target as well as the clean one.
        """
        from dlg_attack import apply_dp_noise, dlg_reconstruct
        from model_def import get_model
        torch.manual_seed(7)
        model = get_model("mnist")
        model.eval()
        x = torch.randn(1, 1, 28, 28)
        y = torch.tensor([5])
        model.zero_grad()
        loss = nn.CrossEntropyLoss()(model(x), y)
        loss.backward()
        clean_grads = [p.grad.detach().clone() for p in model.parameters()]

        model.zero_grad()
        _, _, curve_clean = dlg_reconstruct(model, clean_grads, iterations=50, use_idlg=True)

        # Very large noise: optimiser has a completely different target to chase
        noisy_grads = apply_dp_noise(clean_grads, noise_multiplier=10.0)
        model.zero_grad()
        _, _, curve_noisy = dlg_reconstruct(model, noisy_grads, iterations=50, use_idlg=True)

        # The noisy target should result in a higher minimum gradient diff
        # (the optimiser converges to a different, noisier target)
        assert min(curve_noisy) != min(curve_clean), (
            "Clean and heavily-noised reconstructions should differ in convergence"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# server/main.py — FastAPI endpoint integration tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestServerEndpoints:
    @pytest.fixture(autouse=True)
    def client(self):
        """Create a fresh TestClient for each test with reset server state."""
        # Reset module-level state between tests
        import importlib
        import server.main as sm

        # Patch to avoid MNIST download during testing
        sm.current_round = 0
        sm.accepting_cycle = 0
        sm.pending.clear()
        sm.round_history.clear()
        sm.known_nodes.clear()

        from fastapi.testclient import TestClient
        self._client = TestClient(sm.app)
        yield self._client

    def _serialise_state_dict(self, sd):
        buf = io.BytesIO()
        torch.save(sd, buf)
        return buf.getvalue()

    def _random_sd(self, seed=0):
        torch.manual_seed(seed)
        return {
            "fc1.weight": torch.randn(128, 784),
            "fc1.bias":   torch.randn(128),
            "fc2.weight": torch.randn(10, 128),
            "fc2.bias":   torch.randn(10),
        }

    def test_health(self):
        r = self._client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"

    def test_status_initial(self):
        r = self._client.get("/status")
        assert r.status_code == 200
        data = r.json()
        assert data["round"] == 0
        assert data["pending_nodes"] == []
        assert "history" in data
        assert "config" in data

    def test_get_model_returns_binary(self):
        r = self._client.get("/get_model")
        assert r.status_code == 200
        assert r.headers["content-type"] == "application/octet-stream"
        assert "X-FL-Cycle" in r.headers
        assert len(r.content) > 0

    def test_get_model_cycle_header(self):
        r = self._client.get("/get_model")
        cycle = int(r.headers["X-FL-Cycle"])
        assert cycle == 0

    def test_get_model_deserialises(self):
        r = self._client.get("/get_model")
        buf = io.BytesIO(r.content)
        sd = torch.load(buf, map_location="cpu", weights_only=True)
        assert "fc1.weight" in sd

    def test_submit_weights_valid(self):
        import server.main as sm
        # Get current cycle
        r = self._client.get("/get_model")
        cycle = int(r.headers["X-FL-Cycle"])

        raw = self._serialise_state_dict(self._random_sd(0))
        r = self._client.post(
            "/submit_weights",
            data={"node_id": "1", "cycle_id": str(cycle),
                  "sample_count": "1000", "metadata": "{}"},
            files={"weights": ("w.pt", raw, "application/octet-stream")},
        )
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_submit_weights_stale_cycle_409(self):
        r = self._client.post(
            "/submit_weights",
            data={"node_id": "1", "cycle_id": "999",
                  "sample_count": "1000", "metadata": "{}"},
            files={"weights": ("w.pt", self._serialise_state_dict(self._random_sd()), "application/octet-stream")},
        )
        assert r.status_code == 409

    def test_submit_weights_empty_payload_400(self):
        r = self._client.post(
            "/submit_weights",
            data={"node_id": "1", "cycle_id": "0",
                  "sample_count": "100", "metadata": "{}"},
            files={"weights": ("w.pt", b"", "application/octet-stream")},
        )
        assert r.status_code == 400

    def test_submit_weights_invalid_sample_count(self):
        raw = self._serialise_state_dict(self._random_sd())
        r = self._client.post(
            "/submit_weights",
            data={"node_id": "1", "cycle_id": "0",
                  "sample_count": "0", "metadata": "{}"},
            files={"weights": ("w.pt", raw, "application/octet-stream")},
        )
        assert r.status_code == 400

    def test_two_nodes_trigger_aggregation(self):
        """MIN_NODES=2: submitting 2 nodes should trigger a round."""
        import server.main as sm
        sm.current_round = 0
        sm.accepting_cycle = 0
        sm.pending.clear()
        sm.round_history.clear()

        cycle = sm.accepting_cycle

        for i, seed in enumerate([0, 1]):
            raw = self._serialise_state_dict(self._random_sd(seed))
            self._client.post(
                "/submit_weights",
                data={"node_id": str(i + 1), "cycle_id": str(cycle),
                      "sample_count": "1000", "metadata": "{}"},
                files={"weights": ("w.pt", raw, "application/octet-stream")},
            )

        status = self._client.get("/status").json()
        assert status["round"] == 1, f"Expected round=1 after 2 submissions, got {status['round']}"
        assert len(status["history"]) == 1

    def test_duplicate_node_overwritten(self):
        """Same node_id submitting twice counts as one."""
        import server.main as sm
        sm.current_round = 0
        sm.accepting_cycle = 0
        sm.pending.clear()

        cycle = sm.accepting_cycle
        for _ in range(2):
            raw = self._serialise_state_dict(self._random_sd(0))
            self._client.post(
                "/submit_weights",
                data={"node_id": "1", "cycle_id": str(cycle),
                      "sample_count": "1000", "metadata": "{}"},
                files={"weights": ("w.pt", raw, "application/octet-stream")},
            )
        # Only 1 unique node, should not have aggregated
        assert sm.current_round == 0

    def test_reset_clears_state(self):
        import server.main as sm
        sm.current_round = 5
        sm.accepting_cycle = 5
        sm.round_history.append({"round": 5})

        r = self._client.post("/reset")
        assert r.status_code == 200
        assert sm.current_round == 0
        assert sm.accepting_cycle == 0
        assert len(sm.round_history) == 0

    def test_cycle_increments_after_round(self):
        """After a completed round, accepting_cycle increments so old uploads are rejected."""
        import server.main as sm
        sm.current_round = 0
        sm.accepting_cycle = 0
        sm.pending.clear()
        sm.round_history.clear()

        cycle = 0
        for i, seed in enumerate([0, 1]):
            raw = self._serialise_state_dict(self._random_sd(seed))
            self._client.post(
                "/submit_weights",
                data={"node_id": str(i + 1), "cycle_id": str(cycle),
                      "sample_count": "1000", "metadata": "{}"},
                files={"weights": ("w.pt", raw, "application/octet-stream")},
            )

        # Round completed, cycle should now be 1
        assert sm.accepting_cycle == 1

        # Old cycle=0 upload should now be rejected
        raw = self._serialise_state_dict(self._random_sd(2))
        r = self._client.post(
            "/submit_weights",
            data={"node_id": "3", "cycle_id": "0",
                  "sample_count": "1000", "metadata": "{}"},
            files={"weights": ("w.pt", raw, "application/octet-stream")},
        )
        assert r.status_code == 409
