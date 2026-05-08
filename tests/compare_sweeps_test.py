# -*- coding: utf-8 -*-
"""Tests for compare_sweeps.py pure functions (no wandb/GPU required).

Since compare_sweeps.py imports torch (not available locally), we mock
the torch/torch_geometric imports and test only the pure numpy functions.
"""
import pytest
import os
import sys
import types
import tempfile
import numpy as np
import pandas as pd

# Mock torch and related modules so we can import pure functions
# without having PyTorch installed locally.
_mock_modules = [
    'torch', 'torch.nn', 'torch.nn.functional',
    'torch_geometric', 'torch_geometric.loader', 'torch_geometric.nn',
    'torch_geometric.nn.pool', 'torch_geometric.utils',
    'wandb', 'tqdm',
    'modelling.gnn.delta_egnn_model',
    'modelling.gnn.settings_only_egnn_model',
    'modelling.gnn.delta_egnn_attention_model',
    'modelling.gnn.egnn_layer',
    'modelling.gnn.egnn_attention_layer',
]
for mod_name in _mock_modules:
    sys.modules[mod_name] = types.ModuleType(mod_name)

# torch.no_grad is used as a decorator
sys.modules['torch'].no_grad = lambda: lambda fn: fn
# torch.cuda.is_available
_torch_cuda = types.ModuleType('torch.cuda')
_torch_cuda.is_available = lambda: False
_torch_cuda.empty_cache = lambda: None
sys.modules['torch'].cuda = _torch_cuda
sys.modules['torch.cuda'] = _torch_cuda
# torch.device
sys.modules['torch'].device = lambda x: x

# Add fake classes so the import doesn't fail on 'from X import Y'
sys.modules['modelling.gnn.delta_egnn_model'].DeltaGNN = type('DeltaGNN', (), {})
sys.modules['modelling.gnn.settings_only_egnn_model'].SettingsOnlyGNN = type('SettingsOnlyGNN', (), {})
sys.modules['modelling.gnn.delta_egnn_attention_model'].DeltaAttentionGNN = type('DeltaAttentionGNN', (), {})
sys.modules['torch_geometric.loader'].DataLoader = type('DataLoader', (), {})
sys.modules['tqdm'].tqdm = lambda x, **kw: x

# Fake sklearn if needed
try:
    from sklearn.metrics import mean_absolute_error
except ImportError:
    sys.modules['sklearn'] = types.ModuleType('sklearn')
    sys.modules['sklearn.metrics'] = types.ModuleType('sklearn.metrics')
    sys.modules['sklearn.metrics'].mean_absolute_error = lambda y, p: float(np.mean(np.abs(np.array(y) - np.array(p))))

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from modelling.gnn.compare_sweeps import (
    calculate_rmsle,
    calculate_smape,
    calculate_mag_acc,
    compute_metrics_for_targets,
    ensemble_arrays,
    arrays_to_csv,
)


# ============================================================
# calculate_rmsle
# ============================================================

class TestRMSLE:
    def test_identical_values(self):
        y = np.array([1.0, 2.0, 5.0, 10.0])
        assert calculate_rmsle(y, y) == pytest.approx(0.0, abs=1e-6)

    def test_known_values(self):
        # log(|1| + eps) vs log(|e| + eps) ~ 0 vs 1 -> RMSLE ~ 1.0
        y_true = np.array([1.0])
        y_pred = np.array([np.e])
        result = calculate_rmsle(y_true, y_pred)
        assert result == pytest.approx(1.0, abs=0.01)

    def test_zeros_dont_crash(self):
        result = calculate_rmsle(np.array([0.0]), np.array([0.0]))
        assert np.isfinite(result)

    def test_negative_values_use_abs(self):
        r1 = calculate_rmsle(np.array([1.0]), np.array([2.0]))
        r2 = calculate_rmsle(np.array([-1.0]), np.array([-2.0]))
        assert r1 == pytest.approx(r2, abs=1e-10)

    def test_symmetry(self):
        a = np.array([1.0, 5.0, 10.0])
        b = np.array([2.0, 3.0, 8.0])
        assert calculate_rmsle(a, b) == pytest.approx(calculate_rmsle(b, a), abs=1e-10)

    def test_positive_result(self):
        result = calculate_rmsle(np.array([1.0]), np.array([100.0]))
        assert result > 0


# ============================================================
# calculate_smape
# ============================================================

class TestSMAPE:
    def test_identical_values(self):
        y = np.array([1.0, 5.0, 100.0])
        assert calculate_smape(y, y) == pytest.approx(0.0, abs=1e-4)

    def test_known_value(self):
        # sMAPE = 200 * |110 - 100| / (|110| + |100|) = 200 * 10 / 210
        result = calculate_smape(np.array([100.0]), np.array([110.0]))
        expected = 200.0 * 10.0 / 210.0
        assert result == pytest.approx(expected, abs=0.01)

    def test_zeros_dont_crash(self):
        result = calculate_smape(np.array([0.0]), np.array([0.0]))
        assert np.isfinite(result)
        assert result == pytest.approx(0.0, abs=1.0)

    def test_range_bounded(self):
        np.random.seed(42)
        y_true = np.random.randn(100)
        y_pred = np.random.randn(100)
        result = calculate_smape(y_true, y_pred)
        assert 0.0 <= result <= 200.0

    def test_symmetry(self):
        a = np.array([1.0, 5.0])
        b = np.array([2.0, 3.0])
        assert calculate_smape(a, b) == pytest.approx(calculate_smape(b, a), abs=1e-10)


# ============================================================
# calculate_mag_acc
# ============================================================

class TestMagAcc:
    def test_perfect_accuracy(self):
        y = np.array([0.005, 0.05, 0.5, 5.0, 50.0])
        assert calculate_mag_acc(y, y) == pytest.approx(1.0)

    def test_zero_accuracy(self):
        y_true = np.array([0.0005, 0.005, 0.05, 0.5, 5.0])
        y_pred = np.array([0.005, 0.05, 0.5, 5.0, 50.0])
        result = calculate_mag_acc(y_true, y_pred)
        assert result == pytest.approx(0.0)

    def test_partial_accuracy(self):
        y_true = np.array([0.5, 0.5, 5.0, 5.0])
        y_pred = np.array([0.5, 50.0, 5.0, 50.0])
        result = calculate_mag_acc(y_true, y_pred)
        assert result == pytest.approx(0.5)

    def test_values_above_max_bin(self):
        result = calculate_mag_acc(np.array([100.0]), np.array([999.0]))
        assert result == pytest.approx(1.0)

    def test_values_below_min_bin(self):
        result = calculate_mag_acc(np.array([1e-5]), np.array([1e-6]))
        assert result == pytest.approx(1.0)


# ============================================================
# ensemble_arrays
# ============================================================

class TestEnsembleArrays:
    def _make_arrays(self, pred, true, sigma, error_norm=None):
        d = {
            'delta_energy': {'pred': pred, 'true': true, 'sigma': sigma},
        }
        if error_norm is not None:
            d['delta_energy']['error_norm'] = error_norm
        return d

    def test_single_model_ensemble(self):
        pred = np.array([1.0, 2.0, 3.0])
        true = np.array([1.1, 2.1, 3.1])
        sigma = np.array([0.1, 0.2, 0.3])
        arrays = [self._make_arrays(pred, true, sigma)]

        result = ensemble_arrays(arrays)
        np.testing.assert_array_almost_equal(result['delta_energy']['pred'], pred)
        np.testing.assert_array_almost_equal(result['delta_energy']['true'], true)
        np.testing.assert_array_almost_equal(result['delta_energy']['sigma_epi'],
                                             np.zeros(3), decimal=10)
        np.testing.assert_array_almost_equal(result['delta_energy']['sigma_alea'], sigma)

    def test_two_identical_models(self):
        pred = np.array([1.0, 2.0])
        true = np.array([1.5, 2.5])
        sigma = np.array([0.1, 0.2])
        arrays = [self._make_arrays(pred, true, sigma)] * 2

        result = ensemble_arrays(arrays)
        np.testing.assert_array_almost_equal(result['delta_energy']['pred'], pred)
        np.testing.assert_array_almost_equal(result['delta_energy']['sigma_epi'],
                                             np.zeros(2), decimal=10)

    def test_two_different_models(self):
        true = np.array([1.0, 2.0])
        pred1 = np.array([1.0, 3.0])
        pred2 = np.array([3.0, 1.0])
        sigma1 = np.array([0.5, 0.5])
        sigma2 = np.array([0.5, 0.5])

        arrays = [
            self._make_arrays(pred1, true, sigma1),
            self._make_arrays(pred2, true, sigma2),
        ]
        result = ensemble_arrays(arrays)

        expected_pred = np.array([2.0, 2.0])
        np.testing.assert_array_almost_equal(result['delta_energy']['pred'], expected_pred)

        # Aleatoric = sqrt(mean(sigma^2)) = sqrt(0.25) = 0.5
        np.testing.assert_array_almost_equal(
            result['delta_energy']['sigma_alea'], np.array([0.5, 0.5]))

        # Epistemic = sqrt(var(preds)) = sqrt(((1-2)^2 + (3-2)^2)/2) = 1.0
        np.testing.assert_array_almost_equal(
            result['delta_energy']['sigma_epi'], np.array([1.0, 1.0]))

        # Total = sqrt(alea^2 + epi^2) = sqrt(0.25 + 1.0) = sqrt(1.25)
        expected_total = np.sqrt(0.25 + 1.0)
        np.testing.assert_array_almost_equal(
            result['delta_energy']['sigma'], np.array([expected_total, expected_total]))

    def test_sigma_decomposition(self):
        """Verify sigma_total^2 = sigma_alea^2 + sigma_epi^2"""
        np.random.seed(42)
        K = 5
        N = 50
        arrays = []
        for _ in range(K):
            arrays.append(self._make_arrays(
                pred=np.random.randn(N),
                true=np.random.randn(N),
                sigma=np.abs(np.random.randn(N)) + 0.1,
            ))
        result = ensemble_arrays(arrays)
        total_sq = result['delta_energy']['sigma'] ** 2
        alea_sq = result['delta_energy']['sigma_alea'] ** 2
        epi_sq = result['delta_energy']['sigma_epi'] ** 2
        np.testing.assert_array_almost_equal(total_sq, alea_sq + epi_sq, decimal=10)

    def test_error_norm_ensembled(self):
        true = np.array([1.0, 2.0])
        err1 = np.array([0.1, 0.2])
        err2 = np.array([0.3, 0.4])
        arrays = [
            self._make_arrays(np.zeros(2), true, np.ones(2), error_norm=err1),
            self._make_arrays(np.zeros(2), true, np.ones(2), error_norm=err2),
        ]
        result = ensemble_arrays(arrays)
        expected = np.array([0.2, 0.3])
        np.testing.assert_array_almost_equal(
            result['delta_energy']['error_norm'], expected)


# ============================================================
# compute_metrics_for_targets
# ============================================================

class TestComputeMetrics:
    def test_perfect_predictions(self):
        arrays = {
            'delta_energy': {
                'pred': np.array([1.0, 2.0, 3.0]),
                'true': np.array([1.0, 2.0, 3.0]),
            },
        }
        metrics = compute_metrics_for_targets(arrays, targets=['delta_energy'])
        assert metrics['delta_energy/MAE'] == pytest.approx(0.0)
        assert metrics['delta_energy/sMAPE'] == pytest.approx(0.0, abs=1e-4)
        assert metrics['delta_energy/MagAcc'] == pytest.approx(1.0)
        assert metrics['delta_energy/RMSLE'] == pytest.approx(0.0, abs=1e-6)

    def test_missing_key_skipped(self):
        arrays = {
            'delta_energy': {
                'pred': np.array([1.0]),
                'true': np.array([2.0]),
            },
        }
        metrics = compute_metrics_for_targets(arrays, targets=['delta_energy', 'delta_gap'])
        assert 'delta_energy/MAE' in metrics
        assert 'delta_gap/MAE' not in metrics

    def test_empty_array_skipped(self):
        arrays = {
            'delta_energy': {
                'pred': np.array([]),
                'true': np.array([]),
            },
        }
        metrics = compute_metrics_for_targets(arrays, targets=['delta_energy'])
        assert 'delta_energy/MAE' not in metrics

    def test_positions_uses_error_norm(self):
        """Position MAE should use error_norm (norm of vector diff), not diff of norms."""
        arrays = {
            'delta_positions': {
                'pred': np.array([3.0, 4.0]),
                'true': np.array([3.0, 4.0]),
                'error_norm': np.array([0.5, 1.0]),
            },
        }
        metrics = compute_metrics_for_targets(arrays, targets=['delta_positions'])
        # MAE should be mean(error_norm) = 0.75, NOT mean(|pred - true|) = 0.0
        assert metrics['delta_positions/MAE'] == pytest.approx(0.75)


# ============================================================
# arrays_to_csv
# ============================================================

class TestArraysToCSV:
    def test_uniform_length_roundtrip(self):
        arrays = {
            'delta_energy': {
                'pred': np.array([1.0, 2.0, 3.0]),
                'true': np.array([1.1, 2.1, 3.1]),
                'sigma': np.array([0.1, 0.2, 0.3]),
            },
        }
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            path = f.name
        try:
            arrays_to_csv(arrays, path, targets=['delta_energy'])
            df = pd.read_csv(path)
            assert 'delta_energy_pred' in df.columns
            assert 'delta_energy_true' in df.columns
            assert 'delta_energy_sigma' in df.columns
            assert len(df) == 3
            np.testing.assert_array_almost_equal(
                df['delta_energy_pred'].values, [1.0, 2.0, 3.0])
        finally:
            os.unlink(path)

    def test_ensemble_extra_columns(self):
        arrays = {
            'delta_energy': {
                'pred': np.array([1.0, 2.0]),
                'true': np.array([1.1, 2.1]),
                'sigma': np.array([0.5, 0.5]),
                'sigma_alea': np.array([0.3, 0.3]),
                'sigma_epi': np.array([0.4, 0.4]),
            },
        }
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            path = f.name
        try:
            arrays_to_csv(arrays, path, targets=['delta_energy'])
            df = pd.read_csv(path)
            assert 'delta_energy_sigma_alea' in df.columns
            assert 'delta_energy_sigma_epi' in df.columns
        finally:
            os.unlink(path)

    def test_mixed_lengths_split_files(self):
        """Positions (per-atom) have different length from graph-level targets."""
        arrays = {
            'delta_energy': {
                'pred': np.array([1.0, 2.0]),
                'true': np.array([1.1, 2.1]),
                'sigma': np.array([0.1, 0.2]),
            },
            'delta_positions': {
                'pred': np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
                'true': np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
                'sigma': np.array([0.01, 0.02, 0.03, 0.04, 0.05]),
            },
        }
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            path = f.name
        pos_path = path.replace('.csv', '_positions.csv')
        try:
            arrays_to_csv(arrays, path, targets=['delta_energy', 'delta_positions'])
            df = pd.read_csv(path)
            assert 'delta_energy_pred' in df.columns
            assert 'delta_positions_pred' not in df.columns
            assert len(df) == 2
            assert os.path.exists(pos_path)
            df_pos = pd.read_csv(pos_path)
            assert 'delta_positions_pred' in df_pos.columns
            assert len(df_pos) == 5
        finally:
            if os.path.exists(path):
                os.unlink(path)
            if os.path.exists(pos_path):
                os.unlink(pos_path)
