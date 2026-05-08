"""Tests for MegaPaiNN surrogate training loop.

Tests cover: train_step loss decrease, evaluate metrics, scaler inverse-transform,
scale_targets correctness, and compute_metrics units.
"""

import os
import sys

import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from modelling.gnn.mega_painn_model import MegaPaiNN  # noqa: E402
from modelling.gnn.train_surrogate import (  # noqa: E402
    compute_metrics,
    evaluate,
    inverse_transform,
    scale_targets,
    train_step,
)


# ---- Helpers ----

def make_model():
    return MegaPaiNN(
        num_layers=2, hidden_dim=16, num_precision_settings=12,
        max_z=10, num_rbf=8, cutoff=5.0, embedding_dim=32, use_film=False)


def make_graph(y_form_val, y_gap_val, num_atoms=3, seed=0):
    """Create a graph with known target values."""
    torch.manual_seed(seed)
    pos = torch.randn(num_atoms, 3)
    z = torch.randint(1, 9, (num_atoms, 1))

    src, dst = [], []
    for i in range(num_atoms):
        for j in range(num_atoms):
            if i != j:
                src.append(i)
                dst.append(j)
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    prec = torch.zeros(1, 12)
    prec[0, 10] = 1.0
    prec[0, 11] = 8.0

    return Data(
        x=pos, z=z, edge_index=edge_index,
        precision_settings=prec,
        y_form=torch.tensor([y_form_val], dtype=torch.float),
        y_gap=torch.tensor([y_gap_val], dtype=torch.float),
    )


def make_dataset(n=20):
    """Create a small dataset with scalers fitted on it."""
    graphs = []
    for i in range(n):
        y_f = -2.0 + 0.2 * i  # range -2 to 2
        y_g = 0.5 + 0.5 * i   # range 0.5 to 10
        g = make_graph(y_f, y_g, seed=i)
        g.split = 'train' if i < 16 else 'val'
        graphs.append(g)

    # Fit scalers on train
    train_forms = np.array([g.y_form.item() for g in graphs if g.split == 'train'])
    train_gaps = np.array([g.y_gap.item() for g in graphs if g.split == 'train'])

    form_scaler = StandardScaler()
    gap_scaler = StandardScaler()
    form_scaler.fit(train_forms.reshape(-1, 1))
    gap_scaler.fit(train_gaps.reshape(-1, 1))

    graphs = scale_targets(graphs, form_scaler, gap_scaler)
    return graphs, form_scaler, gap_scaler


# ---- inverse_transform ----

class TestInverseTransform:

    def test_roundtrip(self):
        """Transform then inverse gives back original values."""
        scaler = StandardScaler()
        orig = np.array([1.0, 2.0, 3.0, 4.0, 5.0]).reshape(-1, 1)
        scaler.fit(orig)
        scaled = scaler.transform(orig).flatten()
        recovered = inverse_transform(scaler, scaled)
        np.testing.assert_allclose(recovered, orig.flatten(), atol=1e-6)


# ---- scale_targets ----

class TestScaleTargets:

    def test_adds_scaled_fields(self):
        """scale_targets adds y_form_scaled and y_gap_scaled to graphs."""
        graphs, form_scaler, gap_scaler = make_dataset(10)
        for g in graphs:
            assert hasattr(g, 'y_form_scaled')
            assert hasattr(g, 'y_gap_scaled')
            assert g.y_form_scaled.shape == (1,)
            assert g.y_gap_scaled.shape == (1,)

    def test_scaled_values_correct(self):
        """Scaled values match manual scaler.transform()."""
        graphs, form_scaler, gap_scaler = make_dataset(10)
        g = graphs[0]
        expected_form = form_scaler.transform(
            g.y_form.numpy().reshape(-1, 1)).flatten()
        np.testing.assert_allclose(
            g.y_form_scaled.numpy(), expected_form, atol=1e-6)


# ---- compute_metrics ----

class TestComputeMetrics:

    def test_perfect_prediction(self):
        """Zero error when predictions match targets."""
        scaler = StandardScaler()
        scaler.fit(np.array([0, 1, 2, 3, 4]).reshape(-1, 1))
        scaled = scaler.transform(np.array([0, 1, 2, 3, 4]).reshape(-1, 1)).flatten()

        metrics = compute_metrics(scaled, scaled, scaler, 'test')
        assert metrics['MAE_test'] == pytest.approx(0.0, abs=1e-6)
        assert metrics['RMSE_test'] == pytest.approx(0.0, abs=1e-6)
        assert metrics['R2_test'] == pytest.approx(1.0, abs=1e-6)

    def test_metrics_in_original_units(self):
        """MAE is computed in original (unscaled) units."""
        scaler = StandardScaler()
        orig = np.array([0, 10, 20, 30, 40]).reshape(-1, 1)
        scaler.fit(orig)

        # Preds off by 1 in original units -> MAE should be ~1.0
        preds_orig = orig.flatten() + 1.0
        preds_scaled = scaler.transform(preds_orig.reshape(-1, 1)).flatten()
        targets_scaled = scaler.transform(orig).flatten()

        metrics = compute_metrics(preds_scaled, targets_scaled, scaler, 'x')
        assert metrics['MAE_x'] == pytest.approx(1.0, abs=1e-5)


# ---- train_step ----

class TestTrainStep:

    def test_loss_decreases(self):
        """Training loss decreases over multiple steps."""
        model = make_model()
        graphs, form_scaler, gap_scaler = make_dataset(20)
        train_graphs = [g for g in graphs if g.split == 'train']
        loader = DataLoader(train_graphs, batch_size=8, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        device = torch.device('cpu')

        losses = []
        for _ in range(10):
            loss = train_step(model, loader, optimizer, 1.0, 1.0, device)
            losses.append(loss)

        # Loss should decrease (first > last)
        assert losses[0] > losses[-1], \
            f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"

    def test_train_step_returns_float(self):
        """train_step returns a scalar float."""
        model = make_model()
        graphs, _, _ = make_dataset(10)
        train_graphs = [g for g in graphs if g.split == 'train']
        loader = DataLoader(train_graphs, batch_size=8)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        loss = train_step(model, loader, optimizer, 1.0, 1.0, torch.device('cpu'))
        assert isinstance(loss, float)


# ---- evaluate ----

class TestEvaluate:

    def test_evaluate_returns_all_metrics(self):
        """evaluate() returns expected metric keys."""
        model = make_model()
        graphs, form_scaler, gap_scaler = make_dataset(20)
        val_graphs = [g for g in graphs if g.split == 'val']
        loader = DataLoader(val_graphs, batch_size=8)
        device = torch.device('cpu')

        metrics = evaluate(
            model, loader, form_scaler, gap_scaler, 1.0, 1.0, device)

        expected_keys = {
            'loss', 'MAE_form', 'RMSE_form', 'R2_form',
            'MAE_gap', 'RMSE_gap', 'R2_gap',
        }
        assert set(metrics.keys()) == expected_keys

    def test_evaluate_metrics_are_finite(self):
        """All metrics are finite numbers."""
        model = make_model()
        graphs, form_scaler, gap_scaler = make_dataset(20)
        val_graphs = [g for g in graphs if g.split == 'val']
        loader = DataLoader(val_graphs, batch_size=8)

        metrics = evaluate(
            model, loader, form_scaler, gap_scaler, 1.0, 1.0, torch.device('cpu'))

        for k, v in metrics.items():
            assert np.isfinite(v), f"{k} is not finite: {v}"

    def test_evaluate_after_training_improves(self):
        """Metrics improve after training vs random init."""
        model = make_model()
        graphs, form_scaler, gap_scaler = make_dataset(20)
        train_graphs = [g for g in graphs if g.split == 'train']
        val_graphs = [g for g in graphs if g.split == 'val']

        train_loader = DataLoader(train_graphs, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=8)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        device = torch.device('cpu')

        # Before training
        m_before = evaluate(
            model, val_loader, form_scaler, gap_scaler, 1.0, 1.0, device)

        # Train 20 steps
        for _ in range(20):
            train_step(model, train_loader, optimizer, 1.0, 1.0, device)

        # After training
        m_after = evaluate(
            model, val_loader, form_scaler, gap_scaler, 1.0, 1.0, device)

        assert m_after['loss'] < m_before['loss'], \
            f"Val loss did not improve: {m_before['loss']:.4f} -> {m_after['loss']:.4f}"
