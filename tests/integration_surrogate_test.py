"""Integration test: AFLOW download -> process -> train MegaPaiNN.

Requires network access. Not run by default.
Run with: pytest tests/integration_surrogate_test.py -v -m integration
"""

import os
import sys

import numpy as np
import pytest
import torch
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts', 'data'))

from modelling.gnn.mega_painn_model import MegaPaiNN  # noqa: E402
from modelling.gnn.train_surrogate import (  # noqa: E402
    evaluate,
    scale_targets,
    train_step,
)
from process_aflow import (  # noqa: E402
    process_aflow,
    query_aflow,
)


@pytest.mark.integration
def test_aflow_to_training_end_to_end(tmp_path):
    """Full pipeline: download AFLOW -> process -> train -> evaluate."""

    raw_path = str(tmp_path / 'aflow_raw.parquet')
    output_path = str(tmp_path / 'aflow_processed.pt')

    # --- 1. Download (limit=10 per nspecies -> ~30 entries) ---
    try:
        query_aflow(raw_path, limit=10)
    except Exception as e:
        pytest.skip(f"AFLOW API unavailable: {e}")

    assert os.path.exists(raw_path), "Raw parquet not created"

    # --- 2. Process -> PyG graphs ---
    process_aflow(raw_path, output_path, cutoff=5.0, seed=42)
    assert os.path.exists(output_path), "Processed .pt not created"

    # --- 3. Validate data ---
    loaded = torch.load(output_path, weights_only=False)
    assert 'graphs' in loaded
    assert 'y_form_scaler' in loaded
    assert 'y_gap_scaler' in loaded

    graphs = loaded['graphs']
    form_scaler = loaded['y_form_scaler']
    gap_scaler = loaded['y_gap_scaler']

    if len(graphs) == 0:
        pytest.skip("No valid graphs after filtering (AFLOW data issue)")

    assert len(graphs) >= 2, f"Need at least 2 graphs, got {len(graphs)}"

    # Check graph fields
    g = graphs[0]
    assert hasattr(g, 'x')
    assert hasattr(g, 'z')
    assert hasattr(g, 'edge_index')
    assert hasattr(g, 'cell')
    assert hasattr(g, 'y_form')
    assert hasattr(g, 'y_gap')
    assert hasattr(g, 'precision_settings')
    assert hasattr(g, 'split')
    assert g.x.shape[1] == 3
    assert g.z.shape[1] == 1
    assert g.cell.shape == (3, 3)
    assert g.precision_settings.shape == (1, 12)
    assert g.edge_index.shape[0] == 2
    assert g.edge_index.shape[1] > 0, "Graph has no edges"

    # Scalers should be fitted
    assert hasattr(form_scaler, 'mean_')
    assert hasattr(gap_scaler, 'mean_')

    # Check splits exist
    splits = set(g.split for g in graphs)
    assert 'train' in splits, f"No train split, got {splits}"

    print(f"  Processed {len(graphs)} graphs, splits: "
          f"{ {s: sum(1 for g in graphs if g.split == s) for s in splits} }")

    # --- 4. Scale targets ---
    graphs = scale_targets(graphs, form_scaler, gap_scaler)
    for g in graphs:
        assert hasattr(g, 'y_form_scaled')
        assert hasattr(g, 'y_gap_scaled')

    # --- 5. Train ---
    train_graphs = [g for g in graphs if g.split == 'train']
    val_graphs = [g for g in graphs if g.split != 'train']

    # If no val graphs, take last train graph as val
    if len(val_graphs) == 0:
        val_graphs = [train_graphs[-1]]

    assert len(train_graphs) > 0, "No training graphs"

    train_loader = DataLoader(train_graphs, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=8, shuffle=False)

    model = MegaPaiNN(
        num_layers=2, hidden_dim=32, num_rbf=8, cutoff=5.0,
        max_z=120, embedding_dim=16, use_film=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    device = torch.device('cpu')

    losses = []
    for epoch in range(5):
        loss = train_step(model, train_loader, optimizer, 1.0, 1.0, device)
        losses.append(loss)
        print(f"  Epoch {epoch + 1}: loss={loss:.6f}")

    assert all(np.isfinite(l) for l in losses), f"Non-finite loss: {losses}"
    assert losses[-1] < losses[0], \
        f"Loss did not decrease: {losses[0]:.6f} -> {losses[-1]:.6f}"

    # --- 6. Evaluate ---
    metrics = evaluate(
        model, val_loader, form_scaler, gap_scaler, 1.0, 1.0, device)

    for k, v in metrics.items():
        assert np.isfinite(v), f"Metric {k} is not finite: {v}"
        print(f"  {k}: {v:.6f}")

    assert metrics['MAE_form'] > 0
    assert metrics['MAE_gap'] > 0

    print(f"\n  Integration test passed: {len(graphs)} graphs, "
          f"final loss={losses[-1]:.6f}")
