import pytest
import torch
import torch.nn as nn
import os
import sys
import numpy as np
from ase import Atoms
from ase.db import connect
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

# --- Imports ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from modelling.gnn.egnn_attention_layer import EGNNAttentionLayer
from modelling.gnn.delta_egnn_attention_model import DeltaAttentionGNN
from modelling.gnn.ase_db_to_graphs import process_database

# --- Constants from your Generator Script ---
REQUIRED_DB_KEYS = [
    'total_energy_per_atom', 'homo_lumo_gap',
    'sum_eigenvalues_per_atom', 'xc_energy_correction_per_atom',
    'xc_potential_correction_per_atom', 'free_atom_electrostatic_energy_per_atom',
    'hartree_energy_correction_per_atom', 'entropy_correction_per_atom',
    'total_energy_T0_per_atom', 'kinetic_energy_per_atom',
    'electrostatic_energy_per_atom', 'multipole_correction_per_atom',
    'final_volume_per_atom', 'relaxed_a_len', 'relaxed_b_len', 'relaxed_c_len',
    'relaxed_alpha_angle', 'relaxed_beta_angle', 'relaxed_gamma_angle',
    'delta_total_energy_per_atom',
    'delta_homo_lumo_gap',
    'delta_aims_free_energy_per_atom',
    'delta_vbm', 'delta_cbm', 'delta_chemical_potential',
    'delta_final_volume_per_atom',
    'delta_relaxed_a_len', 'delta_relaxed_b_len', 'delta_relaxed_c_len',
    'delta_relaxed_alpha_angle', 'delta_relaxed_beta_angle', 'delta_relaxed_gamma_angle',
]


# =============================================
# Test 1: EGNNAttentionLayer unit test
# =============================================
def test_attention_layer_forward_shape():
    """
    Verifies that EGNNAttentionLayer:
    - Accepts (h, pos, edge_index)
    - Returns (h_out, pos_out) with correct shapes
    - h_out preserves node count and feature dim
    - pos_out preserves node count and 3D coords
    """
    in_dim = 32
    hidden_dim = 32
    out_dim = 32
    layer = EGNNAttentionLayer(in_dim, hidden_dim, out_dim)

    # 3 nodes, edges 0->1 and 1->0
    h = torch.randn(3, in_dim)
    pos = torch.randn(3, 3)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)

    h_out, pos_out = layer(h, pos, edge_index)

    assert h_out.shape == (3, out_dim)
    assert pos_out.shape == (3, 3)


def test_attention_layer_removes_self_loops():
    """
    Verifies that self-loops in edge_index do not cause errors
    and are handled (removed) by the layer.
    """
    dim = 16
    layer = EGNNAttentionLayer(dim, dim, dim)

    h = torch.randn(2, dim)
    pos = torch.randn(2, 3)
    # Include self-loops: 0->0 and 1->1
    edge_index = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=torch.long)

    h_out, pos_out = layer(h, pos, edge_index)

    assert h_out.shape == (2, dim)
    assert pos_out.shape == (2, 3)


def test_attention_layer_residual_connection():
    """
    When in_features == out_features, the layer uses a residual connection.
    Verify output differs from input (update happened) but isn't wildly different.
    """
    dim = 16
    layer = EGNNAttentionLayer(dim, dim, dim)

    h = torch.randn(3, dim)
    pos = torch.randn(3, 3)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)

    h_out, pos_out = layer(h, pos, edge_index)

    # Output should differ from input (non-trivial update)
    assert not torch.allclose(h_out, h, atol=1e-6)


def test_attention_layer_no_residual_when_dims_differ():
    """
    When in_features != out_features, residual is disabled.
    """
    layer = EGNNAttentionLayer(in_features_h=16, hidden_features=32, out_features_h=32)
    assert layer.residuals is False

    h = torch.randn(3, 16)
    pos = torch.randn(3, 3)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)

    h_out, pos_out = layer(h, pos, edge_index)

    assert h_out.shape == (3, 32)
    assert pos_out.shape == (3, 3)


def test_attention_layer_equivariance():
    """
    Checks E(3)-equivariance: rotating input positions should rotate output
    positions by the same amount (and leave h unchanged).
    """
    dim = 16
    layer = EGNNAttentionLayer(dim, dim, dim)
    layer.eval()

    h = torch.randn(3, dim)
    pos = torch.randn(3, 3)
    edge_index = torch.tensor([[0, 1, 1, 2, 0, 2], [1, 0, 2, 1, 2, 0]], dtype=torch.long)

    # 90-degree rotation around z-axis
    R = torch.tensor([[0., -1., 0.],
                       [1.,  0., 0.],
                       [0.,  0., 1.]])

    # Forward on original positions
    h_out1, pos_out1 = layer(h, pos, edge_index)

    # Forward on rotated positions
    h_out2, pos_out2 = layer(h, pos @ R.T, edge_index)

    # h should be invariant (same for both)
    assert torch.allclose(h_out1, h_out2, atol=1e-5), \
        "Feature vectors should be invariant under rotation"

    # pos should be equivariant: pos_out2 ≈ pos_out1 @ R^T
    assert torch.allclose(pos_out2, pos_out1 @ R.T, atol=1e-5), \
        "Position outputs should be equivariant under rotation"


# =============================================
# Test 2: DeltaAttentionGNN unit test
# =============================================
def test_delta_attention_gnn_forward_shape():
    """
    Verifies that the attention model accepts the same input format as DeltaGNN
    and returns 4 tuples of (mean, logvar) with correct shapes.
    """
    model = DeltaAttentionGNN(
        num_layers=2,
        hidden_features=16,
        num_cheap_dft_inputs=12,
        num_precision_settings=12,
        num_geo_inputs=7,
        max_z=10
    )

    # Graph 1: 2 atoms. Graph 2: 1 atom. Total N=3, B=2.
    x_input = torch.randn(3, 3)
    z_input = torch.tensor([1, 6, 8], dtype=torch.long)
    dft_scalars = torch.randn(3, 12)
    global_precision_settings = torch.randn(2, 12)
    geo_scalars = torch.randn(2, 7)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    batch = torch.tensor([0, 0, 1], dtype=torch.long)

    data = Data(
        x=x_input,
        z=z_input,
        edge_index=edge_index,
        batch=batch,
        cheap_dft_scalars=dft_scalars,
        precision_settings=global_precision_settings,
        cheap_geometry_scalars=geo_scalars
    )

    (pos_mean, pos_logvar), \
    (energy_mean, energy_logvar), \
    (gap_mean, gap_logvar), \
    (geo_mean, geo_logvar) = model(data)

    # Positions: per-atom [N, 3] and [N, 1]
    assert pos_mean.shape == (3, 3)
    assert pos_logvar.shape == (3, 1)

    # Scalars: per-graph [B]
    assert energy_mean.shape == (2,)
    assert energy_logvar.shape == (2,)
    assert gap_mean.shape == (2,)
    assert gap_logvar.shape == (2,)

    # Geometry: per-graph [B, 7]
    assert geo_mean.shape == (2, 7)
    assert geo_logvar.shape == (2, 7)


def test_delta_attention_gnn_backward():
    """
    Verifies that gradients flow through the entire model without errors.
    """
    model = DeltaAttentionGNN(
        num_layers=2,
        hidden_features=16,
        num_cheap_dft_inputs=12,
        num_precision_settings=12,
        num_geo_inputs=7,
        max_z=10
    )

    x_input = torch.randn(3, 3)
    z_input = torch.tensor([1, 6, 8], dtype=torch.long)
    dft_scalars = torch.randn(3, 12)
    global_precision_settings = torch.randn(2, 12)
    geo_scalars = torch.randn(2, 7)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    batch = torch.tensor([0, 0, 1], dtype=torch.long)

    data = Data(
        x=x_input, z=z_input, edge_index=edge_index, batch=batch,
        cheap_dft_scalars=dft_scalars,
        precision_settings=global_precision_settings,
        cheap_geometry_scalars=geo_scalars
    )

    (pos_mean, pos_logvar), (energy_mean, energy_logvar), \
    (gap_mean, gap_logvar), (geo_mean, geo_logvar) = model(data)

    # Dummy loss combining all outputs
    loss = pos_mean.sum() + pos_logvar.sum() + \
           energy_mean.sum() + energy_logvar.sum() + \
           gap_mean.sum() + gap_logvar.sum() + \
           geo_mean.sum() + geo_logvar.sum()

    loss.backward()

    # Check that all parameters received gradients
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


# =============================================
# Test 3: Full Integration (DB -> Generator -> Model)
# =============================================

def get_dummy_row_data():
    """Helper to populate all required keys with dummy floats."""
    data = {k: 0.5 for k in REQUIRED_DB_KEYS}
    data['delta_total_energy_per_atom'] = -0.1
    data['delta_homo_lumo_gap'] = 0.05
    data['delta_relaxed_atom_positions'] = "None"
    data['k_point_density'] = 2.0
    return data


@pytest.fixture
def mock_db_environment(tmp_path):
    """Creates a temporary ASE DB with enough columns to pass the generator's validation."""
    db_file = tmp_path / "test_attention.db"
    out_file = tmp_path / "test_output.pt"

    dummy_data = get_dummy_row_data()

    with connect(db_file) as db:
        # Graph 1: H2
        dummy_data['delta_relaxed_atom_positions'] = "[[0.01, 0.0, 0.0], [-0.01, 0.0, 0.0]]"
        db.write(Atoms('H2', positions=[[0,0,0],[0,0,0.74]]),
                 split='train',
                 **dummy_data)

        # Graph 2: H2O
        dummy_data['delta_relaxed_atom_positions'] = "[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]"
        db.write(Atoms('H2O', positions=[[0,0,0],[1,0,0],[0,1,0]]),
                 split='train',
                 **dummy_data)

    return str(db_file), str(out_file)


def test_integration_with_generated_db(mock_db_environment):
    """
    Runs the full pipeline:
    1. Generator reads DB, scales data, saves .pt
    2. DeltaAttentionGNN consumes the processed .pt file
    """
    db_path, out_path = mock_db_environment

    # 1. Run Generator
    process_database(db_path, out_path, cutoff=3.0)

    # 2. Load Data
    assert os.path.exists(out_path)
    data_dict = torch.load(out_path, weights_only=False)
    graph_list = data_dict['graphs']

    assert len(graph_list) == 2

    # 3. Simulate Batching
    loader = DataLoader(graph_list, batch_size=2)
    batch_data = next(iter(loader))

    # 4. Initialize Model
    model = DeltaAttentionGNN(
        num_layers=2,
        hidden_features=32,
        num_cheap_dft_inputs=12,
        num_precision_settings=13,  # 12 OHE + 1 K-dens (after bp=11 fix)
        num_geo_inputs=7,
        max_z=100
    )

    # 5. Run Forward Pass
    (pos_mean, pos_logvar), (energy_mean, energy_logvar), \
    (gap_mean, gap_logvar), (geo_mean, geo_logvar) = model(batch_data)

    # 6. Verify Shapes
    # Total atoms = 2 (H2) + 3 (H2O) = 5
    assert pos_mean.shape == (5, 3)
    assert pos_logvar.shape == (5, 1)

    # Total graphs = 2
    assert energy_mean.shape == (2,)
    assert energy_logvar.shape == (2,)
    assert gap_mean.shape == (2,)
    assert gap_logvar.shape == (2,)
    assert geo_mean.shape == (2, 7)
    assert geo_logvar.shape == (2, 7)
