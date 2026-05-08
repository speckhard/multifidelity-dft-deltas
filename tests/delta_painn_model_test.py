import pytest
import torch
import os
import sys
import numpy as np
from ase import Atoms
from ase.db import connect
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from modelling.gnn.painn_layer import (
    GaussianRBF, CosineCutoff, FiLMGenerator,
    PaiNNMessage, PaiNNUpdate, PaiNNInteraction
)
from modelling.gnn.delta_painn_model import DeltaPaiNN
from modelling.gnn.ase_db_to_graphs import process_database


# =============================================
# Test 1: GaussianRBF shape
# =============================================
def test_gaussian_rbf_shape():
    rbf = GaussianRBF(num_rbf=20, cutoff=5.0)
    dist = torch.tensor([1.0, 2.5, 4.9])
    out = rbf(dist)
    assert out.shape == (3, 20)


# =============================================
# Test 2: GaussianRBF peaks at centers
# =============================================
def test_gaussian_rbf_values_at_centers():
    rbf = GaussianRBF(num_rbf=5, cutoff=4.0)
    # Centers at 0.0, 1.0, 2.0, 3.0, 4.0
    dist = torch.tensor([0.0, 1.0, 2.0])
    out = rbf(dist)
    # At dist=0.0, center[0]=0.0 should give peak of 1.0
    assert out[0, 0].item() == pytest.approx(1.0, abs=1e-5)
    # At dist=1.0, center[1]=1.0 should give peak of 1.0
    assert out[1, 1].item() == pytest.approx(1.0, abs=1e-5)
    # At dist=2.0, center[2]=2.0 should give peak of 1.0
    assert out[2, 2].item() == pytest.approx(1.0, abs=1e-5)


# =============================================
# Test 3: CosineCutoff boundary values
# =============================================
def test_cosine_cutoff_boundary():
    cutoff_fn = CosineCutoff(cutoff=5.0)
    dist = torch.tensor([0.0, 2.5, 5.0, 6.0])
    out = cutoff_fn(dist)
    assert out[0].item() == pytest.approx(1.0, abs=1e-5)   # at 0
    assert out[2].item() == pytest.approx(0.0, abs=1e-5)   # at cutoff
    assert out[3].item() == pytest.approx(0.0, abs=1e-5)   # beyond cutoff


# =============================================
# Test 4: PaiNNMessage output shapes
# =============================================
def test_painn_message_shapes():
    F_dim = 16
    msg = PaiNNMessage(hidden_dim=F_dim, num_rbf=10, cutoff=5.0)

    N = 4
    s = torch.randn(N, F_dim)
    v = torch.randn(N, F_dim, 3)
    pos = torch.randn(N, 3)
    edge_index = torch.tensor([[0,1,1,2,2,3], [1,0,2,1,3,2]], dtype=torch.long)

    ds, dv = msg(s, v, edge_index, pos)
    assert ds.shape == (N, F_dim)
    assert dv.shape == (N, F_dim, 3)


# =============================================
# Test 5: PaiNNInteraction preserves shapes
# =============================================
def test_painn_interaction_shapes():
    F_dim = 16
    cond_dim = 24
    layer = PaiNNInteraction(F_dim, num_rbf=10, cutoff=5.0, cond_dim=cond_dim)

    N = 4
    s = torch.randn(N, F_dim)
    v = torch.randn(N, F_dim, 3)
    pos = torch.randn(N, 3)
    cond = torch.randn(N, cond_dim)
    edge_index = torch.tensor([[0,1,1,2,2,3], [1,0,2,1,3,2]], dtype=torch.long)

    s_out, v_out = layer(s, v, edge_index, pos, cond)
    assert s_out.shape == (N, F_dim)
    assert v_out.shape == (N, F_dim, 3)


# =============================================
# Test 6: DeltaPaiNN forward output shapes
# =============================================
def test_delta_painn_forward_shape():
    model = DeltaPaiNN(
        num_layers=2,
        hidden_dim=16,
        num_cheap_dft_inputs=12,
        num_precision_settings=12,
        num_geo_inputs=7,
        max_z=10,
        num_rbf=10,
        cutoff=5.0
    )

    # Graph 1: 2 atoms. Graph 2: 1 atom. Total N=3, B=2.
    data = Data(
        x=torch.randn(3, 3),
        z=torch.tensor([1, 6, 8], dtype=torch.long),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        batch=torch.tensor([0, 0, 1], dtype=torch.long),
        cheap_dft_scalars=torch.randn(3, 12),
        precision_settings=torch.randn(2, 12),
        cheap_geometry_scalars=torch.randn(2, 7)
    )

    (pos_mean, pos_logvar), (e_mean, e_logvar), \
    (gap_mean, gap_logvar), (geo_mean, geo_logvar) = model(data)

    assert pos_mean.shape == (3, 3)
    assert pos_logvar.shape == (3, 1)
    assert e_mean.shape == (2,)
    assert e_logvar.shape == (2,)
    assert gap_mean.shape == (2,)
    assert gap_logvar.shape == (2,)
    assert geo_mean.shape == (2, 7)
    assert geo_logvar.shape == (2, 7)

    # Check denoising vector output is stored
    assert hasattr(model, '_vector_output')
    assert model._vector_output.shape == (3, 3)


# =============================================
# Test 7: Backward pass (gradient flow)
# =============================================
def test_delta_painn_backward():
    model = DeltaPaiNN(
        num_layers=2, hidden_dim=16, num_cheap_dft_inputs=12,
        num_precision_settings=12, num_geo_inputs=7, max_z=10,
        num_rbf=10, cutoff=5.0
    )

    data = Data(
        x=torch.randn(3, 3), z=torch.tensor([1, 6, 8]),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        batch=torch.tensor([0, 0, 1]),
        cheap_dft_scalars=torch.randn(3, 12),
        precision_settings=torch.randn(2, 12),
        cheap_geometry_scalars=torch.randn(2, 7)
    )

    (pos_mean, pos_logvar), (e_mean, e_logvar), \
    (gap_mean, gap_logvar), (geo_mean, geo_logvar) = model(data)

    loss = (pos_mean.sum() + pos_logvar.sum() +
            e_mean.sum() + e_logvar.sum() +
            gap_mean.sum() + gap_logvar.sum() +
            geo_mean.sum() + geo_logvar.sum() +
            model._vector_output.sum())

    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


# =============================================
# Test 8: Scalar invariance under rotation
# =============================================
def test_delta_painn_scalar_invariance():
    """Scalar outputs (energy, gap) should be invariant under rotation."""
    torch.manual_seed(42)
    model = DeltaPaiNN(
        num_layers=2, hidden_dim=16, num_cheap_dft_inputs=12,
        num_precision_settings=12, num_geo_inputs=7, max_z=10,
        num_rbf=10, cutoff=5.0
    )
    model.eval()

    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    # Fully connected graph so edges exist after rotation
    edge_index = torch.tensor([[0,1,1,2,0,2], [1,0,2,1,2,0]], dtype=torch.long)

    shared_dft = torch.randn(3, 12)
    shared_prec = torch.randn(1, 12)
    shared_geo = torch.randn(1, 7)

    data1 = Data(
        x=pos.clone(), z=torch.tensor([1, 6, 8]),
        edge_index=edge_index.clone(),
        batch=torch.tensor([0, 0, 0]),
        cheap_dft_scalars=shared_dft.clone(),
        precision_settings=shared_prec.clone(),
        cheap_geometry_scalars=shared_geo.clone()
    )

    # 90-degree rotation around z-axis
    R = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])

    data2 = Data(
        x=(pos @ R.T).clone(), z=torch.tensor([1, 6, 8]),
        edge_index=edge_index.clone(),
        batch=torch.tensor([0, 0, 0]),
        cheap_dft_scalars=shared_dft.clone(),
        precision_settings=shared_prec.clone(),
        cheap_geometry_scalars=shared_geo.clone()
    )

    with torch.no_grad():
        _, (e1, _), (g1, _), _ = model(data1)
        _, (e2, _), (g2, _), _ = model(data2)

    assert torch.allclose(e1, e2, atol=1e-4), \
        f"Energy not rotation-invariant: {e1} vs {e2}"
    assert torch.allclose(g1, g2, atol=1e-4), \
        f"Gap not rotation-invariant: {g1} vs {g2}"


# =============================================
# Test 8b: Vector equivariance under rotation
# =============================================
def test_delta_painn_vector_equivariance():
    """Position deltas should rotate when input positions are rotated."""
    torch.manual_seed(42)
    model = DeltaPaiNN(
        num_layers=2, hidden_dim=16, num_cheap_dft_inputs=12,
        num_precision_settings=12, num_geo_inputs=7, max_z=10,
        num_rbf=10, cutoff=5.0
    )
    model.eval()

    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    edge_index = torch.tensor([[0,1,1,2,0,2], [1,0,2,1,2,0]], dtype=torch.long)

    shared_dft = torch.randn(3, 12)
    shared_prec = torch.randn(1, 12)
    shared_geo = torch.randn(1, 7)

    data1 = Data(
        x=pos.clone(), z=torch.tensor([1, 6, 8]),
        edge_index=edge_index.clone(),
        batch=torch.tensor([0, 0, 0]),
        cheap_dft_scalars=shared_dft.clone(),
        precision_settings=shared_prec.clone(),
        cheap_geometry_scalars=shared_geo.clone()
    )

    # 90-degree rotation around z-axis
    R = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])

    data2 = Data(
        x=(pos @ R.T).clone(), z=torch.tensor([1, 6, 8]),
        edge_index=edge_index.clone(),
        batch=torch.tensor([0, 0, 0]),
        cheap_dft_scalars=shared_dft.clone(),
        precision_settings=shared_prec.clone(),
        cheap_geometry_scalars=shared_geo.clone()
    )

    with torch.no_grad():
        (pos_delta1, _), _, _, _ = model(data1)
        (pos_delta2, _), _, _, _ = model(data2)

    # Equivariance: rotating input should rotate output
    expected = pos_delta1 @ R.T
    assert torch.allclose(pos_delta2, expected, atol=1e-4), \
        f"Vector output not equivariant:\n  got {pos_delta2}\n  expected {expected}"


# =============================================
# Test 8c: Single-atom graph (z.squeeze bug)
# =============================================
def test_delta_painn_single_atom_graph():
    """Model must handle single-atom graphs without crashing."""
    model = DeltaPaiNN(
        num_layers=2, hidden_dim=16, num_cheap_dft_inputs=12,
        num_precision_settings=12, num_geo_inputs=7, max_z=10,
        num_rbf=10, cutoff=5.0
    )

    data = Data(
        x=torch.randn(1, 3),
        z=torch.tensor([6]),
        edge_index=torch.zeros(2, 0, dtype=torch.long),  # no edges
        batch=torch.tensor([0]),
        cheap_dft_scalars=torch.randn(1, 12),
        precision_settings=torch.randn(1, 12),
        cheap_geometry_scalars=torch.randn(1, 7),
    )

    (pos_mean, pos_logvar), (e_mean, e_logvar), \
    (gap_mean, gap_logvar), (geo_mean, geo_logvar) = model(data)

    assert pos_mean.shape == (1, 3)
    assert pos_logvar.shape == (1, 1)
    assert e_mean.shape == (1,)
    assert geo_mean.shape == (1, 7)


# =============================================
# Test 8d: FiLM conditioning changes output
# =============================================
def test_film_conditioning_changes_output():
    """Different precision settings must produce different predictions."""
    torch.manual_seed(42)
    model = DeltaPaiNN(
        num_layers=2, hidden_dim=16, num_cheap_dft_inputs=12,
        num_precision_settings=12, num_geo_inputs=7, max_z=10,
        num_rbf=10, cutoff=5.0
    )
    model.eval()

    pos = torch.randn(3, 3)
    z = torch.tensor([1, 6, 8])
    edge_index = torch.tensor([[0,1,1,2,0,2], [1,0,2,1,2,0]], dtype=torch.long)
    dft = torch.randn(3, 12)
    geo = torch.randn(1, 7)

    prec_a = torch.randn(1, 12)
    prec_b = torch.randn(1, 12)

    data_a = Data(
        x=pos.clone(), z=z.clone(), edge_index=edge_index.clone(),
        batch=torch.tensor([0, 0, 0]),
        cheap_dft_scalars=dft.clone(),
        precision_settings=prec_a,
        cheap_geometry_scalars=geo.clone()
    )
    data_b = Data(
        x=pos.clone(), z=z.clone(), edge_index=edge_index.clone(),
        batch=torch.tensor([0, 0, 0]),
        cheap_dft_scalars=dft.clone(),
        precision_settings=prec_b,
        cheap_geometry_scalars=geo.clone()
    )

    with torch.no_grad():
        _, (ea, _), _, _ = model(data_a)
        _, (eb, _), _, _ = model(data_b)

    assert not torch.allclose(ea, eb, atol=1e-6), \
        "FiLM conditioning has no effect on model output"


# =============================================
# Test 8e: FiLM identity initialization
# =============================================
def test_film_generator_identity_init():
    """At init, FiLM gamma should be near 1.0 (identity modulation)."""
    torch.manual_seed(0)
    film = FiLMGenerator(cond_dim=24, hidden_dim=16)

    # Zero conditioning input
    cond = torch.zeros(5, 24)
    gamma, beta = film(cond)

    assert gamma.shape == (5, 16)
    assert beta.shape == (5, 16)

    # gamma has a +1.0 bias, so mean should be > 0.5 even at init
    assert gamma.mean().item() > 0.5, \
        f"FiLM gamma not initialized near 1.0: mean={gamma.mean().item()}"


# =============================================
# Test 8f: PaiNNUpdate isolated shape and output
# =============================================
def test_painn_update_shapes():
    """PaiNNUpdate should produce correct shapes and non-trivial output."""
    F_dim = 16
    cond_dim = 24
    update = PaiNNUpdate(F_dim, cond_dim)

    N = 4
    s = torch.randn(N, F_dim)
    v = torch.randn(N, F_dim, 3)
    cond = torch.randn(N, cond_dim)

    ds, dv = update(s, v, cond)
    assert ds.shape == (N, F_dim)
    assert dv.shape == (N, F_dim, 3)

    # Update should be non-trivial (not all zeros)
    assert not torch.allclose(ds, torch.zeros_like(ds), atol=1e-8)
    assert not torch.allclose(dv, torch.zeros_like(dv), atol=1e-8)


# =============================================
# Test 8g: Translation invariance of scalar outputs
# =============================================
def test_delta_painn_translation_invariance():
    """Scalar outputs should be invariant under translation of all atoms."""
    torch.manual_seed(42)
    model = DeltaPaiNN(
        num_layers=2, hidden_dim=16, num_cheap_dft_inputs=12,
        num_precision_settings=12, num_geo_inputs=7, max_z=10,
        num_rbf=10, cutoff=5.0
    )
    model.eval()

    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    edge_index = torch.tensor([[0,1,1,2,0,2], [1,0,2,1,2,0]], dtype=torch.long)
    shared_dft = torch.randn(3, 12)
    shared_prec = torch.randn(1, 12)
    shared_geo = torch.randn(1, 7)

    data1 = Data(
        x=pos.clone(), z=torch.tensor([1, 6, 8]),
        edge_index=edge_index.clone(),
        batch=torch.tensor([0, 0, 0]),
        cheap_dft_scalars=shared_dft.clone(),
        precision_settings=shared_prec.clone(),
        cheap_geometry_scalars=shared_geo.clone()
    )

    # Translate all positions by a large vector
    translation = torch.tensor([10.0, -5.0, 3.0])
    data2 = Data(
        x=(pos + translation).clone(), z=torch.tensor([1, 6, 8]),
        edge_index=edge_index.clone(),
        batch=torch.tensor([0, 0, 0]),
        cheap_dft_scalars=shared_dft.clone(),
        precision_settings=shared_prec.clone(),
        cheap_geometry_scalars=shared_geo.clone()
    )

    with torch.no_grad():
        _, (e1, _), (g1, _), _ = model(data1)
        _, (e2, _), (g2, _), _ = model(data2)

    assert torch.allclose(e1, e2, atol=1e-4), \
        f"Energy not translation-invariant: {e1} vs {e2}"
    assert torch.allclose(g1, g2, atol=1e-4), \
        f"Gap not translation-invariant: {g1} vs {g2}"


# =============================================
# Test 8h: use_film=False runs and produces valid output
# =============================================
def test_delta_painn_use_film_false_runs():
    """Model with use_film=False should forward without errors."""
    model = DeltaPaiNN(
        num_layers=2, hidden_dim=16, num_cheap_dft_inputs=12,
        num_precision_settings=12, num_geo_inputs=7, max_z=10,
        num_rbf=10, cutoff=5.0, use_film=False
    )

    data = Data(
        x=torch.randn(3, 3), z=torch.tensor([1, 6, 8]),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        batch=torch.tensor([0, 0, 1]),
        cheap_dft_scalars=torch.randn(3, 12),
        precision_settings=torch.randn(2, 12),
        cheap_geometry_scalars=torch.randn(2, 7)
    )

    (pos_mean, pos_logvar), (e_mean, e_logvar), \
    (gap_mean, gap_logvar), (geo_mean, geo_logvar) = model(data)

    assert pos_mean.shape == (3, 3)
    assert e_mean.shape == (2,)
    assert gap_mean.shape == (2,)
    assert geo_mean.shape == (2, 7)

    # Gradients should still flow to all parameters
    loss = (pos_mean.sum() + pos_logvar.sum() +
            e_mean.sum() + e_logvar.sum() +
            gap_mean.sum() + gap_logvar.sum() +
            geo_mean.sum() + geo_logvar.sum())
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


# =============================================
# Test 8i: use_film flag changes backbone output
# =============================================
def test_delta_painn_film_flag_affects_backbone():
    """FiLM ON vs OFF should produce different scalar features in the backbone."""
    torch.manual_seed(42)

    model_film = DeltaPaiNN(
        num_layers=2, hidden_dim=16, num_cheap_dft_inputs=12,
        num_precision_settings=12, num_geo_inputs=7, max_z=10,
        num_rbf=10, cutoff=5.0, use_film=True
    )
    model_nofilm = DeltaPaiNN(
        num_layers=2, hidden_dim=16, num_cheap_dft_inputs=12,
        num_precision_settings=12, num_geo_inputs=7, max_z=10,
        num_rbf=10, cutoff=5.0, use_film=False
    )

    # Copy weights from film model to nofilm model so only the FiLM path differs
    shared_state = model_film.state_dict()
    # Remove FiLM-specific keys
    nofilm_keys = set(model_nofilm.state_dict().keys())
    filtered_state = {k: v for k, v in shared_state.items() if k in nofilm_keys}
    model_nofilm.load_state_dict(filtered_state)

    model_film.eval()
    model_nofilm.eval()

    data = Data(
        x=torch.randn(3, 3), z=torch.tensor([1, 6, 8]),
        edge_index=torch.tensor([[0,1,1,2,0,2], [1,0,2,1,2,0]], dtype=torch.long),
        batch=torch.tensor([0, 0, 0]),
        cheap_dft_scalars=torch.randn(3, 12),
        precision_settings=torch.randn(1, 12),
        cheap_geometry_scalars=torch.randn(1, 7)
    )

    with torch.no_grad():
        _, (e_film, _), _, _ = model_film(data)
        _, (e_nofilm, _), _, _ = model_nofilm(data)

    # FiLM should change the backbone output
    assert not torch.allclose(e_film, e_nofilm, atol=1e-6), \
        "use_film=True and use_film=False produce identical output"


# =============================================
# Test 9: Integration with DB -> Generator -> Model
# =============================================
REQUIRED_DB_KEYS = [
    'total_energy_per_atom', 'homo_lumo_gap',
    'sum_eigenvalues_per_atom', 'xc_energy_correction_per_atom',
    'xc_potential_correction_per_atom', 'free_atom_electrostatic_energy_per_atom',
    'hartree_energy_correction_per_atom', 'entropy_correction_per_atom',
    'total_energy_T0_per_atom', 'kinetic_energy_per_atom',
    'electrostatic_energy_per_atom', 'multipole_correction_per_atom',
    'final_volume_per_atom', 'relaxed_a_len', 'relaxed_b_len', 'relaxed_c_len',
    'relaxed_alpha_angle', 'relaxed_beta_angle', 'relaxed_gamma_angle',
    'delta_total_energy_per_atom', 'delta_homo_lumo_gap',
    'delta_aims_free_energy_per_atom',
    'delta_vbm', 'delta_cbm', 'delta_chemical_potential',
    'delta_final_volume_per_atom',
    'delta_relaxed_a_len', 'delta_relaxed_b_len', 'delta_relaxed_c_len',
    'delta_relaxed_alpha_angle', 'delta_relaxed_beta_angle', 'delta_relaxed_gamma_angle',
]

def get_dummy_row_data():
    data = {k: 0.5 for k in REQUIRED_DB_KEYS}
    data['delta_total_energy_per_atom'] = -0.1
    data['delta_homo_lumo_gap'] = 0.05
    data['delta_relaxed_atom_positions'] = "None"
    data['k_point_density'] = 2.0
    return data

@pytest.fixture
def mock_db_environment(tmp_path):
    db_file = tmp_path / "test_painn.db"
    out_file = tmp_path / "test_output.pt"
    dummy_data = get_dummy_row_data()

    with connect(db_file) as db:
        dummy_data['delta_relaxed_atom_positions'] = \
            "[[0.01, 0.0, 0.0], [-0.01, 0.0, 0.0]]"
        db.write(Atoms('H2', positions=[[0,0,0],[0,0,0.74]]),
                 split='train', **dummy_data)

        dummy_data['delta_relaxed_atom_positions'] = \
            "[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]"
        db.write(Atoms('H2O', positions=[[0,0,0],[1,0,0],[0,1,0]]),
                 split='train', **dummy_data)

    return str(db_file), str(out_file)

def test_integration_with_generated_db(mock_db_environment):
    db_path, out_path = mock_db_environment
    process_database(db_path, out_path, cutoff=3.0)

    assert os.path.exists(out_path)
    data_dict = torch.load(out_path, weights_only=False)
    graph_list = data_dict['graphs']
    assert len(graph_list) == 2

    loader = DataLoader(graph_list, batch_size=2)
    batch_data = next(iter(loader))

    model = DeltaPaiNN(
        num_layers=2, hidden_dim=32, num_cheap_dft_inputs=12,
        num_precision_settings=13, num_geo_inputs=7, max_z=100,
        num_rbf=10, cutoff=3.0
    )

    (pos_mean, pos_logvar), (e_mean, e_logvar), \
    (gap_mean, gap_logvar), (geo_mean, geo_logvar) = model(batch_data)

    # Total atoms = 2 (H2) + 3 (H2O) = 5
    assert pos_mean.shape == (5, 3)
    assert pos_logvar.shape == (5, 1)
    assert e_mean.shape == (2,)
    assert e_logvar.shape == (2,)
    assert gap_mean.shape == (2,)
    assert gap_logvar.shape == (2,)
    assert geo_mean.shape == (2, 7)
    assert geo_logvar.shape == (2, 7)
