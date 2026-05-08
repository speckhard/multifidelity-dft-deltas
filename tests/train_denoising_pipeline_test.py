import pytest
import torch
import os
import sys
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from modelling.gnn.train_denoising_pipeline import (
    calibrate_noise_schedule,
    get_per_atom_sigma,
)
from modelling.gnn.delta_painn_model import DeltaPaiNN


def _make_dummy_graphs(precision_vectors, energy_deltas):
    """Helper: create a list of Data objects with given precision/energy pairs."""
    graphs = []
    for prec, delta_e in zip(precision_vectors, energy_deltas):
        g = Data(
            x=torch.randn(2, 3),
            z=torch.tensor([1, 1]),
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            batch=torch.tensor([0, 0]),
            cheap_dft_scalars=torch.randn(2, 12),
            precision_settings=prec.unsqueeze(0),  # [1, 12]
            cheap_geometry_scalars=torch.randn(1, 7),
            delta_total_energy_per_atom=torch.tensor([delta_e]),
            delta_homo_lumo_gap=torch.tensor([0.0]),
            delta_relaxed_atom_positions=torch.randn(2, 3),
            delta_final_volume_per_atom=torch.tensor([0.0]),
            delta_lattice_params=torch.randn(1, 6),
        )
        graphs.append(g)
    return graphs


# =============================================
# Test 10: calibrate_noise_schedule returns valid map
# =============================================
def test_calibrate_noise_schedule():
    """Map should have entries for each unique precision setting,
    with sigma values in [sigma_low, sigma_high]."""
    sigma_low, sigma_high = 0.01, 0.1

    # Two distinct precision settings with different energy errors
    prec_a = torch.zeros(12); prec_a[0] = 1.0   # low precision
    prec_b = torch.zeros(12); prec_b[10] = 1.0   # high precision

    graphs = _make_dummy_graphs(
        [prec_a, prec_a, prec_b, prec_b],
        [0.5, 0.3, 0.05, 0.07]  # group A has higher |delta_e|
    )
    loader = DataLoader(graphs, batch_size=2, shuffle=False)

    fidelity_map = calibrate_noise_schedule(loader, sigma_low, sigma_high)

    assert len(fidelity_map) == 2
    for key, sigma in fidelity_map.items():
        assert sigma_low <= sigma <= sigma_high, \
            f"sigma {sigma} out of range [{sigma_low}, {sigma_high}]"

    # Group A (higher MAE) should get higher sigma
    key_a = tuple(prec_a.tolist())
    key_b = tuple(prec_b.tolist())
    assert fidelity_map[key_a] > fidelity_map[key_b]


# =============================================
# Test 11: calibrate_noise_schedule single group (div-by-zero)
# =============================================
def test_calibrate_noise_schedule_single_group():
    """Single group or identical MAEs should all map to sigma_low."""
    sigma_low, sigma_high = 0.01, 0.1

    prec = torch.zeros(12); prec[3] = 1.0
    graphs = _make_dummy_graphs(
        [prec, prec, prec],
        [0.1, 0.2, 0.15]
    )
    loader = DataLoader(graphs, batch_size=3, shuffle=False)

    fidelity_map = calibrate_noise_schedule(loader, sigma_low, sigma_high)

    assert len(fidelity_map) == 1
    key = tuple(prec.tolist())
    assert fidelity_map[key] == pytest.approx(sigma_low, abs=1e-8)


# =============================================
# Test 12: get_per_atom_sigma shapes
# =============================================
def test_get_per_atom_sigma_shapes():
    prec = torch.zeros(12); prec[0] = 1.0
    fidelity_map = {tuple(prec.tolist()): 0.05}

    # B=2 graphs, N=5 atoms total
    precision_settings = torch.stack([prec, prec])  # [2, 12]
    batch = torch.tensor([0, 0, 0, 1, 1])           # 3 atoms + 2 atoms

    sigma_atom = get_per_atom_sigma(precision_settings, batch, fidelity_map)
    assert sigma_atom.shape == (5, 1)


# =============================================
# Test 13: get_per_atom_sigma correct lookup
# =============================================
def test_get_per_atom_sigma_lookup():
    prec_a = torch.zeros(12); prec_a[0] = 1.0
    prec_b = torch.zeros(12); prec_b[5] = 1.0

    fidelity_map = {
        tuple(prec_a.tolist()): 0.08,
        tuple(prec_b.tolist()): 0.02,
    }

    precision_settings = torch.stack([prec_a, prec_b])  # [2, 12]
    batch = torch.tensor([0, 0, 1, 1, 1])  # 2 atoms graph 0, 3 atoms graph 1

    sigma_atom = get_per_atom_sigma(precision_settings, batch, fidelity_map)

    # Atoms in graph 0 should have sigma=0.08
    assert sigma_atom[0, 0].item() == pytest.approx(0.08, abs=1e-6)
    assert sigma_atom[1, 0].item() == pytest.approx(0.08, abs=1e-6)
    # Atoms in graph 1 should have sigma=0.02
    assert sigma_atom[2, 0].item() == pytest.approx(0.02, abs=1e-6)
    assert sigma_atom[3, 0].item() == pytest.approx(0.02, abs=1e-6)
    assert sigma_atom[4, 0].item() == pytest.approx(0.02, abs=1e-6)


# =============================================
# Test 14: denoise_train_step runs without error
# =============================================
def test_denoise_train_step_runs():
    """One training step should complete and return a finite loss."""
    from omegaconf import OmegaConf
    from modelling.gnn.train_denoising_pipeline import denoise_train_step
    from modelling.gnn.train_pipeline import GaussianNLLLoss

    model = DeltaPaiNN(
        num_layers=1, hidden_dim=16, num_cheap_dft_inputs=12,
        num_precision_settings=12, num_geo_inputs=7, max_z=10,
        num_rbf=10, cutoff=5.0
    )

    prec = torch.zeros(12); prec[2] = 1.0
    graphs = _make_dummy_graphs([prec, prec], [0.1, 0.2])
    loader = DataLoader(graphs, batch_size=2, shuffle=False)

    fidelity_map = {tuple(prec.tolist()): 0.05}

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = GaussianNLLLoss()
    weights = OmegaConf.create({
        'delta_e': 1.0, 'delta_gap': 1.0, 'delta_r': 1.0,
        'delta_vol': 1.0, 'delta_lat': 1.0,
    })
    cfg = OmegaConf.create({
        'denoising_weight': 0.1,
        'noise_scales': {'low': 0.01, 'high': 0.1},
    })

    device = torch.device('cpu')
    loss = denoise_train_step(
        model, loader, optimizer, criterion, weights, device, cfg, fidelity_map
    )

    assert isinstance(loss, float)
    assert not torch.isnan(torch.tensor(loss))
    assert not torch.isinf(torch.tensor(loss))


# =============================================
# Test 15: evaluate() compatibility with DeltaPaiNN
# =============================================
def test_evaluate_with_delta_painn():
    """The shared evaluate() function must work with DeltaPaiNN."""
    from omegaconf import OmegaConf
    from modelling.gnn.train_pipeline import evaluate, GaussianNLLLoss

    model = DeltaPaiNN(
        num_layers=1, hidden_dim=16, num_cheap_dft_inputs=12,
        num_precision_settings=12, num_geo_inputs=7, max_z=10,
        num_rbf=10, cutoff=5.0
    )
    model.eval()

    prec = torch.zeros(12); prec[0] = 1.0
    graphs = _make_dummy_graphs([prec, prec], [0.1, 0.2])
    loader = DataLoader(graphs, batch_size=2, shuffle=False)

    criterion = GaussianNLLLoss()
    weights = OmegaConf.create({
        'delta_e': 1.0, 'delta_gap': 1.0, 'delta_r': 1.0,
        'delta_vol': 1.0, 'delta_lat': 1.0,
    })

    metrics, arrays = evaluate(model, loader, criterion, weights)

    assert 'MAE_delta_energy' in metrics
    assert 'MAE_delta_gap' in metrics
    assert 'MAE_delta_positions' in metrics
    assert 'Val_Loss' in metrics
    assert metrics['MAE_delta_energy'] >= 0
    assert metrics['MAE_delta_gap'] >= 0
    assert len(arrays['delta_energy']['pred']) > 0


# =============================================
# Test 16: Denoising loss sign convention
# =============================================
def test_denoising_loss_sign_convention():
    """Denoising target should be -epsilon (not +epsilon). Sign matters."""
    import torch.nn.functional as F

    model = DeltaPaiNN(
        num_layers=1, hidden_dim=16, num_cheap_dft_inputs=12,
        num_precision_settings=12, num_geo_inputs=7, max_z=10,
        num_rbf=10, cutoff=5.0
    )
    model.eval()

    prec = torch.zeros(12); prec[0] = 1.0
    data = Data(
        x=torch.randn(2, 3),
        z=torch.tensor([1, 1]),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        batch=torch.tensor([0, 0]),
        cheap_dft_scalars=torch.randn(2, 12),
        precision_settings=prec.unsqueeze(0),
        cheap_geometry_scalars=torch.randn(1, 7),
    )

    sigma = 0.1
    epsilon = torch.randn_like(data.x) * sigma

    # Forward on noisy data
    data_noisy = data.clone()
    data_noisy.x = data.x.clone() + epsilon

    with torch.no_grad():
        _ = model(data_noisy)
        vec = model._vector_output

    loss_correct = F.mse_loss(vec, -epsilon)
    loss_wrong = F.mse_loss(vec, epsilon)

    assert torch.isfinite(loss_correct)
    assert torch.isfinite(loss_wrong)
    # The two losses should differ (sign matters for score matching)
    assert not torch.allclose(loss_correct, loss_wrong), \
        "Sign of epsilon doesn't affect loss — denoising target may be wrong"


# =============================================
# Test 17: get_per_atom_sigma fallback for unseen keys
# =============================================
def test_get_per_atom_sigma_unseen_key_fallback():
    """Unseen precision settings should use fallback sigma, not crash."""
    prec_known = torch.zeros(12); prec_known[0] = 1.0
    prec_unknown = torch.zeros(12); prec_unknown[11] = 1.0

    fidelity_map = {tuple(prec_known.tolist()): 0.05}
    expected_fallback = 0.05  # mean of [0.05]

    precision_settings = torch.stack([prec_known, prec_unknown])  # [2, 12]
    batch = torch.tensor([0, 0, 1, 1])  # 2 atoms per graph

    sigma_atom = get_per_atom_sigma(precision_settings, batch, fidelity_map)

    assert sigma_atom.shape == (4, 1)
    # Known key atoms get 0.05
    assert sigma_atom[0, 0].item() == pytest.approx(0.05, abs=1e-6)
    assert sigma_atom[1, 0].item() == pytest.approx(0.05, abs=1e-6)
    # Unknown key atoms get fallback (mean of known sigmas)
    assert sigma_atom[2, 0].item() == pytest.approx(expected_fallback, abs=1e-6)
    assert sigma_atom[3, 0].item() == pytest.approx(expected_fallback, abs=1e-6)
