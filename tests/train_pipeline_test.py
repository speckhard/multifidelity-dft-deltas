import pytest
import torch
import numpy as np
import ase.db
from ase import Atoms
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import os
import sys

# --- Path Setup ---
# Assumes structure:
# root/
#   modelling/gnn/...
#   tests/train_pipeline_test.py
# This puts 'root' in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modelling.gnn.delta_egnn_model import DeltaGNN
from modelling.gnn.ase_db_to_graphs import process_database
from unittest.mock import MagicMock
sys.modules["wandb"] = MagicMock()
try:
    from modelling.gnn.train_pipeline import AsinhL1Loss, evaluate, plot_history, train_step
except ImportError:
    # If the user is running pytest strictly from the tests/ folder
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'modelling', 'gnn'))
    from train_pipeline import AsinhL1Loss, evaluate, plot_history, train_step



DEEP_DFT_KEYS = [
    'sum_eigenvalues_per_atom', 'xc_energy_correction_per_atom', 
    'xc_potential_correction_per_atom', 'free_atom_electrostatic_energy_per_atom', 
    'hartree_energy_correction_per_atom', 'entropy_correction_per_atom', 
    'total_energy_T0_per_atom', 'kinetic_energy_per_atom', 
    'electrostatic_energy_per_atom', 'multipole_correction_per_atom'
]
CHEAP_GEOMETRY_KEYS = [
    'final_volume_per_atom', # Note: key name must match exactly
    'relaxed_a_len', 'relaxed_b_len', 'relaxed_c_len',
    'relaxed_alpha_angle', 'relaxed_beta_angle', 'relaxed_gamma_angle'
]
# Needed so rows aren't skipped
ADDITIONAL_TARGETS = [
    'delta_aims_free_energy_per_atom',
    'delta_vbm', 'delta_cbm', 'delta_chemical_potential', 
    'delta_final_volume_per_atom',
    'delta_relaxed_a_len', 'delta_relaxed_b_len', 'delta_relaxed_c_len',
    'delta_relaxed_alpha_angle', 'delta_relaxed_beta_angle', 'delta_relaxed_gamma_angle',
]

ALL_DUMMY_KEYS = DEEP_DFT_KEYS + CHEAP_GEOMETRY_KEYS + ADDITIONAL_TARGETS


class MockDeltaGNN(torch.nn.Module):
    """
    A dumb model that returns predictable constant outputs for testing math.

    Matches the multi-target heteroscedastic head layout of the real
    DeltaGNN-family models: each head returns (mean, logvar). Used by
    test_evaluate_metrics and test_train_step_runs to test the
    train_pipeline.evaluate / train_step functions in isolation.
    """
    def __init__(self, output_val=0.0):
        super().__init__()
        self.output_val = output_val

        # Dummy parameter so next(model.parameters()) works (evaluate uses this
        # to determine device) AND so loss.backward() has something to differentiate.
        self.dummy_param = torch.nn.Parameter(torch.tensor([0.0]))

    def forward(self, data):
        num_nodes = data.x.shape[0]
        batch_size = data.num_graphs if hasattr(data, 'num_graphs') else 1

        # Connect the output to the parameter (enables .backward() in train_step)
        val = self.output_val + (0 * self.dummy_param.sum())

        # Heads with (mean, logvar) tuples matching the real DeltaGNN architecture:
        #   pos: per-atom (N, 3) mean + (N, 1) logvar
        #   energy/gap: per-graph (B,) mean + (B,) logvar
        #   geo: per-graph (B, 7) mean + (B, 7) logvar (volume + 6 lattice scalars)
        pos_mean = torch.ones((num_nodes, 3), dtype=torch.float) * val
        pos_logvar = torch.zeros((num_nodes, 1), dtype=torch.float) + (0 * self.dummy_param.sum())

        e_mean = torch.ones((batch_size,), dtype=torch.float) * val
        e_logvar = torch.zeros((batch_size,), dtype=torch.float) + (0 * self.dummy_param.sum())

        gap_mean = torch.ones((batch_size,), dtype=torch.float) * val
        gap_logvar = torch.zeros((batch_size,), dtype=torch.float) + (0 * self.dummy_param.sum())

        geo_mean = torch.ones((batch_size, 7), dtype=torch.float) * val
        geo_logvar = torch.zeros((batch_size, 7), dtype=torch.float) + (0 * self.dummy_param.sum())

        return (
            (pos_mean, pos_logvar),
            (e_mean, e_logvar),
            (gap_mean, gap_logvar),
            (geo_mean, geo_logvar),
        )


@pytest.fixture
def mock_loader():
    """Creates a DataLoader with known target values."""
    # Create 2 graphs
    # Graph 1: 2 atoms
    # Graph 2: 3 atoms
    # Total: 5 atoms, 2 graphs
    
    # We set targets to 1.0 everywhere.
    # If Model predicts 0.0, errors should be exactly 1.0 (MAE).
    
    g1 = Data(
        x=torch.randn(2, 3),
        delta_relaxed_atom_positions=torch.ones(2, 3), # Target = 1
        delta_total_energy_per_atom=torch.tensor([1.0]),
        delta_homo_lumo_gap=torch.tensor([1.0]),
        delta_final_volume_per_atom=torch.tensor([1.0]),
        delta_lattice_params=torch.tensor([[1.0]*6]),
        num_graphs=1
    )
    
    g2 = Data(
        x=torch.randn(3, 3),
        delta_relaxed_atom_positions=torch.ones(3, 3), # Target = 1
        delta_total_energy_per_atom=torch.tensor([1.0]),
        delta_homo_lumo_gap=torch.tensor([1.0]),
        delta_final_volume_per_atom=torch.tensor([1.0]),
        delta_lattice_params=torch.tensor([[1.0]*6]),
        num_graphs=1
    )
    
    return DataLoader([g1, g2], batch_size=2)

# --- Constants for Mock Data ---
# These match the strict validation in ase_db_to_graphs.py

def create_dummy_delta_db(tmp_path):
    """Creates a temporary ASE database with valid schema for testing."""
    db_path = tmp_path / "dummy_delta.db"
    # Simple H2 molecule
    atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]])
    
    # Fill required deep keys with dummy floats
    dummy_data = {k: 0.1 for k in ALL_DUMMY_KEYS}
    
    # Helper to stringify arrays for ASE DB
    def format_arr(arr):
        return np.array2string(
            arr, separator=',',
            formatter={'float_kind': lambda x: "%.16f" % x}
        ).replace('\n', '')

    with ase.db.connect(str(db_path)) as db:
        # Data Point 1
        db.write(
            atoms,
            # Metadata
            k_point_density=4.0,
            binary_precision=2, # Maps to index 2 in One-Hot
            
            # Basic Inputs
            total_energy_per_atom=1.4,
            homo_lumo_gap=2.7,
            
            # Core Targets
            delta_total_energy_per_atom=5.0,
            delta_homo_lumo_gap=2.5,
            delta_relaxed_atom_positions=format_arr(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.01]])),
            
            # Flatten dummy data into kwargs
            **dummy_data
        )
        
        # Data Point 2 (Different precision/k-points)
        db.write(
            atoms,
            k_point_density=8.0,
            binary_precision=8,
            
            total_energy_per_atom=1.4,
            homo_lumo_gap=2.7,
            
            delta_total_energy_per_atom=1.0,
            delta_homo_lumo_gap=1.5,
            delta_relaxed_atom_positions=format_arr(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.005]])),
            
            **dummy_data
        )
        
    return str(db_path)



def test_asinh_loss_math():
    """
    Verifies the custom AsinhL1Loss computes the correct value.
    Formula: L1( asinh(pred / scale), asinh(target / scale) )
    """
    criterion = AsinhL1Loss(scale=1.0)

    # Case 1: Perfect prediction
    pred = torch.tensor([0.5])
    target = torch.tensor([0.5])
    loss = criterion(pred, target, is_position=False)
    assert loss.item() == 0.0

    # Case 2: Known error
    # scale=1.0. pred=0, target=1. asinh(0/1)=0. asinh(1/1) approx 0.881
    pred = torch.tensor([0.0])
    target = torch.tensor([1.0])
    loss = criterion(pred, target, is_position=False)

    expected = abs(np.arcsinh(0.0) - np.arcsinh(1.0))
    assert loss.item() == pytest.approx(expected, abs=1e-5)

    # Case 3: Position Scaling — equivalent of "multiply by 10" is scale=0.1
    # asinh(0 / 0.1) - asinh(0.1 / 0.1) = asinh(0) - asinh(1) ≈ 0.881
    pos_criterion = AsinhL1Loss(scale=0.1)
    loss_pos = pos_criterion(pred, torch.tensor([0.1]), is_position=True)
    assert loss_pos.item() == pytest.approx(expected, abs=1e-5)


def test_evaluate_metrics(mock_loader):
    """
    Verifies that the evaluate function correctly aggregates MAE
    across batches and atoms.

    Uses the new evaluate(model, loader, criterion, weights) signature
    that returns a (metrics, arrays) tuple. Metric keys follow the
    `MAE_delta_<target>` naming convention.
    """
    # Model always predicts 0.0; targets are always 1.0 -> MAE = 1.0
    model = MockDeltaGNN(output_val=0.0)
    criterion = AsinhL1Loss(scale=1.0)

    class MockWeights:
        delta_r = 1.0
        delta_e = 1.0
        delta_gap = 1.0
        delta_geo = 1.0  # _get_geo_weights falls back to this for vol+lat

    metrics, arrays = evaluate(model, mock_loader, criterion, MockWeights())

    # 1. Check Keys
    assert "MAE_delta_energy" in metrics
    assert "MAE_delta_gap" in metrics
    assert "MAE_delta_positions" in metrics
    assert "MAE_delta_geo_total" in metrics

    # 2. Check Values
    # Energy: pred=0, target=1 -> MAE = 1.0
    assert metrics["MAE_delta_energy"] == pytest.approx(1.0)
    # Gap: pred=0, target=1 -> MAE = 1.0
    assert metrics["MAE_delta_gap"] == pytest.approx(1.0)

    # Position MAE: per-atom diff is [1,1,1], its L2 norm is sqrt(3) ≈ 1.732.
    # evaluate sums these norms over all atoms then divides by total_atoms.
    # 5 atoms total (2 + 3) * sqrt(3) / 5 = sqrt(3).
    expected_r_mae = np.sqrt(3.0)
    assert metrics["MAE_delta_positions"] == pytest.approx(expected_r_mae, abs=1e-4)

    # Geo total MAE: target_delta_geo is concat([vol(2,1), lat(2,6)]) = (2,7) all ones,
    # geo_mean is (2,7) zeros, F.l1_loss(reduction='sum') = 14, divide by 2 graphs = 7.
    assert metrics["MAE_delta_geo_total"] == pytest.approx(7.0)


def test_plot_history_generates_file(tmp_path):
    """
    Verifies that plot_history creates a PNG file given a history dict.
    """
    # Create dummy history
    history = {
        'train_loss': [0.5, 0.4, 0.3],
        'train_mae_e': [(1, 0.5), (2, 0.4), (3, 0.3)],
        'test_mae_e': [(1, 0.6), (2, 0.5), (3, 0.4)]
    }
    
    save_path = tmp_path / "test_plot.png"
    
    # We don't want to actually show the plot, just save it
    # Use a non-interactive backend to prevent window popping up
    plt.switch_backend('Agg')
    
    plot_history(history, str(save_path))
    
    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_train_step_runs(mock_loader):
    """
    Integration test: Does a single training step run without crashing?
    """
    model = MockDeltaGNN(output_val=0.0)
    
    # REMOVED: model.dummy_param = ... (The class handles this now)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = AsinhL1Loss(scale=1.0)
    
    class MockWeights:
        delta_r = 1.0
        delta_e = 1.0
        delta_gap = 1.0
        delta_geo = 1.0
    
    weights = MockWeights()
    device = torch.device('cpu')
    
    loss = train_step(model, mock_loader, optimizer, criterion, weights, device)
    
    assert isinstance(loss, float)
    assert loss >= 0.0 # Loss should be non-negative


def test_ase_dataset_loading_and_scaling(tmp_path):
    """Verifies that ase_db_to_graphs processes the new keys correctly."""
    # 1. Create DB
    db_path = create_dummy_delta_db(tmp_path)
    output_path = str(tmp_path / 'output_file.pt')
    
    # 2. Run Generator
    process_database(db_path=db_path, output_path=output_path, cutoff=1.0)
    
    # 3. Load Result
    assert os.path.exists(output_path)
    loaded_dict = torch.load(output_path, weights_only=False)
    graph_list = loaded_dict['graphs']
    
    loader = DataLoader(graph_list, batch_size=2)
    assert len(graph_list) == 2

    data = next(iter(loader))
    
    # A. Check Cheap DFT Scalars (Node Level)
    # 2 graphs * 2 atoms = 4 nodes
    # Dims: 2 Basic + 10 Deep = 12
    assert data.cheap_dft_scalars.shape == (4, 12)
    assert data.cheap_dft_scalars.dtype == torch.float
    
    # B. Check Precision Settings (Graph Level)
    # Dims: 12 One-Hot (bp 0..11) + 1 K-density = 13 (after bp=11 fix)
    assert hasattr(data, 'precision_settings')
    assert data.precision_settings.shape == (2, 13)

    # C. Check Cheap Geometry Scalars (Graph Level)
    # Dims: 1 Vol + 6 Lattice = 7
    assert hasattr(data, 'cheap_geometry_scalars')
    assert data.cheap_geometry_scalars.shape == (2, 7)
    
    # D. Check Scalers
    assert 'dft_scaler' in loaded_dict
    assert 'geo_scaler' in loaded_dict


def test_delta_gnn_forward_pass(tmp_path):
    """Verifies the Model accepts the new inputs and returns 4 outputs."""
    # 1. Setup Data
    db_path = create_dummy_delta_db(tmp_path)
    output_path = str(tmp_path / 'output_file.pt')
    process_database(db_path=db_path, output_path=output_path, cutoff=1.0)
    
    loaded_dict = torch.load(output_path, weights_only=False)
    graph_list = loaded_dict['graphs']
    loader = DataLoader(graph_list, batch_size=2) 
    data = next(iter(loader))

    # 2. Setup Model
    model = DeltaGNN(
        num_layers=2,
        hidden_features=32,
        num_cheap_dft_inputs=12,    # New Arg
        num_precision_settings=13,  # 12 OHE + 1 K-dens (after bp=11 fix)
        num_geo_inputs=7,           # New Arg
        max_z=100
    )
    
    # 3. Run Forward Pass
    # Multi-target architecture returns (mean, logvar) tuples for each head.
    (pos_mean, pos_logvar), (e_mean, e_logvar), \
        (gap_mean, gap_logvar), (geo_mean, geo_logvar) = model(data)

    # 4. Check Shapes
    # Positions (Per Node)
    assert pos_mean.shape == (4, 3)
    assert pos_logvar.shape == (4, 1)

    # Energy/Gap (Per Graph)
    assert e_mean.shape == (2,)
    assert gap_mean.shape == (2,)

    # Geometry (Per Graph, 7 dims)
    assert geo_mean.shape == (2, 7)

    # 5. Check Equivariance / Invariance
    # delta_r is a displacement vector. If we translate the input system,
    # the predicted DISPLACEMENT should be the same (vector is invariant to translation of origin).
    T = torch.tensor([10.0, 5.0, 0.0], dtype=torch.float)
    data_translated = data.clone()
    data_translated.x = data.x + T

    (pos_mean_trans, _), _, _, _ = model(data_translated)

    # The output vectors should be identical
    assert torch.allclose(pos_mean, pos_mean_trans, atol=1e-5)
