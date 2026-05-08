import pytest
import os
import sys
import torch
import numpy as np
from ase.db import connect
from ase import Atoms

# --- Import Setup ---
# Add the project root to path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from modelling.gnn.ase_db_to_graphs import process_database

# --- Constants Validation List ---
# We define these here to ensure the Mock DB has every column 
# the script expects. If one is missing, the script skips the row.

# 1. Deep DFT Keys expected by the script
DEEP_DFT_KEYS = [
    'sum_eigenvalues_per_atom', 'xc_energy_correction_per_atom', 
    'xc_potential_correction_per_atom', 'free_atom_electrostatic_energy_per_atom', 
    'hartree_energy_correction_per_atom', 'entropy_correction_per_atom', 
    'total_energy_T0_per_atom', 'kinetic_energy_per_atom', 
    'electrostatic_energy_per_atom', 'multipole_correction_per_atom'
]

# 2. Geometry Keys expected by the script
CHEAP_GEOMETRY_KEYS = [
    'final_volume_per_atom',
    'relaxed_a_len', 'relaxed_b_len', 'relaxed_c_len',
    'relaxed_alpha_angle', 'relaxed_beta_angle', 'relaxed_gamma_angle'
]

# 3. Target Delta Keys
ADDITIONAL_TARGETS = [
    'delta_aims_free_energy_per_atom',
    'delta_vbm', 'delta_cbm', 'delta_chemical_potential', 
    'delta_final_volume_per_atom',
    'delta_relaxed_a_len', 'delta_relaxed_b_len', 'delta_relaxed_c_len',
    'delta_relaxed_alpha_angle', 'delta_relaxed_beta_angle', 'delta_relaxed_gamma_angle',
]

ALL_DUMMY_KEYS = DEEP_DFT_KEYS + CHEAP_GEOMETRY_KEYS + ADDITIONAL_TARGETS

@pytest.fixture
def mock_db_environment(tmp_path):
    """
    Creates a real SQLite database in a temp folder.
    Returns a tuple: (path_to_db, path_to_output)
    """
    db_file = tmp_path / "test_delta.db"
    out_file = tmp_path / "test_output.pt"
    
    # Create a dictionary of dummy values for all the extra required keys
    # so the generator doesn't skip the rows.
    dummy_data = {k: 0.5 for k in ALL_DUMMY_KEYS}
    
    # Add required metadata
    dummy_data['k_point_density'] = 5.0
    dummy_data['binary_precision'] = 1 # Maps to index 1 in OHE
    
    # Add basis info (default to 0 to avoid filtering)
    dummy_data['delta_mean_basis_functions'] = 0.0
    
    # Add spacegroup (mock metadata)
    dummy_data['spacegroup'] = 225

    # --- 1. Create Mock Data ---
    with connect(db_file) as db:
        # Train Row 1
        db.write(Atoms('H2', positions=[[0,0,0],[0,0,1]]), 
                 split='train', 
                 total_energy_per_atom=10.0, homo_lumo_gap=1.0,
                 delta_total_energy_per_atom=0.1, delta_homo_lumo_gap=0.01,
                 delta_relaxed_atom_positions="[[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]",
                 **dummy_data)
        
        # Train Row 2
        db.write(Atoms('H2', positions=[[0,0,0],[0,0,1]]), 
                 split='train', 
                 total_energy_per_atom=20.0, homo_lumo_gap=2.0,
                 delta_total_energy_per_atom=0.2, delta_homo_lumo_gap=0.02,
                 delta_relaxed_atom_positions="[[0.2, 0.2, 0.2], [0.2, 0.2, 0.2]]",
                 **dummy_data)
        
        # Test Row (Outlier for Scaler check)
        # We give this a HUGE energy value to verify the scaler doesn't see it
        db.write(Atoms('H2', positions=[[0,0,0],[0,0,1]]), 
                 split='test', 
                 total_energy_per_atom=1000.0, homo_lumo_gap=5.0,
                 delta_total_energy_per_atom=0.5, delta_homo_lumo_gap=0.05,
                 delta_relaxed_atom_positions="[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]",
                 **dummy_data)

    return str(db_file), str(out_file)


def test_end_to_end_execution(mock_db_environment):
    """Verifies the function runs and produces a valid .pt file."""
    db_path, out_path = mock_db_environment
    
    # Call the function directly with explicit arguments
    process_database(db_path, out_path, cutoff=5.0)
    
    assert os.path.exists(out_path)
    data = torch.load(out_path, weights_only=False)
    
    # Should contain 3 graphs (2 train + 1 test)
    assert len(data['graphs']) == 3


def test_over_convergence_filtering(tmp_path):
    """
    Verifies that rows with delta_basis > 0 AND k_point_density == 8 
    are EXCLUDED from the dataset.
    """
    db_file = tmp_path / "filter_test.db"
    out_file = tmp_path / "filter_out.pt"
    
    dummy_data = {k: 0.5 for k in ALL_DUMMY_KEYS}
    dummy_data['binary_precision'] = 1
    
    with connect(db_file) as db:
        # Row 1: Valid (Standard calc)
        # k=4, delta_basis=0 -> KEEP
        db.write(Atoms('H2'), split='train', k_point_density=4.0, delta_mean_basis_functions=0.0, 
                 total_energy_per_atom=10.0, homo_lumo_gap=1.0, delta_total_energy_per_atom=0.1,
                 delta_homo_lumo_gap=0.1, delta_relaxed_atom_positions="[[0,0,0],[0,0,0]]", **dummy_data)

        # Row 2: Invalid (Over-converged)
        # k=8, delta_basis=1.0 -> SKIP
        db.write(Atoms('H2'), split='train', k_point_density=8.0, delta_mean_basis_functions=1.0, 
                 total_energy_per_atom=10.0, homo_lumo_gap=1.0, delta_total_energy_per_atom=0.1,
                 delta_homo_lumo_gap=0.1, delta_relaxed_atom_positions="[[0,0,0],[0,0,0]]", **dummy_data)
                 
        # Row 3: Valid (High K but standard basis)
        # k=8, delta_basis=0 -> KEEP
        db.write(Atoms('H2'), split='train', k_point_density=8.0, delta_mean_basis_functions=0.0, 
                 total_energy_per_atom=10.0, homo_lumo_gap=1.0, delta_total_energy_per_atom=0.1,
                 delta_homo_lumo_gap=0.1, delta_relaxed_atom_positions="[[0,0,0],[0,0,0]]", **dummy_data)

    process_database(str(db_file), str(out_file), cutoff=5.0)
    
    data = torch.load(str(out_file), weights_only=False)
    
    # Should keep Row 1 and Row 3. Row 2 must be gone.
    assert len(data['graphs']) == 2
    
    # Check that we didn't keep the k=8 one by accident (Row 2)
    # Row 3 is k=8 but basis=0, so we check basis.
    # Note: basis info is not stored in the final graph, so we trust the count.


def test_split_separation_and_scaler_fitting(mock_db_environment):
    """Verifies scaler fitting logic works and does NOT leak test data."""
    db_path, out_path = mock_db_environment
    
    process_database(db_path, out_path, cutoff=5.0)
    data = torch.load(out_path, weights_only=False)
    
    # Retrieve the scaler
    dft_scaler = data['dft_scaler']
    
    # --- Check Leakage ---
    # The 'total_energy_per_atom' is index 0 in BASIC_INPUT_KEYS
    # Train values: 10.0 and 20.0 -> Mean should be 15.0
    # Test value: 1000.0 (Should be ignored)
    
    fitted_mean_energy = dft_scaler.mean_[0] 
    
    assert np.isclose(fitted_mean_energy, 15.0), \
        f"Scaler leaked test data! Expected 15.0, got {fitted_mean_energy}"


def test_graph_structure_and_features(mock_db_environment):
    """Verifies graph structure and new attribute keys."""
    db_path, out_path = mock_db_environment

    process_database(db_path, out_path, cutoff=5.0)
    data_dict = torch.load(out_path, weights_only=False)
    graph = data_dict['graphs'][0]

    # 1. Check Dimensions
    # H2 = 2 nodes. Positions = [2, 3]
    assert graph.x.shape == (2, 3)

    # 2. Check Spacegroup Extraction
    # We mocked 'spacegroup=225' in the DB.
    assert hasattr(graph, 'spacegroup')
    assert graph.spacegroup.item() == 225

    # 3. Check Separated Attributes (New Logic)
    # Z should be [2, 1]
    assert graph.z.shape == (2, 1)

    # cheap_dft_scalars should be [2, 12]
    # (2 Basic Inputs + 10 Deep DFT Inputs = 12 columns)
    assert hasattr(graph, 'cheap_dft_scalars')
    assert graph.cheap_dft_scalars.shape == (2, 12)

    # 4. Check Global Context & Geometry
    assert hasattr(graph, 'precision_settings') # The OHE + K-dens vector
    assert hasattr(graph, 'cheap_geometry_scalars') # The 7 geometry inputs

    # 5. Check Targets
    assert torch.is_tensor(graph.delta_relaxed_atom_positions)
    assert hasattr(graph, 'delta_final_volume_per_atom')
    assert hasattr(graph, 'delta_aims_free_energy_per_atom')


def test_precision_settings_shape_and_layout(mock_db_environment):
    """bp=11 fix regression guard: precision_settings is 13-dim with the
    layout [12-dim bp one-hot, k_point_density scalar at index 12].

    Mock data uses binary_precision=1, k_point_density=5.0.
    """
    db_path, out_path = mock_db_environment

    process_database(db_path, out_path, cutoff=5.0)
    data_dict = torch.load(out_path, weights_only=False)
    graph = data_dict['graphs'][0]

    # Shape: precision_settings is stored as [1, 13] (graph-level, not per-atom)
    assert graph.precision_settings.shape == (1, 13), \
        f"Expected (1, 13), got {tuple(graph.precision_settings.shape)} — bp=11 fix not applied"

    vec = graph.precision_settings.squeeze(0)
    # bp=1 one-hot at index 1
    assert vec[1].item() == 1.0
    # All other one-hot slots zero (indices 0, 2..11)
    assert vec[0].item() == 0.0
    assert vec[2:12].sum().item() == 0.0
    # k_point_density at the trailing index 12
    assert vec[12].item() == pytest.approx(5.0)


def test_bp11_gets_own_one_hot_slot(tmp_path):
    """A row with binary_precision=11 must produce a 13-dim vector with the
    one-hot bit set at index 11 — NOT silently mapped to all-zeros.

    This is the canonical regression guard for the bp=11 silent-zero bug.
    """
    db_file = tmp_path / "bp11.db"
    out_file = tmp_path / "bp11.pt"

    dummy_data = {k: 0.5 for k in ALL_DUMMY_KEYS}
    dummy_data['k_point_density'] = 8.0
    dummy_data['binary_precision'] = 11  # really_tight/tier2 — the formerly-silent case
    dummy_data['delta_mean_basis_functions'] = 0.0
    dummy_data['spacegroup'] = 225

    with connect(db_file) as db:
        db.write(Atoms('H2', positions=[[0, 0, 0], [0, 0, 1]]),
                 split='train',
                 total_energy_per_atom=10.0, homo_lumo_gap=1.0,
                 delta_total_energy_per_atom=0.1, delta_homo_lumo_gap=0.01,
                 delta_relaxed_atom_positions="[[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]",
                 **dummy_data)

    process_database(str(db_file), str(out_file), cutoff=5.0)
    data_dict = torch.load(str(out_file), weights_only=False)
    assert len(data_dict['graphs']) == 1

    vec = data_dict['graphs'][0].precision_settings.squeeze(0)
    assert vec.shape == (13,), f"Expected 13-dim, got {tuple(vec.shape)}"
    # bp=11 has its own slot — this is the WHOLE point of the fix
    assert vec[11].item() == 1.0, "bp=11 silent-zero bug: index 11 should be 1.0"
    # All other one-hot slots zero
    assert vec[:11].sum().item() == 0.0
    # k_point_density at index 12
    assert vec[12].item() == pytest.approx(8.0)
