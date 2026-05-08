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
# Adjust these paths to match your folder structure
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from modelling.gnn.delta_egnn_model import DeltaGNN
from modelling.gnn.ase_db_to_graphs import process_database

# --- Constants from your Generator Script ---
# We need these to populate the Mock DB so the generator doesn't skip rows
# Add all keys that MUST be present on the DB row
REQUIRED_DB_KEYS = [
    # Basic Inputs
    'total_energy_per_atom', 'homo_lumo_gap',
    # Deep DFT Inputs
    'sum_eigenvalues_per_atom', 'xc_energy_correction_per_atom', 
    'xc_potential_correction_per_atom', 'free_atom_electrostatic_energy_per_atom', 
    'hartree_energy_correction_per_atom', 'entropy_correction_per_atom', 
    'total_energy_T0_per_atom', 'kinetic_energy_per_atom', 
    'electrostatic_energy_per_atom', 'multipole_correction_per_atom',
    # Geometry Inputs
    'final_volume_per_atom', 'relaxed_a_len', 'relaxed_b_len', 'relaxed_c_len',
    'relaxed_alpha_angle', 'relaxed_beta_angle', 'relaxed_gamma_angle',
    
    # CORE TARGETS (The generator checks for ALL of these)
    'delta_total_energy_per_atom', 
    'delta_homo_lumo_gap',
    # NOTE: delta_relaxed_atom_positions needs a specific string input, handled below
    
    # DELTA TARGET BASES
    'delta_aims_free_energy_per_atom',
    'delta_vbm', 'delta_cbm', 'delta_chemical_potential', 
    'delta_final_volume_per_atom',
    'delta_relaxed_a_len', 'delta_relaxed_b_len', 'delta_relaxed_c_len',
    'delta_relaxed_alpha_angle', 'delta_relaxed_beta_angle', 'delta_relaxed_gamma_angle',
]


# --- Test 1: Unit Test with Manual Data ---
def test_delta_gnn_forward_shape():
    """
    Verifies that the model accepts the separated input format:
    x: Positions
    z: Atomic Numbers
    cheap_dft_scalars: Node-level physics features
    global_context: Graph-level settings
    cheap_geometry_scalars: Graph-level geometry inputs
    """
    # 1. Setup Model
    # Dimensions based on your generator:
    # 2 (Basic) + 10 (Deep) = 12 DFT inputs
    # 11 (Precision) + 1 (K-point) = 12 Context inputs
    model = DeltaGNN(
        num_layers=2,
        hidden_features=16,
        num_cheap_dft_inputs=12,    
        num_precision_settings=12,  
        num_geo_inputs=7,
        max_z=10
    )

    # 2. Create Dummy Data (2 Graphs in one Batch)
    # Graph 1: 2 atoms. Graph 2: 1 atom. Total Nodes = 3.
    
    # Node Level Inputs [N=3]
    x_input = torch.randn(3, 3) 
    z_input = torch.tensor([1, 6, 8], dtype=torch.long)
    # 12 DFT scalars per node
    dft_scalars = torch.randn(3, 12) 
    
    # Graph Level Inputs [B=2]
    # 12 Context features (Precision OHE + K-density) per graph
    global_precision_settings = torch.randn(2, 12)
    # 7 Geometry scalars per graph
    geo_scalars = torch.randn(2, 7)
    
    # Connectivity
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    batch = torch.tensor([0, 0, 1], dtype=torch.long) # Atoms 0,1 -> Graph 0. Atom 2 -> Graph 1.

    data = Data(
        x=x_input, 
        z=z_input, 
        edge_index=edge_index, 
        batch=batch,
        cheap_dft_scalars=dft_scalars,
        precision_settings=global_precision_settings,
        cheap_geometry_scalars=geo_scalars
    )

    # 3. Forward Pass
    # Multi-target architecture returns (mean, logvar) tuples for each head.
    (pos_mean, pos_logvar), (e_mean, e_logvar), \
        (gap_mean, gap_logvar), (geo_mean, geo_logvar) = model(data)

    # 4. Assertions
    # Positions live on each node [N=3, 3]
    assert pos_mean.shape == (3, 3)
    assert pos_logvar.shape == (3, 1)

    # Scalars/Globals should match number of graphs [B=2]
    assert e_mean.shape == (2,)
    assert e_logvar.shape == (2,)
    assert gap_mean.shape == (2,)
    assert gap_logvar.shape == (2,)

    # Geometry delta is a vector of 7 [B=2, 7]
    assert geo_mean.shape == (2, 7)
    assert geo_logvar.shape == (2, 7)


# --- Test 2: Full Integration (DB -> Gen -> Model) ---

def get_dummy_row_data():
    """Helper to populate all required keys with dummy floats so the generator accepts the row."""
    data = {k: 0.5 for k in REQUIRED_DB_KEYS}
    # Specific overrides
    data['delta_total_energy_per_atom'] = -0.1
    data['delta_homo_lumo_gap'] = 0.05
    data['delta_relaxed_atom_positions'] = "None" # Generator handles None parsing or valid string
    # Add k-point density (required)
    data['k_point_density'] = 2.0

    return data

@pytest.fixture
def mock_db_environment(tmp_path):
    """Creates a temporary ASE DB with enough columns to pass the generator's validation."""
    db_file = tmp_path / "test_delta.db"
    out_file = tmp_path / "test_output.pt"
  
    dummy_data = get_dummy_row_data()

    with connect(db_file) as db:
        # 1. H2 (Graph 1)
        # Needs valid array string for positions delta
        dummy_data['delta_relaxed_atom_positions'] = "[[0.01, 0.0, 0.0], [-0.01, 0.0, 0.0]]"

        db.write(Atoms('H2', positions=[[0,0,0],[0,0,0.74]]), 
                 split='train', 
                 **dummy_data)

        # 2. H2O (Graph 2)
        dummy_data['delta_relaxed_atom_positions'] = "[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]"
        
        db.write(Atoms('H2O', positions=[[0,0,0],[1,0,0],[0,1,0]]), 
                 split='train', 
                 **dummy_data)

    return str(db_file), str(out_file)

def test_integration_with_generated_db(mock_db_environment):
    """
    Runs the full pipeline:
    1. Generator reads DB, scales data, saves .pt
    2. Model consumes the processed .pt file
    """
    db_path, out_path = mock_db_environment

    # 1. Run Generator
    # This will print "Processing..." and "Fitting scalers..."
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
    # We must know the input dimensions. 
    # Based on 'process_database', we have 12 DFT scalars (2 basic + 10 deep)
    
    model = DeltaGNN(
        num_layers=2,
        hidden_features=32,
        num_cheap_dft_inputs=12,
        num_precision_settings=13, # 12 OHE + 1 K-dens (after bp=11 fix)
        num_geo_inputs=7,
        max_z=100
    )

    # 5. Run Forward Pass
    # Multi-target architecture returns (mean, logvar) tuples for each head.
    (pos_mean, pos_logvar), (e_mean, e_logvar), \
        (gap_mean, gap_logvar), (geo_mean, geo_logvar) = model(batch_data)

    # 6. Verify Shapes
    # Total atoms = 2 (H2) + 3 (H2O) = 5
    assert pos_mean.shape == (5, 3)
    assert pos_logvar.shape == (5, 1)

    # Total graphs = 2
    assert e_mean.shape == (2,)
    assert gap_mean.shape == (2,)
    assert geo_mean.shape == (2, 7)
