import pytest
import ase.db
from ase import Atoms
import numpy as np
import os
import sys

# Ensure we can import from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parsing.create_delta_dataset import create_delta_dataset


def test_create_delta_dataset_logic(tmp_path):
    """
    Tests the delta dataset creation logic with 3 specific datapoints:
    1. ICSD=10, Prec=2, K=4  (Target: Should be processed, delta calculated)
    2. ICSD=10, Prec=11, K=8 (Reference: Should be USED but NOT SAVED)
    3. ICSD=6,  Prec=6, K=4  (Orphan: No reference with K=8, Should be discarded)
    """
    
    # Define temporary paths
    input_db_path = tmp_path / "test_input.json"
    output_db_path = tmp_path / "test_output.db"
    output_csv_path = tmp_path / "output.csv"
    
    # --- 1. Create Input Data ---
    
    def format_arr(arr):
        return np.array2string(
            arr, separator=',',
            formatter={'float_kind': lambda x: "%.16f" % x}
        ).replace('\n', '')

    # Dummy Atoms object
    atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]])
    
    with ase.db.connect(str(input_db_path)) as db:
        
        # Datapoint 1: ICSD=10, Precision=2, K-points=4 (Target)
        db.write(
            atoms,
            ICSD_number=10,
            binary_precision=2,
            k_point_density=4,   # <--- ADDED THIS
            total_energy=-100.0,
            relaxed_atom_positions=format_arr(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.75]]))
        )
        
        # Datapoint 2: ICSD=10, Precision=11, K-points=8 (Reference)
        # MUST have k_point_density=8 to be selected as reference
        db.write(
            atoms,
            ICSD_number=10,
            binary_precision=11,
            k_point_density=8,   # <--- ADDED THIS (Vital for new logic)
            total_energy=-105.0,
            relaxed_atom_positions=format_arr(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]))
        )
        
        # Datapoint 3: ICSD=6, Precision=6 (Orphan)
        # Even if this had k=8, it's precision 6, so it's not a reference.
        # Since no Prec 11 / K 8 exists for ICSD 6, this is discarded.
        db.write(
            atoms,
            ICSD_number=6,
            binary_precision=6,
            k_point_density=4,   # <--- ADDED THIS
            total_energy=-50.0
        )
        
    # --- 2. Run the Function ---
    create_delta_dataset(
        str(input_db_path), str(output_db_path), str(output_csv_path),
        train_ratio=0.8, seed=5)
    
    # --- 3. Verify Output ---
    
    assert os.path.exists(output_db_path)
    
    with ase.db.connect(str(output_db_path)) as db:
        rows = list(db.select())
        
        # Check Count: Should contain EXACTLY 1 row (ICSD 10: Prec 2).
        assert len(rows) == 1
        
        # Fetch rows
        row_prec2 = db.get(ICSD_number=10, binary_precision=2)
        
        # Verify Reference Row is missing (it shouldn't be saved to output)
        with pytest.raises(KeyError):
            db.get(ICSD_number=10, binary_precision=11)
            
        # Verify Orphan Row is missing
        orphans = list(db.select(ICSD_number=6))
        assert len(orphans) == 0
        
        # Check Delta Calculation
        # Target (-100/2 = -50) - Ref (-105/2 = -52.5) = +2.5
        assert hasattr(row_prec2, 'delta_total_energy_per_atom')
        assert row_prec2.delta_total_energy_per_atom == pytest.approx(2.5)
        
        # Check that we did NOT add the delta_k feature (per your request)
        assert not hasattr(row_prec2, 'delta_k_point_density')

    # Verify CSV
    assert os.path.exists(output_csv_path)