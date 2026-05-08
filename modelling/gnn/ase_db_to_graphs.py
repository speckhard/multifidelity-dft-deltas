import torch
import numpy as np
import os
import ast
import sys
import spglib
from absl import app, flags, logging
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph
from ase.db import connect
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


# --- ABSL Flags ---
FLAGS = flags.FLAGS
flags.DEFINE_string('db_path', 'delta_learning.db', 'Path to the input ASE database file.')
flags.DEFINE_string('output_path', 'processed_delta_graphs.pt', 'Path to the output PyTorch .pt file.')
flags.DEFINE_float('cutoff', 5.0, 'Cutoff radius for graph creation.')

# --- Constants ---

# 1. Existing Core Targets (Prediction Goals)
CORE_TARGETS = [
    'delta_total_energy_per_atom', 
    'delta_homo_lumo_gap',
    'delta_relaxed_atom_positions'
]

# 2. New Scalar Targets (Base names)
ADDITIONAL_SCALAR_BASES = [
    'aims_free_energy_per_atom',
    'vbm', 'cbm', 'chemical_potential', 
    'final_volume_per_atom',
    'relaxed_a_len', 'relaxed_b_len', 'relaxed_c_len',
    'relaxed_alpha_angle', 'relaxed_beta_angle', 'relaxed_gamma_angle',
]

TARGET_KEYS = CORE_TARGETS + [f'delta_{k}' for k in ADDITIONAL_SCALAR_BASES]

# --- INPUT FEATURES ---

# A. Basic Physics Inputs
BASIC_INPUT_KEYS = [
    'total_energy_per_atom', 
    'homo_lumo_gap'
]

# B. Deep DFT Inputs
DEEP_DFT_KEYS = [
    'sum_eigenvalues_per_atom', 
    'xc_energy_correction_per_atom', 
    'xc_potential_correction_per_atom',
    'free_atom_electrostatic_energy_per_atom', 
    'hartree_energy_correction_per_atom', 
    'entropy_correction_per_atom', 
    'total_energy_T0_per_atom', 
    'kinetic_energy_per_atom', 
    'electrostatic_energy_per_atom', 
    'multipole_correction_per_atom'
]

ALL_DFT_SCALARS = BASIC_INPUT_KEYS + DEEP_DFT_KEYS

# C. Cheap Geometry Inputs
CHEAP_GEOMETRY_KEYS = [
    'final_volume_per_atom',
    'relaxed_a_len', 'relaxed_b_len', 'relaxed_c_len',
    'relaxed_alpha_angle', 'relaxed_beta_angle', 'relaxed_gamma_angle'
]


def parse_array_string(array_str):
    """Helper to parse stringified arrays from ASE DB"""
    if array_str is None: return None
    try:
        return np.array(ast.literal_eval(array_str)).reshape(-1, 3)
    except (ValueError, SyntaxError, AttributeError):
        return None


def process_database(db_path, output_path, cutoff):
    """
    Core logic: Reads DB, processes graphs, fits scalers (on train), saves .pt file.
    Includes filtering for unphysical volume changes.
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database {db_path} not found.")

    print(f"--- Processing {db_path} ---")
    print(f"--- Cheap DFT Scalars: {len(ALL_DFT_SCALARS)} keys ---")
    print(f"--- Geometry Inputs: {len(CHEAP_GEOMETRY_KEYS)} keys ---")
    
    # --- OUTLIER CONFIGURATION ---
    # Any volume change > 10 A^3/atom is considered unphysical/failed DFT
    VOLUME_THRESHOLD = 10.0 
    
    raw_data_list = []
    
    # Storage for Scaler Fitting (Train only)
    train_dft_inputs = []
    train_geo_inputs = []

    # Diagnostics
    missing_key_counts = {}
    split_counts = {'train': 0, 'val': 0, 'test': 0, 'unknown': 0}
    skipped_outliers = 0
    skipped_overconverged = 0  # <--- 1. Initialize Counter

    # 1. Extraction Loop
    with connect(db_path) as db:
        for row in tqdm(db.select(), desc="Extracting graphs", disable=None):
            
            # --- VALIDATION ---
            if not hasattr(row, 'k_point_density'):
                 continue

            # --- MISSING KEY CHECK ---
            all_required = TARGET_KEYS + ALL_DFT_SCALARS + CHEAP_GEOMETRY_KEYS
            missing_keys = [key for key in all_required if not hasattr(row, key)]

            if missing_keys:
                for k in missing_keys:
                    missing_key_counts[k] = missing_key_counts.get(k, 0) + 1
                continue

            has_none = False
            for k in all_required:
                if getattr(row, k) is None:
                    has_none = True; break
            if has_none: continue

            # --- NEW FILTER: Remove "Over-Converged" Points ---
            # Condition: delta_mean_basis_functions > 0 AND k_point_density == 8
            # These are calculation settings higher than the reference standard.
            if hasattr(row, 'delta_mean_basis_functions'):
                d_basis = float(getattr(row, 'delta_mean_basis_functions'))
                k_dens_val = float(getattr(row, 'k_point_density'))
                
                # Check condition (using epsilon for float comparison safety)
                if d_basis > 0 and abs(k_dens_val - 8.0) < 1e-5:
                    skipped_overconverged += 1
                    continue  # Skip this row


            # --- SPLIT HANDLING ---
            # Default to 'train' if missing, but we track it to be sure
            split_type = getattr(row, 'split', 'train')
            
            # --- 0. Graph Structure ---
            atoms = row.toatoms()
            positions = torch.tensor(atoms.get_positions(), dtype=torch.float)
            Z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long).unsqueeze(1)

            # --- 1. Targets (Deltas) & Outlier Check ---
            targets = {}
            lattice_deltas = []
            is_outlier = False
            
            for key in TARGET_KEYS:
                raw_val = getattr(row, key)
                
                # --- OUTLIER FILTER ---
                if key == 'delta_final_volume_per_atom':
                    if abs(float(raw_val)) >= VOLUME_THRESHOLD:
                        is_outlier = True
                        break # Stop parsing this row immediately
                # ---------------------
                
                if key == 'delta_relaxed_atom_positions':
                    targets[key] = torch.tensor(parse_array_string(raw_val), dtype=torch.float)
                elif 'delta_relaxed_' in key and ('len' in key or 'angle' in key):
                    lattice_deltas.append(float(raw_val))
                else:
                    targets[key] = torch.tensor(raw_val, dtype=torch.float)

            # Spacegroup precedence:
            #   1. DB-provided 'spacegroup' field (source of truth when present)
            #   2. spglib recompute from the atoms object (fallback)
            #   3. 0 = unknown
            sg = 0
            db_sg = getattr(row, 'spacegroup', None)
            if db_sg is not None:
                try:
                    sg = int(db_sg)
                except (TypeError, ValueError):
                    sg = 0

            if sg == 0:
                try:
                    import spglib
                    # spglib needs a tuple: (cell, positions, atomic_numbers)
                    cell = (atoms.get_cell(), atoms.get_scaled_positions(), atoms.get_atomic_numbers())
                    dataset = spglib.get_symmetry_dataset(cell, symprec=1e-5)
                    if dataset:
                        sg = dataset['number']
                except ImportError:
                    pass # spglib not installed, keep as 0

            # 1-230 is valid. 0 means unknown.
            spacegroup_tensor = torch.tensor([sg], dtype=torch.long)

            if is_outlier:
                skipped_outliers += 1
                continue # Skip this graph entirely

            targets['delta_lattice_params'] = torch.tensor(lattice_deltas, dtype=torch.float).unsqueeze(0)

            # --- 2. Cheap DFT Scalars (Raw) ---
            dft_scalars_raw = [getattr(row, k) for k in ALL_DFT_SCALARS]
            
            # --- 3. Global Context ---
            # bp=11 fix: 12-dim one-hot for bp ∈ {0..11} (was 11-dim, silently
            # mapping bp=11 (really_tight/tier2) to all-zeros). Total precision
            # vector is now 13-dim: [12-dim bp one-hot, k_point_density scalar].
            P = getattr(row, 'binary_precision', 0)
            P_ohe = torch.zeros(12, dtype=torch.float)
            if 0 <= P <= 11: P_ohe[P] = 1.0

            k_dens = float(getattr(row, 'k_point_density'))

            precision_settings = torch.cat([P_ohe, torch.tensor([k_dens], dtype=torch.float)])

            # --- 4. Cheap Geometry Inputs ---
            geo_features_raw = [getattr(row, k) for k in CHEAP_GEOMETRY_KEYS]

            # --- Edge Index ---
            edge_index = radius_graph(positions, r=cutoff, loop=False)

            data = Data(
                x=positions,
                z=Z,
                spacegroup=spacegroup_tensor,
                _raw_dft_scalars=torch.tensor(dft_scalars_raw, dtype=torch.float).unsqueeze(0),
                _raw_geo_features=torch.tensor(geo_features_raw, dtype=torch.float).unsqueeze(0),
                precision_settings=precision_settings.unsqueeze(0),    
                edge_index=edge_index,
                split=split_type, 
                **targets
            )

            raw_data_list.append(data) 
            
            # Track counts (only if we didn't skip it)
            if split_type not in split_counts:
                split_counts['unknown'] += 1
            else:
                split_counts[split_type] += 1

            # --- Collect Data for Scalers (Train Only) ---
            # CRITICAL: Only fit scalers on training data to prevent data leakage
            if split_type == 'train':
                train_dft_inputs.append(dft_scalars_raw)
                train_geo_inputs.append(geo_features_raw)


    # --- REPORTING ---
    print("\n" + "="*40)
    print("PROCESSING SUMMARY")
    print("="*40)
    print(f"Total Graphs Extracted: {len(raw_data_list)}")
    print(f"Skipped Outliers (> {VOLUME_THRESHOLD} A^3): {skipped_outliers}")
    print(f"Skipped Over-converged (Basis>0, K=8): {skipped_overconverged}")
    print(f"Split Distribution: {split_counts}")
    
    if split_counts['val'] == 0:
        logging.warning("No 'val' split found! Did you regenerate delta_learning.db with the new script?")
        
    if missing_key_counts:
        print("\nMissing Keys:")
        for k, v in sorted(missing_key_counts.items(), key=lambda item: item[1], reverse=True):
            print(f"{k}: {v}")
    print("="*40 + "\n")

    # 2. Fit Scalers
    print("Fitting scalers on TRAIN data only...")
    dft_scaler = StandardScaler()
    geo_scaler = StandardScaler()
    
    if len(train_dft_inputs) > 0:
        dft_scaler.fit(np.array(train_dft_inputs))
        geo_scaler.fit(np.array(train_geo_inputs))
    else:
        print("WARNING: No training data found. Scalers will be empty/invalid.")

    # 3. Apply Scaling & Construct Final Tensors
    print("Applying transformations...")
    processed_list = []

    for data in tqdm(raw_data_list, desc="Scaling"):
        num_nodes = data.x.shape[0]

        # A. Scale DFT Scalars
        dft_scaled = dft_scaler.transform(data._raw_dft_scalars.numpy()) 
        data.cheap_dft_scalars = torch.tensor(dft_scaled, dtype=torch.float).repeat(num_nodes, 1)

        # B. Scale Geometry Inputs
        geo_feats_scaled = geo_scaler.transform(data._raw_geo_features.numpy()) 
        data.cheap_geometry_scalars = torch.tensor(geo_feats_scaled, dtype=torch.float)

        # Cleanup
        del data._raw_dft_scalars
        del data._raw_geo_features

        processed_list.append(data)

    # 4. Save
    save_dict = {
        'graphs': processed_list,
        'dft_scaler': dft_scaler,
        'geo_scaler': geo_scaler
    }

    torch.save(save_dict, output_path)
    print(f"\nSaved {len(processed_list)} graphs to {output_path}")


def main(argv):
    if len(argv) > 1:
        logging.warning(f"Unknown arguments passed: {argv[1:]}")
        
    process_database(FLAGS.db_path, FLAGS.output_path, FLAGS.cutoff)


if __name__ == "__main__":
    app.run(main)
