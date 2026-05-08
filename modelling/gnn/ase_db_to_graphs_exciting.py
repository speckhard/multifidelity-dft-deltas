"""Build PyG graph tensors from an exciting delta-learning ASE DB.

This is a fork of `ase_db_to_graphs.py` (the aims pipeline) with the
exciting-specific schema baked in:

  * Cheap DFT scalars: 10 per-atom energy components from the INFO.OUT
    decomposition (not the 12 aims scalars).
  * Cheap geometry scalars: 7 (relaxed volume per atom + 3 lattice lengths
    + 3 lattice angles).
  * Precision settings: 8-dim one-hot over APW precision {0.3, 0.4, ...,
    1.0} concatenated with the scalar k-point density (total 9 dims).
  * Targets: delta total energy per atom, delta SCF band gap, delta
    relaxed positions, delta 6-vector lattice parameters.

Scalers are fit on the training split only — no leakage. Volume-delta
outliers above a configurable threshold are dropped (same rationale as
aims: unphysical relaxations from broken DFT runs).

Input DB must have been produced by `create_delta_dataset.py
--reference_selector=exciting` so the `delta_*` keys exist.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
from absl import app, flags, logging
from ase.db import connect
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from parsing.create_delta_dataset import parse_array_string  # noqa: E402


# --- ABSL Flags ---
FLAGS = flags.FLAGS
flags.DEFINE_string(
    'db_path', 'exciting_delta_learning.db',
    'Path to the exciting delta ASE database (from create_delta_dataset).'
)
flags.DEFINE_string(
    'output_path', 'processed_exciting_delta_graphs.pt',
    'Path to the output PyTorch .pt file.'
)
flags.DEFINE_float('cutoff', 5.0, 'Cutoff radius (Å) for graph creation.')
flags.DEFINE_float(
    'volume_threshold', 10.0,
    'Drop rows whose |delta_relaxed_volume_per_atom| exceeds this (Å³/atom).'
)


# --- Constants ---

# APW precisions used on the HU-Berlin Oasis sweep: [0.3, 0.4, ..., 1.0].
# One-hot index = round((APWprecision - 0.3) * 10). Out-of-range values
# leave the one-hot vector zero (a sentinel for "unexpected precision").
APW_PRECISIONS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
NUM_APW_PRECISIONS = len(APW_PRECISIONS)

# 1. Prediction targets. Scalar targets are stored as 0-d tensors; position
#    delta is a (N_atoms, 3) tensor; lattice-param delta is a (1, 6) tensor.
CORE_TARGETS = [
    'delta_total_energy_per_atom',
    'delta_band_gap_scf_eV',
    'delta_relaxed_atom_positions',
]

# 2. Additional scalar targets (base names — the DB column is `delta_<name>`).
ADDITIONAL_SCALAR_BASES = [
    'relaxed_volume_per_atom',
    'relaxed_a_len', 'relaxed_b_len', 'relaxed_c_len',
    'relaxed_alpha_angle', 'relaxed_beta_angle', 'relaxed_gamma_angle',
]

TARGET_KEYS = CORE_TARGETS + [f'delta_{k}' for k in ADDITIONAL_SCALAR_BASES]

# 3. Cheap DFT scalars (10 per-atom energies from INFO.OUT).
ALL_DFT_SCALARS = [
    'total_energy_per_atom',
    'kinetic_energy_per_atom',
    'exchange_energy_per_atom',
    'correlation_energy_per_atom',
    'hartree_energy_per_atom',
    'electron_nuclear_energy_per_atom',
    'nuclear_nuclear_energy_per_atom',
    'xc_potential_energy_per_atom',
    'madelung_energy_per_atom',
    'core_electron_kinetic_energy_per_atom',
]

# 4. Cheap geometry scalars (7 features — same shape as aims).
CHEAP_GEOMETRY_KEYS = [
    'relaxed_volume_per_atom',
    'relaxed_a_len', 'relaxed_b_len', 'relaxed_c_len',
    'relaxed_alpha_angle', 'relaxed_beta_angle', 'relaxed_gamma_angle',
]


def _apw_precision_one_hot(apw_precision):
    """Return an 8-dim float tensor one-hot over APW_PRECISIONS.

    Off-grid values produce an all-zero vector (sentinel). This keeps the
    tensor shape consistent and downstream code can inspect `.sum() == 0`
    to flag it if needed.
    """
    ohe = torch.zeros(NUM_APW_PRECISIONS, dtype=torch.float)
    if apw_precision is None:
        return ohe
    for i, p in enumerate(APW_PRECISIONS):
        if abs(float(apw_precision) - p) < 1e-4:
            ohe[i] = 1.0
            return ohe
    return ohe


def _spacegroup_from_atoms(atoms):
    """Best-effort spacegroup number; returns 0 on any failure."""
    try:
        import spglib
        cell = (atoms.get_cell(),
                atoms.get_scaled_positions(),
                atoms.get_atomic_numbers())
        ds = spglib.get_symmetry_dataset(cell, symprec=1e-5)
        if ds:
            return int(ds['number'])
    except Exception:
        pass
    return 0


def process_database(db_path, output_path, cutoff,
                     volume_threshold=10.0):
    """Read exciting delta DB, build PyG graphs, fit scalers on train, save."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database {db_path} not found.")

    print(f"--- Processing {db_path} (exciting schema) ---")
    print(f"--- Cheap DFT scalars: {len(ALL_DFT_SCALARS)} keys ---")
    print(f"--- Geometry inputs: {len(CHEAP_GEOMETRY_KEYS)} keys ---")
    print(f"--- APW precisions grid: {APW_PRECISIONS} ---")

    raw_data_list = []

    # Storage for scaler fitting (train only — avoid leakage).
    train_dft_inputs = []
    train_geo_inputs = []

    # Diagnostics
    missing_key_counts = {}
    split_counts = {'train': 0, 'val': 0, 'test': 0, 'unknown': 0}
    skipped_outliers = 0
    skipped_missing_precision = 0

    all_required = TARGET_KEYS + ALL_DFT_SCALARS + CHEAP_GEOMETRY_KEYS

    # --- Extraction loop ---
    with connect(db_path) as db:
        for row in tqdm(db.select(), desc="Extracting graphs", disable=None):

            # Must know which precision this row is at.
            apw_prec = getattr(row, 'APWprecision_input', None)
            if apw_prec is None:
                apw_prec = getattr(row, 'APW_precision_path', None)
            if apw_prec is None:
                skipped_missing_precision += 1
                continue

            k_dens_val = getattr(row, 'k_point_density', None)
            if k_dens_val is None:
                k_dens_val = getattr(row, 'k_point_density_path', None)
            if k_dens_val is None:
                skipped_missing_precision += 1
                continue

            # Missing-key check.
            missing_keys = [k for k in all_required if not hasattr(row, k)]
            if missing_keys:
                for k in missing_keys:
                    missing_key_counts[k] = missing_key_counts.get(k, 0) + 1
                continue
            if any(getattr(row, k) is None for k in all_required):
                continue

            # Split label (default to train if missing).
            split_type = getattr(row, 'split', 'train')

            # Graph structure.
            atoms = row.toatoms()
            positions = torch.tensor(atoms.get_positions(), dtype=torch.float)
            Z = torch.tensor(atoms.get_atomic_numbers(),
                             dtype=torch.long).unsqueeze(1)

            # Targets & volume outlier check.
            targets = {}
            lattice_deltas = []
            is_outlier = False

            for key in TARGET_KEYS:
                raw_val = getattr(row, key)

                if key == 'delta_relaxed_volume_per_atom':
                    if abs(float(raw_val)) >= volume_threshold:
                        is_outlier = True
                        break

                if key == 'delta_relaxed_atom_positions':
                    arr = parse_array_string(raw_val)
                    if arr is None:
                        is_outlier = True
                        break
                    targets[key] = torch.tensor(
                        np.asarray(arr).reshape(-1, 3), dtype=torch.float)
                elif 'delta_relaxed_' in key and (
                        'len' in key or 'angle' in key):
                    lattice_deltas.append(float(raw_val))
                else:
                    targets[key] = torch.tensor(float(raw_val),
                                                dtype=torch.float)

            if is_outlier:
                skipped_outliers += 1
                continue

            targets['delta_lattice_params'] = torch.tensor(
                lattice_deltas, dtype=torch.float).unsqueeze(0)

            # Spacegroup.
            spacegroup_tensor = torch.tensor(
                [_spacegroup_from_atoms(atoms)], dtype=torch.long)

            # Cheap DFT scalars (raw — scaled later).
            dft_scalars_raw = [float(getattr(row, k)) for k in ALL_DFT_SCALARS]

            # Global context: 8-way APW one-hot + k-density.
            apw_ohe = _apw_precision_one_hot(apw_prec)
            precision_settings = torch.cat([
                apw_ohe,
                torch.tensor([float(k_dens_val)], dtype=torch.float),
            ])

            # Cheap geometry inputs (raw — scaled later).
            geo_features_raw = [float(getattr(row, k))
                                for k in CHEAP_GEOMETRY_KEYS]

            # Edge index.
            edge_index = radius_graph(positions, r=cutoff, loop=False)

            data = Data(
                x=positions,
                z=Z,
                spacegroup=spacegroup_tensor,
                _raw_dft_scalars=torch.tensor(
                    dft_scalars_raw, dtype=torch.float).unsqueeze(0),
                _raw_geo_features=torch.tensor(
                    geo_features_raw, dtype=torch.float).unsqueeze(0),
                precision_settings=precision_settings.unsqueeze(0),
                edge_index=edge_index,
                split=split_type,
                **targets,
            )

            raw_data_list.append(data)

            if split_type in split_counts:
                split_counts[split_type] += 1
            else:
                split_counts['unknown'] += 1

            if split_type == 'train':
                train_dft_inputs.append(dft_scalars_raw)
                train_geo_inputs.append(geo_features_raw)

    # --- Reporting ---
    print("\n" + "=" * 40)
    print("PROCESSING SUMMARY (exciting)")
    print("=" * 40)
    print(f"Total graphs extracted: {len(raw_data_list)}")
    print(f"Skipped volume outliers (>{volume_threshold} Å³/atom): "
          f"{skipped_outliers}")
    print(f"Skipped missing precision/k_density: {skipped_missing_precision}")
    print(f"Split distribution: {split_counts}")

    if split_counts['val'] == 0:
        logging.warning(
            "No 'val' split found — did you regenerate the delta DB with "
            "create_delta_dataset.py --reference_selector=exciting ?")

    if missing_key_counts:
        print("\nMissing keys:")
        for k, v in sorted(missing_key_counts.items(),
                           key=lambda it: it[1], reverse=True):
            print(f"  {k}: {v}")
    print("=" * 40 + "\n")

    # --- Fit scalers (train only) ---
    print("Fitting scalers on TRAIN split only...")
    dft_scaler = StandardScaler()
    geo_scaler = StandardScaler()
    if train_dft_inputs:
        dft_scaler.fit(np.asarray(train_dft_inputs))
        geo_scaler.fit(np.asarray(train_geo_inputs))
    else:
        print("WARNING: no training data found; scalers not fit.")

    # --- Apply scaling ---
    print("Applying transformations...")
    processed_list = []
    for data in tqdm(raw_data_list, desc="Scaling"):
        num_nodes = data.x.shape[0]

        if train_dft_inputs:
            dft_scaled = dft_scaler.transform(data._raw_dft_scalars.numpy())
            geo_scaled = geo_scaler.transform(data._raw_geo_features.numpy())
        else:
            dft_scaled = data._raw_dft_scalars.numpy()
            geo_scaled = data._raw_geo_features.numpy()

        data.cheap_dft_scalars = torch.tensor(
            dft_scaled, dtype=torch.float).repeat(num_nodes, 1)
        data.cheap_geometry_scalars = torch.tensor(
            geo_scaled, dtype=torch.float)

        del data._raw_dft_scalars
        del data._raw_geo_features

        processed_list.append(data)

    save_dict = {
        'graphs': processed_list,
        'dft_scaler': dft_scaler,
        'geo_scaler': geo_scaler,
        'schema': 'exciting',
        'apw_precisions': list(APW_PRECISIONS),
    }

    torch.save(save_dict, output_path)
    print(f"\nSaved {len(processed_list)} graphs to {output_path}")
    return save_dict


def main(argv):
    if len(argv) > 1:
        logging.warning(f"Unknown arguments passed: {argv[1:]}")
    process_database(
        FLAGS.db_path, FLAGS.output_path, FLAGS.cutoff,
        volume_threshold=FLAGS.volume_threshold,
    )


if __name__ == "__main__":
    app.run(main)
