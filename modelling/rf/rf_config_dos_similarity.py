"""RF configuration for predicting DOS Tanimoto similarity.

Target: `dos_tanimoto` — the Tanimoto coefficient between the cheap calc's
DOS fingerprint and the APW=1.0 reference DOS fingerprint, computed by
`parsing/compute_dos_similarity.py`.  Values live in [0, 1].

Data file: `exciting_delta_learning_tanimoto.csv` (separate from the
energy-prediction CSV to prevent accidental use of Tc as a feature).

Consumers: `rf_trainer.py` with `--config_module=rf_config_dos_similarity`
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    make_scorer,
)


# ---------------------------------------------------------------------------
# 1. Feature cleaning (leakage prevention)
# ---------------------------------------------------------------------------

COLS_TO_DROP_EXPLICIT = [
    # --- The target itself (not auto-dropped: doesn't start with delta_) ---
    "dos_tanimoto",

    # --- Identifiers / metadata ---
    "path", "id", "uid",
    "ICSD_number", "compound_name", "chem_formula",
    "status", "geometry_status", "parse_warnings", "bandgap_source",

    # --- Raw INFO.OUT energy scalars (Hartree) ---
    "total_energy",
    "kinetic_energy",
    "exchange_energy",
    "correlation_energy",
    "hartree_energy",
    "electron_nuclear_energy",
    "nuclear_nuclear_energy",
    "xc_potential_energy",
    "madelung_energy",
    "core_electron_kinetic_energy",
    "coulomb_energy", "effective_potential_energy",
    "coulomb_potential_energy", "sum_of_eigenvalues",

    # --- Per-atom raw energies ---
    "total_energy_per_atom",
    "kinetic_energy_per_atom",
    "exchange_energy_per_atom",
    "correlation_energy_per_atom",
    "hartree_energy_per_atom",
    "electron_nuclear_energy_per_atom",
    "nuclear_nuclear_energy_per_atom",
    "xc_potential_energy_per_atom",
    "madelung_energy_per_atom",
    "core_electron_kinetic_energy_per_atom",
    "relaxed_volume_per_atom",

    # --- Band-gap raw values ---
    "band_gap_scf_eV",
    "band_gap_indirect_bands_eV",
    "band_gap_direct_bands_eV",
    "band_gap_bands_eV",
    "is_metal_scf", "is_metal_bands", "is_direct_bands",

    # --- Fermi / DOS ---
    "fermi_energy", "fermi_energy_Ha", "dos_at_fermi",

    # --- Geometry scalars ---
    "relaxed_volume", "unit_cell_volume_Bohr3",
    "relaxed_a_len", "relaxed_b_len", "relaxed_c_len",
    "relaxed_alpha_angle", "relaxed_beta_angle", "relaxed_gamma_angle",

    # --- Array-valued fields ---
    "relaxed_atom_positions", "relaxed_lattice", "atomic_numbers",
]


# ---------------------------------------------------------------------------
# 2. Target definitions
# ---------------------------------------------------------------------------

TARGET_GROUPS = {
    "dos_tanimoto": ["dos_tanimoto"],
}


# ---------------------------------------------------------------------------
# 3. Custom metrics
# ---------------------------------------------------------------------------

def calculate_smape(y_true, y_pred, epsilon=1e-7):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    numerator = np.abs(y_pred - y_true)
    denominator = np.abs(y_pred) + np.abs(y_true) + epsilon
    return float(np.mean(100.0 * 2.0 * numerator / denominator))


def calculate_mag_acc(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    bins = np.array([1e-3, 1e-2, 1e-1, 1.0, 10.0])
    p_bins = np.digitize(np.abs(y_pred), bins)
    t_bins = np.digitize(np.abs(y_true), bins)
    return float(np.mean(p_bins == t_bins))


def _smape_loss(y, y_pred):
    return calculate_smape(y, y_pred)


def _r2_score(y, y_pred):
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


# ---------------------------------------------------------------------------
# 4. Scorer registry
# ---------------------------------------------------------------------------

SCORERS = {
    "smape": make_scorer(_smape_loss, greater_is_better=False),
    "mae": "neg_mean_absolute_error",
    "r2": make_scorer(_r2_score, greater_is_better=True),
}
