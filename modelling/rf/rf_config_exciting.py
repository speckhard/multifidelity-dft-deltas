"""RF configuration for the exciting delta-learning dataset.

Mirrors `rf_config.py` (FHI-aims) but with:

  * Exciting-specific target columns (`delta_total_energy_per_atom`,
    10-term energy decomposition, three bandgap flavors).
  * NO geometry target groups: the exciting sweep relaxes atomic
    positions only, not cell vectors, so every `delta_relaxed_{a,b,c}_len`
    and `delta_relaxed_{alpha,beta,gamma}_angle` is identically 0 (and
    `delta_relaxed_volume_per_atom` too). Fitting a regressor to constant
    zeros is uninformative.
  * An expanded `COLS_TO_DROP_EXPLICIT` that covers exciting's raw-value
    leakage paths (per-atom energies in Ha, fermi/DOS, raw bandgaps, etc.).

Consumers: `rf_trainer.py` with `--config_module=rf_config_exciting`
(resolved via importlib — see `rf_trainer.resolve_config`).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    make_scorer,
)


# ---------------------------------------------------------------------------
# 1. Feature cleaning (leakage prevention)
# ---------------------------------------------------------------------------

# Columns dropped from X before training. The data loader ALSO drops every
# `delta_*` column (except `delta_monomer_*`), so we only need to list the
# raw / intermediate fields that would otherwise leak target information.
COLS_TO_DROP_EXPLICIT = [
    # --- Identifiers / metadata ---
    "path", "id", "uid",
    "ICSD_number", "compound_name", "chem_formula",
    "status", "geometry_status", "parse_warnings", "bandgap_source",

    # --- Raw INFO.OUT energy scalars (Hartree) ---
    # Each is computed at the current APW, so (current − reference) would
    # exactly reconstruct the delta target. Hard leakage.
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
    # Exciting also emits these correlated energy variants we don't delta-
    # learn but that leak the decomposition sums.
    "coulomb_energy", "effective_potential_energy",
    "coulomb_potential_energy", "sum_of_eigenvalues",

    # --- Per-atom raw energies ---
    # Exact atom-normalized siblings of the above; same leakage.
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

    # --- Band-gap raw values (leak the delta_band_gap_* targets) ---
    "band_gap_scf_eV",
    "band_gap_indirect_bands_eV",
    "band_gap_direct_bands_eV",
    "band_gap_bands_eV",       # back-compat alias for indirect
    "is_metal_scf", "is_metal_bands", "is_direct_bands",

    # --- Fermi / DOS (leak their delta_* versions) ---
    "fermi_energy", "fermi_energy_Ha", "dos_at_fermi",

    # --- Geometry scalars ---
    # Cells are fixed, so these are ICSD-identifying constants. Drop to
    # avoid the model learning an ICSD-specific shortcut.
    "relaxed_volume", "unit_cell_volume_Bohr3",
    "relaxed_a_len", "relaxed_b_len", "relaxed_c_len",
    "relaxed_alpha_angle", "relaxed_beta_angle", "relaxed_gamma_angle",

    # --- Array-valued fields ---
    "relaxed_atom_positions", "relaxed_lattice", "atomic_numbers",
]


# ---------------------------------------------------------------------------
# 2. Target definitions (all `delta_*` produced by create_delta_dataset.py)
# ---------------------------------------------------------------------------

_ENERGY_DECOMPOSITION_TARGETS = [
    "delta_total_energy_per_atom",
    "delta_kinetic_energy_per_atom",
    "delta_exchange_energy_per_atom",
    "delta_correlation_energy_per_atom",
    "delta_hartree_energy_per_atom",
    "delta_electron_nuclear_energy_per_atom",
    "delta_nuclear_nuclear_energy_per_atom",
    "delta_xc_potential_energy_per_atom",
    "delta_madelung_energy_per_atom",
    "delta_core_electron_kinetic_energy_per_atom",
]

_BANDGAP_SCF = ["delta_band_gap_scf_eV"]
_BANDGAP_INDIRECT = ["delta_band_gap_indirect_bands_eV"]
_BANDGAP_DIRECT = ["delta_band_gap_direct_bands_eV"]


TARGET_GROUPS = {
    # Single-target baselines.
    "energy": ["delta_total_energy_per_atom"],
    "bandgap_scf": _BANDGAP_SCF,
    "bandgap_indirect_bands": _BANDGAP_INDIRECT,
    "bandgap_direct_bands": _BANDGAP_DIRECT,
    # Multi-target.
    "energy_decomposition": list(_ENERGY_DECOMPOSITION_TARGETS),
    # Union — every other target group's cols at least once.
    "all_scalar": list(dict.fromkeys(
        _ENERGY_DECOMPOSITION_TARGETS
        + _BANDGAP_SCF + _BANDGAP_INDIRECT + _BANDGAP_DIRECT
    )),
}


# ---------------------------------------------------------------------------
# 3. Custom metrics — identical to rf_config.py (aims)
# ---------------------------------------------------------------------------

def calculate_smape(y_true, y_pred, epsilon=1e-7):
    """sMAPE (symmetric MAPE). Matches GNN + aims RF implementation."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    numerator = np.abs(y_pred - y_true)
    denominator = np.abs(y_pred) + np.abs(y_true) + epsilon
    return float(np.mean(100.0 * 2.0 * numerator / denominator))


def calculate_mag_acc(y_true, y_pred):
    """Magnitude accuracy: same log10 bucket as ground truth."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    bins = np.array([1e-3, 1e-2, 1e-1, 1.0, 10.0])
    p_bins = np.digitize(np.abs(y_pred), bins)
    t_bins = np.digitize(np.abs(y_true), bins)
    return float(np.mean(p_bins == t_bins))


def _smape_loss(y, y_pred):
    return calculate_smape(y, y_pred)


def _rmsle_abs(y, y_pred):
    return np.sqrt(mean_squared_error(
        np.log1p(np.abs(y)), np.log1p(np.abs(y_pred))
    ))


def _asinh_mae(y, y_pred):
    return mean_absolute_error(np.arcsinh(y), np.arcsinh(y_pred))


# ---------------------------------------------------------------------------
# 4. Scorer registry (greater_is_better=False = MINIMIZE)
# ---------------------------------------------------------------------------

SCORERS = {
    "smape": make_scorer(_smape_loss, greater_is_better=False),
    "mae": "neg_mean_absolute_error",
    "mse": "neg_mean_squared_error",
    "rmsle_abs": make_scorer(_rmsle_abs, greater_is_better=False),
    "asinh": make_scorer(_asinh_mae, greater_is_better=False),
}
