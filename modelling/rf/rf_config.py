import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer

# --- 1. Feature Cleaning ---
COLS_TO_DROP_EXPLICIT = [
    'path', 'id', 'uid', 'compound_name', 'chem_formula', 'ICSD_number',
    'total_energy_per_atom', 'aims_free_energy_per_atom',
    'sum_eigenvalues_per_atom', 'xc_energy_correction_per_atom', 'xc_potential_correction_per_atom',
    'free_atom_electrostatic_energy_per_atom', 'hartree_energy_correction_per_atom',
    'entropy_correction_per_atom', 'total_energy_T0_per_atom', 'kinetic_energy_per_atom',
    'electrostatic_energy_per_atom', 'multipole_correction_per_atom',
    'electronic_free_energy_per_atom',
    'vbm', 'cbm', 'homo_lumo_gap', 'chemical_potential',
    'final_volume',
    'relaxed_a_len', 'relaxed_b_len', 'relaxed_c_len',
    'relaxed_alpha_angle', 'relaxed_beta_angle', 'relaxed_gamma_angle',
    'relaxed_atom_positions', 'relaxed_cell',
    'atomic_numbers'
]

# --- 1b. Recommender Feature Sets ---
# Features that require running a DFT calculation. NOT available pre-calculation.
# Note: original_volume, original_*_len, original_*_angle are from the input
# crystal structure (ICSD) and ARE available pre-calculation — do NOT drop them.
COLS_DFT_INITIAL = [
    # 13 energy components from cheap DFT
    'total_energy', 'aims_free_energy', 'sum_eigenvalues',
    'xc_energy_correction', 'xc_potential_correction',
    'free_atom_electrostatic_energy', 'hartree_energy_correction',
    'entropy_correction', 'total_energy_T0', 'kinetic_energy',
    'electrostatic_energy', 'multipole_correction', 'electronic_free_energy',
    # Derived from cheap DFT relaxation
    'final_volume_per_atom',
    # Non-numeric arrays (dropped by select_dtypes anyway, listed for completeness)
    'original_atom_positions', 'original_cell',
]

FEATURE_SETS = {
    'full':    {'extra_drops': []},
    'precalc': {'extra_drops': COLS_DFT_INITIAL},
    'minimal': {'extra_drops': COLS_DFT_INITIAL, 'drop_monomer': True},
}

# --- 2. Target Definitions ---
TARGET_GROUPS = {
    'energy': ['delta_total_energy_per_atom'],
    'bandgap': ['delta_homo_lumo_gap'],
    'volume': ['delta_final_volume_per_atom'],
    'geometry': [
        'delta_final_volume_per_atom',
        'delta_relaxed_a_len', 'delta_relaxed_b_len', 'delta_relaxed_c_len',
        'delta_relaxed_alpha_angle', 'delta_relaxed_beta_angle', 'delta_relaxed_gamma_angle'
    ],
    # Everything except positions (flattened lattice params + energy + gap)
    'all_scalar': [
        'delta_total_energy_per_atom', 'delta_homo_lumo_gap',
        'delta_final_volume_per_atom',
        'delta_relaxed_a_len', 'delta_relaxed_b_len', 'delta_relaxed_c_len',
        'delta_relaxed_alpha_angle', 'delta_relaxed_beta_angle', 'delta_relaxed_gamma_angle'
    ]
}


# --- 3. Custom Metrics ---
def calculate_smape(y_true, y_pred, epsilon=1e-7):
    """
    Symmetric Mean Absolute Percentage Error (sMAPE).
    Matches the GNN implementation: 2 * |pred - target| / (|pred| + |target| + epsilon)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    numerator = np.abs(y_pred - y_true)
    denominator = np.abs(y_pred) + np.abs(y_true) + epsilon
    
    # Formula: 100 * 2 * num / denom
    smape = 100.0 * 2.0 * numerator / denominator
    
    return np.mean(smape)


def calculate_mag_acc(y_true, y_pred):
    """Magnitude Accuracy: Checks if prediction is in the same log10 bucket."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    bins = np.array([1e-3, 1e-2, 1e-1, 1.0, 10.0])
    p_bins = np.digitize(np.abs(y_pred), bins)
    t_bins = np.digitize(np.abs(y_true), bins)
    return np.mean(p_bins == t_bins)

# --- 4. Scorers for RandomSearch ---
# Note: greater_is_better=False means the optimizer tries to MINIMIZE the output.

def smape_loss_func(y, y_pred):
    return calculate_smape(y, y_pred)

def rmsle_abs_func(y, y_pred):
    # RMSLAE: log10(|x| + 1e-4) — matches GNN training loss and validation metric
    eps = 1e-4
    return np.sqrt(mean_squared_error(np.log10(np.abs(y) + eps), np.log10(np.abs(y_pred) + eps)))

def asinh_mae_func(y, y_pred):
    # Robust loss
    return mean_absolute_error(np.arcsinh(y), np.arcsinh(y_pred))

SCORERS = {
    'smape': make_scorer(smape_loss_func, greater_is_better=False),
    'mae': 'neg_mean_absolute_error',
    'mse': 'neg_mean_squared_error',
    'rmsle_abs': make_scorer(rmsle_abs_func, greater_is_better=False),
    'rmslae': make_scorer(rmsle_abs_func, greater_is_better=False),
    'asinh': make_scorer(asinh_mae_func, greater_is_better=False)
}
