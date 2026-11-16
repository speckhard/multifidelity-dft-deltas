"""Ordered FHI-aims RF inputs for the ICML feature ablation.

Ported from speckhard/errorbar_modelling, modelling/rf/feature_contracts.py.
The 102/89/57 counts describe the maintained FHI input schema. The legacy
CSV loader can retain additional numeric fields; these explicit lists let
benchmarking code check its inputs against that schema.
"""

FHI_FULL_FEATURES = (
    'k_point_density',
    'total_energy',
    'aims_free_energy',
    'multipole_correction',
    'electrostatic_energy',
    'kinetic_energy',
    'total_energy_T0',
    'entropy_correction',
    'hartree_energy_correction',
    'free_atom_electrostatic_energy',
    'xc_potential_correction',
    'xc_energy_correction',
    'sum_eigenvalues',
    'binary_precision',
    'max_atomic_EA_half',
    'min_atomic_EA_half',
    'mean_atomic_EA_half',
    'mad_atomic_EA_half',
    'max_atomic_IP_half',
    'min_atomic_IP_half',
    'mean_atomic_IP_half',
    'mad_atomic_IP_half',
    'max_atomic_EA_delta',
    'min_atomic_EA_delta',
    'mean_atomic_EA_delta',
    'mad_atomic_EA_delta',
    'max_atomic_IP_delta',
    'min_atomic_IP_delta',
    'mean_atomic_IP_delta',
    'mad_atomic_IP_delta',
    'max_atomic_HOMO',
    'min_atomic_HOMO',
    'mean_atomic_HOMO',
    'mad_atomic_HOMO',
    'max_atomic_LUMO',
    'min_atomic_LUMO',
    'mean_atomic_LUMO',
    'mad_atomic_LUMO',
    'max_atomic_rs',
    'min_atomic_rs',
    'mean_atomic_rs',
    'mad_atomic_rs',
    'max_atomic_rp',
    'min_atomic_rp',
    'mean_atomic_rp',
    'mad_atomic_rp',
    'max_atomic_rd',
    'min_atomic_rd',
    'mean_atomic_rd',
    'mad_atomic_rd',
    'max_atomic_rf',
    'min_atomic_rf',
    'mean_atomic_rf',
    'mad_atomic_rf',
    'max_monomer_total_energy_per_atom',
    'min_monomer_total_energy_per_atom',
    'mean_monomer_total_energy_per_atom',
    'mad_monomer_total_energy_per_atom',
    'max_monomer_volume_per_atom',
    'min_monomer_volume_per_atom',
    'mean_monomer_volume_per_atom',
    'mad_monomer_volume_per_atom',
    'max_monomer_bandstructure_gap',
    'min_monomer_bandstructure_gap',
    'mean_monomer_bandstructure_gap',
    'mad_monomer_bandstructure_gap',
    'max_monomer_gamma_gap',
    'min_monomer_gamma_gap',
    'mean_monomer_gamma_gap',
    'mad_monomer_gamma_gap',
    'max_valence_electrons',
    'min_valence_electrons',
    'mean_valence_electrons',
    'mad_valence_electrons',
    'max_basis_functions',
    'min_basis_functions',
    'mean_basis_functions',
    'mad_basis_functions',
    'original_volume',
    'original_a_len',
    'original_b_len',
    'original_c_len',
    'original_alpha_angle',
    'original_beta_angle',
    'original_gamma_angle',
    'final_volume_per_atom',
    'delta_max_monomer_total_energy_per_atom',
    'delta_min_monomer_total_energy_per_atom',
    'delta_mean_monomer_total_energy_per_atom',
    'delta_mad_monomer_total_energy_per_atom',
    'delta_max_monomer_volume_per_atom',
    'delta_min_monomer_volume_per_atom',
    'delta_mean_monomer_volume_per_atom',
    'delta_mad_monomer_volume_per_atom',
    'delta_max_monomer_bandstructure_gap',
    'delta_min_monomer_bandstructure_gap',
    'delta_mean_monomer_bandstructure_gap',
    'delta_mad_monomer_bandstructure_gap',
    'delta_max_monomer_gamma_gap',
    'delta_min_monomer_gamma_gap',
    'delta_mean_monomer_gamma_gap',
    'delta_mad_monomer_gamma_gap',
)

FHI_CANDIDATE_OUTPUT_FEATURES = (
    'total_energy',
    'aims_free_energy',
    'multipole_correction',
    'electrostatic_energy',
    'kinetic_energy',
    'total_energy_T0',
    'entropy_correction',
    'hartree_energy_correction',
    'free_atom_electrostatic_energy',
    'xc_potential_correction',
    'xc_energy_correction',
    'sum_eigenvalues',
    'final_volume_per_atom',
)

FHI_TARGET_FEATURES = (
    'delta_total_energy_per_atom',
    'delta_homo_lumo_gap',
    'delta_final_volume_per_atom',
)

# PreCalc removes 12 cheap compound energy outputs and relaxed volume.
FHI_PRECALC_FEATURES = tuple(
    name for name in FHI_FULL_FEATURES
    if name not in FHI_CANDIDATE_OUTPUT_FEATURES
)

# Minimal also removes all 16 elemental-solid descriptors and 16 deltas.
# Input-cell geometry remains available in both pre-calculation regimes.
FHI_MINIMAL_FEATURES = tuple(
    name for name in FHI_PRECALC_FEATURES if "monomer" not in name
)

FHI_FEATURE_SETS = {
    "full": FHI_FULL_FEATURES,
    "precalc": FHI_PRECALC_FEATURES,
    "minimal": FHI_MINIMAL_FEATURES,
}
