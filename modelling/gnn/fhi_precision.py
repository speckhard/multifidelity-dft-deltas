"""Portable canonical mapping mirrored from /Users/dts/Documents/academic/new_delta_calcs/scripts/fhi_precision.py."""
NUMERICAL_SETTINGS = ("light", "tight", "really_tight")
BASIS_SIZES = ("minimal", "standard", "tier1", "tier2")
LEGACY_BASIS_SIZES = ("minimal", "tier1", "standard", "tier2")

def normalize_num_setting(value):
    value = str(value).strip().lower().replace(" ", "_")
    if value == "reallytight": value = "really_tight"
    if value not in NUMERICAL_SETTINGS: raise ValueError(f"unrecognized num_setting: {value!r}")
    return value

def normalize_basis_size(value):
    value = str(value).strip().lower()
    if value not in BASIS_SIZES: raise ValueError(f"unrecognized basis_size: {value!r}")
    return value

def precision_from_settings(num_setting, basis_size, legacy=False):
    num = normalize_num_setting(num_setting)
    basis = normalize_basis_size(basis_size)
    basis_order = LEGACY_BASIS_SIZES if legacy else BASIS_SIZES
    return NUMERICAL_SETTINGS.index(num) * 4 + basis_order.index(basis)
