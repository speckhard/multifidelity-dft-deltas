"""TDD suite for the exciting RF pipeline.

Three groups of tests:

1. `rf_config_exciting` module shape: TARGET_GROUPS keys, target-column
   naming, COLS_TO_DROP_EXPLICIT coverage, SCORERS registry, custom
   metrics. These lock in "never regress" properties of the config.

2. `data_loader.load_and_clean_data(..., drop_cols=...)` — the refactor
   that removes the hard import `from rf_config import ...`. Tests
   verify the function is config-agnostic given a drop list.

3. `rf_trainer`-side config resolution — that `--config_module=<name>`
   importlib-loads the right module and pulls TARGET_GROUPS / SCORERS
   / COLS_TO_DROP_EXPLICIT from it, with `rf_config` as the default
   (aims backwards compat).

Run:
    python -m pytest tests/test_rf_exciting.py -v
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

TESTS_DIR = Path(__file__).resolve().parent
REPO_DIR = TESTS_DIR.parent
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "modelling" / "rf"))


# ---------------------------------------------------------------------------
# 1. rf_config_exciting module shape
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def exciting_cfg():
    return importlib.import_module("rf_config_exciting")


def test_exciting_config_module_imports_cleanly(exciting_cfg):
    assert exciting_cfg is not None


def test_exciting_config_has_target_groups(exciting_cfg):
    assert hasattr(exciting_cfg, "TARGET_GROUPS")
    tg = exciting_cfg.TARGET_GROUPS
    assert isinstance(tg, dict)
    assert len(tg) > 0


def test_exciting_target_groups_expected_keys(exciting_cfg):
    """The exciting sweep should cover: total energy, 3 band-gap flavors,
    the 10-term energy decomposition, and an all-scalar group."""
    tg = exciting_cfg.TARGET_GROUPS
    required = {
        "energy",
        "bandgap_scf",
        "bandgap_indirect_bands",
        "bandgap_direct_bands",
        "energy_decomposition",
        "all_scalar",
    }
    missing = required - set(tg)
    assert not missing, f"missing target groups: {missing}"


def test_exciting_no_geometry_target_groups(exciting_cfg):
    """Cells were fixed during relaxation, so any delta_relaxed_{a_len,
    b_len, c_len, alpha_angle, ..., volume_per_atom} is identically 0.
    The config must NOT ship geometry-only target groups."""
    tg = exciting_cfg.TARGET_GROUPS
    for forbidden in ("geometry", "volume", "lattice", "cell"):
        assert forbidden not in tg, (
            f"{forbidden!r} group should not exist — cells are fixed"
        )


def test_exciting_targets_use_delta_prefix(exciting_cfg):
    """Every target column must be a `delta_*` key (the RF is trained on
    deltas produced by create_delta_dataset.py)."""
    tg = exciting_cfg.TARGET_GROUPS
    for name, cols in tg.items():
        assert isinstance(cols, list) and len(cols) > 0
        for c in cols:
            assert c.startswith("delta_"), (
                f"target {c!r} in group {name!r} missing delta_ prefix"
            )


def test_exciting_energy_decomposition_is_ten_terms(exciting_cfg):
    """The INFO.OUT energy decomposition has 10 per-atom terms."""
    cols = exciting_cfg.TARGET_GROUPS["energy_decomposition"]
    assert len(cols) == 10
    expected = {
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
    }
    assert set(cols) == expected


def test_exciting_bandgap_targets_match_direct_and_indirect(exciting_cfg):
    tg = exciting_cfg.TARGET_GROUPS
    assert tg["bandgap_scf"] == ["delta_band_gap_scf_eV"]
    assert tg["bandgap_indirect_bands"] == ["delta_band_gap_indirect_bands_eV"]
    assert tg["bandgap_direct_bands"] == ["delta_band_gap_direct_bands_eV"]


def test_exciting_all_scalar_is_union(exciting_cfg):
    """all_scalar group should contain each other group's columns at least once."""
    tg = exciting_cfg.TARGET_GROUPS
    all_scalar = set(tg["all_scalar"])
    for key in ("energy_decomposition", "bandgap_scf",
                "bandgap_indirect_bands", "bandgap_direct_bands"):
        for c in tg[key]:
            assert c in all_scalar, (
                f"all_scalar missing {c!r} (from {key!r})"
            )


# --- leakage-prevention drop list ------------------------------------------

def test_exciting_cols_to_drop_is_list(exciting_cfg):
    assert hasattr(exciting_cfg, "COLS_TO_DROP_EXPLICIT")
    assert isinstance(exciting_cfg.COLS_TO_DROP_EXPLICIT, list)


_EXPECTED_DROPS = [
    # Identifiers / metadata
    "path", "id", "uid", "ICSD_number", "compound_name", "chem_formula",
    "status", "geometry_status", "parse_warnings", "bandgap_source",
    # Raw energy scalars in Hartree — would leak delta_*_per_atom targets
    "total_energy", "kinetic_energy", "exchange_energy", "correlation_energy",
    "hartree_energy", "electron_nuclear_energy", "nuclear_nuclear_energy",
    "xc_potential_energy", "madelung_energy", "core_electron_kinetic_energy",
    # Per-atom raw values — same leakage in atomic units
    "total_energy_per_atom", "kinetic_energy_per_atom",
    "exchange_energy_per_atom", "correlation_energy_per_atom",
    "hartree_energy_per_atom", "electron_nuclear_energy_per_atom",
    "nuclear_nuclear_energy_per_atom", "xc_potential_energy_per_atom",
    "madelung_energy_per_atom", "core_electron_kinetic_energy_per_atom",
    "relaxed_volume_per_atom",
    # Band-gap raw values — leak delta_band_gap_*
    "band_gap_scf_eV", "band_gap_indirect_bands_eV",
    "band_gap_direct_bands_eV", "band_gap_bands_eV",
    "is_metal_scf", "is_metal_bands", "is_direct_bands",
    # Fermi / DOS — leak their delta versions
    "fermi_energy", "dos_at_fermi", "fermi_energy_Ha",
    # Geometry scalars (constant by construction, but still drop to avoid
    # accidental one-hot of the ICSD identity leaking into X)
    "relaxed_volume", "relaxed_a_len", "relaxed_b_len", "relaxed_c_len",
    "relaxed_alpha_angle", "relaxed_beta_angle", "relaxed_gamma_angle",
    # Arrays — un-numeric, fallback path-reparse
    "relaxed_atom_positions", "relaxed_lattice", "atomic_numbers",
]


@pytest.mark.parametrize("col", _EXPECTED_DROPS)
def test_exciting_drop_list_covers_leakage(exciting_cfg, col):
    assert col in exciting_cfg.COLS_TO_DROP_EXPLICIT, (
        f"{col!r} must be in COLS_TO_DROP_EXPLICIT to prevent leakage"
    )


# --- scorer / metric surface -----------------------------------------------

def test_exciting_scorers_keys(exciting_cfg):
    assert hasattr(exciting_cfg, "SCORERS")
    assert set(exciting_cfg.SCORERS) == {"smape", "mae", "mse",
                                          "rmsle_abs", "asinh"}


def test_exciting_custom_metrics_callable(exciting_cfg):
    from numpy import array
    assert callable(exciting_cfg.calculate_smape)
    val = exciting_cfg.calculate_smape(array([1.0, 2.0]), array([1.1, 1.9]))
    assert val >= 0
    assert callable(exciting_cfg.calculate_mag_acc)


# ---------------------------------------------------------------------------
# 2. data_loader: drop_cols-as-arg refactor
# ---------------------------------------------------------------------------

@pytest.fixture
def synth_delta_csv(tmp_path) -> Path:
    """Minimal CSV with one delta target + one leakage col + one plain
    input + a split column."""
    rows = []
    for split in ("train", "train", "val", "test"):
        rows.append({
            "split": split,
            "ICSD_number": 42,
            "path": "/fake",
            # plain input feature (should survive)
            "num_atoms": 4,
            # would-be leakage: raw energy
            "total_energy": -100.0,
            # target
            "delta_total_energy_per_atom": 0.01,
            # unrelated delta (always dropped)
            "delta_band_gap_scf_eV": 0.05,
            # delta_monomer_* — should be kept as feature (existing carve-out)
            "delta_monomer_volume": 1.23,
        })
    csv = tmp_path / "delta.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return csv


def test_data_loader_accepts_drop_cols_arg(synth_delta_csv):
    """After the refactor, load_and_clean_data must take `drop_cols`
    explicitly — no hidden import of COLS_TO_DROP_EXPLICIT."""
    import data_loader
    import inspect
    sig = inspect.signature(data_loader.load_and_clean_data)
    assert "drop_cols" in sig.parameters, (
        "load_and_clean_data must accept drop_cols kwarg"
    )


def test_data_loader_drops_listed_cols(synth_delta_csv):
    """Cols in drop_cols are removed from X."""
    import data_loader
    drop = ["total_energy", "path", "ICSD_number"]
    df, X = data_loader.load_and_clean_data(str(synth_delta_csv), drop_cols=drop)
    for c in drop:
        assert c not in X.columns, f"{c!r} should have been dropped from X"


def test_data_loader_keeps_plain_features(synth_delta_csv):
    import data_loader
    df, X = data_loader.load_and_clean_data(str(synth_delta_csv), drop_cols=[])
    assert "num_atoms" in X.columns


def test_data_loader_always_drops_delta_except_monomer(synth_delta_csv):
    """Non-negotiable carve-out — prevents target leakage regardless of
    what the config-provided drop list does or doesn't contain."""
    import data_loader
    df, X = data_loader.load_and_clean_data(str(synth_delta_csv), drop_cols=[])
    assert "delta_total_energy_per_atom" not in X.columns
    assert "delta_band_gap_scf_eV" not in X.columns
    # Carve-out: delta_monomer_* stays
    assert "delta_monomer_volume" in X.columns


def test_data_loader_returns_full_df_unchanged(synth_delta_csv):
    """`df` return still carries `split` and target cols — needed for
    get_train_test_split downstream."""
    import data_loader
    df, _ = data_loader.load_and_clean_data(str(synth_delta_csv), drop_cols=[])
    assert "split" in df.columns
    assert "delta_total_energy_per_atom" in df.columns


def test_get_train_test_split_by_split_column(synth_delta_csv):
    """Split train+val vs test by df.split; behavior unchanged by refactor."""
    import data_loader
    df, X = data_loader.load_and_clean_data(str(synth_delta_csv), drop_cols=[])
    X_train, y_train, X_test, y_test = data_loader.get_train_test_split(
        df, X, ["delta_total_energy_per_atom"]
    )
    # 3 train+val rows, 1 test row.
    assert len(X_train) == 3
    assert len(X_test) == 1


# ---------------------------------------------------------------------------
# 3. rf_trainer: --config_module flag resolution
# ---------------------------------------------------------------------------

def test_trainer_declares_config_module_flag():
    """--config_module must exist as an absl FLAG; default is rf_config
    (aims backwards compat). We inspect without running main()."""
    import rf_trainer
    from absl import flags as absl_flags
    FLAGS = absl_flags.FLAGS
    # abseil flags are lazily-parsed; force parse on a no-op argv.
    try:
        FLAGS([__file__])
    except absl_flags.DuplicateFlagError:
        pass
    assert "config_module" in FLAGS, (
        "rf_trainer must declare --config_module"
    )
    assert FLAGS["config_module"].default == "rf_config", (
        "default must be 'rf_config' for aims backwards compat"
    )


def test_trainer_resolve_config_loads_exciting_module():
    """The helper (or __main__ code) that imports the config module by
    name must return an object that exposes TARGET_GROUPS, SCORERS, and
    COLS_TO_DROP_EXPLICIT."""
    import rf_trainer
    # Expect a module-level helper `resolve_config(name) -> module`.
    assert hasattr(rf_trainer, "resolve_config"), (
        "rf_trainer must expose resolve_config(name) for test/CLI use"
    )
    cfg = rf_trainer.resolve_config("rf_config_exciting")
    assert hasattr(cfg, "TARGET_GROUPS")
    assert hasattr(cfg, "SCORERS")
    assert hasattr(cfg, "COLS_TO_DROP_EXPLICIT")
    assert "energy" in cfg.TARGET_GROUPS


def test_trainer_resolve_config_bad_name_raises():
    import rf_trainer
    with pytest.raises((ImportError, ModuleNotFoundError)):
        rf_trainer.resolve_config("definitely_not_a_real_module_xyz")


def test_trainer_resolve_config_aims_still_works():
    """Default aims config must still resolve without regression."""
    import rf_trainer
    cfg = rf_trainer.resolve_config("rf_config")
    assert "energy" in cfg.TARGET_GROUPS
    assert "delta_total_energy_per_atom" in cfg.TARGET_GROUPS["energy"]
