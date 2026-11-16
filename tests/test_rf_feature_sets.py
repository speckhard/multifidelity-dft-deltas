"""Regressions for the FHI feature ablations and CLI-to-model wiring."""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from absl.testing import flagsaver
from sklearn.model_selection import RandomizedSearchCV

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modelling" / "rf"))

import rf_config
import rf_trainer
from data_loader import load_and_clean_data
from feature_contracts import FHI_FEATURE_SETS, FHI_FULL_FEATURES


@pytest.fixture
def fhi_csv(tmp_path):
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(rng.normal(size=(16, 102)), columns=FHI_FULL_FEATURES)
    frame["k_point_density"] = 4
    frame["split"] = ["train"] * 10 + ["val"] * 2 + ["test"] * 4
    frame["delta_total_energy_per_atom"] = rng.normal(size=16)
    frame["delta_mean_basis_functions"] = -1
    frame["total_energy_per_atom"] = 123.0
    frame["homo_lumo_gap"] = 456.0
    frame["ICSD_number"] = range(16)
    frame["original_cell"] = "[[1, 0, 0], [0, 1, 0], [0, 0, 1]]"
    path = tmp_path / "fhi.csv"
    frame.to_csv(path, index=False)
    return path


@pytest.mark.parametrize("mode,count", [("full", 102), ("precalc", 89), ("minimal", 57)])
def test_trainer_saves_model_with_selected_inputs(fhi_csv, tmp_path, monkeypatch, mode, count):
    # Exercise main() with real CV, fitting and joblib persistence, but a tiny
    # forest instead of the production 40-candidate sweep.
    def small_search(**kwargs):
        kwargs["param_distributions"] = {"n_estimators": [2], "max_depth": [2]}
        kwargs["n_iter"] = 1
        return RandomizedSearchCV(**kwargs)

    monkeypatch.setattr(rf_trainer, "RandomizedSearchCV", small_search)
    rf_trainer.FLAGS(["rf-feature-test"])
    with flagsaver.flagsaver(
        data_file=str(fhi_csv), output_dir=str(tmp_path), config_module="rf_config",
        feature_set=mode, target_key="energy", metric_key="mae", n_jobs=1,
    ):
        rf_trainer.main([])

    model = joblib.load(tmp_path / "rf_energy_mae_best.joblib")
    names = list(model.feature_names_in_)
    assert model.n_features_in_ == count
    assert names == list(FHI_FEATURE_SETS[mode])
    assert "original_volume" in names and "original_a_len" in names
    assert "delta_total_energy_per_atom" not in names
    assert "total_energy_per_atom" not in names
    assert "homo_lumo_gap" not in names
    assert "ICSD_number" not in names
    assert ("total_energy" in names) == (mode == "full")
    assert ("final_volume_per_atom" in names) == (mode == "full")
    assert ("max_monomer_gamma_gap" in names) == (mode != "minimal")
    assert ("delta_max_monomer_gamma_gap" in names) == (mode != "minimal")


def test_minimal_removes_additional_monomer_columns(fhi_csv):
    frame = pd.read_csv(fhi_csv)
    frame["extra_monomer_descriptor"] = 1.0
    frame["delta_extra_monomer_descriptor"] = 2.0
    frame.to_csv(fhi_csv, index=False)
    _, X = load_and_clean_data(
        fhi_csv, drop_cols=rf_config.COLS_TO_DROP_EXPLICIT,
        **rf_config.FEATURE_SETS["minimal"],
    )
    assert not any("monomer" in name for name in X.columns)
    assert len(X.columns) == 57


@pytest.mark.parametrize("mode", ["precalc", "minimal", "unknown"])
def test_exciting_rejects_unsupported_ablations(tmp_path, mode):
    rf_trainer.FLAGS(["rf-feature-test"])
    with flagsaver.flagsaver(
        config_module="rf_config_exciting", feature_set=mode,
        output_dir=str(tmp_path),
    ):
        with pytest.raises(ValueError, match="not supported by rf_config_exciting"):
            rf_trainer.main([])
