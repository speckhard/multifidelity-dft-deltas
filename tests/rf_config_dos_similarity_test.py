"""Tests for rf_config_dos_similarity — written BEFORE the implementation (TDD)."""

import os
import sys

import numpy as np
import pytest

sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "modelling", "rf")
)

from rf_config_dos_similarity import (
    COLS_TO_DROP_EXPLICIT,
    SCORERS,
    TARGET_GROUPS,
    calculate_mag_acc,
    calculate_smape,
)


class TestRequiredExports:
    def test_has_target_groups(self):
        assert isinstance(TARGET_GROUPS, dict)

    def test_has_scorers(self):
        assert isinstance(SCORERS, dict)

    def test_has_cols_to_drop(self):
        assert isinstance(COLS_TO_DROP_EXPLICIT, list)

    def test_has_calculate_smape(self):
        assert callable(calculate_smape)

    def test_has_calculate_mag_acc(self):
        assert callable(calculate_mag_acc)


class TestTargetGroups:
    def test_dos_tanimoto_target_exists(self):
        assert "dos_tanimoto" in TARGET_GROUPS

    def test_dos_tanimoto_target_columns(self):
        assert TARGET_GROUPS["dos_tanimoto"] == ["dos_tanimoto"]


class TestLeakagePrevention:
    def test_dos_tanimoto_dropped_from_features(self):
        """dos_tanimoto does NOT start with delta_, so must be explicitly dropped."""
        assert "dos_tanimoto" in COLS_TO_DROP_EXPLICIT

    def test_raw_energies_dropped(self):
        assert "total_energy" in COLS_TO_DROP_EXPLICIT

    def test_bandgaps_dropped(self):
        assert "band_gap_scf_eV" in COLS_TO_DROP_EXPLICIT

    def test_identifiers_dropped(self):
        assert "ICSD_number" in COLS_TO_DROP_EXPLICIT
        assert "path" in COLS_TO_DROP_EXPLICIT


class TestScorers:
    def test_mae_scorer_exists(self):
        assert "mae" in SCORERS

    def test_smape_scorer_exists(self):
        assert "smape" in SCORERS

    def test_r2_scorer_exists(self):
        assert "r2" in SCORERS


class TestMetricsOnBoundedData:
    """DOS Tanimoto lives in [0, 1] — metrics must handle this range."""

    def test_smape_perfect_prediction(self):
        y = np.array([0.9, 0.8, 1.0])
        assert calculate_smape(y, y) == pytest.approx(0.0, abs=1e-5)

    def test_smape_imperfect_prediction(self):
        y_true = np.array([0.9, 0.8, 0.7])
        y_pred = np.array([0.85, 0.75, 0.65])
        result = calculate_smape(y_true, y_pred)
        assert 0 < result < 100

    def test_mag_acc_returns_fraction(self):
        y = np.array([0.9, 0.5, 0.1])
        result = calculate_mag_acc(y, y)
        assert 0.0 <= result <= 1.0
