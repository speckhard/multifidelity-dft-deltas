"""Tests for parsing/reference_selectors.py.

We exercise both selectors against three row shapes:

  * plain dict,
  * object with matching attributes,
  * object with `key_value_pairs` dict (mimics ase.db.row.AtomsRow).

No ase.db required — these tests are purely about the selector predicates.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parsing.reference_selectors import (  # noqa: E402
    SELECTORS,
    aims_selector,
    exciting_selector,
    get_selector,
)


# ---------------------------------------------------------------------------
# aims selector
# ---------------------------------------------------------------------------

def test_aims_selector_accepts_reference_dict():
    row = {"binary_precision": 11, "k_point_density": 8}
    assert aims_selector(row) is True


def test_aims_selector_accepts_float_k_density():
    row = {"binary_precision": 11, "k_point_density": 8.0}
    assert aims_selector(row) is True


def test_aims_selector_rejects_wrong_precision():
    assert aims_selector({"binary_precision": 10, "k_point_density": 8}) is False


def test_aims_selector_rejects_wrong_k_density():
    assert aims_selector({"binary_precision": 11, "k_point_density": 4}) is False


def test_aims_selector_rejects_missing_fields():
    assert aims_selector({"binary_precision": 11}) is False
    assert aims_selector({"k_point_density": 8}) is False
    assert aims_selector({}) is False


def test_aims_selector_handles_ase_row_like_object():
    """ase.db.Row exposes fields both as attributes and via key_value_pairs."""
    row = SimpleNamespace(binary_precision=11, k_point_density=8,
                          key_value_pairs={"binary_precision": 11,
                                           "k_point_density": 8})
    assert aims_selector(row) is True


def test_aims_selector_handles_kvp_only():
    row = SimpleNamespace(key_value_pairs={"binary_precision": 11,
                                           "k_point_density": 8})
    assert aims_selector(row) is True


# ---------------------------------------------------------------------------
# exciting selector
# ---------------------------------------------------------------------------

def test_exciting_selector_accepts_reference_from_input_xml():
    row = {"APWprecision_input": 1.0, "k_point_density": 8}
    assert exciting_selector(row) is True


def test_exciting_selector_falls_back_to_path_fields():
    """When input.xml fields missing, selector falls back to path-regex fields."""
    row = {"APW_precision_path": 1.0, "k_point_density_path": 8}
    assert exciting_selector(row) is True


def test_exciting_selector_rejects_precision_0_9():
    row = {"APWprecision_input": 0.9, "k_point_density": 8}
    assert exciting_selector(row) is False


def test_exciting_selector_rejects_low_k_density():
    row = {"APWprecision_input": 1.0, "k_point_density": 4}
    assert exciting_selector(row) is False


def test_exciting_selector_ignores_rmt_scaling():
    """rmt_scaling is held fixed per-sweep; the selector must not gate on it."""
    row = {"APWprecision_input": 1.0, "k_point_density": 8, "rmt_scaling_path": 0.85}
    assert exciting_selector(row) is True


def test_exciting_selector_rejects_missing_fields():
    assert exciting_selector({"APWprecision_input": 1.0}) is False
    assert exciting_selector({"k_point_density": 8}) is False


def test_exciting_selector_handles_string_values_parseable():
    """Some ase.db round-trips store numbers as strings — must still work."""
    row = {"APWprecision_input": "1.0", "k_point_density": "8"}
    assert exciting_selector(row) is True


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

def test_registry_has_expected_selectors():
    assert set(SELECTORS) == {"aims", "exciting"}
    assert SELECTORS["aims"] is aims_selector
    assert SELECTORS["exciting"] is exciting_selector


def test_get_selector_happy_path():
    assert get_selector("aims") is aims_selector
    assert get_selector("exciting") is exciting_selector


def test_get_selector_unknown_name_raises():
    with pytest.raises(ValueError, match="Unknown reference selector"):
        get_selector("does_not_exist")
