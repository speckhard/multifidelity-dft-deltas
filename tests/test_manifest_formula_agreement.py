import pytest

from modelling.gnn.manifest_graph_builder import (
    ManifestError,
    _DFT_KEYS,
    _GEO_KEYS,
    _TARGET_KEYS,
    verify_row_agreement,
)


class _FakeAtoms:
    def __init__(self, formula, atomic_numbers):
        self._formula = formula
        self._atomic_numbers = atomic_numbers

    def get_chemical_formula(self):
        return self._formula

    def get_atomic_numbers(self):
        return self._atomic_numbers


class _FakeDbRow:
    def __init__(self, formula, atomic_numbers):
        self._atoms = _FakeAtoms(formula, atomic_numbers)
        self.ICSD_number = 1
        for key in _DFT_KEYS + _GEO_KEYS + _TARGET_KEYS:
            setattr(self, key, 1.0)
        self.k_point_density = 8
        self.rel_setting = "atomic_zora"
        self.basis_size = "tier1"
        self.num_setting = "light"
        self.binary_precision = 2

    def toatoms(self):
        return self._atoms


def _manifest_row(formula, atomic_numbers):
    row = {
        "source_row_id": "legacy_hf:000001",
        "formula": formula,
        "_atomic_numbers": atomic_numbers,
        "atomic_numbers": str(atomic_numbers),
        "ICSD_number": "1",
        "k_point_density": "8",
        "rel_setting": "atomic_zora",
        "basis_size": "tier1",
        "num_setting": "light",
        "binary_precision": "2",
    }
    for key in _DFT_KEYS + _GEO_KEYS + _TARGET_KEYS:
        row[key] = "1.0"
    return row


def test_formula_order_permutation_is_accepted():
    manifest_row = _manifest_row("Ti2O4", [22, 22, 8, 8, 8, 8])
    db_row = _FakeDbRow("O4Ti2", [22, 22, 8, 8, 8, 8])

    verify_row_agreement(manifest_row, db_row)


@pytest.mark.parametrize("manifest_formula, db_formula", [("Ti2O4", "TiO2"), ("Ti2O4", "Ti2O3")])
def test_formula_mismatch_is_rejected(manifest_formula, db_formula):
    manifest_row = _manifest_row(manifest_formula, [22, 22, 8, 8, 8, 8])
    db_row = _FakeDbRow(db_formula, [22, 22, 8, 8, 8, 8])

    with pytest.raises(ManifestError, match="semantic mismatch formula"):
        verify_row_agreement(manifest_row, db_row)


def test_zero_count_formula_is_rejected():
    manifest_row = _manifest_row("Ti0O4", [22, 22, 8, 8, 8, 8])
    db_row = _FakeDbRow("O4Ti2", [22, 22, 8, 8, 8, 8])

    with pytest.raises(ManifestError, match="formula"):
        verify_row_agreement(manifest_row, db_row)


def test_malformed_formula_is_rejected():
    manifest_row = _manifest_row("Ti2O4", [22, 22, 8, 8, 8, 8])
    db_row = _FakeDbRow("bad formula", [22, 22, 8, 8, 8, 8])

    with pytest.raises(ManifestError, match="formula parse failure for legacy_hf:000001 field formula"):
        verify_row_agreement(manifest_row, db_row)


def test_empty_formula_is_rejected():
    manifest_row = _manifest_row("", [22, 22, 8, 8, 8, 8])
    db_row = _FakeDbRow("O4Ti2", [22, 22, 8, 8, 8, 8])

    with pytest.raises(ManifestError, match="formula parse failure for legacy_hf:000001 field formula"):
        verify_row_agreement(manifest_row, db_row)
