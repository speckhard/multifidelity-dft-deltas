"""Tests for parsing/compute_dos_similarity.py — written BEFORE the implementation (TDD)."""

import os
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parsing.compute_dos_similarity import (
    compute_tanimoto,
    derive_reference_path,
    parse_dos_to_arrays,
)


# ---- Fixtures ----


def _write_dos_xml(path: Path, energies_Ha: np.ndarray, dos_vals: np.ndarray):
    """Write a minimal dos.xml file matching the exciting XML schema."""
    root = ET.Element("dos")
    ET.SubElement(root, "title").text = "test"
    totaldos = ET.SubElement(root, "totaldos")
    diagram = ET.SubElement(totaldos, "diagram")
    for e, d in zip(energies_Ha, dos_vals):
        ET.SubElement(diagram, "point", e=str(e), dos=str(d))
    tree = ET.ElementTree(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(path), xml_declaration=True)


@pytest.fixture
def dos_xml_pair(tmp_path):
    """Create a pair of dos.xml files (cheap + reference) with known DOS curves."""
    energies_Ha = np.linspace(-0.3, 0.3, 200)

    dos_cheap = np.exp(-((energies_Ha - 0.05) ** 2) / 0.01) * 100
    dos_ref = np.exp(-((energies_Ha - 0.04) ** 2) / 0.01) * 100

    cheap_dir = tmp_path / "precision_0_3" / "rmt_scaling_0_95" / "8" / "NaCl_ICSD_1234"
    ref_dir = tmp_path / "precision_1_0" / "rmt_scaling_0_95" / "8" / "NaCl_ICSD_1234"

    _write_dos_xml(cheap_dir / "dos.xml", energies_Ha, dos_cheap)
    _write_dos_xml(ref_dir / "dos.xml", energies_Ha, dos_ref)

    return cheap_dir, ref_dir, energies_Ha, dos_cheap, dos_ref


# ---- Tests: parse_dos_to_arrays ----


class TestParseDosToArrays:
    def test_returns_eV_energies_and_dos(self, dos_xml_pair):
        cheap_dir, _, energies_Ha, dos_cheap, _ = dos_xml_pair
        energies_eV, dos_vals = parse_dos_to_arrays(str(cheap_dir / "dos.xml"))

        HA_TO_EV = 27.211386245988
        np.testing.assert_allclose(energies_eV, energies_Ha * HA_TO_EV, rtol=1e-10)
        np.testing.assert_allclose(dos_vals, dos_cheap, rtol=1e-10)

    def test_returns_numpy_arrays(self, dos_xml_pair):
        cheap_dir, *_ = dos_xml_pair
        energies, dos = parse_dos_to_arrays(str(cheap_dir / "dos.xml"))
        assert isinstance(energies, np.ndarray)
        assert isinstance(dos, np.ndarray)

    def test_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            parse_dos_to_arrays("/nonexistent/dos.xml")


# ---- Tests: derive_reference_path ----


class TestDeriveReferencePath:
    def test_replaces_precision_in_path(self):
        cheap = "/data/precision_0_3/rmt_scaling_0_95/8/NaCl_ICSD_1234"
        ref = derive_reference_path(cheap, apw_precision=0.3)
        assert ref == "/data/precision_1_0/rmt_scaling_0_95/8/NaCl_ICSD_1234"

    def test_handles_various_precisions(self):
        for prec in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            prec_str = str(prec).replace(".", "_")
            cheap = f"/data/precision_{prec_str}/rmt/mat"
            ref = derive_reference_path(cheap, apw_precision=prec)
            assert ref == "/data/precision_1_0/rmt/mat"

    def test_precision_1_0_maps_to_itself(self):
        cheap = "/data/precision_1_0/rmt/mat"
        ref = derive_reference_path(cheap, apw_precision=1.0)
        assert ref == "/data/precision_1_0/rmt/mat"


# ---- Tests: compute_tanimoto ----


class TestComputeTanimoto:
    def test_identical_dos_returns_one(self):
        energies = np.linspace(-10, 5, 500)
        dos = np.exp(-(energies**2) / 2) * 100
        tc = compute_tanimoto(energies, dos, energies, dos)
        assert tc == pytest.approx(1.0)

    def test_different_dos_less_than_one(self):
        energies = np.linspace(-10, 5, 500)
        dos1 = np.exp(-((energies - 1) ** 2) / 2) * 100
        dos2 = np.exp(-((energies + 1) ** 2) / 2) * 100
        tc = compute_tanimoto(energies, dos1, energies, dos2)
        assert 0.0 < tc < 1.0

    def test_returns_float(self):
        energies = np.linspace(-10, 5, 500)
        dos = np.ones_like(energies)
        tc = compute_tanimoto(energies, dos, energies, dos)
        assert isinstance(tc, float)


# ---- Tests: end-to-end CSV creation ----


class TestAddTanimotoToCsv:
    def test_output_csv_has_tanimoto_column(self, dos_xml_pair):
        from parsing.compute_dos_similarity import add_tanimoto_to_csv

        cheap_dir, ref_dir, *_ = dos_xml_pair

        input_csv = cheap_dir.parent.parent.parent.parent / "input.csv"
        output_csv = cheap_dir.parent.parent.parent.parent / "output_tanimoto.csv"

        df = pd.DataFrame(
            {
                "path": [str(cheap_dir)],
                "APW_precision_path": [0.3],
                "ICSD_number": [1234],
                "split": ["train"],
                "total_energy": [-100.0],
                "delta_total_energy": [-0.5],
            }
        )
        df.to_csv(str(input_csv), index=False)

        add_tanimoto_to_csv(str(input_csv), str(output_csv))

        result = pd.read_csv(str(output_csv))
        assert "dos_tanimoto" in result.columns
        assert len(result) == 1
        tc = result["dos_tanimoto"].iloc[0]
        assert 0.0 < tc <= 1.0

    def test_missing_dos_xml_produces_nan(self, tmp_path):
        from parsing.compute_dos_similarity import add_tanimoto_to_csv

        input_csv = tmp_path / "input.csv"
        output_csv = tmp_path / "output.csv"

        df = pd.DataFrame(
            {
                "path": ["/nonexistent/precision_0_3/rmt/mat"],
                "APW_precision_path": [0.3],
                "ICSD_number": [9999],
                "split": ["test"],
                "total_energy": [-50.0],
                "delta_total_energy": [-0.1],
            }
        )
        df.to_csv(str(input_csv), index=False)

        add_tanimoto_to_csv(str(input_csv), str(output_csv))

        result = pd.read_csv(str(output_csv))
        assert "dos_tanimoto" in result.columns
        assert pd.isna(result["dos_tanimoto"].iloc[0])

    def test_preserves_all_original_columns(self, dos_xml_pair):
        from parsing.compute_dos_similarity import add_tanimoto_to_csv

        cheap_dir, *_ = dos_xml_pair
        input_csv = cheap_dir.parent.parent.parent.parent / "input.csv"
        output_csv = cheap_dir.parent.parent.parent.parent / "out.csv"

        df = pd.DataFrame(
            {
                "path": [str(cheap_dir)],
                "APW_precision_path": [0.3],
                "ICSD_number": [1234],
                "split": ["train"],
                "total_energy": [-100.0],
                "delta_total_energy": [-0.5],
                "some_feature": [42.0],
            }
        )
        df.to_csv(str(input_csv), index=False)

        add_tanimoto_to_csv(str(input_csv), str(output_csv))

        result = pd.read_csv(str(output_csv))
        for col in df.columns:
            assert col in result.columns
