"""Compute DOS Tanimoto similarity between cheap and reference exciting calcs.

Reads the delta-learning CSV, finds paired dos.xml files on disk, computes
the Tanimoto coefficient via nomad_dos_fingerprints, and writes a new CSV
with a `dos_tanimoto` column appended.

Usage:
    python parsing/compute_dos_similarity.py \
        --input_csv /u/dansp/oasis_data/exciting_delta_learning.csv \
        --output_csv /u/dansp/oasis_data/exciting_delta_learning_tanimoto.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from nomad_dos_fingerprints import DOSFingerprint, Grid
from tqdm import tqdm

HA_TO_EV = 27.211386245988
DEFAULT_GRID_ID = "dg_cut:56:-2:7:(-10, 5)"


def parse_dos_to_arrays(dos_xml_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse a dos.xml file and return (energies_eV, dos_values) as numpy arrays.

    Raises FileNotFoundError if the file doesn't exist.
    """
    path = Path(dos_xml_path)
    if not path.is_file():
        raise FileNotFoundError(f"dos.xml not found: {dos_xml_path}")

    import xml.etree.ElementTree as ET

    root = ET.parse(str(path)).getroot()
    points = root.find("totaldos").find("diagram").findall("point")
    energies_Ha = np.array([float(p.attrib["e"]) for p in points])
    dos_vals = np.array([float(p.attrib["dos"]) for p in points])
    return energies_Ha * HA_TO_EV, dos_vals


def derive_reference_path(cheap_path: str, apw_precision: float) -> str:
    """Replace the precision directory component with precision_1_0."""
    prec_str = f"precision_{str(apw_precision).replace('.', '_')}"
    return cheap_path.replace(prec_str, "precision_1_0")


def compute_tanimoto(
    energies1_eV: np.ndarray,
    dos1: np.ndarray,
    energies2_eV: np.ndarray,
    dos2: np.ndarray,
    grid_id: str = DEFAULT_GRID_ID,
) -> float:
    """Compute the Tanimoto coefficient between two DOS curves.

    Bypasses _convert_dos (which assumes Joule input) and feeds
    pre-converted eV energies directly into the binning/fingerprinting steps.
    """
    grid = Grid.create(grid_id=grid_id)

    fp1 = DOSFingerprint()
    raw_e1, raw_d1 = fp1._integrate_to_bins(energies1_eV, dos1)
    fp1.grid_id = grid.get_grid_id()
    fp1.indices, fp1.bins = fp1._calculate_bytes(raw_e1, raw_d1, grid)

    fp2 = DOSFingerprint()
    raw_e2, raw_d2 = fp2._integrate_to_bins(energies2_eV, dos2)
    fp2.grid_id = grid.get_grid_id()
    fp2.indices, fp2.bins = fp2._calculate_bytes(raw_e2, raw_d2, grid)

    return float(fp1.get_similarity(fp2))


def add_tanimoto_to_csv(input_csv: str, output_csv: str) -> None:
    """Read delta-learning CSV, compute DOS Tanimoto for each row, write new CSV."""
    df = pd.read_csv(input_csv)
    tanimoto_values = np.full(len(df), np.nan)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Computing DOS Tanimoto"):
        cheap_path = row["path"]
        apw = row["APW_precision_path"]

        ref_path = derive_reference_path(cheap_path, apw)
        cheap_dos_xml = Path(cheap_path) / "dos.xml"
        ref_dos_xml = Path(ref_path) / "dos.xml"

        if not cheap_dos_xml.is_file() or not ref_dos_xml.is_file():
            continue

        try:
            e_cheap, d_cheap = parse_dos_to_arrays(str(cheap_dos_xml))
            e_ref, d_ref = parse_dos_to_arrays(str(ref_dos_xml))
            tanimoto_values[idx] = compute_tanimoto(e_cheap, d_cheap, e_ref, d_ref)
        except Exception as e:
            print(f"Row {idx} (ICSD={row.get('ICSD_number', '?')}): {e}")

    df["dos_tanimoto"] = tanimoto_values
    df.to_csv(output_csv, index=False)

    valid = np.sum(~np.isnan(tanimoto_values))
    print(f"Wrote {output_csv}: {valid}/{len(df)} rows with valid Tanimoto")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_csv", required=True, help="Input delta-learning CSV")
    parser.add_argument("--output_csv", required=True, help="Output CSV with dos_tanimoto column")
    args = parser.parse_args()
    add_tanimoto_to_csv(args.input_csv, args.output_csv)


if __name__ == "__main__":
    main()
