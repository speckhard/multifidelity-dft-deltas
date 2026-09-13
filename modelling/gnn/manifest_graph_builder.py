"""Fail-closed graph selection driven by the repaired legacy manifest.

The manifest is an inclusion list: source_row_id is resolved only to ASE's
numeric row id.  No uid or physical-field join is performed here.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import os
import re
import subprocess
from pathlib import Path

import numpy as np
from ase.formula import Formula

from .fhi_precision import (
    normalize_basis_size, normalize_num_setting, precision_from_settings,
)


class ManifestError(ValueError):
    pass


_SOURCE_RE = re.compile(r"^legacy_hf:([0-9]+)$")
_SPLITS = {"train", "val", "test"}
_MODES = {"legacy_encoding_11bp_plus_k": (12, True), "corrected_encoding_12bp_plus_k": (13, False)}
_DFT_KEYS = ("total_energy_per_atom", "homo_lumo_gap", "sum_eigenvalues_per_atom",
             "xc_energy_correction_per_atom", "xc_potential_correction_per_atom",
             "free_atom_electrostatic_energy_per_atom", "hartree_energy_correction_per_atom",
             "entropy_correction_per_atom", "total_energy_T0_per_atom", "kinetic_energy_per_atom",
             "electrostatic_energy_per_atom", "multipole_correction_per_atom")
_GEO_KEYS = ("final_volume_per_atom", "relaxed_a_len", "relaxed_b_len", "relaxed_c_len",
             "relaxed_alpha_angle", "relaxed_beta_angle", "relaxed_gamma_angle")
_TARGET_KEYS = ("delta_total_energy_per_atom", "delta_homo_lumo_gap", "delta_final_volume_per_atom")
_REQUIRED = {"source_row_id", "split", "ICSD_number", "formula", "atomic_numbers",
             "k_point_density", "rel_setting", "basis_size", "num_setting", "binary_precision",
             *_DFT_KEYS, *_GEO_KEYS, *_TARGET_KEYS}


def parse_legacy_source_row_id(value):
    match = _SOURCE_RE.fullmatch(str(value).strip())
    if not match:
        raise ManifestError(f"malformed source_row_id: {value!r}")
    parsed = int(match.group(1))
    if parsed < 1:
        raise ManifestError(f"source_row_id must be positive: {value!r}")
    return parsed


def _atomic_numbers(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        try:
            result = [int(x) for x in value]
        except (TypeError, ValueError) as exc:
            raise ManifestError(f"invalid atomic_numbers: {value!r}") from exc
        if any(x < 1 for x in result):
            raise ManifestError(f"invalid atomic_numbers: {value!r}")
        return result
    if not isinstance(value, str):
        raise ManifestError(f"invalid atomic_numbers: {value!r}")
    try:
        parsed = ast.literal_eval(str(value))
        if not isinstance(parsed, (list, tuple)):
            raise ValueError("not a sequence")
        result = [int(x) for x in parsed]
        if any(x < 1 for x in result):
            raise ValueError("invalid atomic number")
        return result
    except (ValueError, SyntaxError, TypeError):
        # Accept only the exact serialization emitted by the export scripts.
        match = re.fullmatch(r"\[\s*np\.int(?:8|16|32|64)\(\s*[1-9][0-9]*\s*\)(?:\s*,\s*np\.int(?:8|16|32|64)\(\s*[1-9][0-9]*\s*\))*\s*\]", value)
        if not match:
            raise ManifestError(f"invalid atomic_numbers: {value!r}")
        return [int(x) for x in re.findall(r"np\.int(?:8|16|32|64)\(\s*([0-9]+)\s*\)", value)]


def load_manifest(path, db_max_id=None):
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = _REQUIRED - set(reader.fieldnames or ())
        if missing:
            raise ManifestError(f"manifest missing columns: {sorted(missing)}")
        rows = list(reader)
    seen = set()
    for row in rows:
        source_id = parse_legacy_source_row_id(row["source_row_id"])
        if source_id in seen:
            raise ManifestError(f"duplicate source row id: {row['source_row_id']}")
        seen.add(source_id)
        if db_max_id is not None and source_id > int(db_max_id):
            raise ManifestError(f"source row id out of range: {source_id}")
        if row["split"] not in _SPLITS:
            raise ManifestError(f"unknown split: {row['split']!r}")
        try:
            row["_ase_id"] = source_id
            row["normalized_num_setting"] = normalize_num_setting(row["num_setting"])
            row["normalized_basis_size"] = normalize_basis_size(row["basis_size"])
            row["k_point_density"] = _integral(row["k_point_density"], "k_point_density", minimum=1)
            row["binary_precision"] = _integral(row["binary_precision"], "binary_precision", minimum=0, maximum=11)
            row["_atomic_numbers"] = _atomic_numbers(row["atomic_numbers"])
            expected = precision_from_settings(row["normalized_num_setting"], row["normalized_basis_size"])
            legacy_expected = precision_from_settings(row["normalized_num_setting"], row["normalized_basis_size"], legacy=True)
            if row["binary_precision"] not in {expected, legacy_expected}:
                raise ManifestError(f"binary_precision inconsistent with settings: {row['source_row_id']}")
            for field in _DFT_KEYS + _GEO_KEYS + _TARGET_KEYS:
                value = float(row[field])
                if not np.isfinite(value):
                    raise ManifestError(f"nonfinite {field}: {row['source_row_id']}")
        except (TypeError, ValueError) as exc:
            raise ManifestError(f"invalid manifest row {row['source_row_id']}: {exc}") from exc
    if not rows:
        raise ManifestError("manifest is empty")
    return rows


def _integral(value, name, minimum=None, maximum=None):
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        raise ManifestError(f"invalid {name}: {value!r}")
    if not np.isfinite(number) or not number.is_integer():
        raise ManifestError(f"{name} must be integral: {value!r}")
    number = int(number)
    if minimum is not None and number < minimum or maximum is not None and number > maximum:
        raise ManifestError(f"{name} out of range: {value!r}")
    return number


def canonical_precision_vector(row, k_density=None, mode="corrected_encoding_12bp_plus_k"):
    """Encode physical settings using one of two explicit, auditable modes."""
    if mode not in _MODES:
        raise ManifestError(f"unknown encoding mode: {mode!r}")
    num = row.get("normalized_num_setting") or normalize_num_setting(row.get("num_setting"))
    basis = row.get("normalized_basis_size") or normalize_basis_size(row.get("basis_size"))
    dimension, legacy = _MODES[mode]
    precision = precision_from_settings(num, basis, legacy=legacy)
    density = row.get("k_point_density") if k_density is None else k_density
    vector = [0.0] * (dimension - 1) + [float(density)]
    if not (legacy and precision == 11):
        vector[precision] = 1.0
    return vector


def _value(row, name, atoms):
    if name == "formula":
        return atoms.get_chemical_formula()
    if name == "atomic_numbers":
        return list(map(int, atoms.get_atomic_numbers()))
    return getattr(row, name, None)


def _formula_count_map(value, source_row_id, field):
    try:
        counts = Formula(str(value)).count()
    except Exception as exc:
        raise ManifestError(f"formula parse failure for {source_row_id} field {field}") from exc
    if not counts:
        raise ManifestError(f"formula parse failure for {source_row_id} field {field}")
    normalized = {}
    for element, count in counts.items():
        if count <= 0:
            raise ManifestError(f"formula parse failure for {source_row_id} field {field}")
        integer_count = int(count)
        if integer_count != count:
            raise ManifestError(f"formula parse failure for {source_row_id} field {field}")
        normalized[element] = integer_count
    return normalized


def verify_row_agreement(manifest_row, db_row, tolerance=1e-7):
    atoms = db_row.toatoms()
    checks = {"ICSD_number": "ICSD_number", "formula": "formula", "atomic_numbers": "atomic_numbers",
              "k_point_density": "k_point_density", "rel_setting": "rel_setting",
              "basis_size": "basis_size", "num_setting": "num_setting", "binary_precision": "binary_precision"}
    for manifest_name, db_name in checks.items():
        expected = manifest_row[manifest_name]
        actual = _value(db_row, db_name, atoms)
        if manifest_name == "atomic_numbers":
            expected, actual = manifest_row["_atomic_numbers"], actual
        elif manifest_name == "formula":
            expected_counts = _formula_count_map(expected, manifest_row["source_row_id"], manifest_name)
            actual_counts = _formula_count_map(actual, manifest_row["source_row_id"], manifest_name)
            if expected_counts != actual_counts:
                raise ManifestError(f"semantic mismatch {manifest_name} for {manifest_row['source_row_id']}")
            continue
        if actual is None:
            raise ManifestError(f"row {manifest_row['source_row_id']} missing DB field {db_name}")
        try:
            if not np.isclose(float(expected), float(actual), rtol=tolerance, atol=tolerance):
                raise ManifestError(f"semantic mismatch {manifest_name} for {manifest_row['source_row_id']}")
        except (ValueError, TypeError):
            if str(expected).strip().lower() != str(actual).strip().lower():
                raise ManifestError(f"semantic mismatch {manifest_name} for {manifest_row['source_row_id']}")
    # When present in both sources, compare the scalar labels that identify the
    # selected calculation.  Missing optional labels are handled by the
    # manifest schema validator, while mismatches are never silently accepted.
    for name in _DFT_KEYS + _GEO_KEYS + _TARGET_KEYS:
        if not hasattr(db_row, name):
            raise ManifestError(f"row {manifest_row['source_row_id']} missing DB field {name}")
        try:
            expected = float(manifest_row[name])
            actual = float(getattr(db_row, name))
            if not np.isfinite(actual) or not np.isclose(expected, actual, rtol=tolerance, atol=tolerance):
                raise ManifestError(f"semantic mismatch {name} for {manifest_row['source_row_id']}")
        except (TypeError, ValueError) as exc:
            raise ManifestError(f"invalid scalar {name} for {manifest_row['source_row_id']}") from exc


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest_graphs(db_path, manifest_path, output_path, cutoff=5.0, encoding_mode="corrected_encoding_12bp_plus_k"):
    """Build exactly the manifest rows and save auditable metadata."""
    import torch
    from ase.db import connect
    from torch_geometric.data import Data
    from sklearn.preprocessing import StandardScaler
    if encoding_mode not in _MODES:
        raise ManifestError(f"unknown encoding mode: {encoding_mode!r}")
    db_path, manifest_path, output_path = map(Path, (db_path, manifest_path, output_path))
    with connect(db_path) as db:
        # Do not scan the source database merely to discover its maximum id;
        # each manifest id is checked by the indexed ``db.get`` below.
        manifest = load_manifest(manifest_path)
        selected = []
        for item in manifest:
            try:
                db_row = db.get(id=item["_ase_id"])
            except Exception as exc:
                raise ManifestError(f"source row id out of range: {item['_ase_id']}") from exc
            if db_row is None:
                raise ManifestError(f"source row id out of range: {item['_ase_id']}")
            verify_row_agreement(item, db_row)
            selected.append((item, db_row))
        train_dft = np.asarray([[float(getattr(row, key)) for key in _DFT_KEYS]
                                for item, row in selected if item["split"] == "train"], dtype=float)
        train_geo = np.asarray([[float(getattr(row, key)) for key in _GEO_KEYS]
                                for item, row in selected if item["split"] == "train"], dtype=float)
        if len(train_dft) == 0:
            raise ManifestError("manifest must contain at least one train row for scaler fitting")
        dft_scaler, geo_scaler = StandardScaler().fit(train_dft), StandardScaler().fit(train_geo)
        graphs = []
        for item, db_row in selected:
            atoms = db_row.toatoms()
            pos = torch.tensor(atoms.get_positions(), dtype=torch.float)
            z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long).view(-1, 1)
            distances = torch.cdist(pos, pos)
            edge_index = (torch.nonzero((distances <= cutoff) & (distances > 0), as_tuple=False).t().contiguous())
            dft = dft_scaler.transform([[float(getattr(db_row, key)) for key in _DFT_KEYS]])
            geo = geo_scaler.transform([[float(getattr(db_row, key)) for key in _GEO_KEYS]])
            kwargs = {key: torch.tensor(float(getattr(db_row, key)), dtype=torch.float) for key in _TARGET_KEYS}
            kwargs.update({"base_total_energy_per_atom": torch.tensor(float(getattr(db_row, "total_energy_per_atom")), dtype=torch.float),
                           "base_homo_lumo_gap": torch.tensor(float(getattr(db_row, "homo_lumo_gap")), dtype=torch.float),
                           "base_final_volume_per_atom": torch.tensor(float(getattr(db_row, "final_volume_per_atom")), dtype=torch.float)})
            graphs.append(Data(x=pos, z=z, edge_index=edge_index,
                                precision_settings=torch.tensor([canonical_precision_vector(item, mode=encoding_mode)]),
                                cheap_dft_scalars=torch.tensor(dft, dtype=torch.float).repeat(len(atoms), 1),
                                cheap_geometry_scalars=torch.tensor(geo, dtype=torch.float),
                                split=item["split"], ase_id=item["_ase_id"], **kwargs))
    counts = {split: sum(row["split"] == split for row in manifest) for split in _SPLITS}
    payload = {"graphs": graphs, "manifest": manifest, "metadata": {
        "source_db_sha256": _sha256(db_path), "selection_manifest_sha256": _sha256(manifest_path),
        "code_commit": _git_commit(),
        "encoding_name": encoding_mode, "encoding_dimension": _MODES[encoding_mode][0],
        "counts": counts, "filtering_reasons": {},
        "scaler_train_only_provenance": {"fit_split": "train", "fit_count": counts["train"],
                                          "dft_columns": list(_DFT_KEYS), "geometry_columns": list(_GEO_KEYS)},
    }}
    payload["dft_scaler"], payload["geo_scaler"] = dft_scaler, geo_scaler
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, output_path)
    return payload


def _git_commit():
    repo = Path(__file__).resolve().parents[2]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
