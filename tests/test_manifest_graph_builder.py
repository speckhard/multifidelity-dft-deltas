import csv
import pytest
from ase import Atoms
from ase.db import connect

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
from modelling.gnn.manifest_graph_builder import ManifestError, build_manifest_graphs, canonical_precision_vector, load_manifest

DFT = ("total_energy_per_atom", "homo_lumo_gap", "sum_eigenvalues_per_atom", "xc_energy_correction_per_atom", "xc_potential_correction_per_atom", "free_atom_electrostatic_energy_per_atom", "hartree_energy_correction_per_atom", "entropy_correction_per_atom", "total_energy_T0_per_atom", "kinetic_energy_per_atom", "electrostatic_energy_per_atom", "multipole_correction_per_atom")
GEO = ("final_volume_per_atom", "relaxed_a_len", "relaxed_b_len", "relaxed_c_len", "relaxed_alpha_angle", "relaxed_beta_angle", "relaxed_gamma_angle")
TARGET = ("delta_total_energy_per_atom", "delta_homo_lumo_gap", "delta_final_volume_per_atom")

def make_row(split, energy, source):
    row = {k: "1.0" for k in DFT + GEO + TARGET}
    row.update(source_row_id=source, split=split, ICSD_number="1", formula="H2", atomic_numbers="[1, 1]", k_point_density="8", rel_setting="atomic_zora", basis_size="tier1", num_setting="light", binary_precision="2")
    row["total_energy_per_atom"] = str(energy)
    return row

@pytest.fixture
def fixture_files(tmp_path):
    db, manifest, output = tmp_path / "x.db", tmp_path / "m.csv", tmp_path / "o.pt"
    with connect(db) as handle:
        for split, energy in (("train", 1.0), ("val", 999999.0), ("test", 3.0)):
            values = {k: 1.0 for k in DFT + GEO + TARGET}; values.update(ICSD_number=1, k_point_density=8, rel_setting="atomic_zora", basis_size="tier1", num_setting="light", binary_precision=2, total_energy_per_atom=energy)
            handle.write(Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]]), **values)
    rows = [make_row("train", 1, "legacy_hf:000001"), make_row("val", 999999, "legacy_hf:000002"), make_row("test", 3, "legacy_hf:000003")]
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    return db, manifest, output

def test_build_graphs_and_train_only_scaler(fixture_files):
    payload = build_manifest_graphs(*fixture_files)
    assert payload["metadata"]["counts"] == {"train": 1, "val": 1, "test": 1}
    assert payload["metadata"]["source_db_sha256"] and payload["metadata"]["selection_manifest_sha256"]
    assert payload["dft_scaler"].mean_[0] == pytest.approx(1.0)
    graph = payload["graphs"][0]
    assert graph.cheap_dft_scalars.shape == (2, 12) and graph.cheap_geometry_scalars.shape == (1, 7)
    assert all(hasattr(graph, key) for key in TARGET)

def test_encoding_modes_and_swaps():
    row = {"normalized_num_setting": "light", "normalized_basis_size": "tier1", "k_point_density": 8}
    assert canonical_precision_vector(row, mode="corrected_encoding_12bp_plus_k").index(1.0) == 2
    assert canonical_precision_vector({**row, "normalized_basis_size": "standard"}, mode="corrected_encoding_12bp_plus_k").index(1.0) == 1
    legacy = canonical_precision_vector({"normalized_num_setting": "really_tight", "normalized_basis_size": "tier2", "k_point_density": 8}, mode="legacy_encoding_11bp_plus_k")
    assert len(legacy) == 12 and sum(legacy[:11]) == 0
    with pytest.raises(ManifestError): canonical_precision_vector(row, mode="unknown")

@pytest.mark.parametrize("bad", ["8.5", "nan", "-1"])
def test_rejects_bad_numeric_and_atomic_serialization(tmp_path, bad):
    row = make_row("train", 1, "legacy_hf:000001"); row["k_point_density"] = bad; row["atomic_numbers"] = "garbage 1"
    path = tmp_path / "bad.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=row); writer.writeheader(); writer.writerow(row)
    with pytest.raises(ManifestError): load_manifest(path)
