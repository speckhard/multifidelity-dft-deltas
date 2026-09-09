import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
from torch_geometric.data import Data

from modelling.gnn.retraining_contract import (
    require_device,
    resolve_active_targets,
    validate_fixed_splits,
)
from modelling.gnn.train_scalar_retraining import (
    _outputs,
    run_training,
    smape_signed_targets,
)


def _graph(split, shift=0.0):
    return Data(
        x=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        z=torch.tensor([[1], [1]]),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        precision_settings=torch.zeros((1, 13)),
        cheap_dft_scalars=torch.zeros((2, 12)),
        cheap_geometry_scalars=torch.zeros((1, 7)),
        delta_total_energy_per_atom=torch.tensor(0.1 + shift),
        delta_homo_lumo_gap=torch.tensor(0.2 + shift),
        delta_final_volume_per_atom=torch.tensor(0.3 + shift),
        base_total_energy_per_atom=torch.tensor(-10.0),
        base_homo_lumo_gap=torch.tensor(1.0),
        base_final_volume_per_atom=torch.tensor(10.0),
        split=split,
    )


def _payload(path):
    graphs = [_graph("train"), _graph("train", 0.01), _graph("val"), _graph("test")]
    torch.save(
        {
            "graphs": graphs,
            "metadata": {
                "encoding_dimension": 13,
                "selection_manifest_sha256": "fixture-manifest",
                "source_db_sha256": "fixture-db",
            },
        },
        path,
    )


def _config(run_name, evaluate_test=False):
    return {
        "model": {
            "num_layers": 1,
            "hidden_features": 8,
            "num_precision_settings": 13,
            "use_film": True,
        },
        "training": {
            "active_targets": ["energy", "gap", "volume"],
            "allow_cpu_for_tests": True,
            "batch_size": 2,
            "epochs": 1,
            "eval_interval": 1,
            "evaluate_test_at_end": evaluate_test,
            "learning_rate": 1e-4,
            "loss_function": "smape",
            "run_name": run_name,
            "seed": 1,
        },
    }


def test_smape_uses_signed_values_without_magnitude_transform():
    prediction = torch.tensor([2.0])
    target = torch.tensor([1.0])
    assert smape_signed_targets(prediction, target) == pytest.approx(66.66666, rel=1e-4)
    assert smape_signed_targets(prediction + 10.0, target + 10.0) == pytest.approx(
        8.695652, rel=1e-4
    )


def test_output_mapping_uses_delta_gnn_contract():
    output = [
        (torch.tensor([[99.0]]), None),
        (torch.tensor([1.0]), None),
        (torch.tensor([2.0]), None),
        (torch.tensor([[3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]]), None),
    ]
    mapped = _outputs(output)
    assert mapped["energy"].item() == 1
    assert mapped["gap"].item() == 2
    assert mapped["volume"].item() == 3


def test_training_reload_and_default_test_gate(tmp_path):
    payload_path = tmp_path / "graphs.pt"
    output_path = tmp_path / "runs"
    _payload(payload_path)

    metrics = run_training(payload_path, output_path, _config("fixture_no_test"))

    assert set(metrics) == {"validation"}
    run_directory = output_path / "fixture_no_test_seed1"
    manifest = json.loads((run_directory / "run_manifest.json").read_text())
    stored_metrics = json.loads((run_directory / "final_metrics.json").read_text())
    checkpoint = torch.load(manifest["checkpoint"], map_location="cpu", weights_only=False)
    assert stored_metrics == metrics
    assert checkpoint["resolved_config"]["model"]["use_film"] is True
    assert checkpoint["selection"]["criterion"] == "validation_active_target_loss"
    assert "test" not in stored_metrics
    assert "uncertainty" not in json.dumps(stored_metrics).lower()
    assert not hasattr(_graph("train"), "delta_relaxed_atom_positions")
    assert not hasattr(_graph("train"), "delta_lattice_params")


def test_explicit_test_evaluation_and_run_collision(tmp_path):
    payload_path = tmp_path / "graphs.pt"
    output_path = tmp_path / "runs"
    config = _config("fixture_with_test", evaluate_test=True)
    _payload(payload_path)

    metrics = run_training(payload_path, output_path, config)
    assert set(metrics) == {"validation", "test"}
    with pytest.raises(FileExistsError):
        run_training(payload_path, output_path, config)


def test_nonfinite_active_target_fails_before_training(tmp_path):
    payload_path = tmp_path / "bad.pt"
    graph = _graph("train")
    graph.delta_homo_lumo_gap = torch.tensor(float("nan"))
    torch.save(
        {"graphs": [graph, _graph("val"), _graph("test")], "metadata": {"encoding_dimension": 13}},
        payload_path,
    )
    with pytest.raises(ValueError, match="gap"):
        run_training(payload_path, tmp_path / "runs", _config("bad_target"))


def test_unknown_target_strict_splits_and_cpu_guard():
    with pytest.raises(ValueError):
        resolve_active_targets({"active_targets": ["positions"]})
    with pytest.raises(ValueError):
        resolve_active_targets({"active_targets": ["energy", "energy"]})
    with pytest.raises(ValueError):
        validate_fixed_splits([type("Graph", (), {"split": "train"})()])
    with pytest.raises(RuntimeError):
        require_device({"training": {}}, "cpu")
