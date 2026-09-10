import json
import math

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


class _DeterministicEvaluator:
    def __init__(self, val_losses, test_losses=None):
        self.val_losses = list(val_losses)
        self.test_losses = list(test_losses or [])
        self.val_calls = 0
        self.test_calls = 0
        self.models = []
        self.splits = []

    def __call__(self, model, loader, active_targets, weights, training, device):
        split = getattr(loader.dataset[0], "split", "")
        self.models.append(id(model))
        self.splits.append(split)
        if split == "test":
            self.test_calls += 1
            value = (
                self.test_losses[self.test_calls - 1]
                if self.test_calls <= len(self.test_losses)
                else 0.0
            )
        else:
            if self.val_calls >= len(self.val_losses):
                value = self.val_losses[-1]
            else:
                value = self.val_losses[self.val_calls]
            self.val_calls += 1
        return {"selection_loss": value}


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


def test_eval_schedule_includes_final_nondivisible_boundary(tmp_path, monkeypatch):
    payload_path = tmp_path / "graphs.pt"
    output_path = tmp_path / "runs"
    _payload(payload_path)
    evaluator = _DeterministicEvaluator([10.0, 9.0, 8.0])
    monkeypatch.setattr("modelling.gnn.train_scalar_retraining._evaluate", evaluator)

    config = _config("boundary_eval")
    config["training"]["epochs"] = 5
    config["training"]["eval_interval"] = 2
    metrics = run_training(payload_path, output_path, config)

    manifest = json.loads((output_path / "boundary_eval_seed1" / "run_manifest.json").read_text())
    assert evaluator.val_calls == 4
    assert manifest["selected_epoch"] == 5
    assert manifest["completed_epochs"] == 5
    assert manifest["stopped_early"] is False
    assert manifest["early_stopping"]["enabled"] is False
    assert metrics["validation"]["selection_loss"] == 8.0


def test_strict_checkpoint_updates_under_min_delta_with_cumulative_early_stop_reset(tmp_path, monkeypatch):
    payload_path = tmp_path / "graphs.pt"
    output_path = tmp_path / "runs"
    _payload(payload_path)
    evaluator = _DeterministicEvaluator([1.0, 0.96, 0.93, 0.92])
    monkeypatch.setattr("modelling.gnn.train_scalar_retraining._evaluate", evaluator)

    config = _config("min_delta_reset")
    config["training"]["epochs"] = 4
    config["training"]["eval_interval"] = 1
    config["training"]["early_stopping_patience"] = 2
    config["training"]["early_stopping_min_delta"] = 0.05
    metrics = run_training(payload_path, output_path, config)

    manifest = json.loads((output_path / "min_delta_reset_seed1" / "run_manifest.json").read_text())
    assert manifest["selected_epoch"] == 4
    assert manifest["early_stopping"]["remaining_patience"] == 1
    assert manifest["early_stopping"]["stopped_early"] is False
    assert metrics["validation"]["selection_loss"] == 0.92

    checkpoint = torch.load(manifest["checkpoint"], map_location="cpu", weights_only=False)
    assert checkpoint["selection"]["value"] == 0.92
    assert checkpoint["resolved_config"] == manifest["resolved_config"]


def test_absent_early_stopping_preserves_full_completion_and_persists_runtime_metadata(tmp_path, monkeypatch):
    payload_path = tmp_path / "graphs.pt"
    output_path = tmp_path / "runs"
    _payload(payload_path)
    evaluator = _DeterministicEvaluator([4.0, 3.0, 2.0])
    monkeypatch.setattr("modelling.gnn.train_scalar_retraining._evaluate", evaluator)

    config = _config("no_early_stop")
    config["training"]["epochs"] = 4
    config["training"]["eval_interval"] = 2
    metrics = run_training(payload_path, output_path, config)

    manifest_path = output_path / "no_early_stop_seed1" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["stopped_early"] is False
    assert manifest["completed_epochs"] == 4
    assert manifest["early_stopping"]["enabled"] is False
    assert math.isfinite(manifest["total_runtime_seconds"])
    assert manifest["model_parameter_count"] > 0
    assert "cuda_device_name" in manifest
    assert "peak_allocated_cuda_bytes" in manifest

    resolved = manifest["resolved_config"]["training"]
    assert resolved["early_stopping_patience"] is None
    assert resolved["early_stopping_min_delta"] == 0.0
    assert manifest["early_stopping"]["enabled"] is False


@pytest.mark.parametrize(
    "bad_patience",
    [True, 1.5, "3", float("nan"), float("inf"), 0, -1],
)
def test_invalid_early_stopping_patience_rejected_before_run_directory_creation(
    tmp_path, monkeypatch, bad_patience
):
    payload_path = tmp_path / "graphs.pt"
    output_path = tmp_path / "runs"
    _payload(payload_path)
    config = _config("bad_patience")
    config["training"]["early_stopping_patience"] = bad_patience
    with pytest.raises(ValueError, match="early_stopping_patience"):
        run_training(payload_path, output_path, config)
    assert not (output_path / "bad_patience_seed1").exists()


def test_nonfinite_selection_loss_rejected_and_not_saved_as_checkpoint(tmp_path, monkeypatch):
    payload_path = tmp_path / "graphs.pt"
    output_path = tmp_path / "runs"
    _payload(payload_path)
    evaluator = _DeterministicEvaluator([float("nan"), 0.7])
    monkeypatch.setattr("modelling.gnn.train_scalar_retraining._evaluate", evaluator)

    config = _config("nonfinite_eval")
    config["training"]["epochs"] = 2
    config["training"]["eval_interval"] = 2
    with pytest.raises(ValueError, match="selection_loss"):
        run_training(payload_path, output_path, config)

    run_dir = output_path / "nonfinite_eval_seed1"
    assert run_dir.exists()
    assert not (run_dir / "nonfinite_eval_seed1_best_val_loss.pt").exists()


def test_early_stopping_true_stop_with_model_identity_split(tmp_path, monkeypatch):
    payload_path = tmp_path / "graphs.pt"
    output_path = tmp_path / "runs"
    _payload(payload_path)
    evaluator = _DeterministicEvaluator([1.0, 1.0, 1.0, 1.0, 0.5])
    monkeypatch.setattr("modelling.gnn.train_scalar_retraining._evaluate", evaluator)

    config = _config("true_stop")
    config["training"]["epochs"] = 10
    config["training"]["eval_interval"] = 2
    config["training"]["early_stopping_patience"] = 2
    config["training"]["early_stopping_min_delta"] = 0.0
    run_training(payload_path, output_path, config)

    manifest = json.loads((output_path / "true_stop_seed1" / "run_manifest.json").read_text())
    assert manifest["completed_epochs"] == 6
    assert manifest["stopped_early"] is True
    assert manifest["selected_epoch"] == 2
    assert manifest["early_stopping"]["remaining_patience"] == 0
    assert manifest["early_stopping"]["stopped_early"] is True
    assert evaluator.val_calls >= 4
    assert len(evaluator.models) >= 4
    assert evaluator.models[0] == evaluator.models[1] == evaluator.models[2]
    assert evaluator.models[-1] not in evaluator.models[:-1]
    assert evaluator.splits[0] == "val"
    assert evaluator.splits[1] == "val"
    assert evaluator.splits[2] == "val"
    assert evaluator.splits[3] == "val"


@pytest.mark.parametrize(
    "bad_min_delta",
    [True, "0.1", -0.1, float("nan"), float("inf")],
)
def test_invalid_early_stopping_min_delta_rejected_before_run_directory_creation(
    tmp_path, bad_min_delta
):
    payload_path = tmp_path / "graphs.pt"
    output_path = tmp_path / "runs"
    _payload(payload_path)
    config = _config("bad_min_delta")
    config["training"]["early_stopping_patience"] = 2
    config["training"]["early_stopping_min_delta"] = bad_min_delta
    with pytest.raises(ValueError, match="early_stopping_min_delta"):
        run_training(payload_path, output_path, config)
    assert not (output_path / "bad_min_delta_seed1").exists()
