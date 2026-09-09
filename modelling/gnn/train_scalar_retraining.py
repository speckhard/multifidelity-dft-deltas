"""Focused scalar-only retraining runner for manifest graph payloads."""

from __future__ import annotations

import copy
import json
import os
import platform
import subprocess
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from .delta_egnn_model import DeltaGNN
from .retraining_contract import (
    TARGETS,
    require_device,
    resolve_active_targets,
    validate_fixed_splits,
)


BASE_FIELDS = {
    "energy": "base_total_energy_per_atom",
    "gap": "base_homo_lumo_gap",
    "volume": "base_final_volume_per_atom",
}


def smape_signed_targets(prediction, target, epsilon=1e-7):
    """Standard sMAPE evaluated on signed (not magnitude-transformed) values."""
    numerator = torch.abs(prediction - target)
    denominator = torch.abs(prediction) + torch.abs(target) + epsilon
    return float((200.0 * numerator / denominator).mean().item())


def _smape_loss(prediction, target, epsilon):
    numerator = torch.abs(prediction - target)
    denominator = torch.abs(prediction) + torch.abs(target) + epsilon
    return (200.0 * numerator / denominator).mean()


def _outputs(model_output):
    """Map DeltaGNN's position, energy, gap, geometry tuple to scalar means."""
    return {
        "energy": model_output[1][0],
        "gap": model_output[2][0],
        "volume": model_output[3][0][:, 0],
    }


def _model_config(config):
    source = config.get("model", {})
    return {
        "num_layers": int(source.get("num_layers", 3)),
        "hidden_features": int(source.get("hidden_features", 128)),
        "num_cheap_dft_inputs": int(source.get("num_cheap_dft_inputs", 12)),
        "num_precision_settings": int(source.get("num_precision_settings", 13)),
        "num_geo_inputs": int(source.get("num_geo_inputs", 7)),
        "max_z": int(source.get("max_z", 120)),
        "use_film": bool(source.get("use_film", True)),
    }


def _make_model(config, device):
    return DeltaGNN(**_model_config(config)).to(device)


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _atomic_torch(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _targets(batch, active_targets):
    return {name: getattr(batch, TARGETS[name]) for name in active_targets}


def _check_targets(graphs, active_targets):
    for graph in graphs:
        for name in active_targets:
            value = getattr(graph, TARGETS[name], None)
            base = getattr(graph, BASE_FIELDS[name], None)
            if value is None or not torch.isfinite(value).all():
                raise ValueError(f"nonfinite or missing active target: {name}")
            if base is None or not torch.isfinite(base).all():
                raise ValueError(f"nonfinite or missing base value: {name}")


def _target_weights(training, active_targets):
    configured = training.get("target_weights", {})
    weights = {name: float(configured.get(name, 1.0)) for name in active_targets}
    if any(not torch.isfinite(torch.tensor(value)) or value <= 0 for value in weights.values()):
        raise ValueError(f"active target weights must be finite and positive: {weights}")
    return weights


def _batch_loss(predictions, targets, active_targets, weights, training):
    loss_name = str(training.get("loss_function", "smape")).lower()
    epsilon = float(training.get("smape_epsilon", 1e-4))
    losses = []
    for name in active_targets:
        if loss_name == "smape":
            loss = _smape_loss(predictions[name], targets[name], epsilon)
        elif loss_name == "mse":
            loss = torch.nn.functional.mse_loss(predictions[name], targets[name])
        else:
            raise ValueError(f"unsupported scalar loss_function: {loss_name!r}")
        losses.append(weights[name] * loss)
    return torch.stack(losses).sum()


def _evaluate(model, loader, active_targets, weights, training, device):
    model.eval()
    collected = {
        name: {"prediction": [], "target": [], "corrected_prediction": [], "corrected_target": []}
        for name in active_targets
    }
    weighted_loss = 0.0
    graph_count = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            predictions = _outputs(model(batch))
            targets = _targets(batch, active_targets)
            loss = _batch_loss(predictions, targets, active_targets, weights, training)
            weighted_loss += float(loss.item()) * batch.num_graphs
            graph_count += batch.num_graphs
            for name in active_targets:
                base = getattr(batch, BASE_FIELDS[name])
                collected[name]["prediction"].append(predictions[name].detach())
                collected[name]["target"].append(targets[name].detach())
                collected[name]["corrected_prediction"].append((base + predictions[name]).detach())
                collected[name]["corrected_target"].append((base + targets[name]).detach())

    metrics = {"selection_loss": weighted_loss / graph_count}
    for name, values in collected.items():
        prediction = torch.cat(values["prediction"])
        target = torch.cat(values["target"])
        corrected_prediction = torch.cat(values["corrected_prediction"])
        corrected_target = torch.cat(values["corrected_target"])
        metrics[name] = {
            "mae_delta": float(torch.abs(prediction - target).mean().item()),
            "smape_delta_signed_targets": smape_signed_targets(prediction, target),
            "mae_corrected": float(torch.abs(corrected_prediction - corrected_target).mean().item()),
            "smape_corrected_signed_targets": smape_signed_targets(corrected_prediction, corrected_target),
        }
    return metrics


def run_training(payload_path, output_dir, config):
    training = config.get("training", config)
    active_targets = resolve_active_targets(config)
    payload_path = Path(payload_path).resolve()
    payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    splits = validate_fixed_splits(payload["graphs"])
    _check_targets(payload["graphs"], active_targets)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    require_device(config, device)
    seed = int(training.get("seed", 0))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    batch_size = int(training.get("batch_size", 32))
    loaders = {
        name: DataLoader(items, batch_size=batch_size, shuffle=(name == "train"))
        for name, items in splits.items()
    }
    model_config = _model_config(config)
    encoding_dimension = payload.get("metadata", {}).get("encoding_dimension")
    if encoding_dimension is not None and int(encoding_dimension) != model_config["num_precision_settings"]:
        raise ValueError(
            "dataset precision dimension does not match model: "
            f"{encoding_dimension} != {model_config['num_precision_settings']}"
        )
    normalized_config = copy.deepcopy(config)
    normalized_config["model"] = model_config
    normalized_config.setdefault("training", {})["seed"] = seed

    run_name = str(training.get("run_name", "scalar_retraining")).strip()
    if not run_name or any(character in run_name for character in "/\\"):
        raise ValueError(f"invalid run_name: {run_name!r}")
    run_directory = Path(output_dir).resolve() / f"{run_name}_seed{seed}"
    if run_directory.exists():
        raise FileExistsError(f"run directory already exists: {run_directory}")
    run_directory.mkdir(parents=True)
    checkpoint_path = run_directory / f"{run_name}_seed{seed}_best_val_loss.pt"

    model = _make_model(normalized_config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training.get("learning_rate", 1e-4)))
    target_weights = _target_weights(training, active_targets)
    epochs = int(training.get("epochs", 1))
    interval = int(training.get("eval_interval", 1))
    if epochs < 1 or interval < 1:
        raise ValueError("epochs and eval_interval must be positive")

    best = None
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in loaders["train"]:
            batch = batch.to(device)
            predictions = _outputs(model(batch))
            targets = _targets(batch, active_targets)
            loss = _batch_loss(predictions, targets, active_targets, target_weights, training)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        if epoch % interval == 0 or epoch == epochs:
            validation_metrics = _evaluate(
                model, loaders["val"], active_targets, target_weights, training, device
            )
            criterion_value = validation_metrics["selection_loss"]
            if best is None or criterion_value < best["value"]:
                best = {"value": criterion_value, "epoch": epoch}
                _atomic_torch(
                    checkpoint_path,
                    {
                        "model_state": model.state_dict(),
                        "resolved_config": normalized_config,
                        "epoch": epoch,
                        "selection": {
                            "criterion": "validation_active_target_loss",
                            "value": criterion_value,
                        },
                        "dataset_metadata": payload.get("metadata", {}),
                        "code_commit": _git_commit(),
                    },
                )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    fresh_model = _make_model(checkpoint["resolved_config"], device)
    fresh_model.load_state_dict(checkpoint["model_state"])
    final_metrics = {
        "validation": _evaluate(
            fresh_model, loaders["val"], active_targets, target_weights, training, device
        )
    }
    if bool(training.get("evaluate_test_at_end", False)):
        final_metrics["test"] = _evaluate(
            fresh_model, loaders["test"], active_targets, target_weights, training, device
        )

    run_manifest = {
        "active_targets": active_targets,
        "checkpoint": str(checkpoint_path),
        "code_commit": _git_commit(),
        "dataset_metadata": payload.get("metadata", {}),
        "dataset_path": str(payload_path),
        "device": str(device),
        "platform": platform.platform(),
        "resolved_config": normalized_config,
        "selected_epoch": checkpoint["epoch"],
        "selection": checkpoint["selection"],
        "torch_version": torch.__version__,
    }
    _atomic_json(run_directory / "final_metrics.json", final_metrics)
    _atomic_json(run_directory / "run_manifest.json", run_manifest)
    return final_metrics


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
