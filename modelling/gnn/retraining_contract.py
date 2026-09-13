"""Small, fail-closed contracts for scalar retraining and evaluation."""
from __future__ import annotations

import torch

TARGETS = {
    "energy": "delta_total_energy_per_atom",
    "gap": "delta_homo_lumo_gap",
    "volume": "delta_final_volume_per_atom",
}

BASE_FIELDS = {
    "energy": "base_total_energy_per_atom",
    "gap": "base_homo_lumo_gap",
    "volume": "base_final_volume_per_atom",
}


def resolve_active_targets(config):
    active = config.get("training", config).get("active_targets", ["energy", "gap", "volume"])
    active = list(active)
    unknown = set(active) - set(TARGETS)
    if unknown or not active or len(set(active)) != len(active):
        raise ValueError(f"invalid active_targets: {active!r}")
    return active


def validate_fixed_splits(graphs):
    labels = [getattr(graph, "split", None) for graph in graphs]
    if set(labels) != {"train", "val", "test"} or any(labels.count(x) == 0 for x in ("train", "val", "test")):
        raise ValueError("retraining requires nonempty train, val, and test splits with exact labels")
    return {name: [g for g in graphs if g.split == name] for name in ("train", "val", "test")}


REQUIRED_REPAIRED_METADATA_KEYS = (
    "encoding_dimension",
    "selection_manifest_sha256",
    "source_db_sha256",
)
REQUIRED_SCALAR_TARGETS = ("energy", "gap", "volume")


def _is_finite_scalar(value):
    return bool(
        torch.is_tensor(value)
        and value.numel() == 1
        and torch.isfinite(value).all()
    )


def validate_repaired_payload(payload, num_precision_settings):
    if not isinstance(payload, dict):
        raise TypeError("repaired payload must be a dict")

    if "graphs" not in payload or "metadata" not in payload:
        raise ValueError("repaired payload missing required top-level keys: graphs, metadata")

    graphs = payload["graphs"]
    if not isinstance(graphs, (list, tuple)) or not graphs:
        raise ValueError("repaired payload graphs must be a non-empty sequence")

    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("repaired payload metadata must be a dict")

    missing_metadata = [
        key for key in REQUIRED_REPAIRED_METADATA_KEYS if key not in metadata
    ]
    if missing_metadata:
        raise ValueError(
            "repaired payload missing required metadata keys: "
            + ", ".join(missing_metadata)
        )

    encoding_dimension = metadata["encoding_dimension"]
    if not isinstance(encoding_dimension, int) or encoding_dimension <= 0:
        raise ValueError("repaired payload metadata encoding_dimension must be a positive integer")
    if encoding_dimension != num_precision_settings:
        raise ValueError(
            "repaired payload metadata encoding_dimension must match model.num_precision_settings"
        )

    selection_manifest_sha256 = metadata["selection_manifest_sha256"]
    source_db_sha256 = metadata["source_db_sha256"]
    if not isinstance(selection_manifest_sha256, str) or len(selection_manifest_sha256) == 0:
        raise ValueError(
            "repaired payload metadata selection_manifest_sha256 must be a non-empty string"
        )
    if not isinstance(source_db_sha256, str) or len(source_db_sha256) == 0:
        raise ValueError(
            "repaired payload metadata source_db_sha256 must be a non-empty string"
        )

    for index, graph in enumerate(graphs):
        precision_settings = getattr(graph, "precision_settings", None)
        if not torch.is_tensor(precision_settings):
            raise ValueError(f"graph {index} precision_settings must be a tensor")
        if precision_settings.shape[-1] != num_precision_settings:
            raise ValueError(
                f"graph {index} precision_settings last dimension mismatch: "
                f"{precision_settings.shape[-1]} != {num_precision_settings}"
            )
        for target_name in REQUIRED_SCALAR_TARGETS:
            target_field = TARGETS[target_name]
            base_field = BASE_FIELDS[target_name]
            target_value = getattr(graph, target_field, None)
            base_value = getattr(graph, base_field, None)
            if not _is_finite_scalar(target_value):
                raise ValueError(f"graph {index} has non-finite or missing target field: {target_field}")
            if not _is_finite_scalar(base_value):
                raise ValueError(f"graph {index} has non-finite or missing base field: {base_field}")

    return validate_fixed_splits(graphs)


def require_device(config, device):
    allow_cpu = bool(config.get("training", config).get("allow_cpu_for_tests", False))
    if str(device).startswith("cuda"):
        return
    if not allow_cpu:
        raise RuntimeError("campaign retraining requires CUDA; set allow_cpu_for_tests only for fixtures")
