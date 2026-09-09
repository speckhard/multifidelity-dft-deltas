"""Small, fail-closed contracts for scalar retraining and evaluation."""
from __future__ import annotations

TARGETS = {
    "energy": "delta_total_energy_per_atom",
    "gap": "delta_homo_lumo_gap",
    "volume": "delta_final_volume_per_atom",
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


def require_device(config, device):
    allow_cpu = bool(config.get("training", config).get("allow_cpu_for_tests", False))
    if str(device).startswith("cuda"):
        return
    if not allow_cpu:
        raise RuntimeError("campaign retraining requires CUDA; set allow_cpu_for_tests only for fixtures")
