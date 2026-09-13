import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from omegaconf import OmegaConf
from unittest.mock import MagicMock

from modelling.gnn import train_pipeline


def _graph(split, precision_dim=13, target_nan=False):
    value = float("nan") if target_nan else 0.1
    return Data(
        x=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32),
        z=torch.tensor([[1], [1]], dtype=torch.long),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        num_nodes=2,
        precision_settings=torch.zeros((1, precision_dim), dtype=torch.float32),
        cheap_dft_scalars=torch.zeros((2, 12), dtype=torch.float32),
        cheap_geometry_scalars=torch.zeros((1, 7), dtype=torch.float32),
        delta_total_energy_per_atom=torch.tensor([value], dtype=torch.float32),
        delta_homo_lumo_gap=torch.tensor([value], dtype=torch.float32),
        delta_final_volume_per_atom=torch.tensor([value], dtype=torch.float32),
        base_total_energy_per_atom=torch.tensor([0.0], dtype=torch.float32),
        base_homo_lumo_gap=torch.tensor([0.0], dtype=torch.float32),
        base_final_volume_per_atom=torch.tensor([0.0], dtype=torch.float32),
        split=split,
    )


def _config(data_file, strict=False):
    return OmegaConf.create(
        {
            "data_file": str(data_file),
            "strict_repaired_data": strict,
            "wandb_project": "test_project",
            "wandb_group": "test_group",
            "job_name": "test_job",
            "model": {
                "num_layers": 3,
                "hidden_features": 32,
                "hidden_dim": 32,
                "num_precision_settings": 13,
                "num_cheap_dft_inputs": 12,
                "num_geo_inputs": 7,
                "max_z": 100,
            },
            "training": {"batch_size": 1, "epochs": 1, "learning_rate": 1e-3},
            "weights": {
                "delta_e": 1.0,
                "delta_gap": 1.0,
                "delta_r": 1.0,
                "delta_geo": 1.0,
            },
        }
    )


def _valid_payload():
    return {
        "graphs": [_graph("train"), _graph("val"), _graph("test")],
        "metadata": {
            "encoding_dimension": 13,
            "selection_manifest_sha256": "manifest-sha",
            "source_db_sha256": "source-sha",
        },
    }


def test_split_graphs_legacy_preserves_fallback_behavior():
    graphs = [_graph("train"), _graph("train"), _graph("test")]
    split = train_pipeline._split_graphs_for_training(graphs, strict_mode=False)
    assert len(split["train"]) == 1
    assert len(split["val"]) == 1
    assert len(split["test"]) == 1


def test_preflight_rejects_invalid_strict_payload_before_wandb_init(tmp_path, monkeypatch):
    cfg = _config(tmp_path / "missing_metadata.pt", strict=True)
    payload = _valid_payload()
    payload["metadata"].pop("selection_manifest_sha256")

    monkeypatch.setattr(train_pipeline.os.path, "exists", lambda _: True)
    monkeypatch.setattr(train_pipeline.torch, "load", lambda *args, **kwargs: payload)

    called = {"init": 0}

    def init(*args, **kwargs):
        called["init"] += 1
        return MagicMock()

    monkeypatch.setattr(train_pipeline.wandb, "init", init)

    with pytest.raises(ValueError, match="missing required metadata keys"):
        train_pipeline._prepare_training_payload_and_init(cfg)

    assert called["init"] == 0


def test_preflight_rejects_invalid_precision_dimension_before_wandb_init(tmp_path, monkeypatch):
    cfg = _config(tmp_path / "bad_precision.pt", strict=True)
    payload = _valid_payload()
    payload["graphs"][0].precision_settings = torch.zeros((1, 12))

    monkeypatch.setattr(train_pipeline.os.path, "exists", lambda _: True)
    monkeypatch.setattr(train_pipeline.torch, "load", lambda *args, **kwargs: payload)
    called = {"init": False}

    def init(*args, **kwargs):
        called["init"] = True
        return MagicMock()

    monkeypatch.setattr(train_pipeline.wandb, "init", init)

    with pytest.raises(ValueError, match="precision_settings"):
        train_pipeline._prepare_training_payload_and_init(cfg)

    assert called["init"] is False


def test_preflight_rejects_non_integer_encoding_dimension_before_wandb_init(tmp_path, monkeypatch):
    cfg = _config(tmp_path / "bad_encoding_type.pt", strict=True)
    payload = _valid_payload()
    payload["metadata"]["encoding_dimension"] = "13"

    monkeypatch.setattr(train_pipeline.os.path, "exists", lambda _: True)
    monkeypatch.setattr(train_pipeline.torch, "load", lambda *args, **kwargs: payload)
    called = {"init": False}

    def init(*args, **kwargs):
        called["init"] = True
        return MagicMock()

    monkeypatch.setattr(train_pipeline.wandb, "init", init)

    with pytest.raises(ValueError, match="positive integer"):
        train_pipeline._prepare_training_payload_and_init(cfg)

    assert called["init"] is False


def test_preflight_rejects_encoding_dimension_mismatch_before_wandb_init(tmp_path, monkeypatch):
    cfg = _config(tmp_path / "bad_encoding_dim.pt", strict=True)
    payload = _valid_payload()
    payload["metadata"]["encoding_dimension"] = 12

    monkeypatch.setattr(train_pipeline.os.path, "exists", lambda _: True)
    monkeypatch.setattr(train_pipeline.torch, "load", lambda *args, **kwargs: payload)
    called = {"init": False}

    def init(*args, **kwargs):
        called["init"] = True
        return MagicMock()

    monkeypatch.setattr(train_pipeline.wandb, "init", init)

    with pytest.raises(ValueError, match="match model.num_precision_settings"):
        train_pipeline._prepare_training_payload_and_init(cfg)

    assert called["init"] is False


def test_preflight_rejects_empty_metadata_hashes_before_wandb_init(tmp_path, monkeypatch):
    cfg = _config(tmp_path / "empty_hash.pt", strict=True)
    payload = _valid_payload()
    payload["metadata"]["selection_manifest_sha256"] = ""

    monkeypatch.setattr(train_pipeline.os.path, "exists", lambda _: True)
    monkeypatch.setattr(train_pipeline.torch, "load", lambda *args, **kwargs: payload)
    called = {"init": False}

    def init(*args, **kwargs):
        called["init"] = True
        return MagicMock()

    monkeypatch.setattr(train_pipeline.wandb, "init", init)

    with pytest.raises(ValueError, match="selection_manifest_sha256"):
        train_pipeline._prepare_training_payload_and_init(cfg)

    assert called["init"] is False


def test_preflight_rejects_source_db_hash_empty_before_wandb_init(tmp_path, monkeypatch):
    cfg = _config(tmp_path / "empty_hash_source.pt", strict=True)
    payload = _valid_payload()
    payload["metadata"]["source_db_sha256"] = ""

    monkeypatch.setattr(train_pipeline.os.path, "exists", lambda _: True)
    monkeypatch.setattr(train_pipeline.torch, "load", lambda *args, **kwargs: payload)
    called = {"init": False}

    def init(*args, **kwargs):
        called["init"] = True
        return MagicMock()

    monkeypatch.setattr(train_pipeline.wandb, "init", init)

    with pytest.raises(ValueError, match="source_db_sha256"):
        train_pipeline._prepare_training_payload_and_init(cfg)

    assert called["init"] is False


def test_preflight_rejects_nonfinite_scalar_fields_before_wandb_init(tmp_path, monkeypatch):
    cfg = _config(tmp_path / "bad_values.pt", strict=True)
    payload = _valid_payload()
    payload["graphs"][0] = _graph("train", target_nan=True)

    monkeypatch.setattr(train_pipeline.os.path, "exists", lambda _: True)
    monkeypatch.setattr(train_pipeline.torch, "load", lambda *args, **kwargs: payload)
    called = {"init": False}

    def init(*args, **kwargs):
        called["init"] = True
        return MagicMock()

    monkeypatch.setattr(train_pipeline.wandb, "init", init)

    with pytest.raises(ValueError, match="non-finite or missing target field"):
        train_pipeline._prepare_training_payload_and_init(cfg)

    assert called["init"] is False


def test_preflight_rejects_non_scalar_fields_before_wandb_init(tmp_path, monkeypatch):
    cfg = _config(tmp_path / "bad_scalar.pt", strict=True)
    payload = _valid_payload()
    payload["graphs"][0].delta_total_energy_per_atom = torch.tensor([0.1, 0.2])

    monkeypatch.setattr(train_pipeline.os.path, "exists", lambda _: True)
    monkeypatch.setattr(train_pipeline.torch, "load", lambda *args, **kwargs: payload)
    called = {"init": False}

    def init(*args, **kwargs):
        called["init"] = True
        return MagicMock()

    monkeypatch.setattr(train_pipeline.wandb, "init", init)

    with pytest.raises(ValueError, match="non-finite or missing target field"):
        train_pipeline._prepare_training_payload_and_init(cfg)

    assert called["init"] is False


def _full_graph(split, precision_dim=13, pos_val=1.0, lat_val=1.0):
    return Data(
        x=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32),
        z=torch.tensor([[1], [1]], dtype=torch.long),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        num_nodes=2,
        precision_settings=torch.zeros((1, precision_dim), dtype=torch.float32),
        cheap_dft_scalars=torch.zeros((2, 12), dtype=torch.float32),
        cheap_geometry_scalars=torch.zeros((1, 7), dtype=torch.float32),
        delta_total_energy_per_atom=torch.tensor([0.1], dtype=torch.float32),
        delta_homo_lumo_gap=torch.tensor([0.2], dtype=torch.float32),
        delta_final_volume_per_atom=torch.tensor([0.3], dtype=torch.float32),
        delta_relaxed_atom_positions=(torch.ones((2, 3), dtype=torch.float32) * pos_val),
        delta_lattice_params=(torch.ones((1, 6), dtype=torch.float32) * lat_val),
        base_total_energy_per_atom=torch.tensor([0.0], dtype=torch.float32),
        base_homo_lumo_gap=torch.tensor([0.0], dtype=torch.float32),
        base_final_volume_per_atom=torch.tensor([0.0], dtype=torch.float32),
        split=split,
    )


def _scalar_only_graph(split, precision_dim=13, energy=1.0, gap=2.0, volume=4.0):
    return Data(
        x=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32),
        z=torch.tensor([[1], [1]], dtype=torch.long),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        num_nodes=2,
        precision_settings=torch.zeros((1, precision_dim), dtype=torch.float32),
        cheap_dft_scalars=torch.zeros((2, 12), dtype=torch.float32),
        cheap_geometry_scalars=torch.zeros((1, 7), dtype=torch.float32),
        delta_total_energy_per_atom=torch.tensor([energy], dtype=torch.float32),
        delta_homo_lumo_gap=torch.tensor([gap], dtype=torch.float32),
        delta_final_volume_per_atom=torch.tensor([volume], dtype=torch.float32),
        base_total_energy_per_atom=torch.tensor([0.0], dtype=torch.float32),
        base_homo_lumo_gap=torch.tensor([0.0], dtype=torch.float32),
        base_final_volume_per_atom=torch.tensor([0.0], dtype=torch.float32),
        split=split,
    )


class _StrictScalarModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._pipeline_sentinel = torch.nn.Parameter(torch.zeros(()))

    def forward(self, data):
        n_nodes = data.x.shape[0]
        batch_size = data.num_graphs
        device = data.x.device

        pos = torch.zeros((n_nodes, 3), device=device)
        pos_logvar = torch.zeros((n_nodes, 1), device=device)
        e = torch.tensor([2.0] * batch_size, device=device)
        e_logvar = torch.zeros(batch_size, device=device)
        g = torch.tensor([4.0] * batch_size, device=device)
        g_logvar = torch.zeros(batch_size, device=device)
        geo = torch.stack([
            torch.tensor([8.0] * batch_size, device=device),
            torch.tensor([0.0] * batch_size, device=device),
            torch.tensor([0.0] * batch_size, device=device),
            torch.tensor([0.0] * batch_size, device=device),
            torch.tensor([0.0] * batch_size, device=device),
            torch.tensor([0.0] * batch_size, device=device),
            torch.tensor([0.0] * batch_size, device=device),
        ], dim=1)
        geo_logvar = torch.zeros((batch_size, 7), device=device)
        return ( (pos, pos_logvar), (e, e_logvar), (g, g_logvar), (geo, geo_logvar) )


class _LegacyModel(_StrictScalarModel):
    def __init__(self):
        super().__init__()

    def forward(self, data):
        n_nodes = data.x.shape[0]
        batch_size = data.num_graphs
        device = data.x.device

        pos = torch.zeros((n_nodes, 3), device=device)
        pos_logvar = torch.zeros((n_nodes, 1), device=device)
        e = torch.tensor([1.0] * batch_size, device=device)
        e_logvar = torch.zeros(batch_size, device=device)
        g = torch.tensor([2.0] * batch_size, device=device)
        g_logvar = torch.zeros(batch_size, device=device)
        geo = torch.ones((batch_size, 7), device=device)
        geo_logvar = torch.zeros((batch_size, 7), device=device)
        return ( (pos, pos_logvar), (e, e_logvar), (g, g_logvar), (geo, geo_logvar) )


def _smape_scalar(value, target, epsilon=1e-7):
    return 200.0 * abs(value - target) / (abs(value) + abs(target) + epsilon)


def test_evaluate_strict_scalar_only_returns_scalar_smape_mean_and_no_lattice_metrics(tmp_path):
    train_graph = _scalar_only_graph("train")
    val_graph = _scalar_only_graph("val", energy=2.0, gap=4.0, volume=8.0)
    loader = DataLoader([train_graph, val_graph], batch_size=2, shuffle=False)

    class _Weights:
        delta_r = 1.0
        delta_e = 1.0
        delta_gap = 1.0
        delta_geo = 1.0
        delta_vol = 1.0
        delta_lat = 1.0

    metrics, _ = train_pipeline.evaluate(
        _StrictScalarModel(),
        loader,
        train_pipeline.PlainL1Loss(),
        _Weights(),
        strict_mode=True,
    )

    expected = (
        _smape_scalar(2.0, 1.0) +
        _smape_scalar(4.0, 2.0) +
        _smape_scalar(8.0, 4.0) +
        _smape_scalar(1.0, 1.0) +
        _smape_scalar(2.0, 2.0) +
        _smape_scalar(4.0, 4.0)
    ) / 6.0

    assert metrics["sMAPE_scalar_mean"] == pytest.approx(expected, rel=1e-7)
    assert metrics["sMAPE_Scalar_Mean"] == pytest.approx(expected, rel=1e-7)
    assert metrics["MAE_delta_positions"] == float("inf")
    assert metrics["sMAPE_delta_a"] == float("inf")
    assert metrics["RMSLAE_delta_b"] == float("inf")


def test_main_supports_plain_l1_loss_and_strict_scalar_mode(monkeypatch, tmp_path):
    called = {"criterion": None, "strict": False}

    def fake_train_step(model, loader, optimizer, criterion, weights, device, strict_mode=False):
        called["criterion"] = type(criterion).__name__
        called["strict"] = strict_mode
        return 1.0

    def _base_metrics():
        return {
            "Val_Loss": 1.0,
            "MAE_delta_energy": 1.0,
            "MAE_delta_gap": 1.0,
            "MAE_delta_positions": 1.0,
            "MAE_delta_geo_total": 1.0,
            "MAE_delta_volume": 1.0,
            "MAE_delta_a": 1.0,
            "MAE_delta_b": 1.0,
            "MAE_delta_c": 1.0,
            "MAE_delta_alpha": 1.0,
            "MAE_delta_beta": 1.0,
            "MAE_delta_gamma": 1.0,
            "sMAPE_scalar_mean": 1.0,
            "sMAPE_Scalar_Mean": 1.0,
            "sMAPE_delta_energy": 1.0,
            "sMAPE_delta_gap": 1.0,
            "sMAPE_delta_positions": 1.0,
            "sMAPE_delta_volume": 1.0,
            "sMAPE_delta_a": 1.0,
            "sMAPE_delta_b": 1.0,
            "sMAPE_delta_c": 1.0,
            "sMAPE_delta_alpha": 1.0,
            "sMAPE_delta_beta": 1.0,
            "sMAPE_delta_gamma": 1.0,
            "RMSLAE_delta_energy": 1.0,
            "RMSLAE_delta_gap": 1.0,
            "RMSLAE_delta_positions": 1.0,
            "RMSLAE_delta_volume": 1.0,
            "RMSLAE_delta_a": 1.0,
            "RMSLAE_delta_b": 1.0,
            "RMSLAE_delta_c": 1.0,
            "RMSLAE_delta_alpha": 1.0,
            "RMSLAE_delta_beta": 1.0,
            "RMSLAE_delta_gamma": 1.0,
            "MagAcc_delta_energy": 1.0,
            "MagAcc_delta_gap": 1.0,
            "MagAcc_delta_positions": 1.0,
            "MagAcc_delta_volume": 1.0,
            "MagAcc_delta_a": 1.0,
            "MagAcc_delta_b": 1.0,
            "MagAcc_delta_c": 1.0,
            "MagAcc_delta_alpha": 1.0,
            "MagAcc_delta_beta": 1.0,
            "MagAcc_delta_gamma": 1.0,
        }

    def fake_evaluate(model, loader, criterion, weights, strict_mode=False):
        return _base_metrics(), {}

    def fake_payload_and_init(cfg):
        return {
            "train": [_scalar_only_graph("train"), _scalar_only_graph("train")],
            "val": [_scalar_only_graph("val")],
            "test": [_scalar_only_graph("test")],
        }

    run = MagicMock()
    run.summary = {}
    monkeypatch.setattr(train_pipeline, "_prepare_training_payload_and_init", fake_payload_and_init)
    monkeypatch.setattr(train_pipeline.wandb, "init", lambda *args, **kwargs: run)
    monkeypatch.setattr(train_pipeline.wandb, "run", run)
    monkeypatch.setattr(train_pipeline.wandb, "log", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_pipeline.wandb, "save", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_pipeline.wandb, "finish", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_pipeline.wandb, "Image", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_pipeline, "train_step", fake_train_step)
    monkeypatch.setattr(train_pipeline, "evaluate", fake_evaluate)
    monkeypatch.setattr(train_pipeline, "log_element_embeddings", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_pipeline, "plot_history", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_pipeline, "plot_log_regression", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_pipeline, "plot_uncertainty_calibration", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_pipeline, "save_history", lambda *args, **kwargs: None)

    class _DummyModel(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.p = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, batch):
            raise RuntimeError("forward should not be called in this test")

    monkeypatch.setattr(train_pipeline, "DeltaGNN", _DummyModel)

    cfg = _config(tmp_path / "strict_main.pt")
    cfg.strict_repaired_data = True
    cfg.training = OmegaConf.create({
        "batch_size": 2,
        "epochs": 12,
        "learning_rate": 1e-3,
        "loss_function": "l1",
        "eval_interval": 1,
        "early_stopping_patience": 20,
        "early_stopping_min_delta": 1e-5,
        "active_targets": ["energy", "gap", "volume"],
    })

    train_pipeline.main(cfg)
    assert called["criterion"] == "PlainL1Loss"
    assert called["strict"] is True


def test_main_eval_interval_and_early_stop_config_respected(monkeypatch, tmp_path):
    calls = {"train_step": 0, "evaluate": 0}

    def fake_train_step(*args, **kwargs):
        calls["train_step"] += 1
        return 1.0

    metrics = [
        {"Val_Loss": 0.9, "sMAPE_Scalar_Mean": 10.0},
    ]
    # first epoch is best, then one regression triggers early-stop at patience=1
    def fake_evaluate(*args, **kwargs):
        calls["evaluate"] += 1
        idx = min(calls["evaluate"] - 1, len(metrics) - 1)
        base = metrics[0].copy()
        if idx > 0:
            base["Val_Loss"] += 1.0
        base["sMAPE_Scalar_Mean"] = 100.0 + idx
        value = base.copy()
        value.update({
            "MAE_delta_energy": 1.0, "MAE_delta_gap": 1.0, "MAE_delta_positions": 1.0, "MAE_delta_geo_total": 1.0,
            "MAE_delta_volume": 1.0, "MAE_delta_a": 1.0, "MAE_delta_b": 1.0, "MAE_delta_c": 1.0,
            "MAE_delta_alpha": 1.0, "MAE_delta_beta": 1.0, "MAE_delta_gamma": 1.0,
            "sMAPE_scalar_mean": 1.0, "sMAPE_delta_energy": 1.0, "sMAPE_delta_gap": 1.0, "sMAPE_delta_positions": 1.0,
            "sMAPE_delta_volume": 1.0, "sMAPE_delta_a": 1.0, "sMAPE_delta_b": 1.0, "sMAPE_delta_c": 1.0,
            "sMAPE_delta_alpha": 1.0, "sMAPE_delta_beta": 1.0, "sMAPE_delta_gamma": 1.0,
            "RMSLAE_delta_energy": 1.0, "RMSLAE_delta_gap": 1.0, "RMSLAE_delta_positions": 1.0, "RMSLAE_delta_volume": 1.0,
            "RMSLAE_delta_a": 1.0, "RMSLAE_delta_b": 1.0, "RMSLAE_delta_c": 1.0, "RMSLAE_delta_alpha": 1.0,
            "RMSLAE_delta_beta": 1.0, "RMSLAE_delta_gamma": 1.0,
            "MagAcc_delta_energy": 1.0, "MagAcc_delta_gap": 1.0, "MagAcc_delta_positions": 1.0, "MagAcc_delta_volume": 1.0,
            "MagAcc_delta_a": 1.0, "MagAcc_delta_b": 1.0, "MagAcc_delta_c": 1.0, "MagAcc_delta_alpha": 1.0,
            "MagAcc_delta_beta": 1.0, "MagAcc_delta_gamma": 1.0,
        })
        return value, {}

    def fake_payload_and_init(cfg):
        return {
            "train": [_full_graph("train"), _full_graph("train")],
            "val": [_full_graph("val")],
            "test": [_full_graph("test")],
        }

    run = MagicMock()
    run.summary = {}
    monkeypatch.setattr(train_pipeline, "_prepare_training_payload_and_init", fake_payload_and_init)
    monkeypatch.setattr(train_pipeline.wandb, "init", lambda *args, **kwargs: run)
    monkeypatch.setattr(train_pipeline.wandb, "run", run)
    monkeypatch.setattr(train_pipeline.wandb, "log", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_pipeline.wandb, "save", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_pipeline.wandb, "finish", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_pipeline.wandb, "Image", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_pipeline, "train_step", fake_train_step)
    monkeypatch.setattr(train_pipeline, "evaluate", fake_evaluate)
    monkeypatch.setattr(train_pipeline, "log_element_embeddings", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_pipeline, "plot_history", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_pipeline, "plot_log_regression", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_pipeline, "plot_uncertainty_calibration", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_pipeline, "save_history", lambda *args, **kwargs: None)

    class _DummyModel(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.p = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, batch):
            raise RuntimeError("forward should not be called in this test")

    monkeypatch.setattr(train_pipeline, "DeltaGNN", _DummyModel)

    cfg = _config(tmp_path / "schedule_main.pt")
    cfg.strict_repaired_data = False
    cfg.training = OmegaConf.create({
        "batch_size": 2,
        "epochs": 12,
        "learning_rate": 1e-3,
        "loss_function": "smape",
        "eval_interval": 2,
        "early_stopping_patience": 1,
        "early_stopping_min_delta": 1e-5,
        "early_stop_metric": "Val/sMAPE_Scalar_Mean",
        "active_targets": ["energy", "gap", "volume"],
    })

    train_pipeline.main(cfg)
    assert calls["train_step"] == 2
    assert calls["evaluate"] == 4


def test_main_does_not_evaluate_test_loader_when_evaluate_test_at_end_is_false(monkeypatch, tmp_path):
    calls = {
        "train_step": 0,
        "evaluate": 0,
        "val_evaluate": 0,
        "test_evaluate": 0,
        "train_loader": 0,
        "val_loader": 0,
        "test_loader": 0,
    }
    real_dataloader = train_pipeline.DataLoader

    def fake_train_step(*args, **kwargs):
        calls["train_step"] += 1
        return 1.0

    def _fake_metric_payload():
        return {
            "Val_Loss": 1.0,
            "sMAPE_Scalar_Mean": 1.0,
            "sMAPE_scalar_mean": 1.0,
            "MAE_delta_energy": 1.0, "MAE_delta_gap": 1.0, "MAE_delta_positions": 1.0, "MAE_delta_geo_total": 1.0,
            "MAE_delta_volume": 1.0, "MAE_delta_a": 1.0, "MAE_delta_b": 1.0, "MAE_delta_c": 1.0,
            "MAE_delta_alpha": 1.0, "MAE_delta_beta": 1.0, "MAE_delta_gamma": 1.0,
            "sMAPE_delta_energy": 1.0, "sMAPE_delta_gap": 1.0, "sMAPE_delta_positions": 1.0, "sMAPE_delta_volume": 1.0,
            "sMAPE_delta_a": 1.0, "sMAPE_delta_b": 1.0, "sMAPE_delta_c": 1.0, "sMAPE_delta_alpha": 1.0, "sMAPE_delta_beta": 1.0, "sMAPE_delta_gamma": 1.0,
            "RMSLAE_delta_energy": 1.0, "RMSLAE_delta_gap": 1.0, "RMSLAE_delta_positions": 1.0, "RMSLAE_delta_volume": 1.0,
            "RMSLAE_delta_a": 1.0, "RMSLAE_delta_b": 1.0, "RMSLAE_delta_c": 1.0, "RMSLAE_delta_alpha": 1.0,
            "RMSLAE_delta_beta": 1.0, "RMSLAE_delta_gamma": 1.0,
            "MagAcc_delta_energy": 1.0, "MagAcc_delta_gap": 1.0, "MagAcc_delta_positions": 1.0, "MagAcc_delta_volume": 1.0,
            "MagAcc_delta_a": 1.0, "MagAcc_delta_b": 1.0, "MagAcc_delta_c": 1.0, "MagAcc_delta_alpha": 1.0,
            "MagAcc_delta_beta": 1.0, "MagAcc_delta_gamma": 1.0,
        }

    def fake_evaluate(model, loader, criterion, weights, strict_mode=False):
        calls["evaluate"] += 1
        split = getattr(loader.dataset[0], "split", "")
        if split == "test":
            calls["test_evaluate"] += 1
        elif split == "val":
            calls["val_evaluate"] += 1
        else:
            calls["val_evaluate"] += 1
        return _fake_metric_payload(), {}

    def fake_dataloader(dataset, *args, **kwargs):
        split = getattr(dataset[0], "split", "unknown")
        if split == "train":
            calls["train_loader"] += 1
        elif split == "val":
            calls["val_loader"] += 1
        elif split == "test":
            calls["test_loader"] += 1
        return real_dataloader(dataset, *args, **kwargs)

    def fake_payload_and_init(cfg):
        return {
            "train": [_full_graph("train"), _full_graph("train")],
            "val": [_full_graph("val")],
            "test": [_full_graph("test")],
        }

    run = MagicMock()
    run.summary = {}
    monkeypatch.setattr(train_pipeline, "_prepare_training_payload_and_init", fake_payload_and_init)
    monkeypatch.setattr(train_pipeline.wandb, "init", lambda *args, **kwargs: run)
    monkeypatch.setattr(train_pipeline.wandb, "run", run)
    monkeypatch.setattr(train_pipeline.wandb, "log", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_pipeline.wandb, "save", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_pipeline.wandb, "finish", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_pipeline.wandb, "Image", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_pipeline, "train_step", fake_train_step)
    monkeypatch.setattr(train_pipeline, "DataLoader", fake_dataloader)
    monkeypatch.setattr(train_pipeline, "evaluate", fake_evaluate)
    monkeypatch.setattr(train_pipeline, "log_element_embeddings", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_pipeline, "plot_history", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_pipeline, "plot_log_regression", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_pipeline, "plot_uncertainty_calibration", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_pipeline, "save_history", lambda *args, **kwargs: None)

    class _DummyModel(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.p = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, batch):
            raise RuntimeError("forward should not be called in this test")

    monkeypatch.setattr(train_pipeline, "DeltaGNN", _DummyModel)

    cfg = _config(tmp_path / "no_test_eval_main.pt")
    cfg.training = OmegaConf.create({
        "batch_size": 2,
        "epochs": 2,
        "learning_rate": 1e-3,
        "loss_function": "smape",
        "eval_interval": 1,
        "early_stopping_patience": 10,
        "early_stopping_min_delta": 0.0,
        "early_stop_metric": "Val/sMAPE_Scalar_Mean",
        "evaluate_test_at_end": False,
        "active_targets": ["energy", "gap", "volume"],
    })

    train_pipeline.main(cfg)
    assert calls["train_step"] == 2
    assert calls["evaluate"] == 4
    assert calls["val_evaluate"] == 4
    assert calls["test_evaluate"] == 0
    assert calls["train_loader"] == 1
    assert calls["val_loader"] == 1
    assert calls["test_loader"] == 0


def test_main_evaluate_test_at_end_true_restores_best_checkpoint_and_runs_test(monkeypatch, tmp_path):
    calls = {
        "train_step": 0,
        "evaluate": 0,
        "val_evaluate": 0,
        "test_evaluate": 0,
        "load": 0,
        "save": [],
        "test_loader": 0,
        "load_strict": [],
        "sequence": [],
    }
    real_dataloader = train_pipeline.DataLoader
    saved_checkpoint = {"path": None}

    def fake_train_step(*args, **kwargs):
        calls["train_step"] += 1
        return 1.0

    def _fake_metric_payload(value):
        base = {
            "Val_Loss": 1.0,
            "sMAPE_Scalar_Mean": value,
            "sMAPE_scalar_mean": value,
            "MAE_delta_energy": 1.0, "MAE_delta_gap": 1.0, "MAE_delta_positions": 1.0, "MAE_delta_geo_total": 1.0,
            "MAE_delta_volume": 1.0, "MAE_delta_a": 1.0, "MAE_delta_b": 1.0, "MAE_delta_c": 1.0,
            "MAE_delta_alpha": 1.0, "MAE_delta_beta": 1.0, "MAE_delta_gamma": 1.0,
            "sMAPE_delta_energy": 1.0, "sMAPE_delta_gap": 1.0, "sMAPE_delta_positions": 1.0, "sMAPE_delta_volume": 1.0,
            "sMAPE_delta_a": 1.0, "sMAPE_delta_b": 1.0, "sMAPE_delta_c": 1.0, "sMAPE_delta_alpha": 1.0, "sMAPE_delta_beta": 1.0, "sMAPE_delta_gamma": 1.0,
            "RMSLAE_delta_energy": 1.0, "RMSLAE_delta_gap": 1.0, "RMSLAE_delta_positions": 1.0, "RMSLAE_delta_volume": 1.0,
            "RMSLAE_delta_a": 1.0, "RMSLAE_delta_b": 1.0, "RMSLAE_delta_c": 1.0, "RMSLAE_delta_alpha": 1.0,
            "RMSLAE_delta_beta": 1.0, "RMSLAE_delta_gamma": 1.0,
            "MagAcc_delta_energy": 1.0, "MagAcc_delta_gap": 1.0, "MagAcc_delta_positions": 1.0, "MagAcc_delta_volume": 1.0,
            "MagAcc_delta_a": 1.0, "MagAcc_delta_b": 1.0, "MagAcc_delta_c": 1.0, "MagAcc_delta_alpha": 1.0,
            "MagAcc_delta_beta": 1.0, "MagAcc_delta_gamma": 1.0,
        }
        return base

    def fake_evaluate(model, loader, criterion, weights, strict_mode=False):
        calls["evaluate"] += 1
        split = getattr(loader.dataset[0], "split", "")
        if split == "test":
            calls["test_evaluate"] += 1
            calls["sequence"].append("evaluate:test")
            payload = _fake_metric_payload(3.0)
        elif split == "val":
            payload = _fake_metric_payload(2.0 if calls["val_evaluate"] == 0 else 1.5)
            calls["val_evaluate"] += 1
        else:
            payload = _fake_metric_payload(2.0)
            calls["val_evaluate"] += 1
        return payload, {}

    def fake_dataloader(dataset, *args, **kwargs):
        split = getattr(dataset[0], "split", "unknown")
        if split == "test":
            calls["test_loader"] += 1
            calls["sequence"].append("dataloader:test")
        return real_dataloader(dataset, *args, **kwargs)

    def fake_payload_and_init(cfg):
        return {
            "train": [_full_graph("train"), _full_graph("train")],
            "val": [_full_graph("val")],
            "test": [_full_graph("test")],
        }

    run = MagicMock()
    run.summary = {}
    monkeypatch.setattr(train_pipeline, "_prepare_training_payload_and_init", fake_payload_and_init)
    monkeypatch.setattr(train_pipeline.wandb, "init", lambda *args, **kwargs: run)
    monkeypatch.setattr(train_pipeline.wandb, "run", run)
    monkeypatch.setattr(train_pipeline.wandb, "log", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_pipeline.wandb, "save", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_pipeline.wandb, "finish", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_pipeline.wandb, "Image", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_pipeline, "train_step", fake_train_step)
    monkeypatch.setattr(train_pipeline, "DataLoader", fake_dataloader)
    monkeypatch.setattr(train_pipeline, "evaluate", fake_evaluate)
    monkeypatch.setattr(train_pipeline, "log_element_embeddings", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_pipeline, "plot_history", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_pipeline, "plot_log_regression", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_pipeline, "plot_uncertainty_calibration", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_pipeline, "save_history", lambda *args, **kwargs: None)

    class _LoadStateTrackingModel(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.load_call_count = 0
            self.p = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, batch):
            raise RuntimeError("forward should not be called in this test")

        def load_state_dict(self, state_dict, strict=True):
            self.load_call_count += 1
            calls["load_strict"].append(strict)
            return {"missing_keys": [], "unexpected_keys": []}

    model_holder = {"model": None}

    def _init_model(*args, **kwargs):
        model_holder["model"] = _LoadStateTrackingModel()
        return model_holder["model"]

    monkeypatch.setattr(train_pipeline, "DeltaGNN", _init_model)

    def fake_save(obj, path):
        calls["save"].append(path)
        if path == "best_target_smape_model.pth":
            saved_checkpoint["path"] = path

    def fake_torch_load(path, *args, **kwargs):
        calls["load"] += 1
        return {}

    monkeypatch.setattr(train_pipeline.torch, "save", fake_save)
    monkeypatch.setattr(train_pipeline.torch, "load", fake_torch_load)

    cfg = _config(tmp_path / "test_eval_end_main.pt")
    cfg.training = OmegaConf.create({
        "batch_size": 2,
        "epochs": 2,
        "learning_rate": 1e-3,
        "loss_function": "smape",
        "eval_interval": 1,
        "early_stopping_patience": 10,
        "early_stopping_min_delta": 0.0,
        "early_stop_metric": "Val/sMAPE_Scalar_Mean",
        "evaluate_test_at_end": True,
        "active_targets": ["energy", "gap", "volume"],
    })

    train_pipeline.main(cfg)
    assert calls["train_step"] == 2
    assert calls["evaluate"] >= 4
    assert calls["test_evaluate"] == 1
    assert "best_target_smape_model.pth" in calls["save"]
    assert calls["load"] >= 1
    assert saved_checkpoint["path"] == "best_target_smape_model.pth"
    assert model_holder["model"] is not None
    assert model_holder["model"].load_call_count >= 1
    assert calls["test_loader"] == 1
    assert calls["load_strict"] == [True]
    assert calls["sequence"].count("dataloader:test") == 1
    assert calls["sequence"].index("dataloader:test") < calls["sequence"].index("evaluate:test")


def test_main_early_stop_metric_alias_resolves_to_raw_scalar_metric(monkeypatch, tmp_path):
    def fake_train_step(*args, **kwargs):
        return 1.0

    metric_values = [12.0, 11.0]
    calls = {"evaluate": 0, "val_evaluate": 0}
    current_val_metric = {"value": None}

    def fake_evaluate(*args, **kwargs):
        loader = args[1]
        is_val = len(loader.dataset) == 1
        idx = calls["val_evaluate"] if is_val else calls["val_evaluate"] - 1
        idx = max(min(idx, len(metric_values) - 1), 0)
        calls["evaluate"] += 1
        if is_val:
            calls["val_evaluate"] += 1
            current_val_metric["value"] = metric_values[idx]
            value = current_val_metric["value"]
        else:
            value = current_val_metric["value"] or metric_values[0]
        value_dict = {
            "Val_Loss": 1.0,
            "MAE_delta_energy": 1.0,
            "MAE_delta_gap": 1.0,
            "MAE_delta_positions": 1.0,
            "MAE_delta_geo_total": 1.0,
            "MAE_delta_volume": 1.0,
            "MAE_delta_a": 1.0,
            "MAE_delta_b": 1.0,
            "MAE_delta_c": 1.0,
            "MAE_delta_alpha": 1.0,
            "MAE_delta_beta": 1.0,
            "MAE_delta_gamma": 1.0,
            "sMAPE_scalar_mean": value,
            "sMAPE_Scalar_Mean": value,
            "sMAPE_delta_energy": 1.0,
            "sMAPE_delta_gap": 1.0,
            "sMAPE_delta_positions": 1.0,
            "sMAPE_delta_volume": 1.0,
            "sMAPE_delta_a": 1.0,
            "sMAPE_delta_b": 1.0,
            "sMAPE_delta_c": 1.0,
            "sMAPE_delta_alpha": 1.0,
            "sMAPE_delta_beta": 1.0,
            "sMAPE_delta_gamma": 1.0,
            "RMSLAE_delta_energy": 1.0,
            "RMSLAE_delta_gap": 1.0,
            "RMSLAE_delta_positions": 1.0,
            "RMSLAE_delta_volume": 1.0,
            "RMSLAE_delta_a": 1.0,
            "RMSLAE_delta_b": 1.0,
            "RMSLAE_delta_c": 1.0,
            "RMSLAE_delta_alpha": 1.0,
            "RMSLAE_delta_beta": 1.0,
            "RMSLAE_delta_gamma": 1.0,
            "MagAcc_delta_energy": 1.0,
            "MagAcc_delta_gap": 1.0,
            "MagAcc_delta_positions": 1.0,
            "MagAcc_delta_volume": 1.0,
            "MagAcc_delta_a": 1.0,
            "MagAcc_delta_b": 1.0,
            "MagAcc_delta_c": 1.0,
            "MagAcc_delta_alpha": 1.0,
            "MagAcc_delta_beta": 1.0,
            "MagAcc_delta_gamma": 1.0,
        }
        return value_dict, {}

    def fake_payload_and_init(cfg):
        return {
            "train": [_full_graph("train"), _full_graph("train")],
            "val": [_full_graph("val")],
            "test": [_full_graph("test")],
        }

    run = MagicMock()
    run.summary = {}
    monkeypatch.setattr(train_pipeline, "_prepare_training_payload_and_init", fake_payload_and_init)
    monkeypatch.setattr(train_pipeline.wandb, "init", lambda *args, **kwargs: run)
    monkeypatch.setattr(train_pipeline.wandb, "run", run)
    monkeypatch.setattr(train_pipeline.wandb, "log", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_pipeline.wandb, "save", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_pipeline.wandb, "finish", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_pipeline.wandb, "Image", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_pipeline, "train_step", fake_train_step)
    monkeypatch.setattr(train_pipeline, "evaluate", fake_evaluate)
    monkeypatch.setattr(train_pipeline, "log_element_embeddings", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_pipeline, "plot_history", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_pipeline, "plot_log_regression", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_pipeline, "plot_uncertainty_calibration", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_pipeline, "save_history", lambda *args, **kwargs: None)

    saved_best_target_metric = []
    def fake_save(state, path):
        if path == "best_target_smape_model.pth":
            saved_best_target_metric.append(current_val_metric["value"])
        return None

    monkeypatch.setattr(train_pipeline.torch, "save", fake_save)

    class _DummyModel(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.p = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, batch):
            raise RuntimeError("forward should not be called in this test")

    monkeypatch.setattr(train_pipeline, "DeltaGNN", _DummyModel)

    cfg = _config(tmp_path / "alias_main.pt")
    cfg.strict_repaired_data = False
    cfg.training = OmegaConf.create({
        "batch_size": 2,
        "epochs": 3,
        "learning_rate": 1e-3,
        "loss_function": "smape",
        "eval_interval": 1,
        "early_stopping_patience": 10,
        "early_stopping_min_delta": 0.0,
        "early_stop_metric": "Val/sMAPE_Scalar_Mean",
        "active_targets": ["energy", "gap", "volume"],
    })

    train_pipeline.main(cfg)

    assert calls["evaluate"] == 6
    assert calls["val_evaluate"] == 3
    assert saved_best_target_metric == [12.0, 11.0]


def test_evaluate_legacy_full_targets_remains_finite_and_non_scalar_metrics():
    class Weights:
        delta_r = 1.0
        delta_e = 1.0
        delta_gap = 1.0
        delta_geo = 1.0
        delta_vol = 1.0
        delta_lat = 1.0

    loader = DataLoader([_full_graph("train"), _full_graph("val")], batch_size=2, shuffle=False)
    metrics, _ = train_pipeline.evaluate(
        _LegacyModel(),
        loader,
        train_pipeline.AsinhL1Loss(scale=1.0),
        Weights(),
        strict_mode=False,
    )

    for key in [
        "MAE_delta_positions", "MAE_delta_volume", "sMAPE_delta_energy", "sMAPE_delta_gap",
        "sMAPE_delta_volume", "RMSLAE_delta_energy", "RMSLAE_delta_gap", "RMSLAE_delta_volume",
    ]:
        assert metrics[key] != float("inf")
