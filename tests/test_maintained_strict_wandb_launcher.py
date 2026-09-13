import re
import subprocess
from pathlib import Path

import pytest

try:
    import yaml  # type: ignore
except ImportError as exc:  # pragma: no cover - environment dependent
    pytest.skip(f"PyYAML required for launcher/config assertions: {exc}", allow_module_level=True)


REPO_ROOT = Path(__file__).resolve().parents[1]
HU_JOBS_ROOT = REPO_ROOT / "hpc" / "hu"
HPC_REPAIRED_PAYLOAD = "/home/physik/speckhad/delta_data_gpu/retraining_2026-09-09/data/legacy_fhi_aims_dedup_corrected_12bp_plus_k_2026-09-10.pt"
LAUNCHER = HU_JOBS_ROOT / "run_strict_mt_wandb_sweep_array.sbatch"


def test_strict_repaired_base_configs_lock_scalar_contracts():
    def load_yaml(path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    egnn_path = REPO_ROOT / "modelling/gnn/conf/strict_repaired_attention_mt_config.yaml"
    painn_path = REPO_ROOT / "modelling/gnn/conf/strict_repaired_painn_attention_mt_config.yaml"

    egnn = load_yaml(egnn_path)
    painn = load_yaml(painn_path)

    for cfg in (egnn, painn):
        assert cfg.get("strict_repaired_data") is True
        assert cfg.get("data_file") == HPC_REPAIRED_PAYLOAD
        assert cfg.get("wandb_project") == "gnn-repaired-fhi-2026"
        assert cfg.get("wandb_entity") == "dtts"
        assert cfg["training"]["batch_size"] == 32
        assert cfg["training"]["epochs"] == 800
        assert cfg["training"]["eval_interval"] == 5
        assert cfg["training"]["active_targets"] == ["energy", "gap", "volume"]
        assert cfg["training"]["early_stop_metric"] == "Val/sMAPE_Scalar_Mean"
        assert cfg["training"].get("evaluate_test_at_end") is False
        assert cfg["training"].get("seed") == 42
        assert cfg.get("job_name", "__not_none__") is None

        model = cfg["model"]
        assert model["num_cheap_dft_inputs"] == 12
        assert model["num_precision_settings"] == 13
        assert model["num_geo_inputs"] == 7

        assert set(cfg["weights"]) >= {
            "delta_e", "delta_gap", "delta_vol", "delta_r", "delta_lat"
        }
        assert cfg["weights"]["delta_e"] == 1.0
        assert cfg["weights"]["delta_gap"] == 1.0
        assert cfg["weights"]["delta_vol"] == 1.0
        assert cfg["weights"]["delta_r"] == 0.0
        assert cfg["weights"]["delta_lat"] == 0.0
        assert "target_weights" not in cfg["training"]

    assert egnn["model_type"] == "attention_film"
    assert egnn["model"]["num_layers"] == 4
    assert egnn["wandb_group"] == "strict-repaired-mt-egnn"
    assert "hidden_features" in egnn["model"]
    assert "hidden_dim" not in egnn["model"]

    assert painn["model_type"] == "painn_attention"
    assert painn["model"]["num_layers"] == 3
    assert painn["model"]["num_rbf"] == 20
    assert painn["model"]["cutoff"] == 5.0
    assert painn["wandb_group"] == "strict-repaired-mt-painn"


def _command_includes(command, token):
    if isinstance(command, str):
        return token in command
    return any(token in item for item in command)


def test_sweep_yaml_contracts_and_search_spaces():
    with open(HU_JOBS_ROOT / "strict_repaired_egnn_attention_film_hp_sweep.yaml", "r", encoding="utf-8") as f:
        egnn = yaml.safe_load(f)
    with open(HU_JOBS_ROOT / "strict_repaired_painn_attention_film_hp_sweep.yaml", "r", encoding="utf-8") as f:
        painn = yaml.safe_load(f)

    egnn_expected_command = [
        "${env}",
        "${interpreter}",
        "${program}",
        "--config-name=strict_repaired_attention_mt_config",
        "hydra.run.dir=${oc.env:WANDB_HPC_RUN_DIR}",
        "hydra.job.chdir=true",
        "${args_no_hyphens}",
    ]
    painn_expected_command = [
        "${env}",
        "${interpreter}",
        "${program}",
        "--config-name=strict_repaired_painn_attention_mt_config",
        "hydra.run.dir=${oc.env:WANDB_HPC_RUN_DIR}",
        "hydra.job.chdir=true",
        "${args_no_hyphens}",
    ]

    for sweep, expected in ((egnn, egnn_expected_command), (painn, painn_expected_command)):
        assert sweep["entity"] == "dtts"
        assert sweep["project"] == "gnn-repaired-fhi-2026"
        assert sweep["program"] == "modelling/gnn/train_pipeline.py"
        assert sweep["method"] == "random"
        assert sweep["metric"]["name"] == "Val/sMAPE_Scalar_Mean"
        assert sweep["metric"]["goal"] == "minimize"
        assert sweep["command"] == expected
        assert not _command_includes(sweep["command"], "--config-path")

    e_params = egnn["parameters"]
    p_params = painn["parameters"]

    assert set(e_params["model.hidden_features"]["values"]) == {32, 64, 128, 256}
    assert e_params["training.loss_function"]["values"] == ["smape", "asinh_l1", "l1", "nll"]
    assert [float(v) for v in e_params["training.learning_rate"]["values"]] == [1e-3, 5e-4, 1e-4]
    assert e_params["model.num_layers"]["value"] == 4
    assert e_params["training.batch_size"]["value"] == 32
    assert e_params["training.evaluate_test_at_end"]["value"] is False

    assert set(p_params["model.hidden_dim"]["values"]) == {32, 64, 128, 256}
    assert p_params["training.loss_function"]["values"] == ["smape", "asinh_l1", "l1", "nll"]
    assert [float(v) for v in p_params["training.learning_rate"]["values"]] == [1e-3, 5e-4, 1e-4]
    assert p_params["model.num_layers"]["value"] == 3
    assert p_params["training.batch_size"]["value"] == 32
    assert p_params["training.evaluate_test_at_end"]["value"] is False


def test_launcher_is_syntax_valid_and_secret_free():
    assert LAUNCHER.exists()
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "root=\"${HPC_ROOT:-/home/physik/speckhad/delta_data_gpu/retraining_2026-09-09}\"" in text
    assert "repo=\"${HPC_REPO_DIR:-$root/multifidelity-dft-deltas}\"" in text
    assert "venv=\"${HPC_VENV_ACTIVATE:-/home/physik/speckhad/codex_gpu_test/venv-cu126/bin/activate}\"" in text
    assert "source \"$venv\"" in text
    assert "test -n \"${WANDB_API_KEY:-}\"" in text
    assert "test -n \"${WANDB_SWEEP_PATH:-}\"" in text
    assert "test -n \"${DELTA_CODE_COMMIT:-}\"" in text
    assert "python -m wandb agent --count 1 \"$WANDB_SWEEP_PATH\"" in text
    assert "WANDB_HPC_RUN_DIR" in text
    assert re.search(r"A100", text)
    assert "#SBATCH --output=/home/physik/%u/delta_data_gpu/retraining_2026-09-09/runs/stage3/strict_mtsweep_%A_%a.out" in text
    assert re.search(r"task_\$\{SLURM_ARRAY_JOB_ID\}_\$\{SLURM_ARRAY_TASK_ID\}", text)
    assert "#SBATCH --partition=gpu" in text
    assert "#SBATCH --account=speckhad" in text
    assert "#SBATCH --gres=gpu:a10080gb:1" in text
    assert "#SBATCH --array=1-12" in text

    # No embedded long secrets.
    assert re.search(r"WANDB_API_KEY\s*=\s*['\"]?[A-Za-z0-9]{20,}", text) is None
    assert re.search(r"WANDB_SWEEP_PATH\s*=\s*['\"]?[A-Za-z0-9\-/_.]+", text) is None


def test_hpc_sweep_launcher_checks_provenance_contracts():
    repo = REPO_ROOT

    assert repo.joinpath("modelling/gnn/train_pipeline.py").exists()
    text = LAUNCHER.read_text(encoding="utf-8")

    # Enforce strict commit provenance contract.
    assert "module load git/2.51.0" in text
    assert text.index("module load git/2.51.0") < text.index("git_status=\"$(git status --porcelain)\"")
    assert "git_status=\"$(git status --porcelain)\"" in text
    assert "if [ \"$DELTA_CODE_COMMIT\" != \"$current_commit\" ]; then" in text
    assert "current_commit=\"$(git rev-parse --verify HEAD)\"" in text
    assert "test -n \"${DELTA_CODE_COMMIT:-}\"" in text

    # Script should reject dirty state with any porcelain output, including untracked files.
    assert "if [ -n \"$git_status\" ]; then" in text

    assert "torch.cuda.is_available" in text
    assert "--array" in text
    assert HU_JOBS_ROOT.joinpath("strict_repaired_egnn_attention_film_hp_sweep.yaml").exists()
    assert HU_JOBS_ROOT.joinpath("strict_repaired_painn_attention_film_hp_sweep.yaml").exists()


def test_wandb_init_reads_entity_from_config():
    source = (REPO_ROOT / "modelling/gnn/train_pipeline.py").read_text(encoding="utf-8")
    assert 'entity=cfg.get("wandb_entity", "dtts")' in source


def test_sweep_yaml_and_launcher_run_dir_contract_together():
    launch_text = LAUNCHER.read_text(encoding="utf-8")
    assert "WANDB_HPC_RUN_DIR" in launch_text

    with open(HU_JOBS_ROOT / "strict_repaired_egnn_attention_film_hp_sweep.yaml", "r", encoding="utf-8") as f:
        egnn = yaml.safe_load(f)
    with open(HU_JOBS_ROOT / "strict_repaired_painn_attention_film_hp_sweep.yaml", "r", encoding="utf-8") as f:
        painn = yaml.safe_load(f)

    assert any("hydra.run.dir=${oc.env:WANDB_HPC_RUN_DIR}" in token for token in egnn["command"])
    assert any("hydra.run.dir=${oc.env:WANDB_HPC_RUN_DIR}" in token for token in painn["command"])
    assert egnn["command"].count("${args_no_hyphens}") == 1
    assert painn["command"].count("${args_no_hyphens}") == 1


def test_train_pipeline_seed_prefers_training_seed_fallback():
    train_pipeline_source = (REPO_ROOT / "modelling/gnn/train_pipeline.py").read_text(encoding="utf-8")
    assert 'seed = cfg.training.get("seed", cfg.get("seed", None))' in train_pipeline_source
