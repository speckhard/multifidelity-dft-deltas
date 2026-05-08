"""TDD suite for submit_rf_dos_similarity — written BEFORE the implementation.

Validates:
1. Config constants point at the tanimoto CSV and dos_similarity config.
2. Target/metric grid is correct (1 target × 3 metrics = 3 jobs).
3. SLURM template follows CLAUDE.md rules (general partition, exclusive, 72 CPUs).
4. _build_slurm produces valid SLURM content.
5. Dry-run writes files without calling sbatch.

Run:
    python -m pytest tests/test_submit_rf_dos_similarity.py -v
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
REPO_DIR = TESTS_DIR.parent
sys.path.insert(0, str(REPO_DIR / "modelling" / "rf"))


@pytest.fixture(scope="module")
def mod():
    import submit_rf_dos_similarity
    return submit_rf_dos_similarity


# ---------------------------------------------------------------------------
# 1. Config constants
# ---------------------------------------------------------------------------

class TestConfigConstants:
    def test_data_file_points_to_tanimoto_csv(self, mod):
        assert "tanimoto" in mod.DATA_FILE.lower()
        assert mod.DATA_FILE.endswith(".csv")

    def test_config_module_is_dos_similarity(self, mod):
        assert mod.CONFIG_MODULE == "rf_config_dos_similarity"

    def test_targets_list(self, mod):
        assert mod.TARGETS == ["dos_tanimoto"]

    def test_metrics_list(self, mod):
        assert set(mod.METRICS) == {"smape", "mae", "r2"}

    def test_venv_path(self, mod):
        assert "py12_venv" in mod.VENV_PATH

    def test_script_path_points_to_rf_trainer(self, mod):
        assert mod.SCRIPT_PATH.endswith("rf_trainer.py")


# ---------------------------------------------------------------------------
# 2. SLURM template compliance (CLAUDE.md rules)
# ---------------------------------------------------------------------------

class TestSlurmTemplate:
    def test_uses_general_partition(self, mod):
        assert "--partition=general" in mod.SLURM_TEMPLATE

    def test_exclusive_node(self, mod):
        assert "--exclusive" in mod.SLURM_TEMPLATE

    def test_72_cpus(self, mod):
        assert "--cpus-per-task=72" in mod.SLURM_TEMPLATE

    def test_no_small_partition(self, mod):
        assert "small" not in mod.SLURM_TEMPLATE.lower()

    def test_mem_zero(self, mod):
        assert "--mem=0" in mod.SLURM_TEMPLATE


# ---------------------------------------------------------------------------
# 3. _build_slurm output
# ---------------------------------------------------------------------------

class TestBuildSlurm:
    def test_contains_target_and_metric(self, mod):
        slurm = mod._build_slurm("/tmp/test_job", "dos_tanimoto", "r2")
        assert "dos_tanimoto" in slurm
        assert "r2" in slurm

    def test_contains_config_module(self, mod):
        slurm = mod._build_slurm("/tmp/test_job", "dos_tanimoto", "mae")
        assert "--config_module=rf_config_dos_similarity" in slurm

    def test_contains_data_file(self, mod):
        slurm = mod._build_slurm("/tmp/test_job", "dos_tanimoto", "smape")
        assert mod.DATA_FILE in slurm

    def test_contains_job_dir(self, mod):
        slurm = mod._build_slurm("/tmp/my_job_dir", "dos_tanimoto", "smape")
        assert "/tmp/my_job_dir" in slurm


# ---------------------------------------------------------------------------
# 4. Dry-run integration
# ---------------------------------------------------------------------------

class TestDryRun:
    def test_dry_run_writes_slurm_files(self, mod, tmp_path, monkeypatch):
        """Dry-run should create SLURM files but not call sbatch."""
        monkeypatch.setattr(
            sys, "argv",
            ["submit_rf_dos_similarity.py",
             "--root_dir", str(tmp_path),
             "--dry_run"],
        )
        mod.main()
        slurm_files = list(tmp_path.rglob("*.slurm"))
        assert len(slurm_files) == 3  # 1 target × 3 metrics

    def test_dry_run_smoke_writes_one_file(self, mod, tmp_path, monkeypatch):
        """--smoke + --dry_run should produce exactly 1 SLURM file."""
        monkeypatch.setattr(
            sys, "argv",
            ["submit_rf_dos_similarity.py",
             "--root_dir", str(tmp_path),
             "--dry_run", "--smoke"],
        )
        mod.main()
        slurm_files = list(tmp_path.rglob("*.slurm"))
        assert len(slurm_files) == 1
