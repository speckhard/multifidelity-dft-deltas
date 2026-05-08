"""Tests for MegaPaiNN surrogate sweep submission script.

Verifies directory creation, SLURM script content, and grid expansion
without submitting any jobs.
"""

import os
import shutil
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from modelling.gnn.submit_surrogate_sweep import generate_sweep  # noqa: E402


@pytest.fixture
def sweep_dir(tmp_path):
    """Temporary directory for sweep output."""
    d = tmp_path / "test_sweep"
    d.mkdir()
    return str(d)


class TestGenerateSweep:

    def test_correct_job_count(self, sweep_dir):
        """Grid product gives expected number of jobs."""
        grid = {
            'model.num_layers': [3, 5],
            'model.hidden_dim': [128],
            'training.batch_size': [64],
            'training.learning_rate': [1e-3],
        }
        jobs = generate_sweep(
            root_dir=sweep_dir, data_file='dummy.pt',
            wandb_project='test', wandb_group='test',
            grid=grid, dry_run=True)
        assert len(jobs) == 2

    def test_full_grid_count(self, sweep_dir):
        """Default-like grid: 2*2*2*2 = 16 jobs."""
        grid = {
            'model.num_layers': [3, 5],
            'model.hidden_dim': [128, 256],
            'training.batch_size': [64, 128],
            'training.learning_rate': [1e-3, 5e-4],
        }
        jobs = generate_sweep(
            root_dir=sweep_dir, data_file='dummy.pt',
            wandb_project='test', wandb_group='test',
            grid=grid, dry_run=True)
        assert len(jobs) == 16

    def test_directories_created(self, sweep_dir):
        """Each job gets its own subdirectory."""
        grid = {
            'model.num_layers': [3],
            'model.hidden_dim': [128],
            'training.batch_size': [64, 128],
            'training.learning_rate': [1e-3],
        }
        jobs = generate_sweep(
            root_dir=sweep_dir, data_file='dummy.pt',
            wandb_project='test', wandb_group='test',
            grid=grid, dry_run=True)

        for job_name, slurm_path in jobs:
            assert os.path.isdir(os.path.join(sweep_dir, job_name))
            assert os.path.isfile(slurm_path)

    def test_slurm_content_has_overrides(self, sweep_dir):
        """SLURM script contains correct Hydra overrides."""
        grid = {
            'model.num_layers': [5],
            'model.hidden_dim': [256],
            'training.batch_size': [64],
            'training.learning_rate': [0.001],
        }
        jobs = generate_sweep(
            root_dir=sweep_dir, data_file='/path/to/data.pt',
            wandb_project='my_project', wandb_group='my_group',
            grid=grid, dry_run=True)

        _, slurm_path = jobs[0]
        with open(slurm_path) as f:
            content = f.read()

        # Hydra overrides
        assert 'model.num_layers=5' in content
        assert 'model.hidden_dim=256' in content
        assert 'training.batch_size=64' in content
        assert 'training.learning_rate=0.001' in content

        # SLURM directives
        assert '#SBATCH --gres=gpu:a100:1' in content
        assert '#SBATCH --constraint="gpu"' in content

        # Paths
        assert 'data_file=/path/to/data.pt' in content
        assert '++wandb_project=my_project' in content
        assert '++wandb_group=my_group' in content

        # Venv activation
        assert 'source /u/dansp/egnn/py12_venv/bin/activate' in content

        # Correct training script
        assert 'train_surrogate.py' in content

    def test_job_names_readable(self, sweep_dir):
        """Job names contain hyperparameter values."""
        grid = {
            'model.num_layers': [3],
            'model.hidden_dim': [128],
            'training.batch_size': [64],
            'training.learning_rate': [0.0005],
        }
        jobs = generate_sweep(
            root_dir=sweep_dir, data_file='d.pt',
            wandb_project='p', wandb_group='g',
            grid=grid, dry_run=True)

        job_name, _ = jobs[0]
        assert 'surr_' in job_name
        assert 'nl3' in job_name
        assert 'hd128' in job_name
        assert 'bs64' in job_name
        assert 'lr' in job_name

    def test_dry_run_no_sbatch(self, sweep_dir):
        """dry_run=True does not call subprocess (sbatch not available locally)."""
        grid = {
            'model.num_layers': [3],
            'model.hidden_dim': [128],
            'training.batch_size': [64],
            'training.learning_rate': [1e-3],
        }
        # Should not raise even though sbatch doesn't exist locally
        jobs = generate_sweep(
            root_dir=sweep_dir, data_file='d.pt',
            wandb_project='p', wandb_group='g',
            grid=grid, dry_run=True)
        assert len(jobs) == 1
