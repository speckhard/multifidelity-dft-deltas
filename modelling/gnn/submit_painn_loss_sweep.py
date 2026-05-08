"""Loss function sweep: AsinhL1, sMAPE, RMSLE — 3 losses x 3 seeds = 9 jobs.

Fixed config: FiLM=True, Denoising=False, delta_r=0, num_layers=3,
hidden_dim=128, lr=5e-4, batch_size=32.

Usage:
    python submit_painn_loss_sweep.py
    python submit_painn_loss_sweep.py --root_dir=/u/dansp/egnn/delta_painn/loss_sweep
"""

import os
import itertools
import subprocess
from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string('root_dir', '/u/dansp/egnn/delta_painn/loss_sweep', 'Root directory.')
flags.DEFINE_string('data_file',
    '/u/dansp/egnn/relaxation_data_with_kpoints/data/'
    'delta_combined_relaxations_23_51__25_1_2026_egnn_data.pt',
    'Data file path.')
flags.DEFINE_string('wandb_project', 'painn-delta-denoising', 'WandB Project.')

# --- Sweep Grid ---
LOSS_FUNCTIONS = ['asinh_l1', 'smape', 'rmsle']
SEEDS = [42, 123, 7]

SWEEP_CONFIGS = list(itertools.product(LOSS_FUNCTIONS, SEEDS))

FIXED_ARCH = {
    'model.num_layers': 3,
    'model.hidden_dim': 128,
    'training.batch_size': 32,
    'training.learning_rate': 5e-4,
}

# --- SLURM template (standard pipeline, no denoising) ---
SLURM_TEMPLATE = """#!/bin/bash -l
#SBATCH -J {job_name}
#SBATCH -o {run_dir}/djob.out.%j
#SBATCH -e {run_dir}/djob.err.%j
#SBATCH -D {run_dir}
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=120000
#SBATCH --constraint="gpu"

export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK}}
export NUMEXPR_MAX_THREADS=${{SLURM_CPUS_PER_TASK}}

source /u/dansp/egnn/py12_venv/bin/activate

srun python3.12 /u/dansp/egnn/equiformerv2/errorbar_modelling/modelling/gnn/train_pipeline.py \\
    --config-name=painn_denoising_config \\
    hydra.run.dir={run_dir} \\
    job_name={job_name} \\
    data_file={data_file} \\
    ++wandb_project={wandb_project} \\
    ++wandb_group={wandb_group} \\
    ++model_type=painn \\
    ++seed={seed} \\
    ++model.use_film=true \\
    ++training.loss_function={loss_function} \\
    ++weights.delta_r=0.0 \\
    {overrides}
"""


def main(argv):
    del argv
    base_dir = os.path.abspath(FLAGS.root_dir)

    overrides = " ".join(f"{k}={v}" for k, v in FIXED_ARCH.items())

    print(f"Target Directory: {base_dir}")
    print(f"Loss sweep: {len(LOSS_FUNCTIONS)} losses x {len(SEEDS)} seeds = "
          f"{len(SWEEP_CONFIGS)} jobs")
    print(f"Losses: {LOSS_FUNCTIONS}")
    print(f"Fixed: {FIXED_ARCH}")

    for loss_fn, seed in SWEEP_CONFIGS:
        job_name = f"loss_{loss_fn}_s{seed}"
        wandb_group = f"loss_sweep_{loss_fn}"
        run_dir = os.path.join(base_dir, loss_fn, f"seed_{seed}")
        os.makedirs(run_dir, exist_ok=True)

        slurm_content = SLURM_TEMPLATE.format(
            job_name=job_name,
            run_dir=run_dir,
            data_file=FLAGS.data_file,
            wandb_project=FLAGS.wandb_project,
            wandb_group=wandb_group,
            seed=seed,
            loss_function=loss_fn,
            overrides=overrides,
        )

        slurm_path = os.path.join(run_dir, "run.slurm")
        with open(slurm_path, "w") as f:
            f.write(slurm_content)

        print(f"Generated: {slurm_path}")
        subprocess.run(["sbatch", slurm_path])


if __name__ == "__main__":
    app.run(main)
