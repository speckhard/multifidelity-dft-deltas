"""Submit MegaPaiNN surrogate hyperparameter sweep to MPCDF SLURM.

Generates one SLURM job per grid combination, each running train_surrogate.py
with Hydra overrides. Same pattern as submit_sweeps.py.

Usage:
  python modelling/gnn/submit_surrogate_sweep.py --root_dir /u/dansp/egnn/surrogate_sweep
  python modelling/gnn/submit_surrogate_sweep.py --dry_run  # generate scripts only
"""

import itertools
import os
import subprocess

from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string('root_dir', '/u/dansp/egnn/surrogate_sweep',
                    'Root directory for sweep runs.')
flags.DEFINE_string('data_file', 'data/aflow_processed.pt',
                    'Path to processed AFLOW data.')
flags.DEFINE_string('wandb_project', 'mega-painn-surrogate', 'WandB project.')
flags.DEFINE_string('wandb_group', 'sweep_v1', 'WandB group.')
flags.DEFINE_bool('dry_run', False,
                  'Generate SLURM scripts but do not submit.')

# --- Hyperparameter Grid ---
GRID = {
    'model.num_layers': [3, 5],
    'model.hidden_dim': [128, 256],
    'training.batch_size': [64, 128],
    'training.learning_rate': [1e-3, 5e-4],
}

# --- SLURM Template (MPCDF A100 GPU) ---
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

srun python3.12 /u/dansp/egnn/errorbar_modelling/modelling/gnn/train_surrogate.py \\
    hydra.run.dir={run_dir} \\
    job_name={job_name} \\
    data_file={data_file} \\
    ++wandb_project={wandb_project} \\
    ++wandb_group={wandb_group} \\
    {overrides}
"""


def generate_sweep(root_dir, data_file, wandb_project, wandb_group,
                   grid=None, dry_run=False):
    """Generate and optionally submit SLURM jobs for all grid combinations.

    Returns list of (job_name, slurm_path) tuples.
    """
    if grid is None:
        grid = GRID

    base_dir = os.path.abspath(root_dir)
    keys, values = zip(*grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"Sweep directory: {base_dir}")
    print(f"Total jobs: {len(combinations)}")
    if dry_run:
        print("DRY RUN — scripts generated but not submitted.")

    jobs = []
    for params in combinations:
        # Readable job name: surr_nl3_hd128_bs64_lr0.001
        parts = []
        for k, v in params.items():
            short = k.split('.')[-1]
            if short == 'num_layers':
                parts.append(f"nl{v}")
            elif short == 'hidden_dim':
                parts.append(f"hd{v}")
            elif short == 'batch_size':
                parts.append(f"bs{v}")
            elif short == 'learning_rate':
                parts.append(f"lr{v}")
            else:
                parts.append(f"{short[:3]}{v}")
        job_name = "surr_" + "_".join(parts)

        run_dir = os.path.join(base_dir, job_name)
        os.makedirs(run_dir, exist_ok=True)

        overrides = " ".join([f"{k}={v}" for k, v in params.items()])

        slurm_content = SLURM_TEMPLATE.format(
            job_name=job_name,
            run_dir=run_dir,
            data_file=data_file,
            wandb_project=wandb_project,
            wandb_group=wandb_group,
            overrides=overrides,
        )

        slurm_path = os.path.join(run_dir, "run.slurm")
        with open(slurm_path, "w") as f:
            f.write(slurm_content)

        print(f"  Generated: {slurm_path}")

        if not dry_run:
            subprocess.run(["sbatch", slurm_path])

        jobs.append((job_name, slurm_path))

    return jobs


def main(argv):
    del argv
    generate_sweep(
        root_dir=FLAGS.root_dir,
        data_file=FLAGS.data_file,
        wandb_project=FLAGS.wandb_project,
        wandb_group=FLAGS.wandb_group,
        dry_run=FLAGS.dry_run,
    )


if __name__ == "__main__":
    app.run(main)
