"""Phase 3: Denoising hyperparameter sweep — 4x3 = 12 jobs.

Grid: denoising_weight [0.01, 0.05, 0.1, 0.2],
      noise_scales.high [0.05, 0.1, 0.2].

noise_scales.low is kept fixed at 0.01.
Architecture and FiLM settings should be locked from Phase 1/2 results.

Usage:
    python submit_painn_denoising_sweep.py
    python submit_painn_denoising_sweep.py --root_dir=/u/dansp/egnn/painn_denoise_sweep
"""

import os
import itertools
import subprocess
from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string('root_dir', '/u/dansp/egnn/delta_painn/denoising_tuning/sweep_feb_25_2026',
                    'Root directory.')
flags.DEFINE_string('data_file',
    '/u/dansp/egnn/relaxation_data_with_kpoints/data/'
    'delta_combined_relaxations_23_51__25_1_2026_egnn_data.pt',
    'Data file path.')
flags.DEFINE_string('wandb_project', 'painn-delta-denoising', 'WandB Project.')
flags.DEFINE_string('wandb_group', 'denoise_sweep_v1', 'WandB Group.')

# Lock architecture from Phase 2 best — update these after Phase 2
flags.DEFINE_integer('num_layers', 3, 'Number of PaiNN layers.')
flags.DEFINE_integer('hidden_dim', 128, 'Hidden dimension.')
flags.DEFINE_integer('batch_size', 32, 'Batch size.')
flags.DEFINE_float('learning_rate', 5e-4, 'Learning rate.')

# --- Denoising Hyperparameter Grid ---
DENOISE_GRID = {
    'denoising_weight': [0.01, 0.05, 0.1, 0.2],
    'noise_scales.high': [0.05, 0.1, 0.2],
}

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

srun python3.12 /u/dansp/egnn/equiformerv2/errorbar_modelling/modelling/gnn/train_denoising_pipeline.py \\
    hydra.run.dir={run_dir} \\
    job_name={job_name} \\
    data_file={data_file} \\
    ++wandb_project={wandb_project} \\
    ++wandb_group={wandb_group} \\
    model.num_layers={num_layers} \\
    model.hidden_dim={hidden_dim} \\
    training.batch_size={batch_size} \\
    training.learning_rate={learning_rate} \\
    {overrides}
"""


def main(argv):
    del argv
    base_dir = os.path.abspath(FLAGS.root_dir)

    keys, values = zip(*DENOISE_GRID.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"Target Directory: {base_dir}")
    print(f"Architecture: layers={FLAGS.num_layers}, dim={FLAGS.hidden_dim}, "
          f"bs={FLAGS.batch_size}, lr={FLAGS.learning_rate}")
    print(f"Denoising grid: {len(combinations)} combinations")

    for params in combinations:
        dw = params['denoising_weight']
        nh = params['noise_scales.high']
        job_name = f"denoise_dw{dw}_nh{nh}"
        run_dir = os.path.join(base_dir, job_name)
        os.makedirs(run_dir, exist_ok=True)

        overrides = " ".join(f"{k}={v}" for k, v in params.items())

        slurm_content = SLURM_TEMPLATE.format(
            job_name=job_name,
            run_dir=run_dir,
            data_file=FLAGS.data_file,
            wandb_project=FLAGS.wandb_project,
            wandb_group=FLAGS.wandb_group,
            num_layers=FLAGS.num_layers,
            hidden_dim=FLAGS.hidden_dim,
            batch_size=FLAGS.batch_size,
            learning_rate=FLAGS.learning_rate,
            overrides=overrides,
        )

        slurm_path = os.path.join(run_dir, "run.slurm")
        with open(slurm_path, "w") as f:
            f.write(slurm_content)

        print(f"Generated: {slurm_path}")
        subprocess.run(["sbatch", slurm_path])


if __name__ == "__main__":
    app.run(main)
