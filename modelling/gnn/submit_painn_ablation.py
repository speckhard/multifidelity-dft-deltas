"""Phase 1: Ablation study — 2x2 factorial (FiLM x Denoising) x 3 seeds = 12 jobs.

Fixed architecture: num_layers=3, hidden_dim=128, lr=5e-4, batch_size=32.

Usage:
    python submit_painn_ablation.py
    python submit_painn_ablation.py --root_dir=/u/dansp/egnn/painn_ablation
"""

import os
import itertools
import subprocess
from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string('root_dir', '/u/dansp/egnn/delta_painn/ablation/sweep_feb_25_2026', 'Root directory.')
flags.DEFINE_string('data_file',
    '/u/dansp/egnn/relaxation_data_with_kpoints/data/'
    'delta_combined_relaxations_23_51__25_1_2026_egnn_data.pt',
    'Data file path.')
flags.DEFINE_string('wandb_project', 'painn-delta-denoising', 'WandB Project.')
flags.DEFINE_string('wandb_group', 'ablation_v1', 'WandB Group.')
flags.DEFINE_string('extra_overrides', '', 'Extra Hydra overrides appended to each job.')

# --- Ablation Grid ---
# FiLM on/off x Denoising on/off x 3 seeds = 12 jobs
ABLATION_CONFIGS = list(itertools.product(
    [True, False],    # use_film
    [True, False],    # use_denoising
    [42, 123, 7],     # random seeds
))

# Fixed architecture
FIXED_ARCH = {
    'model.num_layers': 3,
    'model.hidden_dim': 128,
    'training.batch_size': 32,
    'training.learning_rate': 5e-4,
}

# --- SLURM template for denoising pipeline (FiLM on/off, denoising ON) ---
SLURM_DENOISING = """#!/bin/bash -l
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
    ++seed={seed} \\
    ++model.use_film={use_film} \\
    {overrides}
"""

# --- SLURM template for standard pipeline (denoising OFF) ---
SLURM_STANDARD = """#!/bin/bash -l
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
    ++model.use_film={use_film} \\
    {overrides}
"""


def main(argv):
    del argv
    base_dir = os.path.abspath(FLAGS.root_dir)

    overrides = " ".join(f"{k}={v}" for k, v in FIXED_ARCH.items())
    if FLAGS.extra_overrides:
        overrides += " " + FLAGS.extra_overrides

    print(f"Target Directory: {base_dir}")
    print(f"Ablation: 2 (FiLM) x 2 (Denoising) x 3 (seeds) = "
          f"{len(ABLATION_CONFIGS)} jobs")
    print(f"Fixed: {FIXED_ARCH}")

    for use_film, use_denoising, seed in ABLATION_CONFIGS:
        film_tag = "film" if use_film else "nofilm"
        denoise_tag = "denoise" if use_denoising else "nodenoise"
        job_name = f"ablation_{film_tag}_{denoise_tag}_s{seed}"
        run_dir = os.path.join(base_dir, job_name)
        os.makedirs(run_dir, exist_ok=True)

        template_args = dict(
            job_name=job_name,
            run_dir=run_dir,
            data_file=FLAGS.data_file,
            wandb_project=FLAGS.wandb_project,
            wandb_group=FLAGS.wandb_group,
            seed=seed,
            use_film=str(use_film).lower(),
            overrides=overrides,
        )

        if use_denoising:
            slurm_content = SLURM_DENOISING.format(**template_args)
        else:
            slurm_content = SLURM_STANDARD.format(**template_args)

        slurm_path = os.path.join(run_dir, "run.slurm")
        with open(slurm_path, "w") as f:
            f.write(slurm_content)

        print(f"Generated: {slurm_path}")
        subprocess.run(["sbatch", slurm_path])


if __name__ == "__main__":
    app.run(main)
