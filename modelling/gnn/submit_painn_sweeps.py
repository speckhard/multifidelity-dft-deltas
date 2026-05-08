"""Phase 2: PaiNN architecture sweep — 3x3x3x2 = 54 jobs.

Grid: batch_size [16,32,64], learning_rate [1e-3,5e-4,1e-4],
      num_layers [2,3,4], hidden_dim [64,128].

FiLM and denoising settings should be locked based on Phase 1 results.
Adjust the flags below accordingly.

Usage:
    python submit_painn_sweeps.py
    python submit_painn_sweeps.py --use_denoising=true --use_film=true
    python submit_painn_sweeps.py --loss_function=smape
"""

import os
import itertools
import subprocess
from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string('root_dir', '/u/dansp/egnn/delta_painn/arch_sweep/sweep_feb_25_2026', 'Root directory.')
flags.DEFINE_string('data_file',
    '/u/dansp/egnn/relaxation_data_with_kpoints/data/'
    'delta_combined_relaxations_23_51__25_1_2026_egnn_data.pt',
    'Data file path.')
flags.DEFINE_string('wandb_project', 'painn-delta-denoising', 'WandB Project.')
flags.DEFINE_string('wandb_group', 'arch_sweep_v1', 'WandB Group.')
flags.DEFINE_string('loss_function', 'nll', 'Loss function: "nll" or "smape".')

# Lock based on Phase 1 ablation results
flags.DEFINE_boolean('use_film', True, 'Enable FiLM conditioning.')
flags.DEFINE_boolean('use_denoising', True, 'Enable denoising auxiliary loss.')

# --- Hyperparameter Grid ---
GRID = {
    'training.batch_size': [16, 32, 64],
    'training.learning_rate': [1e-3, 5e-4, 1e-4],
    'model.num_layers': [2, 3, 4],
    'model.hidden_dim': [64, 128],
}

# --- SLURM template for denoising pipeline ---
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
    ++training.loss_function={loss_function} \\
    ++model.use_film={use_film} \\
    {overrides}
"""

# --- SLURM template for standard pipeline (no denoising) ---
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
    ++training.loss_function={loss_function} \\
    ++model.use_film={use_film} \\
    {overrides}
"""


def main(argv):
    del argv
    base_dir = os.path.abspath(FLAGS.root_dir)

    keys, values = zip(*GRID.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    use_denoising = FLAGS.use_denoising
    use_film = FLAGS.use_film

    print(f"Target Directory: {base_dir}")
    print(f"FiLM: {use_film}, Denoising: {use_denoising}")
    print(f"Loss: {FLAGS.loss_function}")
    print(f"Grid: {len(combinations)} combinations")

    for params in combinations:
        parts = []
        for k, v in params.items():
            short_key = k.split('.')[-1][:2]
            if 'learning_rate' in k:
                short_key = 'lr'
            parts.append(f"{short_key}{v}")

        job_name = f"painn_{FLAGS.loss_function}_" + "_".join(parts)
        run_dir = os.path.join(base_dir, job_name)
        os.makedirs(run_dir, exist_ok=True)

        overrides = " ".join(f"{k}={v}" for k, v in params.items())

        template_args = dict(
            job_name=job_name,
            run_dir=run_dir,
            data_file=FLAGS.data_file,
            wandb_project=FLAGS.wandb_project,
            wandb_group=FLAGS.wandb_group,
            loss_function=FLAGS.loss_function,
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
