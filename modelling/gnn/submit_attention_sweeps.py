import os
import itertools
import subprocess
from absl import app, flags

# --- 1. Define Flags ---
FLAGS = flags.FLAGS

flags.DEFINE_string('root_dir', '/u/dansp/egnn/attention_sweep', 'Root directory.')
flags.DEFINE_string('data_file', '/u/dansp/egnn/relaxation_data_with_kpoints/data/delta_combined_relaxations_11_32_11__1_1_2026_egnn_data.pt', 'Data file path.')
flags.DEFINE_string('wandb_project', 'egnn-attention', 'WandB Project.')
flags.DEFINE_string('wandb_group', 'attention_sweep_v1', 'WandB Group.')
flags.DEFINE_string('loss_function', 'nll', 'Loss function to use: "nll" or "smape"')

# --- 2. Define the Hyperparameter Grid ---
# Same grid as the DeltaGNN sweep for fair comparison
GRID = {
    'training.batch_size': [16, 32, 64],
    'training.learning_rate': [1e-3, 5e-4, 1e-4],
    'model.num_layers': [3, 4, 5],
    'model.hidden_features': [64, 128],
}

# --- 3. Slurm Configuration ---
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

# Attention model: use attention_config and set model_type
srun python3.12 /u/dansp/egnn/errorbar_modelling/modelling/gnn/train_pipeline.py \\
    --config-name=attention_config \\
    hydra.run.dir={run_dir} \\
    job_name={job_name} \\
    data_file={data_file} \\
    ++wandb_project={wandb_project} \\
    ++wandb_group={wandb_group} \\
    ++training.loss_function={loss_function} \\
    ++model_type=attention \\
    {overrides}
"""

def main(argv):
    del argv
    base_experiment_dir = os.path.abspath(FLAGS.root_dir)

    keys, values = zip(*GRID.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"Target Directory: {base_experiment_dir}")
    print(f"Loss Function: {FLAGS.loss_function}")
    print(f"Model Type: attention")
    print(f"Total Combinations: {len(combinations)}")

    for _, params in enumerate(combinations):
        parts = []
        for k, v in params.items():
            short_key = k.split('.')[-1][:2]
            if 'learning_rate' in k: short_key = 'lr'
            parts.append(f"{short_key}{v}")

        job_name = f"att_{FLAGS.loss_function}_" + "_".join(parts)
        run_dir = os.path.join(base_experiment_dir, job_name)
        os.makedirs(run_dir, exist_ok=True)

        overrides = " ".join([f"{k}={v}" for k, v in params.items()])

        slurm_content = SLURM_TEMPLATE.format(
            job_name=job_name,
            run_dir=run_dir,
            data_file=FLAGS.data_file,
            wandb_project=FLAGS.wandb_project,
            wandb_group=FLAGS.wandb_group,
            loss_function=FLAGS.loss_function,
            overrides=overrides
        )

        slurm_path = os.path.join(run_dir, "run.slurm")
        with open(slurm_path, "w") as f:
            f.write(slurm_content)

        print(f"Generated: {slurm_path}")
        subprocess.run(["sbatch", slurm_path])

if __name__ == "__main__":
    app.run(main)
