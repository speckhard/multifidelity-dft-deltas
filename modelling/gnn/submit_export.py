"""Submit job on MPCDF to export results from best model."""

import os
import subprocess
from datetime import datetime
from absl import app, flags

FLAGS = flags.FLAGS

# Config Flags
flags.DEFINE_string('root_dir', '/u/dansp/egnn/parsed_results/export_runs', 'Directory where export folder will be created')
flags.DEFINE_string('data_file', '/u/dansp/egnn/relaxation_data_with_kpoints/data/delta_combined_relaxations_23_51__25_1_2026_egnn_data.pt', 'Data file path')
flags.DEFINE_string('wandb_group', 'sweep_gnll_val_split_v2', 'WandB Group to pick best model from')
flags.DEFINE_string('wandb_project', 'egnn-delta-learning', 'WandB Project Name')
flags.DEFINE_string('venv_path', '/u/dansp/egnn/py12_venv/bin/activate', 'Path to venv activate')

# Default to the export script
flags.DEFINE_string('script_path', '/u/dansp/egnn/errorbar_modelling/modelling/gnn/export_results.py', 'Path to the export worker script')

# SLURM Template
# Optimized for Inference: 1 GPU, 16 CPUs, 64GB RAM is plenty for export
SLURM_TEMPLATE = """#!/bin/bash -l
#SBATCH -J export_{date_str}
#SBATCH -o {output_dir}/export.out.%j
#SBATCH -e {output_dir}/export.err.%j
#SBATCH -D {output_dir}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64000
#SBATCH --constraint="gpu"
#SBATCH --partition="gpudev"

export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK}}
export NUMEXPR_MAX_THREADS=${{SLURM_CPUS_PER_TASK}}

source {venv_path}

echo "Running Export Job..."
echo "Output Directory: {output_dir}"

srun python3.12 {script_path} \\
    --output_file={output_path} \\
    --data_file={data_file} \\
    --wandb_group={wandb_group} \\
    --wandb_project={wandb_project}
"""

def main(argv):
    # 1. Setup timestamped directory
    now = datetime.now()
    date_str = now.strftime('%d_%m_%Y_%H_%M')
    folder_name = f"export_{date_str}"
    full_output_dir = os.path.join(FLAGS.root_dir, folder_name)

    os.makedirs(full_output_dir, exist_ok=True)
    print(f"Created Export Directory: {full_output_dir}")

    # Define the final pickle path
    output_pkl_path = os.path.join(full_output_dir, "thesis_data_export.pkl")

    # 2. Generate SLURM Script
    slurm_content = SLURM_TEMPLATE.format(
        date_str=date_str,
        output_dir=full_output_dir,
        output_path=output_pkl_path,
        venv_path=FLAGS.venv_path,
        script_path=FLAGS.script_path,
        data_file=FLAGS.data_file,
        wandb_group=FLAGS.wandb_group,
        wandb_project=FLAGS.wandb_project
    )

    slurm_path = os.path.join(full_output_dir, "run_export.slurm")
    with open(slurm_path, "w") as f:
        f.write(slurm_content)

    # 3. Submit
    print(f"Submitting job...")
    subprocess.run(["sbatch", slurm_path])
    print(f"Job submitted! Output will be saved to:\n  {output_pkl_path}")

if __name__ == "__main__":
    app.run(main)
