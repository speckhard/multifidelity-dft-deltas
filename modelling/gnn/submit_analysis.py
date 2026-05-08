import os
import subprocess
from datetime import datetime
from absl import app, flags

FLAGS = flags.FLAGS

# Config Flags
flags.DEFINE_string('root_dir', '/u/dansp/egnn/parsed_results/sweep_v2_new_metrics_jan_19_2026', 'Base directory')
flags.DEFINE_string('data_file', '/u/dansp/egnn/relaxation_data_with_kpoints/data/delta_combined_relaxations_22_23__12_1_2026_egnn_data.pt', 'Data file path')
flags.DEFINE_string('wandb_group', 'sweep_v2_new_metrics', 'WandB Group')
flags.DEFINE_string('venv_path', '/u/dansp/egnn/py12_venv/bin/activate', 'Path to venv activate')
flags.DEFINE_string('script_path', '/u/dansp/egnn/errorbar_modelling/modelling/gnn/analyze_best_model.py', 'Path to worker script')

# SLURM Template
SLURM_TEMPLATE = """#!/bin/bash -l
#SBATCH -J analysis_{date_str}
#SBATCH -o {output_dir}/analysis.out.%j
#SBATCH -e {output_dir}/analysis.err.%j
#SBATCH -D {output_dir}
#SBATCH --nodes=1
#SBATCH --cpus-per-task=72
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:a100:4
#SBATCH --constraint="gpu"
#SBATCH --partition="gpudev"

export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK}}
export NUMEXPR_MAX_THREADS=${{SLURM_CPUS_PER_TASK}}

source {venv_path}

echo "Running Analysis..."
echo "Output Directory: {output_dir}"

srun python3.12 {script_path} \\
    --output_dir={output_dir} \\
    --data_file={data_file} \\
    --wandb_group={wandb_group} \\
    --wandb_project=egnn-delta-learning
"""

def main(argv):
    # 1. Setup timestamped directory
    now = datetime.now()
    date_str = now.strftime('%d_%m_%Y_%H_%M')
    folder_name = f"analysis_outputs_{date_str}"
    full_output_dir = os.path.join(FLAGS.root_dir, folder_name)

    os.makedirs(full_output_dir, exist_ok=True)
    print(f"Created Analysis Directory: {full_output_dir}")

    # 2. Generate SLURM Script
    slurm_content = SLURM_TEMPLATE.format(
        date_str=date_str,
        output_dir=full_output_dir,
        venv_path=FLAGS.venv_path,
        script_path=FLAGS.script_path,
        data_file=FLAGS.data_file,
        wandb_group=FLAGS.wandb_group
    )

    slurm_path = os.path.join(full_output_dir, "run_analysis.slurm")
    with open(slurm_path, "w") as f:
        f.write(slurm_content)

    # 3. Submit
    print(f"Submitting job...")
    subprocess.run(["sbatch", slurm_path])
    print(f"Done! Monitor output in: {full_output_dir}/analysis.out.<jobid>")

if __name__ == "__main__":
    app.run(main)
