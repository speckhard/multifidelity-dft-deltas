"""Submit SLURM job on MPCDF to run compare_sweeps.py."""

import os
import subprocess
from datetime import datetime
from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string('root_dir', '/u/dansp/egnn/parsed_results/comparison_runs', 'Base directory for output')
flags.DEFINE_string('data_file',
                    '/u/dansp/egnn/relaxation_data_with_kpoints/data/delta_combined_relaxations_23_51__25_1_2026_egnn_data.pt',
                    'Data file path')
flags.DEFINE_string('venv_path', '/u/dansp/egnn/py12_venv/bin/activate', 'Path to venv activate')
flags.DEFINE_string('script_path', '/u/dansp/egnn/errorbar_modelling/modelling/gnn/compare_sweeps.py', 'Path to compare script')
flags.DEFINE_string('select_metric', 'Val/MAE_Energy', 'Val metric for model selection')
flags.DEFINE_string('select_mode', 'min', 'min or max for metric selection')
flags.DEFINE_integer('top_k', 5, 'Number of top models for ensemble')
flags.DEFINE_string('time_limit', '02:00:00', 'SLURM time limit')

SLURM_TEMPLATE = """#!/bin/bash -l
#SBATCH -J comparison_{date_str}
#SBATCH -o {output_dir}/comparison.out.%j
#SBATCH -e {output_dir}/comparison.err.%j
#SBATCH -D {output_dir}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time={time_limit}
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=120000
#SBATCH --constraint="gpu"

export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK}}
export NUMEXPR_MAX_THREADS=${{SLURM_CPUS_PER_TASK}}

source {venv_path}

echo "Running Model Comparison..."
echo "Output Directory: {output_dir}"

srun python {script_path} \\
    --output_dir={output_dir} \\
    --data_file={data_file} \\
    --select_metric={select_metric} \\
    --select_mode={select_mode} \\
    --top_k={top_k} \\
    --batch_size=64
"""


def main(argv):
    now = datetime.now()
    date_str = now.strftime('%d_%m_%Y_%H_%M')
    folder_name = f"comparison_{date_str}"
    full_output_dir = os.path.join(FLAGS.root_dir, folder_name)

    os.makedirs(full_output_dir, exist_ok=True)
    print(f"Created output directory: {full_output_dir}")

    slurm_content = SLURM_TEMPLATE.format(
        date_str=date_str,
        output_dir=full_output_dir,
        venv_path=FLAGS.venv_path,
        script_path=FLAGS.script_path,
        data_file=FLAGS.data_file,
        select_metric=FLAGS.select_metric,
        select_mode=FLAGS.select_mode,
        top_k=FLAGS.top_k,
        time_limit=FLAGS.time_limit,
    )

    slurm_path = os.path.join(full_output_dir, "run_comparison.slurm")
    with open(slurm_path, "w") as f:
        f.write(slurm_content)

    print(f"Submitting job...")
    subprocess.run(["sbatch", slurm_path])
    print(f"Job submitted! Results will be in:\n  {full_output_dir}")


if __name__ == "__main__":
    app.run(main)
