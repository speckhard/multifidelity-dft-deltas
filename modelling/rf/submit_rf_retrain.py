"""Submit SLURM job to retrain RF models with EGNN-matched filters."""

import os
import subprocess
from datetime import datetime
from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string('root_dir', '/u/dansp/egnn/rf_results', 'Base output directory')
flags.DEFINE_string('data_file',
                    '/u/dansp/egnn/relaxation_data_with_kpoints/csv_data/'
                    'delta_combined_relaxations_23_22__12_1_2026_kpoints_included_no_duplicates.csv',
                    'RF CSV data file')
flags.DEFINE_string('model_dir',
                    '/u/dansp/egnn/rf_results/rf_sweep_2026_02_13',
                    'Directory with original RF sweep models')
flags.DEFINE_string('venv_path', '/u/dansp/egnn/py12_venv/bin/activate', 'Path to venv activate')
flags.DEFINE_string('script_path',
                    '/u/dansp/egnn/errorbar_modelling/modelling/rf/rf_retrain_filtered.py',
                    'Path to retrain script')

SLURM_TEMPLATE = """#!/bin/bash -l
#SBATCH -J rf_retrain_{date_str}
#SBATCH -o {output_dir}/rf_retrain.out.%j
#SBATCH -e {output_dir}/rf_retrain.err.%j
#SBATCH -D {output_dir}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --partition=general
#SBATCH --time=12:00:00
#SBATCH --mem=0

export OMP_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=${{SLURM_CPUS_PER_TASK}}

source {venv_path}

echo "Retraining RF models with EGNN-matched filters..."
echo "Output: {output_dir}"

srun python {script_path} \\
    --data_file={data_file} \\
    --model_dir={model_dir} \\
    --output_dir={output_dir}
"""


def main(argv):
    now = datetime.now()
    date_str = now.strftime('%d_%m_%Y_%H_%M')
    folder_name = f"rf_retrain_filtered_{date_str}"
    full_output_dir = os.path.join(FLAGS.root_dir, folder_name)

    os.makedirs(full_output_dir, exist_ok=True)
    print(f"Created output directory: {full_output_dir}")

    slurm_content = SLURM_TEMPLATE.format(
        date_str=date_str,
        output_dir=full_output_dir,
        venv_path=FLAGS.venv_path,
        script_path=FLAGS.script_path,
        data_file=FLAGS.data_file,
        model_dir=FLAGS.model_dir,
    )

    slurm_path = os.path.join(full_output_dir, "run_rf_retrain.slurm")
    with open(slurm_path, "w") as f:
        f.write(slurm_content)

    print(f"Submitting job...")
    subprocess.run(["sbatch", slurm_path])
    print(f"Job submitted! Results will be in:\n  {full_output_dir}")


if __name__ == "__main__":
    app.run(main)
