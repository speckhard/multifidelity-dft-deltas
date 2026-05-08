"""
Submit 10 SLURM jobs for the XGBoost sweep: 5 targets x 2 model types.

Each job runs 4 losses (mae, smape, asinh, rmslae) sequentially on an exclusive
node, using full-node CV parallelism (n_jobs=72, xgb_n_jobs=1).

Usage:
    python submit_rf_jobs.py              # submit all 10 jobs
    python submit_rf_jobs.py --dry_run    # write .slurm files but don't sbatch
"""
import os
import subprocess
import argparse
from datetime import datetime

# --- Configuration ---
ROOT_DIR = '/u/dansp/egnn/rf_results'
DATA_FILE = '/u/dansp/egnn/relaxation_data_with_kpoints/csv_data/delta_combined_relaxations_23_22__12_1_2026_kpoints_included_no_duplicates.csv'
VENV_PATH = '/u/dansp/egnn/py12_venv/bin/activate'
SCRIPT_PATH = '/u/dansp/egnn/errorbar_modelling/modelling/rf/rf_trainer.py'

TARGETS = ['energy', 'bandgap', 'volume', 'geometry', 'all_scalar']
MODELS = ['xgb_gbdt', 'xgb_rf']
LOSSES = ['mae', 'smape', 'asinh', 'rmslae']

N_ITER = 100
CV_N_JOBS = 72
XGB_N_JOBS = 1

# Each job runs 4 losses sequentially. Single-target jobs finish in ~2-3h,
# multi-target (geometry=7, all_scalar=9) in ~6-8h. 12h covers all.
SLURM_TEMPLATE = """#!/bin/bash -l
#SBATCH -J xgb_{model}_{target}
#SBATCH -o {job_dir}/job.out.%j
#SBATCH -e {job_dir}/job.err.%j
#SBATCH -D {job_dir}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --partition=general
#SBATCH --time=12:00:00
#SBATCH --mem=0

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

source {venv_path}

echo "=== Job start: $(date) ==="
echo "Node: $SLURMD_NODENAME"
echo "Model: {model}, Target: {target}"
echo "Losses: {losses}"

{run_commands}

echo "=== Job end: $(date) ==="
"""

RUN_TEMPLATE = """
echo "--- [{loss_idx}/4] Loss: {loss} ($(date)) ---"
python3.12 {script_path} \\
    --data_file={data_file} \\
    --output_dir={loss_dir} \\
    --target_key={target} \\
    --metric_key={loss} \\
    --model_type={model} \\
    --n_iter={n_iter} \\
    --n_jobs={cv_n_jobs} \\
    --xgb_n_jobs={xgb_n_jobs}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry_run', action='store_true',
                        help='Write .slurm files but do not submit')
    args = parser.parse_args()

    date_str = datetime.now().strftime('%Y_%m_%d')
    base_output = os.path.join(ROOT_DIR, f"xgb_sweep_{date_str}")
    os.makedirs(base_output, exist_ok=True)

    print(f"Sweep output: {base_output}")
    print(f"Jobs: {len(TARGETS)} targets x {len(MODELS)} models = {len(TARGETS) * len(MODELS)}")
    if args.dry_run:
        print("DRY RUN — .slurm files will be written but not submitted.\n")

    for model in MODELS:
        for target in TARGETS:
            job_name = f"{model}_{target}"
            job_dir = os.path.join(base_output, job_name)
            os.makedirs(job_dir, exist_ok=True)

            # Build sequential run commands for all 4 losses
            run_commands = []
            for idx, loss in enumerate(LOSSES, 1):
                loss_dir = os.path.join(job_dir, loss)
                os.makedirs(loss_dir, exist_ok=True)

                run_commands.append(RUN_TEMPLATE.format(
                    loss_idx=idx,
                    loss=loss,
                    script_path=SCRIPT_PATH,
                    data_file=DATA_FILE,
                    loss_dir=loss_dir,
                    target=target,
                    model=model,
                    n_iter=N_ITER,
                    cv_n_jobs=CV_N_JOBS,
                    xgb_n_jobs=XGB_N_JOBS,
                ))

            slurm_content = SLURM_TEMPLATE.format(
                model=model,
                target=target,
                job_dir=job_dir,
                venv_path=VENV_PATH,
                losses=', '.join(LOSSES),
                run_commands=''.join(run_commands),
            )

            slurm_path = os.path.join(job_dir, "run.slurm")
            with open(slurm_path, "w") as f:
                f.write(slurm_content)

            if args.dry_run:
                print(f"  [DRY RUN] Wrote {slurm_path}")
            else:
                print(f"  Submitting {job_name}...")
                result = subprocess.run(["sbatch", slurm_path], capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"    ERROR: sbatch failed: {result.stderr.strip()}")
                else:
                    print(f"    {result.stdout.strip()}")

    print(f"\nDone. {len(TARGETS) * len(MODELS)} jobs {'written' if args.dry_run else 'submitted'}.")


if __name__ == "__main__":
    main()
