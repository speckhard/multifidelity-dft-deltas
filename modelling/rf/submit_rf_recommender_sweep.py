"""RF Recommender Feature Ablation Sweep.

Trains sklearn RandomForestRegressor across 3 feature sets × 3 targets × 4 losses = 36 jobs.

Feature sets:
  - full:    All ~102 numeric features (baseline, same as existing RF sweep)
  - precalc: ~80 features (drop DFT energy components + initial geometry)
  - minimal: ~50 features (elemental properties + precision settings only)

Usage:
    python submit_rf_recommender_sweep.py
    python submit_rf_recommender_sweep.py --dry_run
"""

import os
import subprocess
import itertools
from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string('root_dir', '/u/dansp/egnn/rf_results/rf_recommender_sweep_2026_03_20',
                    'Root output directory.')
flags.DEFINE_string('data_file',
    '/u/dansp/egnn/relaxation_data_with_kpoints/csv_data/'
    'delta_combined_relaxations_23_22__12_1_2026_kpoints_included_no_duplicates.csv',
    'CSV data file path.')
flags.DEFINE_boolean('dry_run', False, 'Generate SLURM scripts without submitting.')

# ──────────────────────────────────────────────────────────────────────
# Sweep grid
# ──────────────────────────────────────────────────────────────────────

FEATURE_SETS = ['full', 'precalc', 'minimal']
TARGETS = ['energy', 'bandgap', 'volume']
LOSSES = ['smape', 'mae', 'rmslae', 'asinh']

# ──────────────────────────────────────────────────────────────────────
# SLURM template (CPU-only, general partition per CLAUDE.md)
# ──────────────────────────────────────────────────────────────────────

SLURM_TEMPLATE = """#!/bin/bash -l
#SBATCH -J {job_name}
#SBATCH -o {run_dir}/job.out.%j
#SBATCH -e {run_dir}/job.err.%j
#SBATCH -D {run_dir}
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --partition=general
#SBATCH --time=12:00:00
#SBATCH --mem=0

export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK}}
export NUMEXPR_MAX_THREADS=${{SLURM_CPUS_PER_TASK}}

source /u/dansp/egnn/py12_venv/bin/activate

cd /raven/u/dansp/egnn/errorbar_modelling/modelling/rf

srun python3.12 rf_trainer.py \\
    --data_file={data_file} \\
    --output_dir={run_dir} \\
    --target_key={target} \\
    --metric_key={loss} \\
    --feature_set={feature_set} \\
    --n_iter=40 \\
    --n_jobs=-1
"""


def main(argv):
    del argv
    base_dir = os.path.abspath(FLAGS.root_dir)

    combos = list(itertools.product(FEATURE_SETS, TARGETS, LOSSES))

    print(f"Output directory : {base_dir}")
    print(f"Total jobs       : {len(combos)}")
    print(f"  Feature sets   : {FEATURE_SETS}")
    print(f"  Targets        : {TARGETS}")
    print(f"  Losses         : {LOSSES}")
    print(f"  Dry run        : {FLAGS.dry_run}")
    print()

    submitted = 0
    for feature_set, target, loss in combos:
        job_name = f"rf_{feature_set}_{target}_{loss}"
        run_dir = os.path.join(base_dir, f"sklearn_rf_{feature_set}_{target}", loss)
        os.makedirs(run_dir, exist_ok=True)

        slurm_content = SLURM_TEMPLATE.format(
            job_name=job_name,
            run_dir=run_dir,
            data_file=FLAGS.data_file,
            target=target,
            loss=loss,
            feature_set=feature_set,
        )

        slurm_path = os.path.join(run_dir, "run.slurm")
        with open(slurm_path, "w") as f:
            f.write(slurm_content)

        if FLAGS.dry_run:
            print(f"[DRY RUN] {slurm_path}")
        else:
            result = subprocess.run(["sbatch", slurm_path],
                                    capture_output=True, text=True)
            if result.returncode != 0:
                print(f"FAILED: {slurm_path}\n  stderr: {result.stderr.strip()}")
            else:
                print(f"Submitted: {job_name} -> {result.stdout.strip()}")
            submitted += 1

    print(f"\nDone. {'Generated' if FLAGS.dry_run else 'Submitted'} {len(combos)} jobs.")


if __name__ == "__main__":
    app.run(main)
