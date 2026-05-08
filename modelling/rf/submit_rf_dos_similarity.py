"""SLURM submitter for the DOS Tanimoto similarity RF sweep.

Predicts how similar a cheap calc's DOS is to the APW=1.0 reference,
using scalar features only.  Uses the separate tanimoto CSV produced by
`parsing/compute_dos_similarity.py` and the `rf_config_dos_similarity`
config module.

1 target (dos_tanimoto) × 3 metrics (smape, mae, r2) = 3 jobs.

CPU-job SLURM block per the project rules in CLAUDE.md.

Usage:
    python submit_rf_dos_similarity.py                  # full sweep (3 jobs)
    python submit_rf_dos_similarity.py --smoke          # smoke-test: 1 job
    python submit_rf_dos_similarity.py --dry_run        # write SLURM files only
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# --- Config ---
ROOT_DIR = "/u/dansp/egnn/rf_results"
DATA_FILE = "/u/dansp/oasis_data/exciting_delta_learning_tanimoto.csv"
VENV_PATH = "/u/dansp/egnn/py12_venv/bin/activate"
SCRIPT_PATH = str(Path(__file__).resolve().parent / "rf_trainer.py")
CONFIG_MODULE = "rf_config_dos_similarity"

TARGETS = ["dos_tanimoto"]
METRICS = ["smape", "mae", "r2"]


SLURM_TEMPLATE = """#!/bin/bash -l
#SBATCH -J rf_dos_{target}_{metric}
#SBATCH -o {job_dir}/job.out.%j
#SBATCH -e {job_dir}/job.err.%j
#SBATCH -D {job_dir}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --partition=general
#SBATCH --time=12:00:00
#SBATCH --mem=0

export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK}}

source {venv_path}

cd {rf_dir}
srun python3.12 {script_path} \\
    --data_file={data_file} \\
    --output_dir={job_dir} \\
    --target_key={target} \\
    --metric_key={metric} \\
    --config_module={config_module} \\
    --n_iter=100 \\
    --n_jobs=72
"""


def _build_slurm(job_dir, target, metric):
    return SLURM_TEMPLATE.format(
        target=target,
        metric=metric,
        job_dir=job_dir,
        venv_path=VENV_PATH,
        script_path=SCRIPT_PATH,
        data_file=DATA_FILE,
        config_module=CONFIG_MODULE,
        rf_dir=str(Path(SCRIPT_PATH).parent),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root_dir", default=ROOT_DIR,
                   help="Output root (sweep dir created under this).")
    p.add_argument("--data_file", default=DATA_FILE,
                   help="Path to the tanimoto CSV.")
    p.add_argument("--targets", nargs="+", default=None,
                   help="Subset of target groups (default: all).")
    p.add_argument("--metrics", nargs="+", default=None,
                   help="Subset of scorers (default: all 3).")
    p.add_argument("--smoke", action="store_true",
                   help="Smoke test: submit only dos_tanimoto × smape (1 job).")
    p.add_argument("--dry_run", action="store_true",
                   help="Write SLURM files but don't sbatch.")
    p.add_argument("--sweep_name", default=None,
                   help="Name of sweep dir (default: rf_dos_sim_sweep_<date>).")
    args = p.parse_args()

    targets = args.targets or TARGETS
    metrics = args.metrics or METRICS
    if args.smoke:
        targets, metrics = ["dos_tanimoto"], ["smape"]

    for t in targets:
        if t not in TARGETS:
            print(f"ERROR: unknown target {t!r}; known: {TARGETS}")
            sys.exit(1)
    for m in metrics:
        if m not in METRICS:
            print(f"ERROR: unknown metric {m!r}; known: {METRICS}")
            sys.exit(1)

    if not Path(args.data_file).exists():
        print(f"WARNING: data file does not exist yet: {args.data_file}")
        print("Run `parsing/compute_dos_similarity.py` first.")

    sweep_name = args.sweep_name or (
        f"rf_dos_sim_sweep_{datetime.now().strftime('%Y_%m_%d')}"
    )
    base_output = os.path.join(args.root_dir, sweep_name)
    os.makedirs(base_output, exist_ok=True)
    print(f"Sweep dir: {base_output}")
    print(f"Grid: {len(targets)} targets × {len(metrics)} metrics = "
          f"{len(targets) * len(metrics)} jobs")

    submitted = []
    for target in targets:
        for metric in metrics:
            job_name = f"{target}_{metric}"
            job_dir = os.path.join(base_output, job_name)
            os.makedirs(job_dir, exist_ok=True)

            slurm_path = os.path.join(job_dir, "run.slurm")
            with open(slurm_path, "w") as fh:
                fh.write(_build_slurm(job_dir, target, metric))

            if args.dry_run:
                print(f"  [dry-run] wrote {slurm_path}")
                continue

            print(f"  Submitting {job_name}...")
            res = subprocess.run(["sbatch", slurm_path],
                                 capture_output=True, text=True)
            if res.returncode != 0:
                print(f"  sbatch FAILED ({res.returncode}): {res.stderr}")
            else:
                print(f"  {res.stdout.strip()}")
                submitted.append(job_name)

    print()
    if args.dry_run:
        print(f"Dry run complete — {len(targets) * len(metrics)} SLURM "
              f"file(s) written to {base_output}.")
    else:
        print(f"Submitted {len(submitted)} job(s) to {base_output}.")


if __name__ == "__main__":
    main()
