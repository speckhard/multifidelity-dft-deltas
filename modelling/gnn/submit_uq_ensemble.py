#!/usr/bin/env python3.12
"""Generate and optionally submit SLURM jobs for UQ ensemble + NLL training.

Produces 22 SLURM scripts:
  - 10 seeds x recommender mode (egnn_film, smape, lr=1e-4, h=256, no cheap DFT)
  - 10 seeds x full mode       (egnn_film, smape, lr=1e-4, h=128, with cheap DFT)
  -  1 recommender NLL          (same arch, nll loss, seed=42)
  -  1 full NLL                 (same arch, nll loss, seed=42)

Usage:
    python submit_uq_ensemble.py                    # generate scripts only
    python submit_uq_ensemble.py --submit           # generate + sbatch all
    python submit_uq_ensemble.py --submit --dry_run # print sbatch commands
"""

import os
import subprocess
from pathlib import Path

from absl import app, flags

FLAGS = flags.FLAGS
flags.DEFINE_bool("submit", False, "Submit jobs via sbatch after generating scripts.")
flags.DEFINE_bool("dry_run", False, "Print sbatch commands instead of running them.")

# ── Paths ──

CODE_DIR = Path("/u/dansp/egnn/errorbar_modelling")
TRAIN_SCRIPT = CODE_DIR / "modelling" / "gnn" / "train_pipeline.py"
DATA_FILE = Path(
    "/u/dansp/egnn/relaxation_data_with_kpoints/data/"
    "delta_combined_relaxations_23_51__25_1_2026_egnn_data.pt"
)
OUTPUT_ROOT = Path("/u/dansp/egnn/delta_painn/uq_ensemble_2026_03_24")
VENV = "/u/dansp/egnn/py12_venv/bin/activate"

# ── Champion Configurations ──

CONFIGS = {
    "recommender": {
        "config_name": "config",
        "hidden_features": 256,
        "learning_rate": 1e-4,
        "use_film": True,
        "num_cheap_dft_inputs": 0,
        "num_geo_inputs": 0,
        "wandb_group_prefix": "uq_rec",
    },
    "full": {
        "config_name": "config",
        "hidden_features": 128,
        "learning_rate": 1e-4,
        "use_film": True,
        "num_cheap_dft_inputs": 12,
        "num_geo_inputs": 7,
        "wandb_group_prefix": "uq_full",
    },
}

# Fixed across all jobs
FIXED = {
    "model_type": "delta",
    "num_layers": 3,
    "batch_size": 32,
    "epochs": 800,
    "smape_epsilon": 1e-4,
    "early_stop_metric": "Val_Loss",
}

ENSEMBLE_SEEDS = list(range(1, 11))  # seeds 1..10
NLL_SEED = 42
WANDB_PROJECT = "gnn-uq-ensemble"

# ── SLURM Template ──

SLURM_TEMPLATE = """\
#!/bin/bash -l
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

source {venv}

srun python3.12 {train_script} \\
    --config-name={config_name} \\
    hydra.run.dir={run_dir} \\
    job_name={job_name} \\
    data_file={data_file} \\
    ++wandb_project={wandb_project} \\
    ++wandb_group={wandb_group} \\
    ++model_type={model_type} \\
    ++seed={seed} \\
    ++training.loss_function={loss_function} \\
    ++training.learning_rate={learning_rate} \\
    ++training.early_stop_metric={early_stop_metric} \\
    model.hidden_features={hidden_features} \\
    ++weights.delta_e=1.0 ++weights.delta_gap=0.0 ++weights.delta_r=0.0 ++weights.delta_vol=0.0 ++weights.delta_lat=0.0 \\
    model.num_layers={num_layers} training.batch_size={batch_size} training.epochs={epochs} \\
    ++training.smape_epsilon={smape_epsilon} \\
    ++model.use_film={use_film} \\
    ++model.num_cheap_dft_inputs={num_cheap_dft_inputs} ++model.num_geo_inputs={num_geo_inputs}
"""


def generate_job(mode: str, seed: int, loss: str) -> tuple[Path, str]:
    """Generate a single SLURM script. Returns (script_path, job_name)."""
    cfg = CONFIGS[mode]

    if loss == "nll":
        subdir = f"{mode}_nll"
        job_name = f"uq_{mode}_nll_seed{seed}"
        wandb_group = f"{cfg['wandb_group_prefix']}_nll"
    else:
        subdir = f"{mode}/seed_{seed:02d}"
        job_name = f"uq_{mode}_seed{seed:02d}"
        wandb_group = f"{cfg['wandb_group_prefix']}_ensemble"

    run_dir = OUTPUT_ROOT / subdir
    run_dir.mkdir(parents=True, exist_ok=True)

    script_content = SLURM_TEMPLATE.format(
        job_name=job_name,
        run_dir=run_dir,
        venv=VENV,
        train_script=TRAIN_SCRIPT,
        config_name=cfg["config_name"],
        data_file=DATA_FILE,
        wandb_project=WANDB_PROJECT,
        wandb_group=wandb_group,
        model_type=FIXED["model_type"],
        seed=seed,
        loss_function=loss,
        learning_rate=cfg["learning_rate"],
        early_stop_metric=FIXED["early_stop_metric"],
        hidden_features=cfg["hidden_features"],
        num_layers=FIXED["num_layers"],
        batch_size=FIXED["batch_size"],
        epochs=FIXED["epochs"],
        smape_epsilon=FIXED["smape_epsilon"],
        use_film=str(cfg["use_film"]).lower(),
        num_cheap_dft_inputs=cfg["num_cheap_dft_inputs"],
        num_geo_inputs=cfg["num_geo_inputs"],
    )

    script_path = run_dir / "run.slurm"
    script_path.write_text(script_content)
    return script_path, job_name


def main(_):
    scripts = []

    # Phase 1: Ensemble jobs (10 seeds x 2 modes = 20 jobs)
    for mode in ["recommender", "full"]:
        for seed in ENSEMBLE_SEEDS:
            path, name = generate_job(mode, seed, loss="smape")
            scripts.append((path, name))

    # Phase 2: NLL jobs (1 per mode = 2 jobs)
    for mode in ["recommender", "full"]:
        path, name = generate_job(mode, NLL_SEED, loss="nll")
        scripts.append((path, name))

    print(f"Generated {len(scripts)} SLURM scripts:")
    for path, name in scripts:
        print(f"  {name}: {path}")

    if FLAGS.submit:
        print(f"\n{'DRY RUN - ' if FLAGS.dry_run else ''}Submitting {len(scripts)} jobs...")
        job_ids = []
        for path, name in scripts:
            cmd = f"sbatch {path}"
            if FLAGS.dry_run:
                print(f"  [DRY RUN] {cmd}")
            else:
                result = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True
                )
                if result.returncode == 0:
                    job_id = result.stdout.strip().split()[-1]
                    job_ids.append(job_id)
                    print(f"  {name}: {job_id}")
                else:
                    print(f"  {name}: FAILED - {result.stderr.strip()}")

        if job_ids and not FLAGS.dry_run:
            print(f"\nSubmitted {len(job_ids)} jobs: {job_ids[0]}–{job_ids[-1]}")
            print(f"Monitor: squeue -u $USER | grep uq_")
    else:
        print("\nScripts generated. Run with --submit to sbatch them.")
        print("Or submit manually: for f in $(find "
              f"{OUTPUT_ROOT} -name run.slurm); do sbatch $f; done")


if __name__ == "__main__":
    app.run(main)
