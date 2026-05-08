#!/usr/bin/env python3.12
"""Generate and submit SLURM jobs for learning curve experiments.

Trains sMAPE-best champion models at data fractions [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
for both full and recommender modes.

MT models predict all targets simultaneously (energy + bandgap + volume).
ST models predict one target each.

Usage:
    python submit_learning_curves.py                    # generate only
    python submit_learning_curves.py --submit           # generate + sbatch
    python submit_learning_curves.py --submit --dry_run # print commands
    python submit_learning_curves.py --submit --sweep=st_full
"""

import subprocess
from pathlib import Path

from absl import app, flags

FLAGS = flags.FLAGS
flags.DEFINE_bool("submit", False, "Submit jobs via sbatch.")
flags.DEFINE_bool("dry_run", False, "Print sbatch commands only.")
flags.DEFINE_string("sweep", "all",
    "Which sweep: mt_full, st_full, st_rec, all")
flags.DEFINE_list("fractions", ["0.5", "0.6", "0.7", "0.8", "0.9", "1.0"],
                  "Training data fractions.")

# ── Paths ──
CODE_DIR = Path("/u/dansp/egnn/errorbar_modelling")
TRAIN_SCRIPT = CODE_DIR / "modelling" / "gnn" / "train_pipeline.py"
DATA_FILE = Path(
    "/u/dansp/egnn/relaxation_data_with_kpoints/data/"
    "delta_combined_relaxations_23_51__25_1_2026_egnn_data.pt"
)
OUTPUT_ROOT = Path("/u/dansp/egnn/delta_painn/learning_curves_2026_04_04")
VENV = "/u/dansp/egnn/py12_venv/bin/activate"
WANDB_PROJECT = "gnn-learning-curves"

# ── All weights on (MT) vs single-target (ST) ──
MT_WEIGHTS = '++weights.delta_e=1.0 ++weights.delta_gap=1.0 ++weights.delta_r=1.0 ++weights.delta_vol=1.0 ++weights.delta_lat=1.0'
ST_WEIGHTS = {
    'energy':  '++weights.delta_e=1.0 ++weights.delta_gap=0.0 ++weights.delta_r=0.0 ++weights.delta_vol=0.0 ++weights.delta_lat=0.0',
    'bandgap': '++weights.delta_e=0.0 ++weights.delta_gap=1.0 ++weights.delta_r=0.0 ++weights.delta_vol=0.0 ++weights.delta_lat=0.0',
    'volume':  '++weights.delta_e=0.0 ++weights.delta_gap=0.0 ++weights.delta_r=0.0 ++weights.delta_vol=1.0 ++weights.delta_lat=0.0',
}

# ── Champion configs ──
# Each: (tag, model_type, config_name, hidden_key, hidden_val, lr, loss, use_film,
#         num_cheap_dft_inputs, num_geo_inputs, seed, weights, targets_label)

CONFIGS = []

# --- MT Full: EGNN+Att+FiLM (energy champion 8.34%) ---
CONFIGS.append({
    'tag': 'mt_full_egnn_att_film',
    'mode': 'full', 'scope': 'mt',
    'model_type': 'attention_film', 'config_name': 'config',
    'hidden_key': 'model.hidden_features', 'hidden_val': 32,
    'lr': 5e-4, 'loss': 'smape', 'seed': 123,
    'use_film': True,
    'num_cheap_dft_inputs': 12, 'num_geo_inputs': 7,
    'weights': MT_WEIGHTS,
    'num_layers': 3,
})

# --- MT Full: PaiNN+Att+FiLM (bandgap 30.97%, volume 23.34%) ---
CONFIGS.append({
    'tag': 'mt_full_painn_att',
    'mode': 'full', 'scope': 'mt',
    'model_type': 'painn_attention', 'config_name': 'painn_attention_config',
    'hidden_key': 'model.hidden_dim', 'hidden_val': 128,
    'lr': 5e-4, 'loss': 'smape', 'seed': 42,
    'use_film': True,
    'num_cheap_dft_inputs': 12, 'num_geo_inputs': 7,
    'weights': MT_WEIGHTS,
    'num_layers': 3,
})

# --- ST Full: Energy — EGNN+FiLM (8.41%) ---
CONFIGS.append({
    'tag': 'st_full_energy_egnn_film',
    'mode': 'full', 'scope': 'st', 'target': 'energy',
    'model_type': 'delta', 'config_name': 'config',
    'hidden_key': 'model.hidden_features', 'hidden_val': 128,
    'lr': 1e-4, 'loss': 'smape', 'seed': 42,
    'use_film': True,
    'num_cheap_dft_inputs': 12, 'num_geo_inputs': 7,
    'weights': ST_WEIGHTS['energy'],
    'num_layers': 3,
})

# --- ST Full: Bandgap — EGNN+FiLM (29.58%) ---
CONFIGS.append({
    'tag': 'st_full_bandgap_egnn_film',
    'mode': 'full', 'scope': 'st', 'target': 'bandgap',
    'model_type': 'delta', 'config_name': 'config',
    'hidden_key': 'model.hidden_features', 'hidden_val': 64,
    'lr': 5e-4, 'loss': 'smape', 'seed': 42,
    'use_film': True,
    'num_cheap_dft_inputs': 12, 'num_geo_inputs': 7,
    'weights': ST_WEIGHTS['bandgap'],
    'num_layers': 3,
})

# --- ST Full: Volume — PaiNN+Att+FiLM (27.35%) ---
CONFIGS.append({
    'tag': 'st_full_volume_painn_att',
    'mode': 'full', 'scope': 'st', 'target': 'volume',
    'model_type': 'painn_attention', 'config_name': 'painn_attention_config',
    'hidden_key': 'model.hidden_dim', 'hidden_val': 256,
    'lr': 1e-4, 'loss': 'smape', 'seed': 42,
    'use_film': True,
    'num_cheap_dft_inputs': 12, 'num_geo_inputs': 7,
    'weights': ST_WEIGHTS['volume'],
    'num_layers': 3,
})

# --- ST Recommender: Energy — EGNN+FiLM (8.10%) ---
CONFIGS.append({
    'tag': 'st_rec_energy_egnn_film',
    'mode': 'recommender', 'scope': 'st', 'target': 'energy',
    'model_type': 'delta', 'config_name': 'config',
    'hidden_key': 'model.hidden_features', 'hidden_val': 256,
    'lr': 1e-4, 'loss': 'smape', 'seed': 42,
    'use_film': True,
    'num_cheap_dft_inputs': 0, 'num_geo_inputs': 0,
    'weights': ST_WEIGHTS['energy'],
    'num_layers': 3,
})

# --- ST Recommender: Bandgap — EGNN+FiLM (32.46%) ---
CONFIGS.append({
    'tag': 'st_rec_bandgap_egnn_film',
    'mode': 'recommender', 'scope': 'st', 'target': 'bandgap',
    'model_type': 'delta', 'config_name': 'config',
    'hidden_key': 'model.hidden_features', 'hidden_val': 64,
    'lr': 5e-4, 'loss': 'smape', 'seed': 42,
    'use_film': True,
    'num_cheap_dft_inputs': 0, 'num_geo_inputs': 0,
    'weights': ST_WEIGHTS['bandgap'],
    'num_layers': 3,
})

# --- ST Recommender: Volume — EGNN+FiLM (28.18% from val, using asinh) ---
CONFIGS.append({
    'tag': 'st_rec_volume_egnn_film',
    'mode': 'recommender', 'scope': 'st', 'target': 'volume',
    'model_type': 'delta', 'config_name': 'config',
    'hidden_key': 'model.hidden_features', 'hidden_val': 64,
    'lr': 5e-4, 'loss': 'asinh_l1', 'seed': 42,
    'use_film': True,
    'num_cheap_dft_inputs': 0, 'num_geo_inputs': 0,
    'weights': ST_WEIGHTS['volume'],
    'num_layers': 3,
})

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
    ++training.loss_function={loss} \\
    ++training.learning_rate={lr} \\
    ++training.early_stop_metric=Val_Loss \\
    ++training.train_fraction={train_fraction} \\
    {hidden_key}={hidden_val} \\
    {weights} \\
    model.num_layers={num_layers} training.batch_size=32 training.epochs=800 \\
    ++training.smape_epsilon=1e-4 \\
    ++model.use_film={use_film} \\
    ++model.num_cheap_dft_inputs={num_cheap_dft_inputs} \\
    ++model.num_geo_inputs={num_geo_inputs}
"""


def main(_):
    fractions = [float(f) for f in FLAGS.fractions]
    sweep = FLAGS.sweep.lower()

    # Filter configs by sweep type
    if sweep == 'all':
        configs = CONFIGS
    elif sweep == 'mt_full':
        configs = [c for c in CONFIGS if c['scope'] == 'mt']
    elif sweep == 'st_full':
        configs = [c for c in CONFIGS if c['scope'] == 'st' and c['mode'] == 'full']
    elif sweep == 'st_rec':
        configs = [c for c in CONFIGS if c['scope'] == 'st' and c['mode'] == 'recommender']
    else:
        raise ValueError(f"Unknown sweep: {sweep}")

    scripts = []
    for cfg in configs:
        for frac in fractions:
            frac_str = f"{frac:.1f}".replace('.', '')
            tag = cfg['tag']
            job_name = f"lc_{tag}_f{frac_str}"
            run_dir = OUTPUT_ROOT / tag / f"frac_{frac_str}"
            run_dir.mkdir(parents=True, exist_ok=True)

            wandb_group = f"lc_{tag}"

            script = SLURM_TEMPLATE.format(
                job_name=job_name,
                run_dir=run_dir,
                venv=VENV,
                train_script=TRAIN_SCRIPT,
                config_name=cfg['config_name'],
                data_file=DATA_FILE,
                wandb_project=WANDB_PROJECT,
                wandb_group=wandb_group,
                model_type=cfg['model_type'],
                seed=cfg['seed'],
                loss=cfg['loss'],
                lr=cfg['lr'],
                train_fraction=frac,
                hidden_key=cfg['hidden_key'],
                hidden_val=cfg['hidden_val'],
                weights=cfg['weights'],
                num_layers=cfg['num_layers'],
                use_film=str(cfg['use_film']).lower(),
                num_cheap_dft_inputs=cfg['num_cheap_dft_inputs'],
                num_geo_inputs=cfg['num_geo_inputs'],
            )

            (run_dir / "run.slurm").write_text(script)
            scripts.append((run_dir / "run.slurm", job_name))

    print(f"Generated {len(scripts)} SLURM scripts ({len(configs)} configs × {len(fractions)} fractions):")
    for cfg in configs:
        print(f"  {cfg['tag']}: {cfg['model_type']} h={cfg['hidden_val']} "
              f"lr={cfg['lr']} {cfg['loss']} seed={cfg['seed']}")
    print()

    if FLAGS.submit:
        print(f"{'DRY RUN — ' if FLAGS.dry_run else ''}Submitting {len(scripts)} jobs...")
        job_ids = []
        for path, name in scripts:
            cmd = f"sbatch {path}"
            if FLAGS.dry_run:
                print(f"  [DRY RUN] {cmd}")
            else:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if result.returncode == 0:
                    jid = result.stdout.strip().split()[-1]
                    job_ids.append(jid)
                    print(f"  {name}: {jid}")
                else:
                    print(f"  {name}: FAILED — {result.stderr.strip()}")
        if job_ids and not FLAGS.dry_run:
            print(f"\nSubmitted {len(job_ids)} jobs: {job_ids[0]}–{job_ids[-1]}")
    else:
        print(f"Run with --submit to sbatch. Or --submit --dry_run to preview.")


if __name__ == "__main__":
    app.run(main)
