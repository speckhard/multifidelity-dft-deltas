"""Single-target GNN sweep: 2 models x 3 targets x 2 losses x 9 HPs = 108 jobs.

Each job trains a GNN (DeltaPaiNN or DeltaEGNN) on a single target
(energy, bandgap, or volume) using either sMAPE or AsinhL1 loss.
All other loss weights are zeroed out so gradients only flow from the
active head.

Hyperparameter grid: 3 learning rates x 3 hidden dims = 9 combos.
Early stopping uses the validation sMAPE of the active target.

Usage:
    python submit_single_target_sweep.py
    python submit_single_target_sweep.py --root_dir=/some/other/dir
    python submit_single_target_sweep.py --dry_run  # Generate scripts only
"""

import os
import itertools
import subprocess
from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string('root_dir', '/u/dansp/egnn/delta_painn/single_target_sweep_2026_03_10',
                    'Root output directory.')
flags.DEFINE_string('data_file',
    '/u/dansp/egnn/relaxation_data_with_kpoints/data/'
    'delta_combined_relaxations_23_51__25_1_2026_egnn_data.pt',
    'Data file path.')
flags.DEFINE_string('wandb_project', 'gnn-single-target-sweep', 'WandB project.')
flags.DEFINE_boolean('dry_run', False, 'Generate SLURM scripts without submitting.')
flags.DEFINE_list('losses', None, 'Comma-separated loss subset to run (e.g. rmslae). Default: all.')
flags.DEFINE_list('hidden_dims', None, 'Comma-separated hidden dim subset (e.g. 256). Default: all.')
flags.DEFINE_list('targets', None, 'Comma-separated target subset (e.g. lattice,all_geo). Default: all.')
flags.DEFINE_list('models', None, 'Comma-separated model name subset (e.g. egnn_film,attention). Default: all.')

# ──────────────────────────────────────────────────────────────────────
# Sweep grid
# ──────────────────────────────────────────────────────────────────────

MODELS = [
    {
        'name': 'painn',
        'model_type': 'painn',
        'config_name': 'painn_denoising_config',
        'hidden_key': 'model.hidden_dim',
        'extra': '++model.use_film=true',
    },
    {
        'name': 'egnn',
        'model_type': 'delta',
        'config_name': 'config',
        'hidden_key': 'model.hidden_features',
        'extra': '',
    },
    {
        'name': 'egnn_film',
        'model_type': 'delta',
        'config_name': 'config',
        'hidden_key': 'model.hidden_features',
        'extra': '++model.use_film=true',
    },
    {
        'name': 'attention',
        'model_type': 'attention',
        'config_name': 'config',
        'hidden_key': 'model.hidden_features',
        'extra': '',
    },
]

TARGETS = {
    'energy': {
        'weights': {'delta_e': 1.0, 'delta_gap': 0.0, 'delta_r': 0.0,
                     'delta_vol': 0.0, 'delta_lat': 0.0},
        'early_stop_metric': 'Val_Loss',
    },
    'bandgap': {
        'weights': {'delta_e': 0.0, 'delta_gap': 1.0, 'delta_r': 0.0,
                     'delta_vol': 0.0, 'delta_lat': 0.0},
        'early_stop_metric': 'Val_Loss',
    },
    'volume': {
        'weights': {'delta_e': 0.0, 'delta_gap': 0.0, 'delta_r': 0.0,
                     'delta_vol': 1.0, 'delta_lat': 0.0},
        'early_stop_metric': 'Val_Loss',
    },
    'lattice': {
        'weights': {'delta_e': 0.0, 'delta_gap': 0.0, 'delta_r': 0.0,
                     'delta_vol': 0.0, 'delta_lat': 1.0},
        'early_stop_metric': 'Val_Loss',
    },
    'all_geo': {
        'weights': {'delta_e': 0.0, 'delta_gap': 0.0, 'delta_r': 0.0,
                     'delta_vol': 1.0, 'delta_lat': 1.0},
        'early_stop_metric': 'Val_Loss',
    },
}

LOSSES = {
    'smape':    {'loss_function': 'smape',    'extra': '++training.smape_epsilon=1e-4'},
    'asinh_l1': {'loss_function': 'asinh_l1', 'extra': '++training.asinh_scale=1e-5'},
    'rmslae':   {'loss_function': 'rmslae',   'extra': ''},
}

LEARNING_RATES = [1e-3, 5e-4, 1e-4]
HIDDEN_DIMS = [32, 64, 128, 256]

# Fixed arch params
FIXED = {
    'model.num_layers': 3,
    'training.batch_size': 32,
    'training.epochs': 800,
}

# ──────────────────────────────────────────────────────────────────────
# SLURM template
# ──────────────────────────────────────────────────────────────────────

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

srun python3.12 /u/dansp/egnn/errorbar_modelling/modelling/gnn/train_pipeline.py \\
    --config-name={config_name} \\
    hydra.run.dir={run_dir} \\
    job_name={job_name} \\
    data_file={data_file} \\
    ++wandb_project={wandb_project} \\
    ++wandb_group={wandb_group} \\
    ++model_type={model_type} \\
    ++seed=42 \\
    ++training.loss_function={loss_function} \\
    ++training.learning_rate={lr} \\
    ++training.early_stop_metric={early_stop_metric} \\
    {hidden_override} \\
    {weight_overrides} \\
    {fixed_overrides} \\
    {loss_extra} \\
    {model_extra}
"""


def main(argv):
    del argv
    base_dir = os.path.abspath(FLAGS.root_dir)

    fixed_overrides = " ".join(f"{k}={v}" for k, v in FIXED.items())

    models = MODELS
    if FLAGS.models:
        models = [m for m in MODELS if m['name'] in FLAGS.models]

    losses = LOSSES
    if FLAGS.losses:
        losses = {k: v for k, v in LOSSES.items() if k in FLAGS.losses}

    hidden_dims = HIDDEN_DIMS
    if FLAGS.hidden_dims:
        hidden_dims = [int(h) for h in FLAGS.hidden_dims]

    targets = TARGETS
    if FLAGS.targets:
        targets = {k: v for k, v in TARGETS.items() if k in FLAGS.targets}

    combos = list(itertools.product(
        models, targets.items(), losses.items(), LEARNING_RATES, hidden_dims))

    print(f"Output directory : {base_dir}")
    print(f"Total jobs       : {len(combos)}")
    print(f"  Models         : {[m['name'] for m in models]}")
    print(f"  Targets        : {list(targets.keys())}")
    print(f"  Losses         : {list(losses.keys())}")
    print(f"  LR grid        : {LEARNING_RATES}")
    print(f"  Hidden dims    : {hidden_dims}")
    print(f"  Dry run        : {FLAGS.dry_run}")
    print()

    submitted = 0
    for model, (tgt_name, tgt_cfg), (loss_name, loss_cfg), lr, hdim in combos:
        job_name = f"st_{model['name']}_{tgt_name}_{loss_name}_lr{lr}_h{hdim}"
        wandb_group = f"st_{model['name']}_{tgt_name}_{loss_name}"
        run_dir = os.path.join(base_dir, model['name'], tgt_name, loss_name,
                               f"lr{lr}_h{hdim}")
        os.makedirs(run_dir, exist_ok=True)

        weight_overrides = " ".join(
            f"++weights.{k}={v}" for k, v in tgt_cfg['weights'].items())

        hidden_override = f"{model['hidden_key']}={hdim}"

        slurm_content = SLURM_TEMPLATE.format(
            job_name=job_name,
            run_dir=run_dir,
            config_name=model['config_name'],
            data_file=FLAGS.data_file,
            wandb_project=FLAGS.wandb_project,
            wandb_group=wandb_group,
            model_type=model['model_type'],
            loss_function=loss_cfg['loss_function'],
            lr=lr,
            early_stop_metric=tgt_cfg['early_stop_metric'],
            hidden_override=hidden_override,
            weight_overrides=weight_overrides,
            fixed_overrides=fixed_overrides,
            loss_extra=loss_cfg['extra'],
            model_extra=model['extra'],
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
