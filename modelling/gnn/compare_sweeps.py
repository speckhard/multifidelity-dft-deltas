"""
Compare top-1 and top-5 ensemble models across multiple EGNN sweep groups.

Runs on MPCDF (requires GPU + wandb access + data file).

Usage:
    python compare_sweeps.py --output_dir /path/to/output --data_file /path/to/data.pt

Outputs:
    - comparison_table.csv: aggregate metrics for all models x splits
    - Per-model prediction CSVs: {sweep_label}_{top1|ensemble5}_{split}.csv
"""

import os
import sys
import torch
import wandb
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
from absl import app, flags
import torch.nn.functional as F

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from modelling.gnn.delta_egnn_model import DeltaGNN
from modelling.gnn.settings_only_egnn_model import SettingsOnlyGNN
from modelling.gnn.delta_egnn_attention_model import DeltaAttentionGNN

FLAGS = flags.FLAGS
flags.DEFINE_string('output_dir', './comparison_output', 'Directory to save all outputs')
flags.DEFINE_string('data_file',
                    '/u/dansp/egnn/relaxation_data_with_kpoints/data/delta_combined_relaxations_23_51__25_1_2026_egnn_data.pt',
                    'Path to .pt data file')
flags.DEFINE_string('select_metric', 'Val/MAE_Energy', 'Val metric to rank runs by')
flags.DEFINE_string('select_mode', 'min', 'min or max')
flags.DEFINE_integer('top_k', 5, 'Number of top models for ensemble')
flags.DEFINE_integer('batch_size', 64, 'Inference batch size')

# ---- Sweep Definitions ----
# Each sweep is: (label, wandb_project, wandb_group)
# UPDATE THESE to match your actual wandb project/group names.
SWEEP_CONFIGS = [
    {
        'label': 'egnn_nll',
        'wandb_project': 'egnn-delta-learning',
        'wandb_group': 'sweep_gnll_val_split_no_overconverged_v3',
    },
    {
        'label': 'egnn_smape',
        'wandb_project': 'egnn-delta-learning',
        'wandb_group': 'sweep_smape_no_overconverged_no_volume_explosions',
    },
    {
        'label': 'egnn_no_cheap_dft',
        'wandb_project': 'egnn-delta-learning',
        'wandb_group': 'sweep_no_cheap_dft_input_NLL_overconverged_no_volume_explosions',
    },
    {
        'label': 'egnn_attention',
        'wandb_project': 'egnn-delta-learning',
        'wandb_group': 'attention_nll_sweep_v1',
    },
    {
        'label': 'egnn_no_lattice',
        'wandb_project': 'egnn-attention',
        'wandb_group': 'no_lattice_sweep_v1',
    },
    {
        'label': 'egnn_no_lattice_smape',
        'wandb_project': 'egnn-attention',
        'wandb_group': 'no_lattice_sweep_smape_v1',
    },
]

WANDB_ENTITY = 'dtts'

# ---- Target keys and names ----
TARGET_KEYS = ['delta_energy', 'delta_gap', 'delta_positions', 'delta_volume']
GEO_KEYS = ['delta_volume', 'delta_a', 'delta_b', 'delta_c',
            'delta_alpha', 'delta_beta', 'delta_gamma']
ALL_KEYS = ['delta_energy', 'delta_gap', 'delta_positions'] + GEO_KEYS


# ============================================================
# Model loading
# ============================================================

def build_model(run_config, device):
    """Instantiate the correct model class from a wandb run config dict."""
    model_type = run_config.get('model_type', 'delta')
    mcfg = run_config['model']

    if model_type == 'settings_only':
        model = SettingsOnlyGNN(
            num_layers=mcfg['num_layers'],
            hidden_features=mcfg['hidden_features'],
            num_precision_settings=mcfg.get('num_precision_settings', 13),
            max_z=mcfg.get('max_z', 120),
        )
    elif model_type == 'attention':
        model = DeltaAttentionGNN(
            num_layers=mcfg['num_layers'],
            hidden_features=mcfg['hidden_features'],
            num_cheap_dft_inputs=mcfg.get('num_cheap_dft_inputs', 12),
            num_precision_settings=mcfg.get('num_precision_settings', 13),
            num_geo_inputs=mcfg.get('num_geo_inputs', 7),
            max_z=mcfg.get('max_z', 120),
        )
    else:
        model = DeltaGNN(
            num_layers=mcfg['num_layers'],
            hidden_features=mcfg['hidden_features'],
            num_cheap_dft_inputs=mcfg.get('num_cheap_dft_inputs', 12),
            num_precision_settings=mcfg.get('num_precision_settings', 13),
            num_geo_inputs=mcfg.get('num_geo_inputs', 7),
            max_z=mcfg.get('max_z', 120),
        )

    return model.to(device)


def download_and_load_model(run, download_dir, device):
    """Download checkpoint from a wandb run and return loaded model."""
    ckpt_name = 'best_delta_model.pth'
    files = [f.name for f in run.files()]
    if ckpt_name not in files:
        print(f"  WARNING: {ckpt_name} not found for run {run.name}. Trying last_model.pth")
        ckpt_name = 'last_model.pth'
        if ckpt_name not in files:
            raise FileNotFoundError(f"No usable checkpoint in run {run.name}. Files: {files}")

    run_dl_dir = os.path.join(download_dir, run.id)
    os.makedirs(run_dl_dir, exist_ok=True)
    run.file(ckpt_name).download(root=run_dl_dir, replace=True)
    ckpt_path = os.path.join(run_dl_dir, ckpt_name)

    model = build_model(run.config, device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    model.eval()
    return model


# ============================================================
# Inference
# ============================================================

@torch.no_grad()
def run_inference(model, loader, device):
    """
    Run model inference and return raw arrays dict.

    Returns dict with keys like 'delta_energy', 'delta_gap', etc.
    Each value is a dict with 'pred', 'true', 'sigma' numpy arrays.
    """
    arrays = {k: {'pred': [], 'true': [], 'sigma': []} for k in ALL_KEYS}

    for data in tqdm(loader, desc="  Inference", leave=False):
        data = data.to(device)
        pred_pos, pred_eng, pred_gap, pred_geo = model(data)

        # Unpack (mean, logvar) tuples
        mu_eng, lv_eng = pred_eng
        mu_gap, lv_gap = pred_gap
        mu_geo, lv_geo = pred_geo
        mu_pos, lv_pos = pred_pos

        # Energy
        arrays['delta_energy']['pred'].append(mu_eng.cpu().numpy())
        arrays['delta_energy']['true'].append(data.delta_total_energy_per_atom.cpu().numpy())
        arrays['delta_energy']['sigma'].append(torch.exp(0.5 * lv_eng).cpu().numpy())

        # Gap
        arrays['delta_gap']['pred'].append(mu_gap.cpu().numpy())
        arrays['delta_gap']['true'].append(data.delta_homo_lumo_gap.cpu().numpy())
        arrays['delta_gap']['sigma'].append(torch.exp(0.5 * lv_gap).cpu().numpy())

        # Positions: compute norm of the displacement error vector,
        # consistent with train_pipeline.py's MAE_delta_positions.
        # We store norm(pred) and norm(true) for scatter plots,
        # but also the displacement error norm for correct MAE.
        pos_diff = mu_pos - data.delta_relaxed_atom_positions
        pos_error_norm = torch.norm(pos_diff, dim=1)  # norm of (pred - true)
        p_pos_norm = torch.norm(mu_pos, dim=1)
        t_pos_norm = torch.norm(data.delta_relaxed_atom_positions, dim=1)
        s_pos = torch.exp(0.5 * lv_pos).squeeze(-1)  # squeeze only last dim, safe for batch=1
        if s_pos.dim() > 1:
            s_pos = s_pos.mean(dim=1)

        arrays['delta_positions']['pred'].append(p_pos_norm.cpu().numpy())
        arrays['delta_positions']['true'].append(t_pos_norm.cpu().numpy())
        arrays['delta_positions']['sigma'].append(s_pos.cpu().numpy())
        arrays['delta_positions'].setdefault('error_norm', []).append(
            pos_error_norm.cpu().numpy()
        )

        # Geometry: volume + 6 lattice params
        t_vol = data.delta_final_volume_per_atom.view(-1, 1)
        t_lat = data.delta_lattice_params.view(-1, 6)
        t_geo = torch.cat([t_vol, t_lat], dim=1)

        for i, key in enumerate(GEO_KEYS):
            arrays[key]['pred'].append(mu_geo[:, i].cpu().numpy())
            arrays[key]['true'].append(t_geo[:, i].cpu().numpy())
            arrays[key]['sigma'].append(torch.exp(0.5 * lv_geo[:, i]).cpu().numpy())

    # Flatten
    for k in arrays:
        for sub in list(arrays[k].keys()):
            if arrays[k][sub]:  # skip empty lists
                arrays[k][sub] = np.concatenate(arrays[k][sub]).flatten()
            else:
                arrays[k][sub] = np.array([])

    return arrays


# ============================================================
# Ensemble logic
# ============================================================

def ensemble_arrays(list_of_arrays):
    """
    Combine predictions from K models into an ensemble.

    Args:
        list_of_arrays: list of K arrays dicts (each from run_inference).

    Returns:
        ensemble arrays dict with same structure, plus 'sigma_epi' and 'sigma_alea' keys.
    """
    K = len(list_of_arrays)
    keys = list_of_arrays[0].keys()
    ensemble = {}

    for key in keys:
        # True values are the same across all models
        true = list_of_arrays[0][key]['true']

        # Stack predictions: [K, N]
        all_preds = np.stack([a[key]['pred'] for a in list_of_arrays], axis=0)
        all_sigmas = np.stack([a[key]['sigma'] for a in list_of_arrays], axis=0)

        # Ensemble mean
        mu_ens = np.mean(all_preds, axis=0)

        # Aleatoric: average of individual variances
        sigma_alea_sq = np.mean(all_sigmas ** 2, axis=0)

        # Epistemic: variance of means (model disagreement)
        sigma_epi_sq = np.var(all_preds, axis=0)

        # Total ensemble std
        sigma_total = np.sqrt(sigma_alea_sq + sigma_epi_sq)

        ensemble[key] = {
            'pred': mu_ens,
            'true': true,
            'sigma': sigma_total,
            'sigma_alea': np.sqrt(sigma_alea_sq),
            'sigma_epi': np.sqrt(sigma_epi_sq),
        }

        # For positions, also ensemble the error_norm
        if 'error_norm' in list_of_arrays[0][key]:
            all_err_norms = np.stack(
                [a[key]['error_norm'] for a in list_of_arrays], axis=0
            )
            ensemble[key]['error_norm'] = np.mean(all_err_norms, axis=0)

    return ensemble


# ============================================================
# Metrics
# ============================================================

def calculate_rmsle(y_true, y_pred, epsilon=1e-7):
    log_pred = np.log(np.abs(y_pred) + epsilon)
    log_true = np.log(np.abs(y_true) + epsilon)
    return np.sqrt(np.mean((log_pred - log_true) ** 2))


def calculate_smape(y_true, y_pred, epsilon=1e-7):
    num = np.abs(y_pred - y_true)
    denom = np.abs(y_pred) + np.abs(y_true) + epsilon
    return 100.0 * 2.0 * np.mean(num / denom)


def calculate_mag_acc(y_true, y_pred):
    bins = np.array([1e-3, 1e-2, 1e-1, 1.0, 10.0])
    p_bins = np.digitize(np.abs(y_pred), bins)
    t_bins = np.digitize(np.abs(y_true), bins)
    return np.mean(p_bins == t_bins)


def compute_metrics_for_targets(arrays, targets=None):
    """Compute MAE, sMAPE, MagAcc, RMSLE for each target key."""
    if targets is None:
        targets = TARGET_KEYS
    metrics = {}
    for key in targets:
        if key not in arrays:
            continue
        y_true = arrays[key]['true']
        y_pred = arrays[key]['pred']
        if len(y_true) == 0:
            continue

        # For positions, MAE should be mean of ||pred - true|| (norm of vector error),
        # not mean of |norm(pred) - norm(true)| (error of norms).
        # This matches train_pipeline.py's MAE_delta_positions.
        if key == 'delta_positions' and 'error_norm' in arrays[key]:
            metrics[f'{key}/MAE'] = np.mean(arrays[key]['error_norm'])
        else:
            metrics[f'{key}/MAE'] = mean_absolute_error(y_true, y_pred)

        metrics[f'{key}/sMAPE'] = calculate_smape(y_true, y_pred)
        metrics[f'{key}/MagAcc'] = calculate_mag_acc(y_true, y_pred)
        metrics[f'{key}/RMSLE'] = calculate_rmsle(y_true, y_pred)
    return metrics


def compute_target_statistics(loaders, device, targets=None):
    """
    Compute mean, std, median, min, max of |true target| for each split.

    These are model-independent and provide essential context for interpreting
    metrics (e.g., MAE / std tells you how much better than a mean-predictor).

    Returns a list of dicts (one per split), ready to become a DataFrame.
    """
    if targets is None:
        targets = TARGET_KEYS

    # We need to extract true values from the data loaders.
    # For graph-level targets we iterate the loader once.
    target_to_attr = {
        'delta_energy': 'delta_total_energy_per_atom',
        'delta_gap': 'delta_homo_lumo_gap',
        'delta_volume': 'delta_final_volume_per_atom',
    }

    rows = []
    for split_name, loader in loaders.items():
        true_vals = {t: [] for t in targets}

        for data in loader:
            data = data.to(device)
            for key in targets:
                if key == 'delta_positions':
                    norms = torch.norm(data.delta_relaxed_atom_positions, dim=1)
                    true_vals[key].append(norms.cpu().numpy())
                elif key in target_to_attr:
                    true_vals[key].append(getattr(data, target_to_attr[key]).cpu().numpy())
                elif key == 'delta_a':
                    true_vals[key].append(data.delta_lattice_params[:, 0].cpu().numpy())
                elif key == 'delta_b':
                    true_vals[key].append(data.delta_lattice_params[:, 1].cpu().numpy())
                elif key == 'delta_c':
                    true_vals[key].append(data.delta_lattice_params[:, 2].cpu().numpy())
                elif key == 'delta_alpha':
                    true_vals[key].append(data.delta_lattice_params[:, 3].cpu().numpy())
                elif key == 'delta_beta':
                    true_vals[key].append(data.delta_lattice_params[:, 4].cpu().numpy())
                elif key == 'delta_gamma':
                    true_vals[key].append(data.delta_lattice_params[:, 5].cpu().numpy())

        row = {'split': split_name}
        for key in targets:
            if not true_vals[key]:
                continue
            vals = np.concatenate(true_vals[key]).flatten()
            abs_vals = np.abs(vals)
            row[f'{key}/mean'] = np.mean(vals)
            row[f'{key}/std'] = np.std(vals)
            row[f'{key}/abs_mean'] = np.mean(abs_vals)
            row[f'{key}/abs_std'] = np.std(abs_vals)
            row[f'{key}/median'] = np.median(vals)
            row[f'{key}/min'] = np.min(vals)
            row[f'{key}/max'] = np.max(vals)
            row[f'{key}/count'] = len(vals)
        rows.append(row)

    return rows


# ============================================================
# CSV I/O
# ============================================================

def arrays_to_csv(arrays, filepath, targets=None):
    """Save raw pred/true/sigma arrays to a CSV file."""
    if targets is None:
        targets = ALL_KEYS
    df_dict = {}
    for key in targets:
        if key not in arrays:
            continue
        df_dict[f'{key}_pred'] = arrays[key]['pred']
        df_dict[f'{key}_true'] = arrays[key]['true']
        df_dict[f'{key}_sigma'] = arrays[key]['sigma']
        # Extra uncertainty columns for ensembles
        if 'sigma_alea' in arrays[key]:
            df_dict[f'{key}_sigma_alea'] = arrays[key]['sigma_alea']
            df_dict[f'{key}_sigma_epi'] = arrays[key]['sigma_epi']

    # Handle different array lengths (positions are per-atom, others per-graph)
    # Group by length and save separately if needed
    lengths = {k: len(v) for k, v in df_dict.items()}
    unique_lengths = set(lengths.values())

    if len(unique_lengths) == 1:
        pd.DataFrame(df_dict).to_csv(filepath, index=False)
    else:
        # Positions are per-atom (different length from graph-level targets)
        # Save graph-level and atom-level separately
        graph_cols = {k: v for k, v in df_dict.items() if 'positions' not in k}
        atom_cols = {k: v for k, v in df_dict.items() if 'positions' in k}

        if graph_cols:
            pd.DataFrame(graph_cols).to_csv(filepath, index=False)
        if atom_cols:
            pos_path = filepath.replace('.csv', '_positions.csv')
            pd.DataFrame(atom_cols).to_csv(pos_path, index=False)

    print(f"  Saved: {filepath}")


# ============================================================
# WandB helpers
# ============================================================

def get_top_k_runs(api, project, group, metric, mode, k):
    """Return the top-k runs from a wandb group sorted by metric."""
    full_path = f"{WANDB_ENTITY}/{project}"
    runs = api.runs(full_path, filters={"group": group})

    scored_runs = []
    for run in runs:
        if run.state != 'finished':
            continue
        if metric not in run.summary:
            continue
        scored_runs.append((run, run.summary[metric]))

    if not scored_runs:
        print(f"  WARNING: No finished runs with metric '{metric}' in {full_path} / {group}")
        return []

    reverse = (mode == 'max')
    scored_runs.sort(key=lambda x: x[1], reverse=reverse)
    top_runs = [r for r, _ in scored_runs[:k]]

    print(f"  Found {len(scored_runs)} valid runs. Top-{k}:")
    for i, (r, v) in enumerate(scored_runs[:k]):
        print(f"    #{i+1}: {r.name} ({metric}={v:.6f})")

    return top_runs


# ============================================================
# Main
# ============================================================

def main(argv):
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Output: {FLAGS.output_dir}")

    # Load data once
    print(f"Loading data: {FLAGS.data_file}")
    loaded = torch.load(FLAGS.data_file, weights_only=False)
    all_graphs = loaded['graphs']

    splits = {
        'train': [d for d in all_graphs if d.split == 'train'],
        'val': [d for d in all_graphs if d.split == 'val'],
        'test': [d for d in all_graphs if d.split == 'test'],
    }
    for name, graphs in splits.items():
        print(f"  {name}: {len(graphs)} graphs")

    loaders = {
        name: DataLoader(graphs, batch_size=FLAGS.batch_size, shuffle=False)
        for name, graphs in splits.items()
        if len(graphs) > 0
    }

    # --- Compute and save target statistics (model-independent) ---
    print("\nComputing target statistics...")
    stats_rows = compute_target_statistics(loaders, device, targets=TARGET_KEYS)
    stats_df = pd.DataFrame(stats_rows)
    stats_path = os.path.join(FLAGS.output_dir, 'target_statistics.csv')
    stats_df.to_csv(stats_path, index=False)
    print(f"Saved target statistics: {stats_path}")
    print(stats_df.to_string(index=False))

    api = wandb.Api()
    dl_dir = os.path.join(FLAGS.output_dir, '_checkpoints')
    all_comparison_rows = []

    for sweep_cfg in SWEEP_CONFIGS:
        label = sweep_cfg['label']
        project = sweep_cfg['wandb_project']
        group = sweep_cfg['wandb_group']

        print(f"\n{'='*60}")
        print(f"Sweep: {label} (project={project}, group={group})")
        print(f"{'='*60}")

        top_runs = get_top_k_runs(
            api, project, group,
            FLAGS.select_metric, FLAGS.select_mode, FLAGS.top_k
        )

        if not top_runs:
            print(f"  SKIPPING sweep {label}: no valid runs found.")
            continue

        # --- Load all top-K models ---
        models = []
        for i, run in enumerate(top_runs):
            print(f"  Loading model #{i+1}: {run.name}")
            try:
                model = download_and_load_model(run, dl_dir, device)
                models.append(model)
            except Exception as e:
                print(f"    ERROR loading model: {e}")

        if not models:
            print(f"  SKIPPING sweep {label}: no models loaded successfully.")
            continue

        # --- Run inference for all models on all splits ---
        all_model_arrays = {}  # {split_name: [arrays_model_0, arrays_model_1, ...]}
        for split_name, loader in loaders.items():
            print(f"\n  Split: {split_name}")
            split_arrays = []
            for i, model in enumerate(models):
                print(f"    Model #{i+1}/{len(models)}")
                arrays = run_inference(model, loader, device)
                split_arrays.append(arrays)
            all_model_arrays[split_name] = split_arrays

        # --- Top-1: use first model's arrays ---
        for split_name in loaders:
            top1_arrays = all_model_arrays[split_name][0]

            # Save CSV
            csv_path = os.path.join(FLAGS.output_dir, f"{label}_top1_{split_name}.csv")
            arrays_to_csv(top1_arrays, csv_path)

            # Compute metrics
            metrics = compute_metrics_for_targets(top1_arrays)
            row = {'model': f'{label}_top1', 'split': split_name}
            row.update(metrics)
            all_comparison_rows.append(row)

        # --- Ensemble-K ---
        if len(models) > 1:
            for split_name in loaders:
                ens_arrays = ensemble_arrays(all_model_arrays[split_name])

                csv_path = os.path.join(FLAGS.output_dir, f"{label}_ensemble{len(models)}_{split_name}.csv")
                arrays_to_csv(ens_arrays, csv_path)

                metrics = compute_metrics_for_targets(ens_arrays)
                row = {'model': f'{label}_ensemble{len(models)}', 'split': split_name}
                row.update(metrics)
                all_comparison_rows.append(row)

        # Free GPU memory
        del models
        torch.cuda.empty_cache()

    # --- Save comparison table ---
    if all_comparison_rows:
        df = pd.DataFrame(all_comparison_rows)

        # Reorder columns
        meta_cols = ['model', 'split']
        metric_cols = sorted([c for c in df.columns if c not in meta_cols])
        df = df[meta_cols + metric_cols]

        csv_path = os.path.join(FLAGS.output_dir, 'comparison_table.csv')
        df.to_csv(csv_path, index=False)
        print(f"\n{'='*60}")
        print(f"Saved comparison table: {csv_path}")

        # Print test-only summary
        test_df = df[df['split'] == 'test'].drop(columns=['split'])
        if not test_df.empty:
            print("\n--- TEST SET SUMMARY ---")
            print(test_df.to_string(index=False))
    else:
        print("\nNo results collected. Check sweep configs and wandb access.")

    print("\nDone!")


if __name__ == "__main__":
    app.run(main)
