"""
Export XGBoost sweep results: load best models, predict on all splits, save CSVs.

Auto-discovers all (model_type, target, loss) combos from the sweep output tree:
    xgb_sweep_*/xgb_gbdt_energy/mae/xgb_gbdt_energy_mae_best.joblib

Handles rmslae specially: compares predictions against |y_true| (magnitude only).

Usage:
    python rf_export.py \
        --data_file /path/to/data.csv \
        --model_dir /path/to/xgb_sweep_2026_03_09 \
        --output_dir ./xgb_export
"""

import os
import sys
import glob
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from absl import app, flags

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from modelling.rf.rf_config import (
    TARGET_GROUPS, LOSS_CONFIGS,
    calculate_smape, calculate_mag_acc, rmsle_abs_func,
)
from modelling.rf.data_loader import load_and_clean_data, get_train_test_split

FLAGS = flags.FLAGS
try:
    flags.DEFINE_string('data_file', '', 'Path to CSV data file')
    flags.DEFINE_string('model_dir', './results', 'Root directory of xgb_sweep output')
    flags.DEFINE_string('output_dir', './xgb_export', 'Directory to save export CSVs')
except flags.DuplicateFlagError:
    pass

# Column name mapping for consistency with GNN comparison tables
COL_NAME_MAP = {
    'delta_total_energy_per_atom': 'delta_energy',
    'delta_homo_lumo_gap': 'delta_gap',
    'delta_final_volume_per_atom': 'delta_volume',
    'delta_relaxed_a_len': 'delta_a',
    'delta_relaxed_b_len': 'delta_b',
    'delta_relaxed_c_len': 'delta_c',
    'delta_relaxed_alpha_angle': 'delta_alpha',
    'delta_relaxed_beta_angle': 'delta_beta',
    'delta_relaxed_gamma_angle': 'delta_gamma',
}

MODELS = ['xgb_gbdt', 'xgb_rf']
TARGETS = ['energy', 'bandgap', 'volume', 'geometry', 'all_scalar']
LOSSES = ['mae', 'smape', 'asinh', 'rmslae']


def compute_metrics(y_true, y_pred, col_name, magnitude_only=False):
    """Compute standardized metrics for a single target column.

    For magnitude_only (rmslae): y_true and y_pred should already be
    non-negative. RMSLAE is computed directly on the values without
    redundant np.abs() calls.
    """
    metrics = {
        f'{col_name}/MAE': mean_absolute_error(y_true, y_pred),
        f'{col_name}/sMAPE': calculate_smape(y_true, y_pred),
        f'{col_name}/MagAcc': calculate_mag_acc(y_true, y_pred),
    }
    if magnitude_only:
        # y_true and y_pred are already non-negative magnitudes
        eps = 1e-4
        metrics[f'{col_name}/RMSLAE'] = np.sqrt(np.mean(
            (np.log(y_true + eps) - np.log(y_pred + eps)) ** 2
        ))
    else:
        metrics[f'{col_name}/RMSLAE'] = rmsle_abs_func(y_true, y_pred)
    return metrics


def discover_models(model_dir):
    """Auto-discover trained models from sweep output tree.

    Looks for: {model_dir}/{model}_{target}/{loss}/*_best.joblib

    Returns list of (model_type, target_key, loss_key, model_path).
    """
    found = []
    for model_type in MODELS:
        for target_key in TARGETS:
            for loss_key in LOSSES:
                combo_dir = os.path.join(
                    model_dir, f"{model_type}_{target_key}", loss_key
                )
                pattern = os.path.join(combo_dir, "*_best.joblib")
                matches = glob.glob(pattern)
                if matches:
                    found.append((model_type, target_key, loss_key, matches[0]))
    return found


def main(argv):
    os.makedirs(FLAGS.output_dir, exist_ok=True)

    # Discover models
    models = discover_models(FLAGS.model_dir)
    if not models:
        print(f"No models found in {FLAGS.model_dir}")
        print(f"Expected structure: {{model_dir}}/xgb_gbdt_energy/mae/*_best.joblib")
        return

    print(f"Found {len(models)} trained models in {FLAGS.model_dir}")

    # Load data once
    df, X = load_and_clean_data(FLAGS.data_file)

    all_rows = []

    for model_type, target_key, loss_key, model_path in models:
        is_magnitude = LOSS_CONFIGS[loss_key]['magnitude_only']
        model_label = f"{model_type}_{target_key}_{loss_key}"

        print(f"\n{'='*60}")
        print(f"{model_label} {'(magnitude)' if is_magnitude else ''}")
        print(f"{'='*60}")

        model = joblib.load(model_path)
        target_cols = TARGET_GROUPS[target_key]
        X_train, y_train, X_test, y_test = get_train_test_split(df, X, target_cols)

        for split_name, X_split, y_split in [('train', X_train, y_train),
                                              ('test', X_test, y_test)]:
            print(f"  {split_name} ({len(X_split)} samples)...")
            y_pred = model.predict(X_split)

            # Normalize shapes
            if len(y_pred.shape) == 1:
                y_pred = y_pred.reshape(-1, 1)
            if hasattr(y_split, 'values'):
                y_true = y_split.values
            else:
                y_true = np.array(y_split)
            if len(y_true.shape) == 1:
                y_true = y_true.reshape(-1, 1)

            csv_dict = {}
            metrics_row = {
                'model': model_label,
                'model_type': model_type,
                'target': target_key,
                'loss': loss_key,
                'split': split_name,
                'magnitude_only': is_magnitude,
            }

            for i, orig_col in enumerate(target_cols):
                std_name = COL_NAME_MAP.get(orig_col, orig_col)
                yp = y_pred[:, i]
                yt = y_true[:, i]

                if is_magnitude:
                    # rmslae: evaluate against |y_true|
                    yt_eval = np.abs(yt)
                    csv_dict[f'{std_name}_pred_magnitude'] = yp
                    csv_dict[f'{std_name}_true_magnitude'] = yt_eval
                else:
                    yt_eval = yt
                    csv_dict[f'{std_name}_pred'] = yp
                    csv_dict[f'{std_name}_true'] = yt_eval

                m = compute_metrics(yt_eval, yp, std_name, magnitude_only=is_magnitude)
                metrics_row.update(m)

                print(f"    {std_name}: MAE={m[f'{std_name}/MAE']:.6f}, "
                      f"sMAPE={m[f'{std_name}/sMAPE']:.2f}%, "
                      f"MagAcc={m[f'{std_name}/MagAcc']:.4f}, "
                      f"RMSLAE={m[f'{std_name}/RMSLAE']:.6f}")

            # Save pred/true CSV
            csv_path = os.path.join(
                FLAGS.output_dir, f"{model_label}_{split_name}.csv"
            )
            pd.DataFrame(csv_dict).to_csv(csv_path, index=False)

            all_rows.append(metrics_row)

    # Save aggregate metrics table
    if all_rows:
        metrics_df = pd.DataFrame(all_rows)
        metrics_path = os.path.join(FLAGS.output_dir, 'xgb_comparison_table.csv')
        metrics_df.to_csv(metrics_path, index=False)
        print(f"\nSaved aggregate metrics: {metrics_path}")

        # Print test-set summary
        test_df = metrics_df[metrics_df['split'] == 'test']
        if not test_df.empty:
            print(f"\n{'='*60}")
            print("XGB TEST SET SUMMARY")
            print(f"{'='*60}")
            # Show compact summary: one row per model, key metrics only
            summary_cols = ['model', 'loss', 'magnitude_only']
            metric_cols = [c for c in test_df.columns if '/' in c]
            print(test_df[summary_cols + metric_cols].to_string(index=False))

    print("\nDone!")


if __name__ == "__main__":
    app.run(main)
