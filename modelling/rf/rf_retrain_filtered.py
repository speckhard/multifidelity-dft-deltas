"""
Retrain best RF models with EGNN-matched data filters and train-only split.

Applies the same filters as ase_db_to_graphs.py:
  1. Over-converged filter (already in data_loader.py)
  2. Volume outlier filter: |delta_final_volume_per_atom| >= 10.0

Then for each target (energy, bandgap, volume), loads the best RF model from
the sweep to extract its hyperparameters, retrains on train-only split, and
evaluates on train/val/test separately.

Usage:
    python rf_retrain_filtered.py \
        --data_file /path/to/data.csv \
        --model_dir /path/to/rf_sweep_results \
        --output_dir /path/to/output
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from absl import app, flags

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

rf_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if rf_dir not in sys.path:
    sys.path.append(rf_dir)

from modelling.rf.data_loader import load_and_clean_data
from modelling.rf.rf_config import TARGET_GROUPS, calculate_smape, calculate_mag_acc, rmsle_abs_func

FLAGS = flags.FLAGS
flags.DEFINE_string('data_file',
                    '/u/dansp/egnn/relaxation_data_with_kpoints/csv_data/'
                    'delta_combined_relaxations_23_22__12_1_2026_kpoints_included_no_duplicates.csv',
                    'Path to RF CSV data file')
flags.DEFINE_string('model_dir',
                    '/u/dansp/egnn/rf_results/rf_sweep_2026_02_13',
                    'Directory containing RF sweep results with best .joblib models')
flags.DEFINE_string('output_dir', './rf_retrain_filtered', 'Output directory')
flags.DEFINE_float('volume_threshold', 10.0,
                   'Volume outlier threshold (matches ase_db_to_graphs.py)')

# Best single-target RF models to retrain (from RF_best_models_parsed_results.md)
RF_MODELS = [
    ('energy', 'smape', ['delta_total_energy_per_atom']),
    ('bandgap', 'smape', ['delta_homo_lumo_gap']),
    ('volume', 'smape', ['delta_final_volume_per_atom']),
]

# Column name mapping for consistency with EGNN comparison table
COL_NAME_MAP = {
    'delta_total_energy_per_atom': 'delta_energy',
    'delta_homo_lumo_gap': 'delta_gap',
    'delta_final_volume_per_atom': 'delta_volume',
}


def compute_metrics(y_true, y_pred, col_name):
    """Compute standardized metrics for a single target column."""
    return {
        f'{col_name}/MAE': mean_absolute_error(y_true, y_pred),
        f'{col_name}/sMAPE': calculate_smape(y_true, y_pred),
        f'{col_name}/MagAcc': calculate_mag_acc(y_true, y_pred),
        f'{col_name}/RMSLE': rmsle_abs_func(y_true, y_pred),
    }


def main(argv):
    os.makedirs(FLAGS.output_dir, exist_ok=True)

    # ---- Load and filter data ----
    df, X = load_and_clean_data(FLAGS.data_file)
    print(f"Rows after over-converged filter: {len(df)}")

    # Apply EGNN-matched volume outlier filter
    vol_outlier = df['delta_final_volume_per_atom'].abs() >= FLAGS.volume_threshold
    n_vol = vol_outlier.sum()
    print(f"Filtering volume outliers (|vol| >= {FLAGS.volume_threshold}): {n_vol} rows removed")
    df = df[~vol_outlier].reset_index(drop=True)
    X = X[~vol_outlier].reset_index(drop=True)
    print(f"Rows after volume filter: {len(df)}")

    # Split masks
    train_mask = df['split'] == 'train'
    val_mask = df['split'] == 'val'
    test_mask = df['split'] == 'test'
    print(f"\nSplit sizes (after filtering):")
    print(f"  Train: {train_mask.sum()}")
    print(f"  Val:   {val_mask.sum()}")
    print(f"  Test:  {test_mask.sum()}")

    all_rows = []

    for target_key, metric_key, target_cols in RF_MODELS:
        model_filename = f"rf_{target_key}_{metric_key}_best.joblib"
        model_path = os.path.join(FLAGS.model_dir, f"{target_key}_{metric_key}", model_filename)

        if not os.path.exists(model_path):
            print(f"\nWARNING: Model not found: {model_path}, skipping.")
            continue

        print(f"\n{'='*60}")
        print(f"Target: {target_key} (metric={metric_key})")
        print(f"{'='*60}")

        # Load saved model to extract hyperparameters
        saved_model = joblib.load(model_path)
        params = saved_model.get_params()
        print(f"  Hyperparams from sweep: n_estimators={params['n_estimators']}, "
              f"max_depth={params['max_depth']}, "
              f"min_samples_split={params['min_samples_split']}, "
              f"min_samples_leaf={params['min_samples_leaf']}")

        # Retrain on train-only split (not train+val as the sweep did)
        y_col = target_cols[0]
        X_train = X[train_mask]
        y_train = df.loc[train_mask, y_col].fillna(0).values

        print(f"  Retraining on train-only split ({len(X_train)} samples)...")
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        print(f"  Done.")

        # Save retrained model
        retrained_path = os.path.join(FLAGS.output_dir, f"rf_{target_key}_filtered_retrained.joblib")
        joblib.dump(model, retrained_path)
        print(f"  Saved retrained model: {retrained_path}")

        # Evaluate on all splits
        std_name = COL_NAME_MAP.get(y_col, y_col)

        for split_name, mask in [('train', train_mask), ('val', val_mask), ('test', test_mask)]:
            X_split = X[mask]
            y_true = df.loc[mask, y_col].fillna(0).values
            y_pred = model.predict(X_split)

            metrics = compute_metrics(y_true, y_pred, std_name)
            row = {'model': f'rf_{target_key}_filtered', 'split': split_name}
            row.update(metrics)
            all_rows.append(row)

            print(f"  {split_name:5s}: MAE={metrics[f'{std_name}/MAE']:.6f}, "
                  f"sMAPE={metrics[f'{std_name}/sMAPE']:.2f}%, "
                  f"MagAcc={metrics[f'{std_name}/MagAcc']:.4f}, "
                  f"RMSLE={metrics[f'{std_name}/RMSLE']:.6f}")

            # Save pred/true CSV
            csv_dict = {
                f'{std_name}_pred': y_pred,
                f'{std_name}_true': y_true,
            }
            csv_path = os.path.join(FLAGS.output_dir, f"rf_{target_key}_{split_name}.csv")
            pd.DataFrame(csv_dict).to_csv(csv_path, index=False)

    # Save aggregate metrics
    if all_rows:
        metrics_df = pd.DataFrame(all_rows)
        metrics_path = os.path.join(FLAGS.output_dir, 'rf_filtered_comparison_table.csv')
        metrics_df.to_csv(metrics_path, index=False)
        print(f"\nSaved RF filtered metrics: {metrics_path}")

        test_df = metrics_df[metrics_df['split'] == 'test'].drop(columns=['split'])
        if not test_df.empty:
            print(f"\n{'='*60}")
            print("RF FILTERED TEST SET SUMMARY")
            print(f"{'='*60}")
            print(test_df.to_string(index=False))

    print("\nDone!")


if __name__ == "__main__":
    app.run(main)
