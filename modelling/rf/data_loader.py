import pandas as pd
import numpy as np


def load_and_clean_data(csv_path, drop_cols=None):
    """Read a delta-learning CSV and return (df, X).

    Args:
        csv_path: path to a `create_delta_dataset.py` CSV.
        drop_cols: list of column names to remove from X (in addition to
            the always-dropped `delta_*` leakage set). Passed in by the
            caller instead of hard-imported — this keeps the loader
            config-agnostic so multiple RF configs (aims, exciting, ...)
            can share it. Default `None` = drop nothing beyond the
            `delta_*` set below.

    Returns:
        df: the raw DataFrame (still has `split` + target cols).
        X:  numeric feature matrix with delta/monomer carve-out applied.
    """
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)

    # 1. Row filtering (aims-only: remove over-converged rows if the field
    #    exists). Silently no-op for datasets that don't have it.
    if 'delta_mean_basis_functions' in df.columns and 'k_point_density' in df.columns:
        overconverged = (
            (df['delta_mean_basis_functions'] > 0)
            & (df['k_point_density'] == 8)
        )
        before = len(df)
        df = df[~overconverged]
        if before != len(df):
            print(f"Rows after filtering over-converged: {len(df)}")

    # 2. Feature selection (X).
    features_to_drop = list(drop_cols) if drop_cols else []

    # Always drop `delta_*` except `delta_monomer_*` — target leakage
    # prevention is a non-negotiable invariant of the loader.
    for col in df.columns:
        if col.startswith('delta_') and 'monomer' not in col:
            features_to_drop.append(col)

    X_raw = df.drop(columns=features_to_drop, errors='ignore')
    X = X_raw.select_dtypes(include=[np.number]).fillna(0)

    print(f"Features selected: {X.shape[1]}")
    return df, X

def get_train_test_split(df, X, target_cols):
    """
    Returns X_train (Train + Val), y_train, X_test, y_test based on 'split' column.
    """
    # Create Masks
    # Combine 'train' and 'val' for the CV process
    train_val_mask = df['split'].isin(['train', 'val'])
    test_mask = df['split'] == 'test'

    X_train = X[train_val_mask]
    X_test = X[test_mask]

    print(f"Split Sizes -> Train+Val: {len(X_train)}, Test: {len(X_test)}")

    # Handle single target vs multi-target
    if len(target_cols) == 1:
        y = df[target_cols[0]].fillna(0)
        y_train = y[train_val_mask]
        y_test = y[test_mask]
    else:
        # For multi-target (e.g. geometry), fillna(0) for safety
        y = df[target_cols].fillna(0)
        y_train = y[train_val_mask]
        y_test = y[test_mask]

    return X_train, y_train, X_test, y_test
