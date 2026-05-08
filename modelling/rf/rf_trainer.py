import importlib
import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from absl import app, flags

from data_loader import load_and_clean_data, get_train_test_split

FLAGS = flags.FLAGS
flags.DEFINE_string('data_file', '', 'Path to CSV')
flags.DEFINE_string('output_dir', './results', 'Dir to save models')
flags.DEFINE_string('target_key', 'energy', 'Key from TARGET_GROUPS')
flags.DEFINE_string('metric_key', 'smape', 'Key from SCORERS')
flags.DEFINE_string('feature_set', 'full', 'Feature set: full, precalc, minimal')
flags.DEFINE_integer('n_iter', 40, 'Random Search Iterations')
flags.DEFINE_integer('n_jobs', -1, 'CPUs to use')
flags.DEFINE_string(
    'config_module', 'rf_config',
    'Module name (importlib-resolved) from which to load TARGET_GROUPS, '
    'SCORERS, and COLS_TO_DROP_EXPLICIT. Default `rf_config` is the aims '
    'sweep; pass `rf_config_exciting` for the exciting sweep.'
)


def resolve_config(name):
    """Import and return the RF config module by name.

    Exposed as a module-level helper so tests + the CLI share one
    resolution path (and so bad names fail with a familiar ImportError).
    The module must define TARGET_GROUPS, SCORERS, COLS_TO_DROP_EXPLICIT,
    `calculate_smape`, and `calculate_mag_acc`.
    """
    return importlib.import_module(name)


def main(argv):
    # 1. Setup: resolve the config module + pull its public surface.
    cfg = resolve_config(FLAGS.config_module)
    TARGET_GROUPS = cfg.TARGET_GROUPS
    SCORERS = cfg.SCORERS
    COLS_TO_DROP_EXPLICIT = cfg.COLS_TO_DROP_EXPLICIT
    calculate_smape = cfg.calculate_smape
    calculate_mag_acc = cfg.calculate_mag_acc

    os.makedirs(FLAGS.output_dir, exist_ok=True)
    target_cols = TARGET_GROUPS[FLAGS.target_key]
    scorer = SCORERS[FLAGS.metric_key]

    print(f"--- Training RF ---")
    print(f"Config module: {FLAGS.config_module}")
    print(f"Target: {FLAGS.target_key} {target_cols}")
    print(f"Metric: {FLAGS.metric_key}")

    # 2. Load Data (pass the config's drop list in explicitly — loader
    #    itself is config-agnostic now).
    df, X = load_and_clean_data(FLAGS.data_file, drop_cols=COLS_TO_DROP_EXPLICIT)
    X_train, y_train, X_test, y_test = get_train_test_split(df, X, target_cols)
    
    # 3. Define Estimator
    # If multiple targets, we assume standard RF supports it (it does), 
    # but for some complex metrics or older sklearn versions, MultiOutputRegressor wrapper is safer.
    # Standard RF works for multi-output regression natively.
    base_rf = RandomForestRegressor(random_state=42)

    # 4. Hyperparameter Grid
    param_dist = {
        'n_estimators': [50, 100, 200, 500],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5],
        'max_features': ['sqrt', 'log2', None]
    }
    
    search = RandomizedSearchCV(
        estimator=base_rf,
        param_distributions=param_dist,
        n_iter=FLAGS.n_iter,
        cv=5,
        scoring=scorer,
        verbose=1,
        random_state=42,
        n_jobs=FLAGS.n_jobs
    )
    
    print("Starting Random Search...")
    search.fit(X_train, y_train)
    
    best_model = search.best_estimator_
    print(f"Best Params: {search.best_params_}")
    
    # 5. Evaluation (Test Set)
    print("Evaluating on Test Set...")
    y_pred = best_model.predict(X_test)
    
    # Handle output shape mismatch if single target
    if len(y_pred.shape) == 1:
        y_pred = y_pred.reshape(-1, 1)
        y_test_np = y_test.values.reshape(-1, 1)
    else:
        y_test_np = y_test.values

    # Calculate Metrics per column
    results = {}
    for i, col_name in enumerate(target_cols):
        yt = y_test_np[:, i]
        yp = y_pred[:, i]
        
        results[col_name] = {
            'MAE': np.mean(np.abs(yt - yp)),
            'sMAPE': calculate_smape(yt, yp),
            'MagAcc': calculate_mag_acc(yt, yp)
        }
        print(f"{col_name}: {results[col_name]}")

    # 6. Save Artifacts
    model_filename = f"rf_{FLAGS.target_key}_{FLAGS.metric_key}_best.joblib"
    model_path = os.path.join(FLAGS.output_dir, model_filename)
    joblib.dump(best_model, model_path)
    
    results_path = os.path.join(FLAGS.output_dir, f"results_{FLAGS.target_key}_{FLAGS.metric_key}.pkl")
    joblib.dump(results, results_path)
    
    print(f"Saved model to {model_path}")

if __name__ == "__main__":
    app.run(main)
