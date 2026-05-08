"""Tests for XGBoost RF/GBDT sweep: config, transforms, model building, training."""
import os
import sys
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import RandomizedSearchCV

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from modelling.rf.rf_config import (
    TRANSFORM_EPS,
    arcsinh_transform,
    arcsinh_inverse,
    log_abs_transform,
    exp_abs_inverse,
    calculate_smape,
    rmsle_abs_func,
    asinh_mae_func,
    LOSS_CONFIGS,
    GBDT_PARAM_GRID,
    RF_PARAM_GRID,
    SCORERS,
)
from modelling.rf.rf_trainer import build_model


# ---- Fixtures ----

@pytest.fixture
def signed_data():
    """Realistic delta values spanning the range of DFT errors."""
    rng = np.random.RandomState(42)
    return np.concatenate([
        rng.uniform(-0.01, 0.01, 30),   # small deltas near zero
        rng.uniform(-1, 1, 30),          # medium deltas
        rng.uniform(-10, 10, 20),        # large deltas
    ])


@pytest.fixture
def synthetic_Xy():
    """Small synthetic dataset for end-to-end training tests."""
    rng = np.random.RandomState(42)
    n, d = 80, 5
    X = rng.randn(n, d)
    y = 0.5 * X[:, 0] - 0.3 * X[:, 1] + 0.1 * rng.randn(n)
    return X, y


@pytest.fixture
def synthetic_Xy_multi():
    """Synthetic dataset with 3 targets for multi-output tests."""
    rng = np.random.RandomState(42)
    n, d = 80, 5
    X = rng.randn(n, d)
    y = np.column_stack([
        0.5 * X[:, 0] + 0.1 * rng.randn(n),
        -0.3 * X[:, 1] + 0.1 * rng.randn(n),
        0.2 * X[:, 2] + 0.1 * rng.randn(n),
    ])
    return X, y


# ============================================================
# Transform Tests
# ============================================================

class TestArcsinhTransform:
    def test_roundtrip_positive(self, signed_data):
        """arcsinh(y/c) -> c*sinh(z) roundtrips for positive values."""
        y = np.abs(signed_data) + 1e-6
        z = arcsinh_transform(y)
        y_recovered = arcsinh_inverse(z)
        np.testing.assert_allclose(y_recovered, y, rtol=1e-10)

    def test_roundtrip_signed(self, signed_data):
        """Roundtrip works for mixed positive/negative values."""
        z = arcsinh_transform(signed_data)
        y_recovered = arcsinh_inverse(z)
        np.testing.assert_allclose(y_recovered, signed_data, rtol=1e-10)

    def test_preserves_sign(self, signed_data):
        """arcsinh is odd -- sign of output matches sign of input."""
        z = arcsinh_transform(signed_data)
        nonzero = signed_data != 0
        np.testing.assert_array_equal(
            np.sign(z[nonzero]), np.sign(signed_data[nonzero])
        )

    def test_monotonically_increasing(self):
        """Transform is monotonically increasing (crucial for tree splits)."""
        y = np.linspace(-10, 10, 1000)
        z = arcsinh_transform(y)
        assert np.all(np.diff(z) > 0)

    def test_log_behavior_for_large_values(self):
        """For |y| >> c, arcsinh(y/c) ~ sign(y)*log(2|y|/c)."""
        y_large = np.array([1.0, 10.0, 100.0])
        z = arcsinh_transform(y_large)
        z_log_approx = np.log(2 * y_large / TRANSFORM_EPS)
        np.testing.assert_allclose(z, z_log_approx, rtol=0.01)

    def test_continuous_through_zero(self):
        """No discontinuity at y=0 (unlike sign(y)*log(|y|+eps))."""
        y = np.array([-1e-8, 0.0, 1e-8])
        z = arcsinh_transform(y)
        assert abs(z[1] - z[0]) < 0.01
        assert abs(z[2] - z[1]) < 0.01

    def test_inverse_overflow_protection(self):
        """sinh(z) should not overflow for extreme z values."""
        z_extreme = np.array([-800, -700, 700, 800])
        result = arcsinh_inverse(z_extreme)
        assert np.all(np.isfinite(result))

    def test_2d_input(self):
        """Transform works on 2D arrays (multi-output targets)."""
        y_2d = np.array([[0.1, -0.5, 3.0], [0.01, 1.0, -7.0]])
        z = arcsinh_transform(y_2d)
        y_recovered = arcsinh_inverse(z)
        assert z.shape == (2, 3)
        np.testing.assert_allclose(y_recovered, y_2d, rtol=1e-10)

    def test_eps_value(self):
        """TRANSFORM_EPS is 1e-4 as specified."""
        assert TRANSFORM_EPS == 1e-4


class TestLogAbsTransform:
    """Tests for the rmslae magnitude transform: log(|y| + eps)."""

    def test_strips_sign(self):
        """log(|y| + eps) maps both positive and negative to same value."""
        y_pos = np.array([0.5, 1.0, 3.0])
        y_neg = -y_pos
        np.testing.assert_array_equal(
            log_abs_transform(y_pos), log_abs_transform(y_neg)
        )

    def test_inverse_always_positive(self):
        """exp(z) - eps is always non-negative for any z."""
        z = np.array([-10, -5, 0, 5, 10])
        result = exp_abs_inverse(z)
        assert np.all(result >= 0)

    def test_roundtrip_positive(self):
        """Roundtrips for positive inputs: exp(log(y+eps)) - eps = y."""
        y = np.array([1e-5, 0.001, 0.1, 1.0, 10.0])
        z = log_abs_transform(y)
        y_rec = exp_abs_inverse(z)
        np.testing.assert_allclose(y_rec, y, rtol=1e-10)

    def test_roundtrip_negative_returns_magnitude(self):
        """Negative inputs roundtrip to their absolute values."""
        y = np.array([-0.5, -1.0, -3.0])
        z = log_abs_transform(y)
        y_rec = exp_abs_inverse(z)
        np.testing.assert_allclose(y_rec, np.abs(y), rtol=1e-10)

    def test_zero_input(self):
        """log(0 + eps) is finite, exp(log(eps)) - eps ~ 0."""
        z = log_abs_transform(np.array([0.0]))
        assert np.isfinite(z[0])
        y_rec = exp_abs_inverse(z)
        np.testing.assert_allclose(y_rec[0], 0.0, atol=1e-12)

    def test_monotonic_for_positive(self):
        """Transform is monotonically increasing for positive values."""
        y = np.linspace(0, 10, 500)
        z = log_abs_transform(y)
        assert np.all(np.diff(z) > 0)

    def test_2d_input(self):
        """Works on 2D arrays for multi-output."""
        y_2d = np.array([[0.1, 0.5], [-3.0, 7.0]])
        z = log_abs_transform(y_2d)
        assert z.shape == (2, 2)
        y_rec = exp_abs_inverse(z)
        np.testing.assert_allclose(y_rec, np.abs(y_2d), rtol=1e-10)

    def test_inverse_overflow_protection(self):
        """exp(z) should not overflow for extreme z."""
        z_extreme = np.array([700, 800, 1000])
        result = exp_abs_inverse(z_extreme)
        assert np.all(np.isfinite(result))


# ============================================================
# Loss Function Tests
# ============================================================

class TestLossFunctions:
    def test_smape_uses_correct_epsilon(self):
        """sMAPE denominator uses eps=1e-4, not 1e-7."""
        y_true = np.array([0.0])
        y_pred = np.array([1e-5])
        result = calculate_smape(y_true, y_pred)
        expected = 100.0 * 2.0 * 1e-5 / (1e-5 + 1e-4)
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_smape_zero_targets_no_nan(self):
        """sMAPE handles all-zero targets without NaN/Inf."""
        y_true = np.zeros(10)
        y_pred = np.zeros(10)
        result = calculate_smape(y_true, y_pred)
        assert np.isfinite(result)
        assert result == 0.0

    def test_smape_perfect_predictions(self):
        """sMAPE is 0 for perfect predictions."""
        y = np.array([0.1, -0.5, 3.0, 0.001])
        assert calculate_smape(y, y) == 0.0

    def test_rmsle_abs_uses_log_not_log1p(self):
        """RMSLE uses log(|y| + eps), NOT log1p(|y|) = log(|y| + 1)."""
        y_true = np.array([0.01])
        y_pred = np.array([0.02])
        result = rmsle_abs_func(y_true, y_pred)
        eps = TRANSFORM_EPS
        expected = np.sqrt(np.mean(
            (np.log(np.abs(y_true) + eps) - np.log(np.abs(y_pred) + eps)) ** 2
        ))
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_rmsle_abs_differs_from_log1p(self):
        """Our RMSLE gives different result than old log1p version."""
        y_true = np.array([0.01, 0.1, 1.0])
        y_pred = np.array([0.02, 0.15, 0.8])
        new_result = rmsle_abs_func(y_true, y_pred)
        old_result = np.sqrt(np.mean(
            (np.log1p(np.abs(y_true)) - np.log1p(np.abs(y_pred))) ** 2
        ))
        assert new_result != pytest.approx(old_result, rel=0.01)

    def test_asinh_mae_uses_scaling(self):
        """asinh_mae uses arcsinh(y/c) with c=1e-4."""
        y_true = np.array([0.1, 1.0, 5.0])
        y_pred = np.array([0.15, 0.8, 5.5])
        result = asinh_mae_func(y_true, y_pred)
        c = TRANSFORM_EPS
        expected = np.mean(np.abs(
            np.arcsinh(y_true / c) - np.arcsinh(y_pred / c)
        ))
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_asinh_mae_differs_from_unscaled(self):
        """Scaled asinh(y/c) differs from unscaled arcsinh(y)."""
        y_true = np.array([0.001, 0.01, 0.1])
        y_pred = np.array([0.002, 0.015, 0.12])
        scaled = asinh_mae_func(y_true, y_pred)
        unscaled = np.mean(np.abs(np.arcsinh(y_true) - np.arcsinh(y_pred)))
        assert scaled != pytest.approx(unscaled, rel=0.01)


# ============================================================
# Config Tests
# ============================================================

class TestLossConfigs:
    def test_all_expected_losses_present(self):
        assert set(LOSS_CONFIGS.keys()) == {'mae', 'smape', 'asinh', 'rmslae'}

    def test_mae_no_transform(self):
        assert LOSS_CONFIGS['mae']['transform'] is None
        assert LOSS_CONFIGS['mae']['inverse'] is None

    def test_mae_uses_absoluteerror(self):
        assert LOSS_CONFIGS['mae']['objective'] == 'reg:absoluteerror'

    def test_smape_uses_absoluteerror(self):
        assert LOSS_CONFIGS['smape']['objective'] == 'reg:absoluteerror'

    def test_asinh_uses_absoluteerror(self):
        assert LOSS_CONFIGS['asinh']['objective'] == 'reg:absoluteerror'

    def test_rmslae_uses_squarederror(self):
        assert LOSS_CONFIGS['rmslae']['objective'] == 'reg:squarederror'

    def test_rmslae_uses_log_abs_transform(self):
        """rmslae uses the sign-stripping log transform."""
        cfg = LOSS_CONFIGS['rmslae']
        assert cfg['transform'] is log_abs_transform
        assert cfg['inverse'] is exp_abs_inverse

    def test_rmslae_is_magnitude_predictor(self):
        """rmslae config is flagged as magnitude predictor."""
        assert LOSS_CONFIGS['rmslae']['magnitude_only'] is True

    def test_signed_losses_not_magnitude(self):
        """mae, smape, asinh are NOT magnitude predictors."""
        for key in ['mae', 'smape', 'asinh']:
            assert LOSS_CONFIGS[key]['magnitude_only'] is False

    def test_smape_asinh_use_arcsinh_transform(self):
        for key in ['smape', 'asinh']:
            cfg = LOSS_CONFIGS[key]
            assert cfg['transform'] is arcsinh_transform
            assert cfg['inverse'] is arcsinh_inverse

    def test_scorers_have_all_losses(self):
        for key in LOSS_CONFIGS:
            assert key in SCORERS

    def test_gbdt_grid_has_learning_rate(self):
        assert 'learning_rate' in GBDT_PARAM_GRID

    def test_rf_grid_no_learning_rate(self):
        assert 'learning_rate' not in RF_PARAM_GRID

    def test_gbdt_grid_matches_spec(self):
        assert GBDT_PARAM_GRID['n_estimators'] == [100, 500, 1000]
        assert GBDT_PARAM_GRID['learning_rate'] == [0.01, 0.05, 0.1]
        assert GBDT_PARAM_GRID['max_depth'] == [3, 5, 7, 9]
        assert GBDT_PARAM_GRID['min_child_weight'] == [1, 5, 10]
        assert GBDT_PARAM_GRID['subsample'] == [0.8, 1.0]
        assert GBDT_PARAM_GRID['colsample_bytree'] == [0.8, 1.0]
        assert GBDT_PARAM_GRID['gamma'] == [0, 0.1, 1.0]


# ============================================================
# Model Building Tests
# ============================================================

class TestBuildModel:
    def test_single_target_no_transform_returns_xgb(self):
        """mae + single target: plain XGBRegressor, no prefix."""
        estimator, grid = build_model('xgb_gbdt', 'mae', n_targets=1)
        from xgboost import XGBRegressor
        assert isinstance(estimator, XGBRegressor)
        assert all(not k.startswith('regressor__') for k in grid)
        assert all(not k.startswith('estimator__') for k in grid)

    def test_single_target_with_transform_wraps(self):
        """smape + single target: TransformedTargetRegressor(XGBRegressor)."""
        estimator, grid = build_model('xgb_gbdt', 'smape', n_targets=1)
        from sklearn.compose import TransformedTargetRegressor
        assert isinstance(estimator, TransformedTargetRegressor)
        assert all(k.startswith('regressor__') for k in grid)
        assert all('estimator__' not in k for k in grid)

    def test_multi_target_no_transform_wraps_multioutput(self):
        """mae + multi target: MultiOutputRegressor(XGBRegressor)."""
        estimator, grid = build_model('xgb_gbdt', 'mae', n_targets=3)
        from sklearn.multioutput import MultiOutputRegressor
        assert isinstance(estimator, MultiOutputRegressor)
        assert all(k.startswith('estimator__') for k in grid)

    def test_multi_target_with_transform_double_wraps(self):
        """smape + multi: TransformedTargetRegressor(MultiOutputRegressor(...))."""
        estimator, grid = build_model('xgb_gbdt', 'smape', n_targets=3)
        from sklearn.compose import TransformedTargetRegressor
        from sklearn.multioutput import MultiOutputRegressor
        assert isinstance(estimator, TransformedTargetRegressor)
        assert isinstance(estimator.regressor, MultiOutputRegressor)
        assert all(k.startswith('regressor__estimator__') for k in grid)

    def test_rf_model_type(self):
        """xgb_rf creates XGBRFRegressor."""
        estimator, grid = build_model('xgb_rf', 'mae', n_targets=1)
        from xgboost import XGBRFRegressor
        assert isinstance(estimator, XGBRFRegressor)
        assert 'learning_rate' not in grid

    def test_gbdt_grid_has_learning_rate_prefixed(self):
        _, grid = build_model('xgb_gbdt', 'smape', n_targets=1)
        assert 'regressor__learning_rate' in grid

    def test_invalid_model_type_raises(self):
        with pytest.raises(ValueError, match="Unknown model_type"):
            build_model('invalid', 'mae', n_targets=1)

    def test_invalid_loss_key_raises(self):
        with pytest.raises(KeyError):
            build_model('xgb_gbdt', 'invalid_loss', n_targets=1)

    def test_objective_propagated_to_xgb(self):
        """The XGBoost objective matches the loss config."""
        from sklearn.compose import TransformedTargetRegressor
        for loss_key, config in LOSS_CONFIGS.items():
            estimator, _ = build_model('xgb_gbdt', loss_key, n_targets=1)
            if isinstance(estimator, TransformedTargetRegressor):
                xgb_model = estimator.regressor
            else:
                xgb_model = estimator
            assert xgb_model.objective == config['objective'], (
                f"Loss {loss_key}: expected {config['objective']}, "
                f"got {xgb_model.objective}"
            )

    def test_n_jobs_propagated(self):
        estimator, _ = build_model('xgb_gbdt', 'mae', n_targets=1, n_jobs=4)
        assert estimator.n_jobs == 4

    def test_n_jobs_propagated_through_wrappers(self):
        estimator, _ = build_model('xgb_gbdt', 'smape', n_targets=1, n_jobs=4)
        assert estimator.regressor.n_jobs == 4

    def test_n_jobs_propagated_through_multi_wrapper(self):
        """n_jobs reaches XGB through MultiOutputRegressor + TransformedTarget."""
        estimator, _ = build_model('xgb_gbdt', 'smape', n_targets=3, n_jobs=4)
        inner_xgb = estimator.regressor.estimator
        assert inner_xgb.n_jobs == 4


# ============================================================
# Param Prefix Integration Tests (with actual RandomizedSearchCV)
# ============================================================

class TestParamPrefixIntegration:
    def test_single_no_transform_params_valid(self, synthetic_Xy):
        X, y = synthetic_Xy
        estimator, grid = build_model('xgb_gbdt', 'mae', n_targets=1, n_jobs=1)
        search = RandomizedSearchCV(
            estimator, grid, n_iter=1, cv=2, random_state=42
        )
        search.fit(X, y)
        assert search.best_estimator_ is not None

    def test_single_with_transform_params_valid(self, synthetic_Xy):
        X, y = synthetic_Xy
        estimator, grid = build_model('xgb_gbdt', 'smape', n_targets=1, n_jobs=1)
        search = RandomizedSearchCV(
            estimator, grid, n_iter=1, cv=2, random_state=42
        )
        search.fit(X, y)
        assert search.best_estimator_ is not None

    def test_multi_with_transform_params_valid(self, synthetic_Xy_multi):
        X, y = synthetic_Xy_multi
        estimator, grid = build_model('xgb_gbdt', 'smape', n_targets=3, n_jobs=1)
        search = RandomizedSearchCV(
            estimator, grid, n_iter=1, cv=2, random_state=42
        )
        search.fit(X, y)
        assert search.best_estimator_ is not None

    def test_rf_params_valid(self, synthetic_Xy):
        X, y = synthetic_Xy
        estimator, grid = build_model('xgb_rf', 'asinh', n_targets=1, n_jobs=1)
        search = RandomizedSearchCV(
            estimator, grid, n_iter=1, cv=2, random_state=42
        )
        search.fit(X, y)
        assert search.best_estimator_ is not None


# ============================================================
# rmslae Magnitude Evaluation Tests
# ============================================================

class TestRmslaeMagnitudeEvaluation:
    """Verify that rmslae predicts magnitudes and is evaluated against |y_true|."""

    def test_rmslae_predictions_are_positive(self, synthetic_Xy):
        """rmslae model predictions are always >= 0 (magnitude only)."""
        X, y = synthetic_Xy
        estimator, _ = build_model('xgb_gbdt', 'rmslae', n_targets=1, n_jobs=1)
        estimator.fit(X, y)
        preds = estimator.predict(X)
        assert np.all(preds >= 0), (
            f"rmslae predictions should be non-negative, got min={preds.min()}"
        )

    def test_rmslae_learns_magnitude_not_sign(self, synthetic_Xy):
        """rmslae predictions correlate with |y|, not signed y."""
        X, y = synthetic_Xy
        estimator, _ = build_model('xgb_gbdt', 'rmslae', n_targets=1, n_jobs=1)
        estimator.fit(X, y)
        preds = estimator.predict(X)
        # Correlation with |y| should be higher than with y
        corr_abs = np.corrcoef(preds, np.abs(y))[0, 1]
        corr_signed = np.corrcoef(preds, y)[0, 1]
        assert abs(corr_abs) >= abs(corr_signed) - 0.1, (
            f"rmslae should correlate with |y| (r={corr_abs:.3f}) "
            f"at least as well as with y (r={corr_signed:.3f})"
        )

    def test_rmslae_transform_strips_sign_during_training(self):
        """During training, log(|y| + eps) makes +0.5 and -0.5 identical targets."""
        y = np.array([0.5, -0.5, 1.0, -1.0])
        z = log_abs_transform(y)
        assert z[0] == z[1]  # +0.5 and -0.5 map to same value
        assert z[2] == z[3]  # +1.0 and -1.0 map to same value

    def test_rmslae_scorer_uses_absolute_values(self):
        """The rmslae scorer compares log magnitudes, not signed values."""
        y_true = np.array([-1.0, -0.5, 0.1, 0.5, 1.0])
        y_pred = np.array([1.0, 0.5, 0.1, 0.5, 1.0])  # correct magnitudes
        # Perfect magnitude predictions should give rmslae = 0
        score = rmsle_abs_func(y_true, y_pred)
        assert score == pytest.approx(0.0, abs=1e-10)

    def test_rmslae_evaluation_against_abs_y_true(self):
        """Correct evaluation: compare rmslae predictions against |y_true|."""
        y_true = np.array([-2.0, -0.5, 0.3, 1.0])
        y_pred_magnitude = np.array([2.0, 0.5, 0.3, 1.0])  # perfect magnitudes

        # Evaluate against |y_true| -- should be perfect
        eps = TRANSFORM_EPS
        rmslae_correct = np.sqrt(np.mean(
            (np.log(np.abs(y_true) + eps) - np.log(y_pred_magnitude + eps)) ** 2
        ))
        assert rmslae_correct == pytest.approx(0.0, abs=1e-10)

        # Evaluate against signed y_true -- would be wrong (penalizes negatives)
        rmslae_wrong = np.sqrt(np.mean(
            (np.log(y_true + eps) - np.log(y_pred_magnitude + eps)) ** 2
        ))
        # This would produce NaN due to log of negative numbers
        assert not np.isfinite(rmslae_wrong)


# ============================================================
# End-to-End Training Tests
# ============================================================

class TestEndToEnd:
    def test_gbdt_mae_trains_and_predicts(self, synthetic_Xy):
        X, y = synthetic_Xy
        estimator, grid = build_model('xgb_gbdt', 'mae', n_targets=1, n_jobs=1)
        search = RandomizedSearchCV(
            estimator, grid, n_iter=2, cv=2, random_state=42,
            scoring='neg_mean_absolute_error',
        )
        search.fit(X, y)
        preds = search.best_estimator_.predict(X)
        assert np.all(np.isfinite(preds))
        mae = np.mean(np.abs(y - preds))
        naive_mae = np.mean(np.abs(y - np.mean(y)))
        assert mae < naive_mae

    def test_gbdt_smape_trains_and_predicts(self, synthetic_Xy):
        """GBDT + smape (arcsinh transform) predicts in original space."""
        X, y = synthetic_Xy
        estimator, _ = build_model('xgb_gbdt', 'smape', n_targets=1, n_jobs=1)
        estimator.fit(X, y)
        preds = estimator.predict(X)
        assert np.all(np.isfinite(preds))
        assert preds.shape == y.shape

    def test_gbdt_rmslae_trains_with_squarederror(self, synthetic_Xy):
        """GBDT + rmslae uses reg:squarederror, predictions are positive."""
        X, y = synthetic_Xy
        estimator, _ = build_model('xgb_gbdt', 'rmslae', n_targets=1, n_jobs=1)
        estimator.fit(X, y)
        preds = estimator.predict(X)
        assert np.all(np.isfinite(preds))
        assert np.all(preds >= 0)

    def test_rf_asinh_trains_and_predicts(self, synthetic_Xy):
        X, y = synthetic_Xy
        estimator, _ = build_model('xgb_rf', 'asinh', n_targets=1, n_jobs=1)
        estimator.fit(X, y)
        preds = estimator.predict(X)
        assert np.all(np.isfinite(preds))

    def test_multi_target_training(self, synthetic_Xy_multi):
        X, y = synthetic_Xy_multi
        estimator, _ = build_model('xgb_gbdt', 'asinh', n_targets=3, n_jobs=1)
        estimator.fit(X, y)
        preds = estimator.predict(X)
        assert preds.shape == y.shape
        assert np.all(np.isfinite(preds))

    def test_multi_target_rmslae(self, synthetic_Xy_multi):
        """Multi-output rmslae (most complex nesting: TTR(MOR(XGB)))."""
        X, y = synthetic_Xy_multi
        estimator, grid = build_model('xgb_gbdt', 'rmslae', n_targets=3, n_jobs=1)
        # Verify double-wrapping
        from sklearn.compose import TransformedTargetRegressor
        from sklearn.multioutput import MultiOutputRegressor
        assert isinstance(estimator, TransformedTargetRegressor)
        assert isinstance(estimator.regressor, MultiOutputRegressor)
        # Train and verify
        estimator.fit(X, y)
        preds = estimator.predict(X)
        assert preds.shape == y.shape
        assert np.all(np.isfinite(preds))
        assert np.all(preds >= 0), "rmslae multi-output predictions must be non-negative"
        # Verify param prefix works with RandomizedSearchCV
        search = RandomizedSearchCV(
            estimator, grid, n_iter=1, cv=2, random_state=42
        )
        search.fit(X, y)
        assert search.best_estimator_ is not None

    def test_multi_target_mae_no_transform(self, synthetic_Xy_multi):
        """Multi-output mae (estimator__ prefix) with RandomizedSearchCV."""
        X, y = synthetic_Xy_multi
        estimator, grid = build_model('xgb_gbdt', 'mae', n_targets=3, n_jobs=1)
        search = RandomizedSearchCV(
            estimator, grid, n_iter=1, cv=2, random_state=42
        )
        search.fit(X, y)
        preds = search.best_estimator_.predict(X)
        assert preds.shape == y.shape
        assert np.all(np.isfinite(preds))

    def test_all_loss_model_combinations(self, synthetic_Xy):
        """Every (model_type, loss_key) combination trains without error."""
        X, y = synthetic_Xy
        for model_type in ['xgb_gbdt', 'xgb_rf']:
            for loss_key in LOSS_CONFIGS:
                estimator, _ = build_model(
                    model_type, loss_key, n_targets=1, n_jobs=1
                )
                estimator.fit(X, y)
                preds = estimator.predict(X)
                assert np.all(np.isfinite(preds)), (
                    f"Failed for {model_type}/{loss_key}"
                )


# ============================================================
# rf_export Tests
# ============================================================

class TestExportDiscoverModels:
    """Test rf_export.py model discovery and metrics."""

    def test_discover_models_finds_correct_paths(self, tmp_path):
        """discover_models finds .joblib files in the expected tree structure."""
        from modelling.rf.rf_export import discover_models
        # Create mock sweep tree
        for model in ['xgb_gbdt', 'xgb_rf']:
            for target in ['energy', 'bandgap']:
                for loss in ['mae', 'smape', 'asinh', 'rmslae']:
                    d = tmp_path / f"{model}_{target}" / loss
                    d.mkdir(parents=True)
                    (d / f"{model}_{target}_{loss}_best.joblib").touch()

        found = discover_models(str(tmp_path))
        assert len(found) == 2 * 2 * 4  # 2 models * 2 targets * 4 losses
        # Check structure of returned tuples
        model_type, target_key, loss_key, path = found[0]
        assert model_type in ['xgb_gbdt', 'xgb_rf']
        assert target_key in ['energy', 'bandgap']
        assert loss_key in ['mae', 'smape', 'asinh', 'rmslae']
        assert path.endswith('_best.joblib')

    def test_discover_models_empty_dir(self, tmp_path):
        """discover_models returns empty list for empty directory."""
        from modelling.rf.rf_export import discover_models
        found = discover_models(str(tmp_path))
        assert found == []

    def test_discover_models_partial_sweep(self, tmp_path):
        """discover_models handles partial sweeps (some combos missing)."""
        from modelling.rf.rf_export import discover_models
        d = tmp_path / "xgb_gbdt_energy" / "mae"
        d.mkdir(parents=True)
        (d / "xgb_gbdt_energy_mae_best.joblib").touch()
        found = discover_models(str(tmp_path))
        assert len(found) == 1

    def test_compute_metrics_signed(self):
        """compute_metrics with magnitude_only=False uses rmsle_abs_func."""
        from modelling.rf.rf_export import compute_metrics
        y_true = np.array([1.0, -2.0, 0.5])
        y_pred = np.array([1.1, -1.8, 0.6])
        m = compute_metrics(y_true, y_pred, 'test', magnitude_only=False)
        assert 'test/MAE' in m
        assert 'test/sMAPE' in m
        assert 'test/MagAcc' in m
        assert 'test/RMSLAE' in m
        # RMSLAE should use np.abs internally
        assert m['test/RMSLAE'] == pytest.approx(
            rmsle_abs_func(y_true, y_pred), rel=1e-10
        )

    def test_compute_metrics_magnitude(self):
        """compute_metrics with magnitude_only=True skips redundant abs()."""
        from modelling.rf.rf_export import compute_metrics
        y_true = np.array([1.0, 2.0, 0.5])  # already non-negative
        y_pred = np.array([1.1, 1.8, 0.6])  # already non-negative
        m = compute_metrics(y_true, y_pred, 'test', magnitude_only=True)
        # Direct computation without abs
        eps = 1e-4
        expected_rmslae = np.sqrt(np.mean(
            (np.log(y_true + eps) - np.log(y_pred + eps)) ** 2
        ))
        assert m['test/RMSLAE'] == pytest.approx(expected_rmslae, rel=1e-10)


# ============================================================
# data_loader Smoke Tests
# ============================================================

class TestDataLoader:
    """Smoke tests for data_loader.py."""

    def test_load_and_clean_filters_overconverged(self, tmp_path):
        """Over-converged rows (delta_mean_basis_functions > 0 & k=8) are removed."""
        from modelling.rf.data_loader import load_and_clean_data
        csv_path = tmp_path / "test_data.csv"
        pd.DataFrame({
            'split': ['train', 'train', 'test', 'test'],
            'delta_mean_basis_functions': [1.0, -0.5, 0.5, -1.0],
            'k_point_density': [8, 8, 4, 8],
            'feature_a': [1.0, 2.0, 3.0, 4.0],
            'delta_total_energy_per_atom': [0.1, 0.2, 0.3, 0.4],
        }).to_csv(csv_path, index=False)
        df, X = load_and_clean_data(str(csv_path))
        # Row 0 (delta_mean=1.0, k=8) should be filtered out
        assert len(df) == 3
        assert 'feature_a' in X.columns

    def test_get_train_test_split(self, tmp_path):
        """Split correctly separates train+val from test."""
        from modelling.rf.data_loader import load_and_clean_data, get_train_test_split
        csv_path = tmp_path / "test_data.csv"
        pd.DataFrame({
            'split': ['train', 'val', 'test', 'test'],
            'delta_mean_basis_functions': [-1, -1, -1, -1],
            'k_point_density': [4, 4, 4, 4],
            'feature_a': [1.0, 2.0, 3.0, 4.0],
            'delta_total_energy_per_atom': [0.1, 0.2, 0.3, 0.4],
        }).to_csv(csv_path, index=False)
        df, X = load_and_clean_data(str(csv_path))
        X_train, y_train, X_test, y_test = get_train_test_split(
            df, X, ['delta_total_energy_per_atom']
        )
        assert len(X_train) == 2  # train + val
        assert len(X_test) == 2   # test
