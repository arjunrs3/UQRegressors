import numpy as np
import torch
import pytest
from sklearn.model_selection import train_test_split

from uqregressors.conformal.cqr import ConformalQuantileRegressor
from uqregressors.tuning.tuning import tune_hyperparams, interval_width

# Optional utility
def generate_data(n_samples=100, n_features=5):
    X = np.random.randn(n_samples, n_features)
    y = np.sin(X[:, 0]) + X[:, 1] ** 2 + np.random.normal(0, 0.1, size=n_samples)
    return X, y

@pytest.mark.parametrize("n_trials", [3])  # Keep short for test speed
def test_cqr_tuning_pipeline(n_trials):
    # Create synthetic dataset
    X, y = generate_data(n_samples=80)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Instantiate untuned base CQR regressor
    cqr = ConformalQuantileRegressor(alpha=0.1, epochs=5, random_seed=42)
    cqr.fit(X_train, y_train)

    # Define Optuna parameter search space
    param_space = {
        "tau_lo": lambda trial: trial.suggest_float("tau_lo", 0.01, 0.1),
        "tau_hi": lambda trial: trial.suggest_float("tau_hi", 0.9, 0.99),
    }

    # Run tuning
    opt_model, best_score, study = tune_hyperparams(
        regressor=cqr,
        param_space=param_space,
        X=X_train,
        y=y_train,
        score_fn=interval_width,
        greater_is_better=False,
        n_trials=n_trials,
        n_splits=2,
        verbose=False,
    )

    # Check model returned
    assert opt_model is not None, "No model returned from tuning"
    assert hasattr(study, "best_trial"), "Optuna study missing best_trial"
    assert np.isfinite(best_score), "Best score is not finite"

    # Predict
    mean, lower, upper = opt_model.predict(X_test)

    # Validate output
    assert mean.shape == y_test.shape
    assert lower.shape == y_test.shape
    assert upper.shape == y_test.shape

    assert isinstance(mean, np.ndarray)
    assert isinstance(lower, np.ndarray)
    assert isinstance(upper, np.ndarray)