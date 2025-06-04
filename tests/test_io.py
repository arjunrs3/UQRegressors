import numpy as np
import torch
import os
import json
from pathlib import Path
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from datetime import datetime

# Replace with actual import paths
from uqregressors.conformal.conformal_ens import ConformalEnsRegressor
from uqregressors.conformal.conformal_deep_ens import ConformalizedDeepEns
from uqregressors.conformal.k_fold_cqr import KFoldCQR
from uqregressors.bayesian.deep_ens import DeepEnsembleRegressor
from uqregressors.bayesian.dropout import MCDropoutRegressor
from uqregressors.bayesian.gaussian_process import GPRegressor
from uqregressors.bayesian.gaussian_process_torch import GPRegressorTorch
from uqregressors.utils.io import save_model, load_model

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Simple synthetic regression dataset
def generate_data(n_samples=200, n_features=5):
    X = np.random.randn(n_samples, n_features)
    y = np.sin(X[:, 0]) + X[:, 1] ** 2 + np.random.normal(0, 0.1, size=n_samples)
    return X, y

def test_regressor_io(regressor_class, regressor_name):
    print(f"\nTesting {regressor_name}...")

    # Create synthetic data
    X, y = generate_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize regressor with short training for speed
    if regressor_class in [GPRegressor]: 
        reg = regressor_class()

    else: 
        reg = regressor_class(epochs=10, random_seed=42)

    # Fit and predict
    reg.fit(X_train, y_train)
    mean_pred, _, _ = reg.predict(X_test)
    mse = mean_squared_error(y_test, mean_pred)

    # Save everything
    save_path = save_model(
        reg,
        metrics={"mse": mse},
        X_train=X_train, 
        y_train=y_train, 
        X_test=X_test, 
        y_test=y_test
    )

    # Load everything
    load_dict = load_model(regressor_class, save_path)
    mse = load_dict["metrics"]["mse"]
    loaded_model = load_dict["model"]
    X_test = load_dict["X_test"]
    y_test = load_dict["y_test"]

    # Re-predict and verify
    mean_pred_loaded, _, _ = loaded_model.predict(X_test)
    loaded_mse = mean_squared_error(y_test, mean_pred_loaded)

    assert np.isclose(loaded_mse, mse), f"MSE mismatch! {mse} vs {loaded_mse}"
    print(f"{regressor_name} passed. MSE = {mse:.4f}")

# List of regressors to test
regressors_to_test = [
    (ConformalEnsRegressor, "ConformalEnsRegressor"),
    (ConformalizedDeepEns, "ConformalDeepEns"), 
    (DeepEnsembleRegressor, "DeepEnsembleRegressor"), 
    (MCDropoutRegressor, "MCDropoutRegressor"), 
    (GPRegressor, "GaussianProcessRegressor"), 
    (GPRegressorTorch, "GPRegressorTorch"), 
    (KFoldCQR, "KFoldCQR")
]

if __name__ == "__main__":
    for reg_cls, reg_name in regressors_to_test:
        test_regressor_io(reg_cls, reg_name)