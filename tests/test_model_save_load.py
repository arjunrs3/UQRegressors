import numpy as np
import torch
import pytest
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from uqregressors.conformal.conformal_ens import ConformalEnsRegressor
from uqregressors.conformal.k_fold_cqr import KFoldCQR
from uqregressors.conformal.cqr import ConformalQuantileRegressor
from uqregressors.bayesian.deep_ens import DeepEnsembleRegressor
from uqregressors.bayesian.dropout import MCDropoutRegressor
from uqregressors.bayesian.gaussian_process import GPRegressor
from uqregressors.bayesian.bbmm_gp import BBMM_GP
from uqregressors.utils.file_manager import FileManager

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Generate synthetic dataset
def generate_data(n_samples=200, n_features=5):
    X = np.random.randn(n_samples, n_features)
    y = np.sin(X[:, 0]) + X[:, 1] ** 2 + np.random.normal(0, 0.1, size=n_samples)
    return X, y

@pytest.mark.parametrize("regressor_class, regressor_name", [
    (DeepEnsembleRegressor, "DeepEnsembleRegressor"), 
    (ConformalEnsRegressor, "ConformalEnsRegressor"),
    (ConformalQuantileRegressor, "ConformalQuantileRegressor"),
    (MCDropoutRegressor, "MCDropoutRegressor"), 
    (GPRegressor, "GaussianProcessRegressor"), 
    (BBMM_GP, "BBMM_GP"), 
    (KFoldCQR, "KFoldCQR")
])
def test_model_save_and_load(regressor_class, regressor_name):
    fm = FileManager()

    X, y = generate_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if regressor_class in [GPRegressor]:
        reg = regressor_class()
    else:
        reg = regressor_class(epochs=10, random_seed=42)

    reg.fit(X_train, y_train)
    mean_pred, _, _ = reg.predict(X_test)
    mse = mean_squared_error(y_test, mean_pred)

    save_path = fm.save_model(
        reg,
        metrics={"mse": mse},
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    load_dict = fm.load_model(regressor_class, save_path, load_logs=True)
    loaded_model = load_dict["model"]
    X_test_loaded = load_dict["X_test"]
    y_test_loaded = load_dict["y_test"]
    mse_loaded = load_dict["metrics"]["mse"]

    mean_pred_loaded, _, _ = loaded_model.predict(X_test_loaded)
    loaded_mse = mean_squared_error(y_test_loaded, mean_pred_loaded)

    assert np.isclose(mse_loaded, loaded_mse), f"{regressor_name}: MSE mismatch {mse_loaded:.4f} vs {loaded_mse:.4f}"