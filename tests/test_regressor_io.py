import numpy as np
import torch
import pytest
from sklearn.model_selection import train_test_split

from uqregressors.conformal.conformal_ens import ConformalEnsRegressor
from uqregressors.conformal.k_fold_cqr import KFoldCQR
from uqregressors.conformal.cqr import ConformalQuantileRegressor
from uqregressors.bayesian.deep_ens import DeepEnsembleRegressor
from uqregressors.bayesian.dropout import MCDropoutRegressor
from uqregressors.bayesian.gaussian_process import GPRegressor
from uqregressors.bayesian.bbmm_gp import BBMM_GP

def generate_data(n_samples=100, n_features=5):
    X = np.random.randn(n_samples, n_features)
    y = np.sin(X[:, 0]) + X[:, 1] ** 2 + np.random.normal(0, 0.1, size=n_samples)
    return X, y

def convert_inputs(X, y, input_type):
    if input_type == "numpy":
        return np.array(X), np.array(y)
    elif input_type == "torch":
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    elif input_type == "list":
        return X.tolist(), y.tolist()
    else:
        raise ValueError(f"Unknown input_type: {input_type}")

def check_output_type_and_shape(output, expected_shape, requires_grad):
    mean, lower, upper = output

    # Check shape
    assert mean.shape == expected_shape, f"Mean shape mismatch: {mean.shape} != {expected_shape}"
    assert lower.shape == expected_shape, f"Lower shape mismatch: {lower.shape} != {expected_shape}"
    assert upper.shape == expected_shape, f"Upper shape mismatch: {upper.shape} != {expected_shape}"

    if requires_grad:
        assert all(torch.is_tensor(t) for t in (mean, lower, upper)), \
            f"Expected torch.Tensor outputs with requires_grad=True, got {type(mean)}, {type(lower)}, {type(upper)}"
    else:
        assert all(isinstance(t, np.ndarray) for t in (mean, lower, upper)), \
            f"Expected np.ndarray outputs with requires_grad=False, got {type(mean)}, {type(lower)}, {type(upper)}"

# List of regressors to test
regressors_to_test = [
    (DeepEnsembleRegressor, "DeepEnsembleRegressor"), 
    (ConformalEnsRegressor, "ConformalEnsRegressor"),
    (ConformalQuantileRegressor, "ConformalQuantileRegressor"),
    (MCDropoutRegressor, "MCDropoutRegressor"), 
    (BBMM_GP, "BBMM_GP"), 
    (KFoldCQR, "KFoldCQR")
]

# All combinations
@pytest.mark.parametrize("regressor_class, regressor_name", regressors_to_test)
@pytest.mark.parametrize("input_type", ["numpy", "torch", "list"])
@pytest.mark.parametrize("requires_grad", [False, True])
def test_regressor_io_types_and_shapes(regressor_class, regressor_name, input_type, requires_grad):
    X, y = generate_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    expected_shape = (X_test.shape[0],)

    # Instantiate regressor
    try:
        reg = regressor_class(epochs=1, requires_grad=requires_grad, random_seed=42)
    except TypeError:
        # Fallback if requires_grad is not supported
        reg = regressor_class(epochs=1, random_seed=42)

    # Convert inputs
    X_in, y_in = convert_inputs(X_train, y_train, input_type)

    # Fit
    reg.fit(X_in, y_in)

    # Predict
    X_test_in, _ = convert_inputs(X_test, y_test, input_type)
    preds = reg.predict(X_test_in)

    # Check outputs
    check_output_type_and_shape(preds, expected_shape, requires_grad)