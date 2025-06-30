import numpy as np
import torch
from sklearn.model_selection import train_test_split
from uqregressors.conformal.conformal_ens import ConformalEnsRegressor
from uqregressors.conformal.k_fold_cqr import KFoldCQR
from uqregressors.conformal.cqr import ConformalQuantileRegressor
from uqregressors.bayesian.deep_ens import DeepEnsembleRegressor
from uqregressors.bayesian.dropout import MCDropoutRegressor
from uqregressors.bayesian.gaussian_process import GPRegressor
from uqregressors.bayesian.bbmm_gp import BBMM_GP
from uqregressors.utils.file_manager import FileManager

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
    if mean.shape != expected_shape or lower.shape != expected_shape or upper.shape != expected_shape:
        raise AssertionError(f"Output shapes mismatch. Expected {expected_shape}, got mean:{mean.shape}, lower:{lower.shape}, upper:{upper.shape}")

    # Check type
    if requires_grad:
        # Expect torch tensors with requires_grad True or False (usually False at output)
        if not (torch.is_tensor(mean) and torch.is_tensor(lower) and torch.is_tensor(upper)):
            raise AssertionError(f"Expected torch.Tensor outputs when requires_grad=True, but got {type(mean)}, {type(lower)}, {type(upper)}")
    else:
        # Expect numpy arrays
        if not (isinstance(mean, np.ndarray) and isinstance(lower, np.ndarray) and isinstance(upper, np.ndarray)):
            raise AssertionError(f"Expected np.ndarray outputs when requires_grad=False, but got {type(mean)}, {type(lower)}, {type(upper)}")

def test_regressor_io(regressor_class, regressor_name):
    print(f"\nTesting {regressor_name} input/output types...")

    X, y = generate_data(n_samples=50, n_features=5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    input_types = ["numpy", "torch", "list"]
    requires_grad_options = [False, True]

    for requires_grad in requires_grad_options:
        # Instantiate model with requires_grad flag if available, else ignore
        try:
            reg = regressor_class(epochs=1, requires_grad=requires_grad, random_seed=42)
        except TypeError:
            # Class doesn't accept requires_grad param
            reg = regressor_class(epochs=1, random_seed=42)

        for input_type in input_types:
            X_in, y_in = convert_inputs(X_train, y_train, input_type)

            try:
                reg.fit(X_in, y_in)
            except Exception as e:
                print(f"Fit failed for input_type={input_type}, requires_grad={requires_grad}: {e}")
                continue

            X_test_in, _ = convert_inputs(X_test, y_test, input_type)

            try:
                preds = reg.predict(X_test_in)
            except Exception as e:
                print(f"Predict failed for input_type={input_type}, requires_grad={requires_grad}: {e}")
                continue

            expected_shape = (X_test.shape[0],)
            try:
                check_output_type_and_shape(preds, expected_shape, requires_grad)
            except AssertionError as e:
                print(f"Output check failed for input_type={input_type}, requires_grad={requires_grad}: {e}")
                continue

            print(f"Passed for input_type={input_type}, requires_grad={requires_grad}")

if __name__ == "__main__":
    regressors_to_test = [
        (DeepEnsembleRegressor, "DeepEnsembleRegressor"), 
        (ConformalEnsRegressor, "ConformalEnsRegressor"),
        (ConformalQuantileRegressor, "ConformalQuantileRegressor"),
        (MCDropoutRegressor, "MCDropoutRegressor"), 
        (BBMM_GP, "BBMM_GP"), 
        (KFoldCQR, "KFoldCQR")
    ]

    for reg_cls, reg_name in regressors_to_test:
        test_regressor_io(reg_cls, reg_name)
