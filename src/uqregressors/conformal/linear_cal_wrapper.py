"""
Linear Calibration Wrapper 
-----------------

This module wraps a UQregressor with linear calibration of the interval width for calibration purposes. 
The underlying regressor must predict a lower, mean, and upper value for each input with some specified confidence. 
"""

import numpy as np 
import torch
from uqregressors.utils.data_loader import validate_and_prepare_inputs, validate_X_input
import pickle 
from pathlib import Path 
from sklearn.base import BaseEstimator, RegressorMixin 
from uqregressors.utils.torch_sklearn_utils import train_test_split
import json 
import copy

class LinearCalWrapper(BaseEstimator, RegressorMixin): 
    def __init__(self, 
                 underlying_regressor=None, 
                 cal_size=0.3, 
                 alpha=0.1):
        self.name = "LinearCal_" + underlying_regressor.name
        self.ur = underlying_regressor 
        self.cal_size = cal_size
        self.fitted = False
        self.alpha = alpha
        self.ur.alpha = alpha
        self.X_cal = None 
        self.y_cal = None
        self.input_dim = self.ur.input_dim

        self.A = None 
        self.b = None         

    def set_alpha(self, alpha): 
        self.alpha = alpha 
        self.ur.alpha = alpha

    def fit(self, X, y): 
        """
        Fit the ensemble on training data. 

        Args: 
            X (array-like or torch.Tensor): Training inputs 
            y (array-like or torch.Tensor): Training targets

        Returns: 
            (ConformalWrapper): Fitted estimator 
        """
        X_tensor, y_tensor = validate_and_prepare_inputs(X, y, device = self.ur.device)
        input_dim = X_tensor.shape[1]
        self.input_dim = input_dim

        X_train, X_cal, y_train, y_cal = train_test_split(X_tensor, y_tensor, test_size=self.cal_size, device=self.ur.device, random_state=self.ur.random_seed)

        self.X_cal = X_cal 
        self.y_cal = y_cal 

        self.ur.fit(X_train, y_train) 
        
        self.fitted = True
        return self

    def transform_widths(self, X_test): 
        requires_grad = copy.copy(self.ur.requires_grad)
        self.ur.requires_grad = True

        mean_raw, lower_raw, upper_raw = self.ur.predict(self.X_cal)
        A_vals = np.linspace(0.5, 2, 150)
        b_vals = np.linspace(-1, 1, 200)

        c  = torch.as_tensor(mean_raw, device=self.ur.device).view(-1, 1, 1)
        lw = torch.as_tensor(mean_raw - lower_raw, device=self.ur.device).view(-1, 1, 1)
        uw = torch.as_tensor(upper_raw - mean_raw, device=self.ur.device).view(-1, 1, 1)
        y  = torch.as_tensor(self.y_cal, device=self.ur.device).view(-1, 1, 1)

        a = torch.Tensor(A_vals[None, :, None], device=self.ur.device)
        b = torch.Tensor(b_vals[None, None, :], device=self.ur.device)

        lower = c - lw * a - b/2 
        upper = c + uw * a + b/2

        alpha = self.alpha 

        under = torch.clamp(lower - y, min=0.0)
        over = torch.clamp(y - upper, min=0.0)

        outside = (under + over) > 0.0
        inside = ~outside

        coverage = inside.float().mean(dim=0)

        score = upper - lower + 2 / alpha * under + 2/ alpha * over 
        total_score = score.sum(dim=0)

        mask = coverage >= (1 - alpha)

        masked_scores = total_score.clone()
        masked_scores[~mask] = float('inf')

        idx = torch.argmin(masked_scores)
        i, j = torch.unravel_index(idx, total_score.shape)

        A = A_vals[i]
        b = b_vals[j]

        self.A = A 
        self.b = b
        
        mean_test, lower_test, upper_test = self.ur.predict(X_test) 

        raw_widths = upper_test - lower_test 
        raw_lw = mean_test - lower_test 
        raw_uw = upper_test - mean_test 

        calibrated_lower = mean_test - A * raw_lw - b/2  
        calibrated_upper = mean_test + A * raw_uw + b/2

        self.ur.requires_grad = requires_grad

        return mean_test, calibrated_lower, calibrated_upper

    def predict(self, X): 
        """
        Predicts the target values with uncertainty estimates, conformalized.

        Args: 
            X (np.ndarray): Feature matrix of shape (n_samples, n_features). 

        Returns:
            (Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]): Tuple containing:
                mean predictions,
                lower bound of the prediction interval,
                upper bound of the prediction interval.
        
        !!! note
            If `requires_grad` is False, all returned arrays are NumPy arrays.
            Otherwise, they are PyTorch tensors with gradients.Returns: 
        """
        if not self.fitted: 
            raise ValueError("Model not yet fit. Please call fit() before predict().")
        
        X_tensor = validate_X_input(X, input_dim=self.input_dim, device=self.ur.device, requires_grad=self.ur.requires_grad)

        mean, lower, upper = self.transform_widths(X_tensor)

        if not self.ur.requires_grad: 
            return mean.detach().cpu().numpy(), lower.detach().cpu().numpy(), upper.detach().cpu().numpy()

        else: 
            return mean, lower, upper
        
    def save(self, path): 
        """
        Save model weights, config, and scalers to disk.

        Args:
            path (str or Path): Directory to save model components.
        """
        path = Path(path)
        config = {"name": self.name,  
                  "cal_size": self.cal_size, 
                  "fitted": self.fitted, 
                  "input_dim": self.input_dim, 
                  "alpha": self.alpha
                  }
        
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=4)

        with open(path / "model_class.pkl", "wb") as f: 
            pickle.dump(self.ur.__class__, f)

        self.ur.save(path / "model")

    @classmethod 
    def load(cls, path, device="cpu", load_logs=False):
        """
        Load a saved conformalized regressor from disk.

        Args:
            path (str or pathlib.Path): Directory path to load the model from.
            device (str or torch.device): Device to load the model onto.
            load_logs (bool): Whether to load training and tuning logs.

        Returns:
            (ConformalWrapper): Loaded model instance.
        """

        path = Path(path)
        with open(path / "config.json", "r") as f: 
            config = json.load(f) 

        fitted = config.pop("fitted", False)
        name = config.pop("name", "ConformalWrapper")
        input_dim = config.pop("input_dim", None)
        A = config.pop("A", None)
        b = config.pop("b", None)

        with open(path / "model_class.pkl", "rb") as f: 
            model_cls = pickle.load(f)

        ur = model_cls.load(path / "model", device=device, load_logs=load_logs)
        model = cls(**config, underlying_regressor=ur)
        model.fitted = fitted
        model.input_dim = input_dim
        model.A = A 
        model.b = b

        extras_path = path / "extras.pt"
        if extras_path.exists():
            extras = torch.load(extras_path, map_location=device, weights_only=False)
            model.conformity_scores = extras.get("conformity_scores", None)
            model.conformal_width = extras.get("conformal_width", None)
            model.X_cal = extras.get("X_cal", None)
            model.y_cal = extras.get("y_cal", None)

        return model
