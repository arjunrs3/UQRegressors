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
from sklearn.isotonic import IsotonicRegression
from uqregressors.utils.torch_sklearn_utils import train_test_split
from scipy.stats import norm
import json 
import copy

class IsotonicCalWrapper(BaseEstimator, RegressorMixin): 
    def __init__(self, 
                 underlying_regressor=None, 
                 cal_size=0.3, 
                 alpha=0.1):
        self.name = "Isotonic_Cal_" + underlying_regressor.name
        self.ur = underlying_regressor 
        self.cal_size = cal_size
        self.fitted = False
        self.alpha = alpha
        self.ur.alpha = alpha
        self.X_cal = None 
        self.y_cal = None
        self.input_dim = self.ur.input_dim

        self.cal_regressor = None   

        self.isotonic_regressor=None
        self.upper_quantile = None 
        self.lower_quantile = None

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

    def fit_isotonic_regressor(self): 
        y_cal_npy = self.y_cal.detach().cpu().numpy().ravel()

        requires_grad = copy.copy(self.ur.requires_grad)
        self.ur.requires_grad = False

        mean_raw, lower_raw, upper_raw = self.ur.predict(self.X_cal)

        alpha = self.alpha 
        z = norm.ppf(1 - alpha / 2)
        std = (upper_raw - lower_raw) / (2 * z)
        std = np.clip(std, 1e-6, None)

        self.ur.requires_grad = requires_grad 
        predicted_CDFs = norm.cdf(y_cal_npy, loc=mean_raw, scale=std)
        predicted_CDFs = np.sort(predicted_CDFs)
        isotonic_x = predicted_CDFs 

        T = len(isotonic_x)
        isotonic_y = 1 / T * np.arange(1, T+1)

        ir = IsotonicRegression(y_min = 0, y_max = 1, out_of_bounds="clip")
        ir.fit(isotonic_x, isotonic_y)
        self.isotonic_regressor = ir 
        return ir

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

        ir = self.fit_isotonic_regressor() 
        X_lookup = np.linspace(0, 1, 10000)
        y_lookup = ir.transform(X_lookup)

        upper_idx = np.searchsorted(y_lookup, 1-self.alpha / 2, side='right')
        lower_idx = np.searchsorted(y_lookup, self.alpha / 2, side='left') - 1

        upper_idx = np.clip(upper_idx, 0, len(X_lookup) - 1)
        lower_idx = np.clip(lower_idx, 0, len(X_lookup) - 1)

        upper_quantile = X_lookup[upper_idx]
        lower_quantile = X_lookup[lower_idx]

        eps = 1e-6
        upper_quantile = np.clip(upper_quantile, eps, 1 - eps)
        lower_quantile = np.clip(lower_quantile, eps, 1 - eps)

        mean, lower, upper = self.ur.predict(X_tensor)
        raw_z = norm.ppf(1 - self.alpha / 2)
        raw_std = (upper - lower) / (2 * raw_z)

        z_upper = norm.ppf(upper_quantile)
        z_lower = norm.ppf(lower_quantile)

        cal_lower = mean + z_lower * raw_std 
        cal_upper = mean + z_upper * raw_std

        self.upper_quantile = upper_quantile 
        self.lower_quantile = lower_quantile 
        
        return mean, cal_lower, cal_upper

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
                  "alpha": self.alpha, 
                  "upper_quantile": self.upper_quantile, 
                  "lower_quantile": self.lower_quantile
                  }
        
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=4)

        with open(path / "model_class.pkl", "wb") as f: 
            pickle.dump(self.ur.__class__, f)

        torch.save({
            "X_cal": self.X_cal, 
            "y_cal": self.y_cal
        }, path / "extras.pt")

        self.ur.save(path / "model")
        with open(path / "isotonic.pkl", "wb") as f:
            pickle.dump(self.isotonic_regressor, f)

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
        name = config.pop("name", "ExponentialCalWrapper")
        input_dim = config.pop("input_dim", None)
        lower_quantile = config.pop("lower_quantile", None)
        upper_quantile = config.pop("upper_quantile", None)

        with open(path / "model_class.pkl", "rb") as f: 
            model_cls = pickle.load(f)

        ur = model_cls.load(path / "model", device=device, load_logs=load_logs)
        model = cls(**config, underlying_regressor=ur)
        model.fitted = fitted
        model.input_dim = input_dim
        model.lower_quantile = lower_quantile 
        model.upper_quantile = upper_quantile

        iso_path = path / "isotonic.pkl"
        if iso_path.exists(): 
            with open(iso_path, "rb") as f: 
                model.isotonic_regressor=pickle.load(f)

        extras_path = path / "extras.pt"
        if extras_path.exists():
            extras = torch.load(extras_path, map_location=device, weights_only=False)
            model.X_cal = extras.get("X_cal", None)
            model.y_cal = extras.get("y_cal", None)

        return model