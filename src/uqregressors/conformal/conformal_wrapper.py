"""
Conformal Wrapper 
-----------------

This module wraps a UQregressor with conformal prediction. The underlying regressor must predict 
a lower, mean, and upper value for each input. 
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

class ConformalWrapper(BaseEstimator, RegressorMixin): 
    def __init__(self, 
                 underlying_regressor=None, 
                 cal_size=0.3):
        self.name = "Conformal_" + underlying_regressor.name
        self.ur = underlying_regressor 
        self.cal_size = cal_size
        self.fitted = False
        self.alpha = self.ur.alpha
        self.X_cal = None 
        self.y_cal = None

        self.conformity_scores = None 
        self.conformal_width = None         

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
        self.compute_conformity_scores(X_cal, y_cal)
        
        self.fitted = True
        return self

    def compute_conformity_scores(self, X_cal, y_cal): 
        requires_grad = copy.copy(self.ur.requires_grad)
        self.ur.requires_grad = True 
        mean, lower, upper = self.ur.predict(X_cal)
        self.ur.requires_grad = requires_grad
        lower_loss = lower.view(-1, 1) - y_cal.view(-1, 1)
        upper_loss = -1 * (upper.view(-1, 1) - y_cal.view(-1, 1))
        self.conformity_scores = torch.max(torch.cat((lower_loss, upper_loss), dim=1), dim=1).values

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

        n = len(self.conformity_scores)
        q = int((1-self.ur.alpha) * (n+1))
        q = min(q, n-1)
        res_quantile = n - q 

        self.conformal_width = torch.topk(self.conformity_scores, res_quantile).values[-1]

        requires_grad = copy.copy(self.ur.requires_grad)
        self.ur.requires_grad = True
        mean, lower, upper = self.ur.predict(X_tensor) 
        lower = lower - self.conformal_width 
        upper = upper + self.conformal_width
        self.ur.requires_grad = requires_grad

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
                  "input_dim": self.input_dim
                  }
        
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=4)

        with open(path / "model_class.pkl", "wb") as f: 
            pickle.dump(self.ur.__class__, f)

        torch.save({
            "conformity_scores": self.conformity_scores, 
            "conformal_width": self.conformal_width, 
            "X_cal": self.X_cal, 
            "y_cal": self.y_cal
        }, path / "extras.pt")

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

        with open(path / "model_class.pkl", "rb") as f: 
            model_cls = pickle.load(f)

        ur = model_cls.load(path / "model", device=device, load_logs=load_logs)
        model = cls(**config, underlying_regressor=ur)
        model.fitted = fitted
        model.input_dim = input_dim

        extras_path = path / "extras.pt"
        if extras_path.exists():
            extras = torch.load(extras_path, map_location=device, weights_only=False)
            model.conformity_scores = extras.get("conformity_scores", None)
            model.conformal_width = extras.get("conformal_width", None)
            model.X_cal = extras.get("X_cal", None)
            model.y_cal = extras.get("y_cal", None)

        return model
