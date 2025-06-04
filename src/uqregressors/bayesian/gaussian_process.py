from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import sklearn
import scipy.stats as st
from pathlib import Path 
import torch 
import json
import pickle 

class GPRegressor: 
    def __init__(self, kernel = RBF(), 
                 alpha=0.1, 
                 gp_kwargs=None):
        self.kernel = kernel 
        self.alpha = alpha 
        self.gp_kwargs = gp_kwargs or {}
        self.model = None

    def fit(self, X, y): 
        model = GaussianProcessRegressor(kernel=self.kernel, **self.gp_kwargs)
        model.fit(X, y)
        self.model = model

    def predict(self, X):
        preds, std = self.model.predict(X, return_std=True)
        z_score = st.norm.ppf(1 - self.alpha / 2)
        mean = preds
        lower = mean - z_score * std
        upper = mean + z_score * std
        return mean, lower, upper
    
    def save(self, path): 
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True) 

        config = {
            k: v for k, v in self.__dict__.items()
            if k not in ["kernel", "model"]
            and not callable(v)
            and not isinstance(v, ())
        }
        config["kernel"] = self.kernel.__class__.__name__

        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=4)

        with open(path / "model.pkl", 'wb') as file: 
            pickle.dump(self, file)

    @classmethod
    def load(cls, path, device="cpu"): 
        path = Path(path)

        with open(path / "model.pkl", 'rb') as file: 
            model = pickle.load(file)
        
        return model
