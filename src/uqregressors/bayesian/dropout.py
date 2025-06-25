import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from uqregressors.utils.activations import get_activation
from uqregressors.utils.logging import Logger
from sklearn.model_selection import train_test_split
from pathlib import Path 
import json 
import pickle
import scipy.stats as st
from scipy.special import logsumexp
from skopt import gp_minimize
from skopt.space import Real


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes, dropout, activation):
        super().__init__()
        layers = []
        for h in hidden_sizes:
            layers.append(nn.Linear(input_dim, h))
            layers.append(activation())
            layers.append(nn.Dropout(dropout))
            input_dim = h
        layers.append(nn.Linear(hidden_sizes[-1], 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class MCDropoutRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        hidden_sizes=[64, 64],
        dropout=0.1,
        tau=1.0,
        tune_tau=False,
        prior_length_scale=1e-2, 
        use_paper_weight_decay=True,
        BO_calls=30,
        BO_epochs=40,
        alpha=0.1,
        activation_str="ReLU",
        n_samples=100,
        learning_rate=1e-3,
        epochs=200,
        batch_size=32,
        optimizer_cls=torch.optim.Adam,
        optimizer_kwargs=None,
        scheduler_cls=None,
        scheduler_kwargs=None,
        loss_fn=torch.nn.functional.mse_loss,
        device="cpu",
        use_wandb=False,
        wandb_project=None,
        wandb_run_name=None,
        random_seed=None,
        scale_data=True, 
        input_scaler=None,
        output_scaler=None
    ):
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.tau = tau
        self.tune_tau = tune_tau
        self.prior_length_scale = prior_length_scale
        self.use_paper_weight_decay = use_paper_weight_decay
        self.BO_calls = BO_calls
        self.alpha = alpha
        self.activation_str = activation_str
        self.n_samples = n_samples
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.BO_epochs = BO_epochs
        self.batch_size = batch_size
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.scheduler_cls = scheduler_cls
        self.scheduler_kwargs = scheduler_kwargs or {}
        self.loss_fn = loss_fn

        self.device = device
        self.model = None

        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name
        self.random_seed = random_seed
        self.input_dim = None

        self.scale_data = scale_data
        self.input_scaler = input_scaler or StandardScaler()
        self.output_scaler = output_scaler or StandardScaler()

    def fit(self, X, y): 
        if self.scale_data: 
            X = self.input_scaler.fit_transform(X)
            y = self.output_scaler.fit_transform(y.reshape(-1, 1))

        if self.tune_tau:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=self.random_seed)
        
            X_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
            y_tensor = torch.tensor(y_train, dtype=torch.float32).to(self.device)
            input_dim = X.shape[1]
            self.input_dim = input_dim

            print("Tuning tau...")
            self.tau = self._tune_tau(X_tensor, y_tensor, X_val, y_val)
            print(f"Best tau found: {self.tau:.4f}")
        
        X_train, y_train = X, y
        X_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        input_dim = X.shape[1]
        self.input_dim = input_dim
        
        self._fit_single_model(X_tensor, y_tensor, mode="FIT")

    def _fit_single_model(self, X_tensor, y_tensor, mode="FIT"):
        if self.random_seed is not None: 
            torch.manual_seed(self.random_seed)
            np.random.seed(self.random_seed)
        
        if mode == "FIT": 
            epochs = self.epochs 
        elif mode == "BO": 
            epochs = self.BO_epochs 
        else: 
            raise ValueError("mode passed to _fit_single_model is unrecognized")
        
        config = {
            "learning_rate": self.learning_rate,
            "epochs": epochs,
            "batch_size": self.batch_size,
        }

        logger = Logger(
            use_wandb=self.use_wandb,
            project_name=self.wandb_project,
            run_name=self.wandb_run_name,
            config=config,
        )

        activation = get_activation(self.activation_str)

        model = MLP(self.input_dim, self.hidden_sizes, self.dropout, activation)
        self.model = model.to(self.device)

        optimizer = self.optimizer_cls(
            self.model.parameters(), lr=self.learning_rate, **self.optimizer_kwargs
        )

        scheduler = None
        if self.scheduler_cls is not None:
            scheduler = self.scheduler_cls(optimizer, **self.scheduler_kwargs)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for xb, yb in dataloader: 
                optimizer.zero_grad()
                preds = self.model(xb)
                loss = self.loss_fn(preds, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss

            if scheduler is not None:
                scheduler.step()

            if epoch % (self.epochs / 20) == 0:
                logger.log({"epoch": epoch, "train_loss": epoch_loss})

        logger.finish()

        return self

    def _tune_tau(self, X_train, y_train, X_val, y_val):
        """
        Use Bayesian optimization to tune tau by maximizing predictive log-likelihood.
        """
        def objective(tau):
            if self.use_paper_weight_decay:
                N = len(X_train)
                l = self.prior_length_scale
                p = 1 - self.dropout  # Keep probability
                weight_decay = (p * l**2) / (2 * N * tau[0])
                print(f"Setting weight decay to {weight_decay:.6f}")
                self.optimizer_kwargs["weight_decay"] = weight_decay

            self._fit_single_model(X_train, y_train, mode="BO")
            # tau is a list with one element
            return -self._predictive_log_likelihood(X_val, y_val, tau[0])

        search_space = [Real(1e-3, 1e3, prior='log-uniform', name='tau')]

        result = gp_minimize(
            func=objective,
            dimensions=search_space,
            n_calls=self.BO_calls,
            n_random_starts=5,
            random_state=self.random_seed or 42,
            verbose=True
        )

        best_tau = result.x[0]
        if self.use_paper_weight_decay:
                N = len(X_train)
                l = self.prior_length_scale
                p = 1 - self.dropout  # Keep probability
                weight_decay = (p * l**2) / (2 * N * best_tau)
                print(f"Setting weight decay to {weight_decay:.6f}")
                self.optimizer_kwargs["weight_decay"] = weight_decay

        return best_tau

    def predict(self, X):
        if self.random_seed is not None: 
            torch.manual_seed(self.random_seed)
            np.random.seed(self.random_seed)
        
        if self.scale_data: 
            X = self.input_scaler.transform(X)
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.model.train()
        preds = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                preds.append(self.model(X).cpu().numpy())
        preds = np.stack(preds, axis=0)
        mean = preds.mean(axis=0)
        variance = np.var(preds, axis=0) + 1 / self.tau 
        std = np.sqrt(variance)
        z_score = st.norm.ppf(1 - self.alpha / 2)
        lower = mean - std * z_score 
        upper = mean + std * z_score

        if self.scale_data: 
            mean = self.output_scaler.inverse_transform(mean).squeeze()
            lower = self.output_scaler.inverse_transform(lower).squeeze()
            upper = self.output_scaler.inverse_transform(upper).squeeze()
        
        else: 
            mean = mean.squeeze() 
            lower = lower.squeeze() 
            upper = upper.squeeze() 

        return mean, lower, upper

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save config (exclude non-serializable or large objects)
        config = {
            k: v for k, v in self.__dict__.items()
            if k not in ["optimizer_cls", "optimizer_kwargs", "scheduler_cls", "scheduler_kwargs"]
            and not callable(v)
            and not isinstance(v, (torch.nn.Module,))
        }
        config["optimizer"] = self.optimizer_cls.__class__.__name__ if self.optimizer_cls is not None else None
        config["scheduler"] = self.optimizer_cls.__class__.__name__ if self.scheduler_cls is not None else None

        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=4)

        with open(path / "extras.pkl", 'wb') as f: 
            pickle.dump([self.optimizer_cls, 
                         self.optimizer_kwargs, self.scheduler_cls, self.scheduler_kwargs], f)

        # Save model weights
        torch.save(self.model.state_dict(), path / f"model.pt")


    @classmethod
    def load(cls, path, device="cpu"):
        path = Path(path)

        # Load config
        with open(path / "config.json", "r") as f:
            config = json.load(f)
        config["device"] = device

        config.pop("optimizer", None)
        config.pop("scheduler", None)
        input_dim = config.pop("input_dim", None)
        model = cls(**config)

        with open(path / "extras.pkl", 'rb') as f: 
            optimizer_cls, optimizer_kwargs, scheduler_cls, scheduler_kwargs = pickle.load(f)

        # Recreate models
        model.input_dim = input_dim
        activation = get_activation(config["activation_str"])

        model.model = MLP(model.input_dim, config["hidden_sizes"], model.dropout, activation).to(device)
        model.model.load_state_dict(torch.load(path / f"model.pt", map_location=device))

        model.optimizer_cls = optimizer_cls 
        model.optimizer_kwargs = optimizer_kwargs 
        model.scheduler_cls = scheduler_cls 
        model.scheduler_kwargs = scheduler_kwargs
        
        return model
    
    def _predictive_log_likelihood(self, X, y_true, tau):
        self.model.train()

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_true = y_true.reshape(-1)
        preds = []

        with torch.no_grad():
            for _ in range(self.n_samples):
                pred = self.model(X_tensor).cpu().numpy().squeeze()
                preds.append(pred)
        preds = np.stack(preds, axis=0)  # Shape: (T, N)

        mean_preds = preds.mean(axis=0)
        var_preds = preds.var(axis=0) + (1 / tau)

        log_likelihoods = (
            -0.5 * np.log(2 * np.pi * var_preds)
            - 0.5 * ((y_true - mean_preds) ** 2) / var_preds
        )

        return np.mean(log_likelihoods)
    