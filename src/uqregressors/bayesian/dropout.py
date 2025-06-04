import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.base import BaseEstimator, RegressorMixin
from uqregressors.utils.activations import get_activation
from uqregressors.utils.logging import Logger
from pathlib import Path 
import json 
import pickle


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
    ):
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.alpha = alpha
        self.activation_str = activation_str
        self.n_samples = n_samples
        self.learning_rate = learning_rate
        self.epochs = epochs
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


    def fit(self, X, y):
        if self.random_seed is not None: 
            torch.manual_seed(self.random_seed)
            np.random.seed(self.random_seed)
        config = {
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
        }

        logger = Logger(
            use_wandb=self.use_wandb,
            project_name=self.wandb_project,
            run_name=self.wandb_run_name,
            config=config,
        )

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        input_dim = X.shape[1]
        self.input_dim = input_dim
        activation = get_activation(self.activation_str)

        model = MLP(input_dim, self.hidden_sizes, self.dropout, activation)
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
        for epoch in range(self.epochs):
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

    def predict(self, X):
        if self.random_seed is not None: 
            torch.manual_seed(self.random_seed)
            np.random.seed(self.random_seed)
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.model.train()
        preds = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                preds.append(self.model(X).cpu().numpy())
        preds = np.stack(preds, axis=0)
        mean = preds.mean(axis=0).squeeze()
        lower = np.percentile(preds, 100 * self.alpha / 2, axis=0).squeeze()
        upper = np.percentile(preds, 100 * (1 - self.alpha / 2), axis=0).squeeze()
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