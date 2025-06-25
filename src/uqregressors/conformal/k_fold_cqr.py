import numpy as np 
import torch 
import torch.nn as nn 
from torch.utils.data import TensorDataset, DataLoader 
from sklearn.base import BaseEstimator, RegressorMixin 
from uqregressors.utils.activations import get_activation 
from uqregressors.utils.logging import Logger
from joblib import Parallel, delayed 
from sklearn.model_selection import KFold
from pathlib import Path 
import json 
import pickle
from sklearn.preprocessing import StandardScaler

class QuantNN(nn.Module): 
    def __init__(self, input_dim, hidden_sizes, dropout, activation): 
        super().__init__()
        layers = []
        for h in hidden_sizes:
            layers.append(nn.Linear(input_dim, h))
            layers.append(activation())
            if dropout is not None: 
                layers.append(nn.Dropout(dropout))
            input_dim = h
        output_layer = nn.Linear(hidden_sizes[-1], 2)
        layers.append(output_layer)
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    

class KFoldCQR(BaseEstimator, RegressorMixin): 
    def __init__(
            self, 
            n_estimators=5,
            hidden_sizes=[64, 64], 
            dropout = None,
            alpha=0.1, 
            tau_lo = None, 
            tau_hi = None, 
            n_jobs=1, 
            activation_str="ReLU",
            learning_rate=1e-3,
            epochs=200,
            batch_size=32,
            optimizer_cls=torch.optim.Adam,
            optimizer_kwargs=None,
            scheduler_cls=None,
            scheduler_kwargs=None,
            loss_fn=None,
            device="cpu",
            use_wandb=False,
            wandb_project=None,
            wandb_run_name=None,
            scale_data = True, 
            input_scaler = None, 
            output_scaler = None,
            random_seed=None
    ):
        self.n_estimators = n_estimators
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.alpha = alpha
        self.tau_lo = tau_lo or alpha / 2 
        self.tau_hi = tau_hi or 1 - alpha / 2
        self.activation_str = activation_str
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.scheduler_cls = scheduler_cls
        self.scheduler_kwargs = scheduler_kwargs or {}
        self.loss_fn = loss_fn or self.quantile_loss
        self.device = device

        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name

        self.n_jobs = n_jobs
        self.random_seed = random_seed
        self.quantiles = torch.tensor([self.alpha / 2, 1 - self.alpha / 2], device=self.device)
        self.models = []
        self.residuals = []
        self.conformal_width = None
        self.input_dim = None
        if self.n_estimators == 1: 
            raise ValueError("n_estimators set to 1. To use a single Quantile Regressor, use a non-ensembled Quantile Regressor class")
        self.scale_data = scale_data 
        self.input_scaler = input_scaler or StandardScaler() 
        self.output_scaler = output_scaler or StandardScaler()


    def quantile_loss(self, preds, y): 
        error = y.view(-1, 1) - preds
        return torch.mean(torch.max(self.quantiles * error, (self.quantiles - 1) * error))

    def _train_single_model(self, X_tensor, y_tensor, input_dim, train_idx, cal_idx, model_idx): 
        if self.random_seed is not None: 
            torch.manual_seed(self.random_seed + model_idx)
            np.random.seed(self.random_seed + model_idx)

        activation = get_activation(self.activation_str)
        model = QuantNN(input_dim, self.hidden_sizes, self.dropout, activation).to(self.device)

        optimizer = self.optimizer_cls(
            model.parameters(), lr=self.learning_rate, **self.optimizer_kwargs
        )
        scheduler = None 
        if self.scheduler_cls: 
            scheduler = self.scheduler_cls(optimizer, **self.scheduler_kwargs)

        dataset = TensorDataset(X_tensor[train_idx], y_tensor[train_idx])
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        logger = Logger(
            use_wandb=self.use_wandb,
            project_name=self.wandb_project,
            run_name=self.wandb_run_name + str(model_idx) if self.wandb_run_name is not None else None,
            config={"n_estimators": self.n_estimators, "learning_rate": self.learning_rate, "epochs": self.epochs},
            name=f"Estimator-{model_idx}"
        )
        
        model.train()
        for epoch in range(self.epochs): 
            model.train()
            epoch_loss = 0.0 
            for xb, yb in dataloader: 
                optimizer.zero_grad() 
                preds = model(xb)
                loss = self.loss_fn(preds, yb)
                loss.backward() 
                optimizer.step() 
                epoch_loss += loss 
            
            if epoch % (self.epochs / 20) == 0:
                logger.log({"epoch": epoch, "train_loss": epoch_loss})

            if scheduler: 
                scheduler.step()

        
        test_X = X_tensor[cal_idx]
        test_y = y_tensor[cal_idx]
        oof_preds = model(test_X)
        loss_matrix =(oof_preds - test_y) * torch.tensor([1, -1], device=self.device)
        residuals = torch.max(loss_matrix, dim=1).values
        logger.finish()
        return model, residuals
    
    def fit(self, X, y): 
        if self.scale_data:
            X = self.input_scaler.fit_transform(X)
            y = self.output_scaler.fit_transform(y.reshape(-1, 1))

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)

        input_dim = X.shape[1]
        self.input_dim = input_dim

        kf = KFold(n_splits=self.n_estimators, shuffle=True)

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._train_single_model)(X_tensor, y_tensor, input_dim, train_idx, cal_idx, i)
            for i, (train_idx, cal_idx) in enumerate(kf.split(X_tensor.detach().cpu().numpy()))
        )

        self.models = [result[0] for result in results]
        self.residuals = torch.cat([result[1] for result in results], dim=0).ravel()

        n = len(self.residuals)
        q = int((1 - self.alpha) * (n + 1))
        q = min(q, n-1)
        self.conformal_width = torch.topk(self.residuals, n-q).values[-1].detach().cpu().numpy()
        return self
    
    def predict(self, X): 
        if self.scale_data: 
            X = self.input_scaler.transform(X)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        preds = [] 

        with torch.no_grad(): 
            for model in self.models: 
                model.eval()
                pred = model(X_tensor).cpu().numpy()
                preds.append(pred)

        preds = np.array(preds)

        means = np.mean(preds, axis=2) 
        mean = np.mean(means, axis=0)
 
        lower_cq = np.mean(preds[:, :, 0], axis=0)
        upper_cq = np.mean(preds[:, :, 1], axis=0)

        lower = lower_cq - self.conformal_width
        upper = upper_cq + self.conformal_width

        if self.scale_data: 
            mean = self.output_scaler.inverse_transform(mean.reshape(-1, 1)).squeeze()
            lower = self.output_scaler.inverse_transform(lower.reshape(-1, 1)).squeeze()
            upper = self.output_scaler.inverse_transform(upper.reshape(-1, 1)).squeeze()

        return mean, lower, upper
    
    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save config (exclude non-serializable or large objects)
        config = {
            k: v for k, v in self.__dict__.items()
            if k not in ["models", "quantiles", "residuals", "conformal_width", "optimizer_cls", "optimizer_kwargs", "scheduler_cls", "scheduler_kwargs"]
            and not callable(v)
            and not isinstance(v, (torch.nn.Module,))
        }

        config["optimizer"] = self.optimizer_cls.__class__.__name__ if self.optimizer_cls is not None else None
        config["scheduler"] = self.optimizer_cls.__class__.__name__ if self.scheduler_cls is not None else None

        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=4)

        # Save model weights
        for i, model in enumerate(self.models):
            torch.save(model.state_dict(), path / f"model_{i}.pt")

        # Save residuals and conformity score
        torch.save({
            "conformal_width": self.conformal_width, 
            "residuals": self.residuals,
            "quantiles": self.quantiles,
        }, path / "extras.pt")

        with open(path / "extras.pkl", 'wb') as f: 
            pickle.dump([self.optimizer_cls, 
                        self.optimizer_kwargs, self.scheduler_cls, self.scheduler_kwargs], f)


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

        # Recreate models
        model.input_dim = input_dim
        activation = get_activation(config["activation_str"])
        model.models = []
        for i in range(config["n_estimators"]):
            m = QuantNN(model.input_dim, config["hidden_sizes"], activation).to(device)
            m.load_state_dict(torch.load(path / f"model_{i}.pt", map_location=device))
            model.models.append(m)

        # Load extras
        extras_path = path / "extras.pt"
        if extras_path.exists():
            extras = torch.load(extras_path, map_location=device, weights_only=False)
            model.conformal_width = extras.get("conformal_width", None)
            model.residuals = extras.get("residuals", None)
            model.quantiles = extras.get("quantiles", None)
        else:
            model.conformal_width = None
            model.residuals = None
            model.quantiles = None

        with open(path / "extras.pkl", 'rb') as f: 
            optimizer_cls, optimizer_kwargs, scheduler_cls, scheduler_kwargs = pickle.load(f)

        model.optimizer_cls = optimizer_cls 
        model.optimizer_kwargs = optimizer_kwargs 
        model.scheduler_cls = scheduler_cls 
        model.scheduler_kwargs = scheduler_kwargs
        return model