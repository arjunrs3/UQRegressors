import numpy as np 
import torch 
import torch.nn as nn 
from torch.utils.data import TensorDataset, DataLoader 
from sklearn.base import BaseEstimator, RegressorMixin 
from sklearn.preprocessing import StandardScaler
from uqregressors.utils.activations import get_activation 
from uqregressors.utils.logging import Logger 
from joblib import Parallel, delayed 
from sklearn.model_selection import train_test_split 
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
            if dropout is not None: 
                layers.append(nn.Dropout(dropout))
            input_dim = h 
        layers.append(nn.Linear(hidden_sizes[-1], 2))
        self.model = nn.Sequential(*layers)

    def forward(self, x): 
        return self.model(x)

class ConformalQuantileRegressor(BaseEstimator, RegressorMixin): 
    def __init__(
            self, 
            name="Conformal_Quantile_Regressor",
            hidden_sizes = [64, 64],
            cal_size = 0.2, 
            dropout = None, 
            alpha = 0.1, 
            tau_lo = None, 
            tau_hi = None,
            activation_str="ReLU",
            learning_rate=1e-3,
            epochs=200, 
            batch_size=32,
            optimizer_cls = torch.optim.Adam, 
            optimizer_kwargs=None, 
            scheduler_cls=None, 
            scheduler_kwargs=None, 
            loss_fn=None, 
            device="cpu", 
            use_wandb=False, 
            wandb_project=None,
            wandb_run_name=None,
            scale_data=True, 
            input_scaler=None,
            output_scaler=None, 
            random_seed=None,
            tuning_loggers = []
    ):
        self.name = name
        self.hidden_sizes = hidden_sizes 
        self.cal_size = cal_size 
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

        self.random_seed = random_seed

        self.quantiles = torch.tensor([self.tau_lo, self.tau_hi], device=self.device)

        self.residuals = [] 
        self.conformal_width = None 
        self.input_dim = None

        self.scale_data = scale_data 
        self.input_scaler = input_scaler or StandardScaler() 
        self.output_scaler = output_scaler or StandardScaler()

        self._loggers = []
        self.training_logs = None
        self.tuning_loggers = tuning_loggers
        self.tuning_logs = None

    def quantile_loss(self, preds, y): 
        error = y.view(-1, 1) - preds 
        return torch.mean(torch.max(self.quantiles * error, (self.quantiles - 1) * error))

    def fit(self, X, y): 
        if self.random_seed is not None: 
            torch.manual_seed(self.random_seed)
            np.random.seed(self.random_seed)

        if self.scale_data: 
            X = self.input_scaler.fit_transform(X)
            y = self.output_scaler.fit_transform(y.reshape(-1, 1))

        X_train, X_cal, y_train, y_cal = train_test_split(X, y, test_size=self.cal_size, random_state=self.random_seed)
        
        X_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_train, dtype=torch.float32).to(self.device)

        input_dim = X.shape[1]
        self.input_dim = input_dim 

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

        activation = get_activation(self.activation_str)

        self.model = MLP(self.input_dim, self.hidden_sizes, self.dropout, activation)
        self.model.to(self.device)

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

        X_cal = torch.tensor(X_cal, dtype=torch.float32).to(self.device)
        y_cal = torch.tensor(y_cal, dtype=torch.float32).to(self.device)

        oof_preds = self.model(X_cal)
        loss_matrix = (oof_preds - y_cal) * torch.tensor([1, -1], device=self.device)
        self.residuals = torch.max(loss_matrix, dim=1).values

        logger.finish()
        self._loggers.append(logger)
        return self

    def predict(self, X): 
        self.model.eval()

        n = len(self.residuals)
        q = int((1 - self.alpha) * (n + 1))
        q = min(q, n-1)
        res_quantile = n-q

        self.conformal_width = torch.topk(self.residuals, res_quantile).values[-1].detach().cpu().numpy()

        if self.random_seed is not None: 
            torch.manual_seed(self.random_seed)
            np.random.seed(self.random_seed)
        
        if self.scale_data: 
            X = self.input_scaler.transform(X)

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        preds = self.model(X_tensor)
        lower_cq = preds[:, 0].unsqueeze(dim=1).detach().cpu().numpy()
        upper_cq = preds[:, 1].unsqueeze(dim=1).detach().cpu().numpy()
        lower = lower_cq - self.conformal_width 
        upper = upper_cq + self.conformal_width 
        mean = (lower + upper) / 2 

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

        config = {
            k: v for k, v in self.__dict__.items()
            if k not in ["model", "residuals", "conformal_width", "optimizer_cls", "optimizer_kwargs", "scheduler_cls", "scheduler_kwargs", "input_scaler", "output_scaler", "quantiles", 
                         "_loggers", "training_logs", "tuning_loggers", "tuning_logs"]
            and not callable(v)
            and not isinstance(v, (torch.nn.Module,))
        }
    

        config["optimizer"] = self.optimizer_cls.__class__.__name__ if self.optimizer_cls is not None else None
        config["scheduler"] = self.scheduler_cls.__class__.__name__ if self.scheduler_cls is not None else None
        config["input_scaler"] = self.input_scaler.__class__.__name__ if self.input_scaler is not None else None 
        config["output_scaler"] = self.output_scaler.__class__.__name__ if self.output_scaler is not None else None

        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=4)

        with open(path / "extras.pkl", 'wb') as f: 
            pickle.dump([self.optimizer_cls, 
                         self.optimizer_kwargs, self.scheduler_cls, self.scheduler_kwargs, self.input_scaler, self.output_scaler], f)

        # Save model weights
        torch.save(self.model.state_dict(), path / f"model.pt")

        torch.save({
            "conformal_width": self.conformal_width, 
            "residuals": self.residuals,
            "quantiles": self.quantiles
        }, path / "extras.pt")

        for i, logger in enumerate(getattr(self, "_loggers", [])):
            logger.save_to_file(path, idx=i, name="estimator")

        for i, logger in enumerate(getattr(self, "tuning_loggers", [])): 
            logger.save_to_file(path, name="tuning", idx=i)

    @classmethod
    def load(cls, path, device="cpu", load_logs=False): 
        path = Path(path)
        with open(path / "config.json", "r") as f:
            config = json.load(f)
        config["device"] = device

        config.pop("optimizer", None)
        config.pop("scheduler", None)
        config.pop("input_scaler", None)
        config.pop("output_scaler", None)
        
        input_dim = config.pop("input_dim", None)
        model = cls(**config)

        with open(path / "extras.pkl", 'rb') as f: 
            optimizer_cls, optimizer_kwargs, scheduler_cls, scheduler_kwargs, input_scaler, output_scaler = pickle.load(f)

        # Recreate models
        model.input_dim = input_dim
        activation = get_activation(config["activation_str"])

        model.model = MLP(model.input_dim, config["hidden_sizes"], model.dropout, activation).to(device)
        model.model.load_state_dict(torch.load(path / f"model.pt", map_location=device))

        extras_path = path / "extras.pt"
        if extras_path.exists():
            extras = torch.load(extras_path, map_location=device, weights_only=False)
            model.residuals = extras.get("residuals", None)
            model.conformal_width = extras.get("conformal_width", None)
            model.quantiles = extras.get("quantiles", None)
        else:
            model.residuals = None
            model.conformal_width = None
            model.quantiles = None

        model.optimizer_cls = optimizer_cls 
        model.optimizer_kwargs = optimizer_kwargs 
        model.scheduler_cls = scheduler_cls 
        model.scheduler_kwargs = scheduler_kwargs
        model.input_scaler = input_scaler 
        model.output_scaler = output_scaler
        
        if load_logs: 
            logs_path = path / "logs"
            training_logs = [] 
            tuning_logs = []
            if logs_path.exists() and logs_path.is_dir(): 
                estimator_log_files = sorted(logs_path.glob("estimator_*.log"))
                for log_file in estimator_log_files:
                    with open(log_file, "r", encoding="utf-8") as f:
                        training_logs.append(f.read())

                tuning_log_files = sorted(logs_path.glob("tuning_*.log"))
                for log_file in tuning_log_files: 
                    with open(log_file, "r", encoding="utf-8") as f: 
                        tuning_logs.append(f.read())

            model.training_logs = training_logs
            model.tuning_logs = tuning_logs

        return model