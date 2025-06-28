import numpy as np
import torch
import torch.nn as nn
import scipy.stats as st
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, RegressorMixin 
from uqregressors.utils.activations import get_activation
from uqregressors.utils.logging import Logger
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import json
import pickle
from sklearn.preprocessing import StandardScaler
import torch.multiprocessing as mp 
import torch.nn.functional as F

mp.set_start_method('spawn', force=True)

class MLP(nn.Module): 
    def __init__(self, input_dim, hidden_sizes, activation):
        super().__init__()
        layers = []
        for h in hidden_sizes:
            layers.append(nn.Linear(input_dim, h))
            layers.append(activation())
            input_dim = h
        output_layer = nn.Linear(hidden_sizes[-1], 2)
        layers.append(output_layer)
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        outputs = self.model(x)
        means = outputs[:, 0]
        unscaled_variances = outputs[:, 1]
        scaled_variance = F.softplus(unscaled_variances) + 1e-6
        scaled_outputs = torch.cat((means.unsqueeze(dim=1), scaled_variance.unsqueeze(dim=1)), dim=1)

        return scaled_outputs
    
class DeepEnsembleRegressor(BaseEstimator, RegressorMixin): 
    def __init__(
        self,
        name = "Deep_Ensemble_Regressor",
        n_estimators=5,
        hidden_sizes=[64, 64],
        alpha=0.1,
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
        n_jobs=1,
        random_seed=None,
        scale_data=True, 
        input_scaler=None,
        scale_output_data=True,
        output_scaler=None, 
        tuning_loggers = [],
    ):
        self.name=name
        self.n_estimators = n_estimators
        self.hidden_sizes = hidden_sizes
        self.alpha = alpha
        self.activation_str = activation_str
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.scheduler_cls = scheduler_cls
        self.scheduler_kwargs = scheduler_kwargs or {}
        self.loss_fn = loss_fn or self.nll_loss
        self.device = device

        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name

        self.n_jobs = n_jobs
        self.random_seed = random_seed
        self.models = []
        self.input_dim = None

        self.scale_data = scale_data
        self.scale_output_data = scale_output_data

        if scale_data: 
            self.input_scaler = input_scaler or StandardScaler()
        if scale_output_data:
            self.output_scaler = output_scaler or StandardScaler()

        self._loggers = []
        self.training_logs = None
        self.tuning_loggers = tuning_loggers
        self.tuning_logs = None

    def nll_loss(self, preds, y): 
        means = preds[:, 0]
        variances = preds[:, 1]
        precision = 1 / variances
        squared_error = (y.view(-1) - means) ** 2
        nll = 0.5 * (torch.log(variances) + precision * squared_error)
        return nll.mean()

    def _train_single_model(self, X_tensor, y_tensor, input_dim, idx): 
        X_tensor = X_tensor.to(self.device)
        gpu = X_tensor.device
        if gpu.type == "cuda" and gpu.index is not None:
            models_on_this_gpu = max(1, self.n_estimators // torch.cuda.device_count())
            memory_fraction = 0.9 / models_on_this_gpu
            torch.cuda.set_per_process_memory_fraction(memory_fraction, device=gpu.index)
        y_tensor = y_tensor.to(self.device)

        if self.random_seed is not None: 
            torch.manual_seed(self.random_seed + idx)
            np.random.seed(self.random_seed + idx)

        activation = get_activation(self.activation_str)
        model = MLP(input_dim, self.hidden_sizes, activation).to(self.device)

        optimizer = self.optimizer_cls(
            model.parameters(), lr=self.learning_rate, **self.optimizer_kwargs
        )
        scheduler = None 
        if self.scheduler_cls: 
            scheduler = self.scheduler_cls(optimizer, **self.scheduler_kwargs)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        logger = Logger(
            use_wandb=self.use_wandb,
            project_name=self.wandb_project,
            run_name=self.wandb_run_name + str(idx) if self.wandb_run_name is not None else None,
            config={"n_estimators": self.n_estimators, "learning_rate": self.learning_rate, "epochs": self.epochs},
            name=f"Estimator-{idx}"
        )
        
        for epoch in range(self.epochs): 
            model.train()
            epoch_loss = 0.0 
            for xb, yb in dataloader: 
                optimizer.zero_grad() 
                preds = model(xb)
                loss = self.loss_fn(preds, yb)
                loss.backward() 
                optimizer.step() 
                epoch_loss += loss.item()
            
            if epoch % (self.epochs / 20) == 0:
                logger.log({"epoch": epoch, "train_loss": epoch_loss})

            if scheduler: 
                scheduler.step()

        logger.finish()
        return model, logger
    
    def fit(self, X, y): 

        if self.scale_data: 
            X = self.input_scaler.fit_transform(X)
        if self.scale_output_data:
            y = self.output_scaler.fit_transform(y.reshape(-1, 1))

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

        input_dim = X.shape[1]
        self.input_dim = input_dim

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._train_single_model)(X_tensor, y_tensor, input_dim, i)
            for i in range(self.n_estimators)
        )

        self.models, self._loggers = zip(*results)

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

        means = preds[:, :, 0]
        variances = preds[:, :, 1]

        mean = means.mean(axis=0)
        variance = np.mean(variances + means ** 2, axis=0) - mean ** 2

        std_mult = st.norm.ppf(1 - self.alpha / 2)

        lower = mean - variance ** 0.5 * std_mult 
        upper = mean + variance ** 0.5 * std_mult 

        if self.scale_output_data: 
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
            if k not in ["models", "optimizer_cls", "optimizer_kwargs", "scheduler_cls", "scheduler_kwargs", 
                         "input_scaler", "output_scaler", "_loggers", "training_logs", "tuning_loggers", "tuning_logs"]
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
        for i, model in enumerate(self.models):
            torch.save(model.state_dict(), path / f"model_{i}.pt")

        for i, logger in enumerate(getattr(self, "_loggers", [])):
            logger.save_to_file(path, idx=i, name="estimator")

        for i, logger in enumerate(getattr(self, "tuning_loggers", [])): 
            logger.save_to_file(path, name="tuning", idx=i)

    @classmethod
    def load(cls, path, device="cpu", load_logs=False):
        path = Path(path)

        # Load config
        with open(path / "config.json", "r") as f:
            config = json.load(f)
        config["device"] = device

        config.pop("optimizer", None)
        config.pop("scheduler", None)
        config.pop("input_scaler", None)
        config.pop("output_scaler", None)

        input_dim = config.pop("input_dim", None)
        model = cls(**config)

        # Recreate models
        model.input_dim = input_dim
        activation = get_activation(config["activation_str"])
        model.models = []
        for i in range(config["n_estimators"]):
            m = MLP(model.input_dim, config["hidden_sizes"], activation).to(device)
            m.load_state_dict(torch.load(path / f"model_{i}.pt", map_location=device))
            model.models.append(m)

        with open(path / "extras.pkl", 'rb') as f: 
            optimizer_cls, optimizer_kwargs, scheduler_cls, scheduler_kwargs, input_scaler, output_scaler = pickle.load(f)

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


