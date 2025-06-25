import numpy as np 
import torch
import torch.nn as nn 
from torch.utils.data import TensorDataset, DataLoader 
from sklearn.base import BaseEstimator, RegressorMixin 
from uqregressors.utils.activations import get_activation 
from uqregressors.utils.logging import Logger 
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from pathlib import Path
import json 
import pickle

def train_test_split(X, cal_size, seed=None):
    if seed is not None: 
        torch.manual_seed(seed)

    n = len(X)
    n_cal = int(np.ceil(cal_size * n))
    all_idx = np.arange(n)
    cal_idx = np.random.randint(n, size=n_cal)
    mask = np.ones(n, dtype=bool)
    mask[cal_idx] = False 
    train_idx = all_idx[mask] 
    return train_idx, cal_idx

class MLP(nn.Module): 
    def __init__(self, input_dim, hidden_sizes, dropout, activation): 
        super().__init__()
        layers = []
        for h in hidden_sizes: 
            layers.append(nn.Linear(input_dim, h))
            layers.append(activation())
            if dropout is not None: 
                layers.append(nn.Dropout(dropout))
            input_dim=h 
        output_layer = nn.Linear(hidden_sizes[-1], 1)
        layers.append(output_layer)
        self.model = nn.Sequential(*layers)

    def forward(self, x): 
        return self.model(x)
    
class ConformalEnsRegressor(BaseEstimator, RegressorMixin): 
    def __init__(self, 
                 n_estimators=5, 
                 hidden_sizes=[64, 64], 
                 alpha=0.1, 
                 dropout=None,
                 pred_with_dropout=False,
                 activation_str="ReLU",
                 cal_size = 0.2, 
                 gamma = 0,
                 learning_rate=1e-3,
                 epochs=200,
                 batch_size=32,
                 optimizer_cls=torch.optim.Adam,
                 optimizer_kwargs=None,
                 scheduler_cls=None,
                 scheduler_kwargs=None,
                 loss_fn=nn.functional.mse_loss,
                 device="cpu", 
                 use_wandb=False, 
                 wandb_project=None, 
                 wandb_run_name=None, 
                 n_jobs=1, 
                 random_seed=None, 
                 scale_data=True, 
                 input_scaler=None,
                 output_scaler=None
    ): 
        self.n_estimators = n_estimators
        self.hidden_sizes = hidden_sizes
        self.alpha = alpha
        self.dropout = dropout
        self.pred_with_dropout = pred_with_dropout
        self.activation_str = activation_str
        self.cal_size = cal_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.scheduler_cls = scheduler_cls
        self.scheduler_kwargs = scheduler_kwargs or {}
        self.loss_fn = loss_fn
        self.device = device

        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name

        self.n_jobs = n_jobs
        self.random_seed = random_seed

        self.scale_data = scale_data 
        self.input_scaler = input_scaler or StandardScaler() 
        self.output_scaler = output_scaler or StandardScaler()

        self.input_dim = None
        self.conformity_score = None
        self.models = []
        self.residuals = []

    def _train_single_model(self, X_tensor, y_tensor, input_dim, train_idx, cal_idx, model_idx): 
        if self.random_seed is not None: 
            torch.manual_seed(self.random_seed + model_idx)
            np.random.seed(self.random_seed + model_idx)

        activation = get_activation(self.activation_str)
        model = MLP(input_dim, self.hidden_sizes, self.dropout, activation).to(self.device)

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

        if self.pred_with_dropout: 
            model.train()
        else: 
            model.eval()

        test_X = X_tensor[cal_idx]
        cal_preds = model(test_X)

        logger.finish()
        return model, cal_preds
    
    def fit(self, X, y): 
        if self.scale_data: 
            X = self.input_scaler.fit_transform(X)
            y = self.output_scaler.fit_transform(y.reshape(-1, 1))
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(self.device)

        input_dim = X.shape[1]
        self.input_dim = input_dim

        train_idx, cal_idx = train_test_split(X_tensor, 0.2, self.random_seed)
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._train_single_model)(X_tensor, y_tensor, input_dim, train_idx, cal_idx, i)
            for i in range(self.n_estimators)
        )

        self.models = [result[0] for result in results]
        cal_preds = torch.stack([result[1] for result in results]).squeeze()
        mean_cal_preds = torch.mean(cal_preds, dim=0).squeeze()
        var_cal_preds = torch.var(cal_preds, dim=0).squeeze()
        std_cal_preds = var_cal_preds ** 0.5 
        self.residuals = torch.abs(mean_cal_preds - y_tensor[cal_idx].squeeze())
    
        conformity_scores = self.residuals / (std_cal_preds + self.gamma)

        n = len(self.residuals)
        q = int((1 - self.alpha) * (n+1)) 
        q = min(q, n-1) 
        self.conformity_score = torch.topk(conformity_scores, n-q).values[-1].detach().cpu().numpy()
        return self 
    
    def predict(self, X): 
        if self.scale_data: 
            X = self.input_scaler.transform(X)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        preds = []

        with torch.no_grad(): 
            for model in self.models: 
                if self.pred_with_dropout: 
                    model.train()
                else: 
                    model.eval()
                pred = model(X_tensor).cpu().numpy() 
                preds.append(pred)

        preds = np.array(preds)[:, :, 0]
        mean = np.mean(preds, axis=0)
        variances = np.var(preds, axis=0, ddof=1)
        stds = variances ** 0.5
        conformal_widths = self.conformity_score * (stds + self.gamma) 
        lower = mean - conformal_widths 
        upper = mean + conformal_widths 

        if self.scale_data: 
            mean = self.output_scaler.inverse_transform(mean.reshape(-1, 1)).squeeze()
            lower = self.output_scaler.inverse_transform(lower.reshape(-1, 1)).squeeze()
            upper = self.output_scaler.inverse_transform(upper.reshape(-1, 1)).squeeze()
        return mean, lower, upper 
    
    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save config
        config = self.__dict__.copy()
        # Remove non-serializable entries
        config.pop("models", None)
        config.pop("residuals", None)
        config.pop("conformity_score", None)
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=4)

        # Save each model
        for i, model in enumerate(self.models):
            torch.save(model.state_dict(), path / f"model_{i}.pt")

        # Save residuals and conformity score
        torch.save({
            "residuals": self.residuals,
            "conformity_score": self.conformity_score
        }, path / "extras.pt")

    @classmethod
    def load(cls, path):
        path = Path(path)

        with open(path / "config.json", "r") as f:
            config = json.load(f)

        model = cls(**config)

        # Recreate models
        model.models = []
        activation = get_activation(model.activation_str)
        input_dim = config["hidden_sizes"][0]  # or store explicitly in config

        for i in range(config["n_estimators"]):
            m = MLP(input_dim, model.hidden_sizes, activation).to(model.device)
            m.load_state_dict(torch.load(path / f"model_{i}.pt", map_location=model.device))
            model.models.append(m)

        # Load extras
        extras = torch.load(path / "extras.pt", map_location=model.device)
        model.residuals = extras["residuals"]
        model.conformity_score = extras["conformity_score"]

        return model
    
    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save config (exclude non-serializable or large objects)
        config = {
            k: v for k, v in self.__dict__.items()
            if k not in ["models", "residuals", "conformity_score", "optimizer_cls", "optimizer_kwargs", "scheduler_cls", "scheduler_kwargs"]
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
            "residuals": self.residuals.cpu(),
            "conformity_score": self.conformity_score
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

        with open(path / "extras.pkl", 'rb') as f: 
            optimizer_cls, optimizer_kwargs, scheduler_cls, scheduler_kwargs = pickle.load(f)
        
        # Recreate models
        model.input_dim = input_dim
        activation = get_activation(config["activation_str"])
        model.models = []
        for i in range(config["n_estimators"]):
            m = MLP(model.input_dim, config["hidden_sizes"], activation).to(device)
            m.load_state_dict(torch.load(path / f"model_{i}.pt", map_location=device))
            model.models.append(m)

        # Load extras
        extras_path = path / "extras.pt"
        if extras_path.exists():
            extras = torch.load(extras_path, map_location=device, weights_only=False)
            model.residuals = extras.get("residuals", None)
            model.conformity_score = extras.get("conformity_score", None)
        else:
            model.residuals = None
            model.conformity_score = None

        model.optimizer_cls = optimizer_cls 
        model.optimizer_kwargs = optimizer_kwargs 
        model.scheduler_cls = scheduler_cls 
        model.scheduler_kwargs = scheduler_kwargs

        return model