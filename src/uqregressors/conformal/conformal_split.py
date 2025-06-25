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
    ):
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
        n = len(self.residuals)
        q = int((1 - self.alpha) * (n + 1))
        q = min(q, n-1)
        self.conformal_width = torch.topk(self.residuals, n-q).values[-1].detach().cpu().numpy()

        logger.finish()
        return self

    def predict(self, X): 
        self.model.eval()
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
            mean = self.output_scaler.inverse_transform(mean)
            lower = self.output_scaler.inverse_transform(lower)
            upper = self.output_scaler.inverse_transform(upper)
        return mean, lower, upper 
    
    def save(self, path): 
        raise NotImplementedError("Save method not implemented")
    
    def load(cls, path, device="cpu"): 
        raise NotImplementedError("Load method not implemented")