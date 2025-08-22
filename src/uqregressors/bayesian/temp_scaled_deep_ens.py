import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, random_split
from uqregressors.utils.data_loader import validate_and_prepare_inputs, validate_X_input
from uqregressors.utils.activations import get_activation
from uqregressors.bayesian.deep_ens import DeepEnsembleRegressor, MLP
import pickle
import scipy.stats as st
from pathlib import Path
import json

class TemperatureScaledDeepEnsembleRegressor(DeepEnsembleRegressor):
    def __init__(
        self,
        *args,
        val_size=0.2,
        lam=0, 
        temp_scaling_epochs=100,
        tau_lr=1e-1,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.val_size = val_size
        self.lam = lam
        self.temp_scaling_epochs = temp_scaling_epochs
        self.tau_lr = tau_lr
        self.tau = None
        self.tau_optimized = False

    def fit(self, X, y):
        X_tensor, y_tensor = validate_and_prepare_inputs(X, y, device=self.device)
        n_val = int(len(X_tensor) * self.val_size)
        n_train = len(X_tensor) - n_val

        train_dataset, val_dataset = random_split(
            TensorDataset(X_tensor, y_tensor), [n_train, n_val],
            generator=torch.Generator().manual_seed(self.random_seed or 42)
        )

        train_X, train_y = zip(*train_dataset)
        val_X, val_y = zip(*val_dataset)

        train_X = torch.stack(train_X)
        train_y = torch.stack(train_y)
        val_X = torch.stack(val_X)
        val_y = torch.stack(val_y)

        # Fit the ensemble on the training data
        super().fit(train_X, train_y)

        # Transform validation set to scaled space: 
        if self.scale_data: 
            val_X = self.input_scaler.transform(val_X)
            val_y = self.output_scaler.transform(val_y)

        # Predict on validation set
        with torch.no_grad():
            val_preds = []
            for model in self.models:
                model.eval()
                val_preds.append(model(val_X.to(self.device)))
            val_preds = torch.stack(val_preds)

        means = val_preds[:, :, 0].mean(dim=0)
        raw_variances = torch.mean(val_preds[:, :, 1] + val_preds[:, :, 0]**2, dim=0) - means**2

        means = means.detach()
        raw_variances = raw_variances.detach()
        val_y = val_y.to(means.device)

        tau = torch.nn.Parameter(torch.tensor(1.0, device=means.device, requires_grad=True))
        optimizer = Adam([tau], lr=self.tau_lr)

        for epoch in range(self.temp_scaling_epochs):
            optimizer.zero_grad()
            scaled_var = raw_variances * tau**2
            precision = 1.0 / scaled_var
            nll = 0.5 * (torch.log(scaled_var) + precision * (val_y.view(-1) - means)**2)
            loss = nll.mean() + self.lam * (tau - 1) ** 2
            loss.backward()
            optimizer.step()

        print(tau)
        print("Gradient:", tau.grad)
        print(f"loss: {loss}")

        self.tau=tau
        self.tau_optimized = True
        return self

    def predict(self, X):
        if not self.fitted:
            raise ValueError("Model not yet fit. Please call fit() before predict().")

        X_tensor = validate_X_input(X, input_dim=self.input_dim, device=self.device, requires_grad=self.requires_grad)
        if self.scale_data:
            X_tensor = self.input_scaler.transform(X_tensor)

        preds = []
        for model in self.models:
            model.eval()
            preds.append(model(X_tensor))
        preds = torch.stack(preds)

        means = preds[:, :, 0]
        variances = preds[:, :, 1]

        mean = means.mean(dim=0)
        variance = torch.mean(variances + means ** 2, dim=0) - mean ** 2
        std = variance.sqrt()

        if self.tau_optimized:
            std = std * self.tau

        std_mult = torch.tensor(st.norm.ppf(1 - self.alpha / 2), device=mean.device)
        lower = mean - std * std_mult
        upper = mean + std * std_mult

        if self.scale_data:
            mean = self.output_scaler.inverse_transform(mean.view(-1, 1)).squeeze()
            lower = self.output_scaler.inverse_transform(lower.view(-1, 1)).squeeze()
            upper = self.output_scaler.inverse_transform(upper.view(-1, 1)).squeeze()

        if not self.requires_grad:
            return mean.detach().cpu().numpy(), lower.detach().cpu().numpy(), upper.detach().cpu().numpy()
        else:
            return mean, lower, upper

    def save(self, path):
        """
        Save the trained ensemble and temperature parameter to disk.
        """
        super().save(path)
        path = Path(path)

        tau_path = path / "temperature.pt"
        torch.save({"tau": self.tau.detach().cpu(), "tau_optimized": self.tau_optimized}, tau_path)

    @classmethod
    def load(cls, path, device="cpu", load_logs=False):
        """
        Load a saved ensemble regressor with temperature scaling from disk.

        Returns:
            (TemperatureScaledDeepEnsembleRegressor): Loaded model instance.
        """
        path = Path(path)

        # Load config 
        with open(path / "config.json", "r") as f: 
            config = json.load(f)
        config["device"] = device 

        config.pop("optimizer", None)
        config.pop("scheduler", None)
        config.pop("input_scaler", None)
        config.pop("output_scaler", None)
        config.pop("tau_optimized", None)

        input_dim = config.pop("input_dim", None)
        fitted = config.pop("fitted", False)
        model = cls(**config)

        # Load temperature info
        tau_path = path / "temperature.pt"
        if not tau_path.exists():
            tau = torch.nn.Parameter(torch.tensor(1.0))
            tau_optimized = False
            print("Temperature file not found, initializing tau to 1")
            #raise FileNotFoundError("Temperature file not found at expected path.")
        else:
            tau_data = torch.load(tau_path, map_location=device)
            tau = tau_data["tau"]
            tau_optimized = tau_data.get("tau_optimized", True)

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
        model.fitted = fitted
        model.tau = torch.nn.Parameter(tau.to(device))
        model.tau_optimized = tau_optimized

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