import gpytorch
import torch
from uqregressors.utils.logging import Logger
import scipy.stats as st
from pathlib import Path
import json
import pickle

class ExactGP(gpytorch.models.ExactGP): 
    def __init__(self, kernel, train_x, train_y, likelihood):
        super(ExactGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel
    
    def forward(self, x): 
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class GPRegressorTorch: 
    def __init__(self, 
                 kernel=gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()), 
                 likelihood=gpytorch.likelihoods.GaussianLikelihood(), 
                 alpha=0.1,
                 learning_rate=1e-3,
                 epochs=200, 
                 optimizer_cls=torch.optim.Adam,
                 optimizer_kwargs=None,
                 scheduler_cls=None,
                 scheduler_kwargs=None,
                 loss_fn=None, 
                 device="cpu", 
                 use_wandb=False,
                 wandb_project=None,
                 wandb_run_name=None, 
                 random_seed=None
            ):
        
        self.kernel = kernel 
        self.likelihood = likelihood
        self.alpha = alpha 
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.optimizer_cls = optimizer_cls 
        self.optimizer_kwargs = optimizer_kwargs or {} 
        self.scheduler_cls = scheduler_cls
        self.scheduler_kwargs = scheduler_kwargs or {}
        self.loss_fn = loss_fn
        self.device = device 
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project 
        self.wandb_run_name = wandb_run_name
        self.model = None
        self.random_seed = random_seed
        self.train_X = None 
        self.train_y = None

    def fit(self, X, y): 
        if self.random_seed is not None: 
            torch.manual_seed(self.random_seed)

        config = {
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
        }

        logger = Logger(
            use_wandb=self.use_wandb,
            project_name=self.wandb_project,
            run_name=self.wandb_run_name,
            config=config,
        )

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)

        self.train_X = X_tensor
        self.train_y = y_tensor

        model = ExactGP(self.kernel, X_tensor, y_tensor, self.likelihood)
        self.model = model.to(self.device)

        self.model.train()
        self.likelihood.train()

        if self.loss_fn == None: 
            self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, model)
            self.loss_fn = self.mll_loss

        optimizer = self.optimizer_cls(
            model.parameters(), lr=self.learning_rate, **self.optimizer_kwargs
        )

        scheduler = None
        if self.scheduler_cls is not None:
            scheduler = self.scheduler_cls(optimizer, **self.scheduler_kwargs)

        for epoch in range(self.epochs): 
            optimizer.zero_grad()
            preds = model(X_tensor)
            loss = self.loss_fn(preds, y_tensor)
            loss.backward()
            optimizer.step() 

            if scheduler is not None:
                scheduler.step()
            if epoch % (self.epochs / 20) == 0:
                logger.log({"epoch": epoch, "train_loss": loss})

    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.model.eval()
        self.likelihood.eval() 

        with torch.no_grad(), gpytorch.settings.fast_pred_var(): 
            preds = self.likelihood(self.model(X))

        with torch.no_grad(): 
            mean = preds.mean.numpy() 
            lower_2std, upper_2std = preds.confidence_region() 
            low_std, up_std = (mean - lower_2std.numpy()) / 2, (upper_2std.numpy() - mean) / 2 

        z_score = st.norm.ppf(1 - self.alpha / 2)
        lower = mean - z_score * low_std
        upper = mean + z_score * up_std
        return mean, lower, upper
    
    def mll_loss(self, preds, y): 
        return -torch.sum(self.mll(preds, y))
    

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save config (exclude non-serializable or large objects)
        config = {
            k: v for k, v in self.__dict__.items()
            if k not in ["model", "kernel", "likelihood", "optimizer_cls", "optimizer_kwargs", "scheduler_cls", "scheduler_kwargs"]
            and not callable(v)
            and not isinstance(v, (torch.nn.Module, torch.Tensor))
        }
        config["optimizer"] = self.optimizer_cls.__class__.__name__ if self.optimizer_cls is not None else None
        config["scheduler"] = self.optimizer_cls.__class__.__name__ if self.scheduler_cls is not None else None

        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=4)

        with open(path / "extras.pkl", 'wb') as f: 
            pickle.dump([self.kernel, self.likelihood, self.optimizer_cls, 
                         self.optimizer_kwargs, self.scheduler_cls, self.scheduler_kwargs], f)

        # Save model weights
        torch.save(self.model.state_dict(), path / f"model.pt")
        torch.save([self.train_X, self.train_y], path / f"train.pt")

    @classmethod
    def load(cls, path, device="cpu"):
        path = Path(path)

        # Load config
        with open(path / "config.json", "r") as f:
            config = json.load(f)
        config["device"] = device

        config.pop("optimizer", None)
        config.pop("scheduler", None)
        model = cls(**config)

        with open(path / "extras.pkl", 'rb') as f: 
            kernel, likelihood, optimizer_cls, optimizer_kwargs, scheduler_cls, scheduler_kwargs = pickle.load(f)

        train_X, train_y = torch.load(path / f"train.pt")
        model.model = ExactGP(kernel, train_X, train_y, likelihood)
        model.model.load_state_dict(torch.load(path / f"model.pt", map_location=device))

        model.optimizer_cls = optimizer_cls 
        model.optimizer_kwargs = optimizer_kwargs 
        model.scheduler_cls = scheduler_cls 
        model.scheduler_kwargs = scheduler_kwargs

        return model