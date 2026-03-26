import math
import torch 
import gpytorch 
import pyro 
from pyro.infer.mcmc import NUTS, MCMC, HMC 
from uqregressors.utils.data_loader import validate_and_prepare_inputs, validate_X_input
from uqregressors.utils.torch_sklearn_utils import TorchStandardScaler
from uqregressors.utils.logging import Logger 
from pathlib import Path 
import json 
import pickle 
import numpy as np 
import scipy.stats as st
import copy
from pyro.infer.autoguide import init_to_median
counter = 0

class ExactGP(gpytorch.models.ExactGP): 
    def __init__(self, kernel, mean_module, train_x, train_y, likelihood): 
        super(ExactGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = mean_module 
        self.covar_module = kernel 

    def forward(self, x): 
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
def gp_log_marginal_likelihood(model, x, y): 
    model.eval()
    model.likelihood.eval()
    global counter
    counter +=1
    #with gpytorch.settings.fast_computations(False, False, False): 
    output = model(x)
    dist = model.likelihood(output)
    return dist.log_prob(y)

def make_pyro_model(train_x, train_y, priors): 
    def pyro_model(): 
        log_lengthscale = pyro.sample("log_lengthscale", priors["log_lengthscale"])
        log_outputscale = pyro.sample("log_outputscale", priors["log_outputscale"])
        log_noise = pyro.sample("log_noise", priors["log_noise"])
        mean = pyro.sample("mean", priors["mean"])

        lengthscale = torch.exp(log_lengthscale)
        outputscale = torch.exp(log_outputscale)
        noise = torch.exp(log_noise)

        kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1], has_lengthscale=True))
        kernel.base_kernel.lengthscale = lengthscale 
        kernel.outputscale = outputscale 

        likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.Positive()) 
        likelihood.noise = noise 

        mean_module = gpytorch.means.ConstantMean() 
        mean_module.constant.data = mean 

        model = ExactGP(kernel, mean_module, train_x, train_y, likelihood)

        log_prob = gp_log_marginal_likelihood(model, train_x, train_y)
        pyro.factor("gp_marginal_likelihood", log_prob)

    return pyro_model

class FBGP: 
    def __init__(self, 
                 name="Fully_Bayesian_GP_Regressor", 
                 kernel=gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()), 
                 likelihood=gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.Positive()), 
                 mean_module=gpytorch.means.ConstantMean(),
                 priors=None, 
                 alpha=0.1, 
                 requires_grad=False, 
                 device="cpu", 
                 num_samples=1000, 
                 warmup_steps=500,
                 scale_data=False, 
                 input_scaler = None, 
                 output_scaler = None, 
                 random_seed=None, 
                 tuning_loggers=[], 
                 logging_frequency=20
                 ): 
        self.name = name 
        self.kernel = kernel 
        self.likelihood = likelihood 
        self.mean_module = mean_module
        self.alpha = alpha 
        self.requires_grad = requires_grad 
        self.device = device 
        self.model = None 
        self.num_samples = num_samples 
        self.warmup_steps = warmup_steps
        self.random_seed = random_seed 
        self.input_dim = None 
        self._loggers = []
        self.logging_frequency = logging_frequency 
        self.tuning_loggers = tuning_loggers 
        self.tuning_logs = None 

        self.scale_data = scale_data 
        if self.scale_data: 
            self.input_scaler = input_scaler or TorchStandardScaler() 
            self.output_scaler = output_scaler or TorchStandardScaler()
        else:
            self.input_scaler = None 
            self.output_scaler = None 

        self.train_X = None 
        self.train_y = None 
        self.fitted = False 
        self.priors = priors

        self.mcmc_run = None 

    def set_alpha(self, alpha): 
        self.alpha = alpha 

    def fit(self, X, y):
        X_tensor, y_tensor = validate_and_prepare_inputs(X, y, device=self.device, requires_grad=self.requires_grad)
        self.input_dim = X_tensor.shape[1]
        
        if self.scale_data:
            if self.requires_grad:
                # Use clone to avoid in-place operations that break gradient flow
                X_tensor_scaled = self.input_scaler.fit_transform(X_tensor.detach()).clone()
                X_tensor_scaled.requires_grad_(True)
                y_tensor_scaled = self.output_scaler.fit_transform(y_tensor.detach()).clone()
                y_tensor_scaled.requires_grad_(True)
                X_tensor = X_tensor_scaled
                y_tensor = y_tensor_scaled
            else:
                X_tensor = self.input_scaler.fit_transform(X_tensor)
                y_tensor = self.output_scaler.fit_transform(y_tensor)

        y_tensor = y_tensor.view(-1)

        self.train_X = X_tensor 
        self.train_y = y_tensor

        if self.random_seed is not None: 
            torch.manual_seed(self.random_seed)

        pyro_model = make_pyro_model(X_tensor, y_tensor, self.priors)
        
        nuts_kernel = NUTS(pyro_model, max_tree_depth=6, init_strategy=init_to_median)
        mcmc_run = MCMC(nuts_kernel, num_samples=self.num_samples, warmup_steps=self.warmup_steps)
        mcmc_run.run()

        self.mcmc_run = mcmc_run 
        Ls = [] 
        alphas = [] 
        lengthscales = [] 
        outputscales = [] 
        means = [] 
        noises = [] 

        samples = self.mcmc_run.get_samples() 
        S = len(samples["log_lengthscale"])

        for i in range(S): 
            lengthscale = torch.exp(samples["log_lengthscale"][i])
            outputscale = torch.exp(samples["log_outputscale"][i])
            noise = torch.exp(samples["log_noise"][i])
            mean = samples["mean"][i]

            kernel = copy.deepcopy(self.kernel)
            likelihood = copy.deepcopy(self.likelihood)
            mean_module = copy.deepcopy(self.mean_module)

            model = ExactGP(kernel, mean_module, X_tensor, y_tensor, likelihood)
            

        self.fitted=True
        
    def predict(self, X): 
        if not self.fitted: 
            raise ValueError("Model not yet fit. Please call fit() before predict().")
        
        X_tensor = validate_X_input(X, device=self.device, requires_grad=True)
        if self.scale_data:
            if self.requires_grad:
                # Use clone to avoid in-place operations that break gradient flow
                X_tensor_scaled = self.input_scaler.transform(X_tensor.detach()).clone()
                X_tensor_scaled.requires_grad_(True)
                X_tensor = X_tensor_scaled
            else:
                X_tensor = self.input_scaler.transform(X_tensor)

        samples = self.mcmc_run.get_samples()

        preds_mean = [] 
        preds_var = []
        
        for i in range(len(samples["log_lengthscale"])): 
            lengthscale = torch.exp(samples["log_lengthscale"][i])
            outputscale = torch.exp(samples["log_outputscale"][i])
            noise = torch.exp(samples["log_noise"][i])
            mean = samples["mean"][i]

            kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=self.train_X.shape[1], has_lengthscale=True))
            kernel.base_kernel.lengthscale = lengthscale 
            kernel.outputscale = outputscale 

            likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.Positive()) 
            likelihood.noise = noise 

            mean_module = gpytorch.means.ConstantMean() 
            mean_module.constant.data = mean 

            model = ExactGP(kernel, mean_module, self.train_X, self.train_y, likelihood)

            model.eval()
            model.likelihood.eval() 

            pred = model.likelihood(model(X_tensor))
            preds_mean.append(pred.mean)
            preds_var.append(pred.variance)

        means = torch.stack(preds_mean) 
        vars = torch.stack(preds_var)

        mean = means.mean(dim=0)
        var = vars.mean(dim=0) + means.var(dim=0)
        
        low_std = var ** 0.5 
        up_std = var ** 0.5

        z_score = st.norm.ppf(1 - self.alpha / 2)
        lower = mean - z_score * low_std
        upper = mean + z_score * up_std

        if self.scale_data: 
            mean = self.output_scaler.inverse_transform(mean.view(-1, 1)).squeeze()
            lower = self.output_scaler.inverse_transform(lower.view(-1, 1)).squeeze()
            upper = self.output_scaler.inverse_transform(upper.view(-1, 1)).squeeze()

        if not self.requires_grad: 
            return mean.detach().cpu().numpy(), lower.detach().cpu().numpy(), upper.detach().cpu().numpy()

        else: 
            return mean, lower, upper