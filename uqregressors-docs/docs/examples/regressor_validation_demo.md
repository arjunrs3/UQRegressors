# Regressor Validation Demo

This notebook demonstrates how to train, tune, validate, and compare uncertainty-aware regression models using the UQRegressors library. It is designed for users new to the code base and provides detailed documentation for each step.

**Covered methods:**
- MC Dropout
- Deep Ensemble
- Split Conformal Quantile Regression (CQR)

---

## 1. Import Required Libraries and Set Up Environment

In this section, we import all necessary Python libraries and modules, and set up the computational environment (CPU/GPU, random seeds). Each import and setup step is explained for clarity.


```python
# Import core Python libraries for data handling and computation
import numpy as np  # Numerical operations
import pandas as pd  # Data manipulation
import torch  # PyTorch for deep learning
import matplotlib.pyplot as plt  # Plotting

# Import UQRegressors components for uncertainty-aware regression
from uqregressors.bayesian.dropout import MCDropoutRegressor
from uqregressors.bayesian.deep_ens import DeepEnsembleRegressor
from uqregressors.conformal.cqr import ConformalQuantileRegressor
from uqregressors.metrics.metrics import compute_all_metrics
from uqregressors.utils.file_manager import FileManager
from uqregressors.utils.torch_sklearn_utils import train_test_split

# Set up device (GPU if available, else CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
```

## 2. Utility Functions for Regressor Validation

This section defines utility functions to streamline the process of training, tuning, and evaluating regressors. Each function is documented to help new users understand its purpose and usage.


```python
# Utility functions for training, tuning, and evaluating regressors
from sklearn.metrics import mean_squared_error
from pathlib import Path
from copy import deepcopy
import optuna

# Set Optuna logging to warning for cleaner output
optuna.logging.set_verbosity(optuna.logging.WARNING)

def test_regressor(model, X, y, dataset_name, test_size, seed=None, 
                   tuning_epochs=None, param_space=None, scoring_fn=None, greater=None,
                   initial_params=None, n_trials=None, n_splits=1):
    """
    Train and evaluate a regressor on a dataset, with optional hyperparameter tuning.
    Returns a dictionary of evaluation metrics.
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Split data into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    # Hyperparameter tuning (if specified)
    if tuning_epochs is not None and param_space is not None:
        epochs_copy = model.epochs
        model.epochs = deepcopy(tuning_epochs)
        from uqregressors.tuning.tuning import tune_hyperparams
        opt_model, opt_score, study = tune_hyperparams(
            regressor=model,
            param_space=param_space,
            X=X_train,
            y=y_train,
            score_fn=scoring_fn,
            greater_is_better=greater,
            initial_params=initial_params,
            n_trials=n_trials,
            n_splits=n_splits,
            verbose=False
        )
        model = opt_model
        model.epochs = epochs_copy

    # Train the model
    model.fit(X_train, y_train)
    mean, lower, upper = model.predict(X_test)

    # Compute evaluation metrics
    metrics = compute_all_metrics(mean, lower, upper, y_test, model.alpha)
    metrics["mse"] = mean_squared_error(y_test, mean)
    return metrics

def run_regressor_test(model, datasets, seed, filename, test_size,
                       BASE_SAVE_DIR=Path.home()/".uqregressors",
                       tuning_epochs=None, param_space=None, scoring_fn=None, greater=None,
                       initial_params=None, n_trials=None, n_splits=1):
    """
    Run a regressor on multiple datasets, save models and metrics, and return save paths.
    """
    saved_results = []
    for name, (X, y) in datasets.items():
        print(f"\nRunning on dataset: {name}")
        metrics = test_regressor(model, X, y, name, seed=seed, test_size=test_size,
                                 tuning_epochs=tuning_epochs, param_space=param_space,
                                 scoring_fn=scoring_fn, greater=greater, initial_params=initial_params,
                                 n_trials=n_trials, n_splits=n_splits)
        print(metrics)
        fm = FileManager(BASE_SAVE_DIR)
        save_path = fm.save_model(model, name=name + "_" + filename, metrics=metrics)
        saved_results.append((model.__class__, name, save_path))
    return saved_results

def print_results(paths):
    """
    Load and print metrics for each saved model.
    """
    fm = FileManager()
    for cls, dataset_name, path in paths:
        results = fm.load_model(cls, path=path, load_logs=False)
        print(f"Results for {dataset_name}")
        print(results["metrics"])
```

## 3. Dataset Preparation and Loading

This section shows how to download, load, and preprocess datasets for regression experiments. We use a synthetic dataset for demonstration and show how to prepare real datasets (e.g., UCI protein) as well.


```python
# Example: Generate a synthetic regression dataset
# This is useful for quick testing and demonstration

def generate_synthetic_data(n_samples=200, n_features=5):
    """Generate a simple synthetic regression dataset."""
    X = np.random.randn(n_samples, n_features)
    y = np.sin(X[:, 0]) + X[:, 1] ** 2 + np.random.normal(0, 0.1, size=n_samples)
    return X, y

# Prepare datasets dictionary for use in validation functions
datasets = {
    "synthetic": generate_synthetic_data()
}

# Example: How to load a real dataset (UCI protein)
# Uncomment and run the following code to download and prepare the UCI protein dataset
# import pandas as pd
# from io import StringIO
# import requests
# uci_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv"
# response = requests.get(uci_url)
# df = pd.read_csv(StringIO(response.text))
# X_protein = df.drop(columns=["RMSD"]).values
# y_protein = df["RMSD"].values
# datasets["protein"] = (X_protein, y_protein)

print("Datasets prepared:", list(datasets.keys()))
```

## 4. MC Dropout Regressor: Training, Tuning, and Validation

This section demonstrates how to configure, tune, train, and validate an MC Dropout regressor. Hyperparameter tuning is performed using Optuna, and results are evaluated using standard metrics.


```python
# Configure and validate an MC Dropout regressor
from uqregressors.tuning.tuning import log_likelihood

# Define the MC Dropout regressor with basic settings
mc_dropout = MCDropoutRegressor(
    hidden_sizes=[50],
    dropout=0.05,
    use_paper_weight_decay=True,
    prior_length_scale=1e-2,
    alpha=0.05,
    n_samples=100,
    epochs=40,  # Fewer epochs for demonstration
    batch_size=32,
    learning_rate=1e-3,
    device=device,
    use_wandb=False
)

# Define hyperparameter search space for tau (aleatoric uncertainty)
param_space = {
    "tau": lambda trial: trial.suggest_float("tau", 1e-2, 1e2, log=True)
}

# Run validation on all datasets (synthetic by default)
mc_save_paths = run_regressor_test(
    mc_dropout, datasets, seed=42, filename="dropout_demo", test_size=0.2,
    tuning_epochs=20, param_space=param_space, scoring_fn=log_likelihood, greater=True, n_trials=10
)

# Print results for each dataset
print_results(mc_save_paths)
```

## 5. Deep Ensemble Regressor: Training and Validation

This section demonstrates how to set up, train, and validate a Deep Ensemble regressor. Any dataset-specific settings (such as learning rate) are explained in comments.


```python
# Configure and validate a Deep Ensemble regressor
# Deep Ensembles are robust and often perform well on a variety of datasets

deep_ens = DeepEnsembleRegressor(
    n_estimators=3,  # Fewer estimators for demonstration
    hidden_sizes=[50],
    n_jobs=1,  # Set to >1 for parallel training if supported
    alpha=0.05,
    batch_size=64,
    learning_rate=1e-2,  # Adjust as needed for your dataset
    epochs=20,
    device=device,
    scale_data=True,
    use_wandb=False
)

deep_ens_save_paths = run_regressor_test(
    deep_ens, datasets, seed=42, filename="deep_ens_demo", test_size=0.2
)

print_results(deep_ens_save_paths)
```

## 6. Split Conformal Quantile Regression: Training, Tuning, and Validation

This section demonstrates how to configure, tune, and validate a Split Conformal Quantile Regressor (CQR). The hyperparameter search space and evaluation metrics are explained in comments.


```python
# Configure and validate a Split Conformal Quantile Regressor (CQR)
from uqregressors.tuning.tuning import interval_width

cqr = ConformalQuantileRegressor(
    hidden_sizes=[32, 32],
    cal_size=0.5,  # Fraction of training data for calibration
    alpha=0.1,
    dropout=0.1,
    epochs=40,
    batch_size=32,
    learning_rate=1e-3,
    optimizer_kwargs={"weight_decay": 1e-6},
    device=device,
    scale_data=True,
    use_wandb=False
)

# Hyperparameter search space for quantile levels
tau_param_space = {
    "tau_lo": lambda trial: trial.suggest_float("tau_lo", 0.03, 0.1),
    "tau_hi": lambda trial: trial.suggest_float("tau_hi", 0.9, 0.97)
}

cqr_save_paths = run_regressor_test(
    cqr, datasets, seed=42, filename="cqr_demo", test_size=0.2,
    tuning_epochs=20, param_space=tau_param_space, scoring_fn=interval_width, greater=False, n_trials=5, n_splits=2
)

print_results(cqr_save_paths)
```

## 7. Visualization of Validation Results

This section provides functions and code to visualize and compare experimental results with published benchmarks. Plots help interpret the performance of each regressor and understand the effect of hyperparameters.


```python
# Visualization: Compare predicted vs. true values and interval widths
# This function can be adapted to visualize results for any regressor

def plot_predictions(X_test, y_test, mean, lower, upper, title="Prediction Intervals"): 
    """Plot predicted mean and uncertainty intervals against true values."""
    plt.figure(figsize=(8, 5))
    plt.plot(y_test, label="True values", marker="o", linestyle="None", alpha=0.5)
    plt.plot(mean, label="Predicted mean", color="red")
    plt.fill_between(range(len(mean)), lower, upper, color="orange", alpha=0.3, label="Prediction interval")
    plt.xlabel("Test sample index")
    plt.ylabel("y")
    plt.title(title)
    plt.legend()
    plt.show()

# Example usage (for the last trained model on the synthetic dataset):
# Load the last saved model and data
fm = FileManager()
last_path = mc_save_paths[-1][2]  # Use MC Dropout as example
load_dict = fm.load_model(MCDropoutRegressor, last_path, load_logs=False)
mean, lower, upper = load_dict["model"].predict(load_dict["X_test"])
plot_predictions(load_dict["X_test"], load_dict["y_test"], mean, lower, upper, title="MC Dropout: Synthetic Data")
```
