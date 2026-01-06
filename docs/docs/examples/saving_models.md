# Example: Saving and Loading Models and Data with UQRegressors

This notebook demonstrates how to train a regression model, save the trained model and associated data to disk, and then load them back for further use or evaluation.

The workflow includes:

1. Generating synthetic data
2. Training a Deep Ensemble regressor
3. Saving the model, metrics, and datasets using the `FileManager` utility
4. Loading the saved model and data
5. Verifying that predictions from the loaded model match the original


## Import Required Libraries

We import the necessary modules from UQRegressors and scikit-learn. The `FileManager` utility handles saving and loading models and data, while `DeepEnsembleRegressor` is used as the example model.


```python
import numpy as np
from sklearn.model_selection import train_test_split
from uqregressors.bayesian.deep_ens import DeepEnsembleRegressor
from uqregressors.utils.file_manager import FileManager
from sklearn.metrics import mean_squared_error
from uqregressors.utils.logging import set_logging_config
set_logging_config(print=False)
```

## Generate Synthetic Data

For demonstration purposes, we generate a simple synthetic regression dataset. The target variable is a nonlinear function of the features, with added Gaussian noise.


```python
# Function to generate synthetic data
def generate_data(n_samples=200, n_features=5):
    X = np.random.randn(n_samples, n_features)
    y = np.sin(X[:, 0]) + X[:, 1] ** 2 + np.random.normal(0, 0.1, size=n_samples)
    return X, y

# Generate data and split into train/test sets
X, y = generate_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## Train a Deep Ensemble Regressor

We instantiate and train a `DeepEnsembleRegressor` on the training data. This model is an ensemble of neural networks, which provides both predictions and uncertainty estimates. For simplicity, we use a small number of epochs.


```python
# Create and train the regressor
reg = DeepEnsembleRegressor(epochs=10, random_seed=42)
reg.fit(X_train, y_train)

# Predict on the test set
mean_pred, lower, upper = reg.predict(X_test)
mse = mean_squared_error(y_test, mean_pred)
print(f"Test MSE: {mse:.4f}")
```

    Test MSE: 1.5120
    

## Save the Model, Metrics, and Datasets

We use the `FileManager` utility to save the trained model, evaluation metrics, and the train/test datasets to disk. This makes it easy to reload the model and data later for reproducibility or further analysis.


```python
# Initialize the FileManager and save everything
fm = FileManager() # Can also be initialized with a 'BASE_DIR', e.g., FileManager(base_dir="C:/my_models")
# When not specified, 'BASE_DIR' defaults to a folder named 'models' in the user's home directory. 

save_path = fm.save_model(
    # Can include a custom path, e.g., save_path="C:/my_models/deep_ensemble_regressor" 
    # or a custom name, e.g., save_name="deep_ensemble_regressor", which will save in 'BASE_DIR'/models/'save_name'
    reg,
    metrics={"mse": mse},
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
)
```

    Model and additional artifacts saved to: C:\Users\arsha\.uqregressors\models\DeepEnsembleRegressor_20250709_115438
    

## Load the Model, Metrics, and Datasets

We demonstrate how to load the saved model, metrics, and datasets using the `FileManager`. This allows you to resume work, evaluate, or make predictions without retraining.


```python
# Load everything back from disk
load_dict = fm.load_model(DeepEnsembleRegressor, save_path, load_logs=True) # Returns a dictionary
loaded_model = load_dict["model"]
X_test_loaded = load_dict["X_test"]
y_test_loaded = load_dict["y_test"]
mse_loaded = load_dict["metrics"]["mse"]
```

    D:\uqregressors\src\uqregressors\bayesian\deep_ens.py:412: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
      m.load_state_dict(torch.load(path / f"model_{i}.pt", map_location=device))
    

## Predict with the Loaded Model and Verify Results

Finally, we use the loaded model to make predictions on the loaded test set and verify that the mean squared error matches the value saved earlier. This confirms that the model and data were saved and loaded correctly.


```python
# Predict with the loaded model and check MSE
mean_pred_loaded, _, _ = loaded_model.predict(X_test_loaded)
loaded_mse = mean_squared_error(y_test_loaded, mean_pred_loaded)
print(f"Loaded MSE: {loaded_mse:.4f} (should match saved: {mse_loaded:.4f})")
```

    Loaded MSE: 1.5120 (should match saved: 1.5120)
    
