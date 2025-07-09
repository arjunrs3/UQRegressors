# Example: Saving and Loading Models and Data with UQRegressors

This notebook demonstrates how to train a regression model, save the trained model and associated data to disk, and then load them back for further use or evaluation.

The workflow includes:
- Generating synthetic data
- Training a Deep Ensemble regressor
- Saving the model, metrics, and datasets using the `FileManager` utility
- Loading the saved model and data
- Verifying that predictions from the loaded model match the original


## 1. Import Required Libraries

We import the necessary modules from UQRegressors and scikit-learn. The `FileManager` utility handles saving and loading models and data, while `DeepEnsembleRegressor` is used as the example model.


```python
import numpy as np
from sklearn.model_selection import train_test_split
from uqregressors.bayesian.deep_ens import DeepEnsembleRegressor
from uqregressors.utils.file_manager import FileManager
from sklearn.metrics import mean_squared_error
```

## 2. Generate Synthetic Data

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

## 3. Train a Deep Ensemble Regressor

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

## 4. Save the Model, Metrics, and Datasets

We use the `FileManager` utility to save the trained model, evaluation metrics, and the train/test datasets to disk. This makes it easy to reload the model and data later for reproducibility or further analysis.


```python
# Initialize the FileManager and save everything
fm = FileManager()
save_path = fm.save_model(
    reg,
    metrics={"mse": mse},
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
)
print(f"Model and data saved to: {save_path}")
```

## 5. Load the Model, Metrics, and Datasets

We demonstrate how to load the saved model, metrics, and datasets using the `FileManager`. This allows you to resume work, evaluate, or make predictions without retraining.


```python
# Load everything back from disk
load_dict = fm.load_model(DeepEnsembleRegressor, save_path, load_logs=True)
loaded_model = load_dict["model"]
X_test_loaded = load_dict["X_test"]
y_test_loaded = load_dict["y_test"]
mse_loaded = load_dict["metrics"]["mse"]
```

## 6. Predict with the Loaded Model and Verify Results

Finally, we use the loaded model to make predictions on the loaded test set and verify that the mean squared error matches the value saved earlier. This confirms that the model and data were saved and loaded correctly.


```python
# Predict with the loaded model and check MSE
mean_pred_loaded, _, _ = loaded_model.predict(X_test_loaded)
loaded_mse = mean_squared_error(y_test_loaded, mean_pred_loaded)
print(f"Loaded MSE: {loaded_mse:.4f} (should match saved: {mse_loaded:.4f})")
```
