## Dataset Creation 


```python
import numpy as np


rng = np.random.RandomState(42)
def true_function(x):
    return np.sin(2 * np.pi * x)

X_test = np.linspace(0, 1, 200).reshape(-1, 1)
y_true = true_function(X_test)

X_train = np.sort(rng.rand(10, 1))
y_train = true_function(X_train).ravel() 
```

## Plotting


```python
import matplotlib.pyplot as plt

def plot_uncertainty_results(mean, lower, upper, model_name): 
    plt.figure(figsize=(10, 6))
    plt.plot(X_test, y_true, 'g--', label="True Function")
    plt.scatter(X_train, y_train, color='black', label="Training data", alpha=0.6)
    plt.plot(X_test, mean, label="Predicted Mean", color="blue")
    plt.fill_between(X_test.ravel(), lower, upper, alpha=0.3, color="blue", label = "Uncertainty Interval")
    plt.legend()
    plt.title(f"{model_name} Uncertainty Test")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
```

## MC Dropout


```python
from src.uqregressors.bayesian.dropout import MCDropoutRegressor

model = MCDropoutRegressor(
    hidden_sizes=[100, 100],
    dropout=0.1,
    alpha=0.1,  # 90% confidence
    n_samples=100,
    epochs=1000,
    learning_rate=1e-3,
    device="cpu",  # use "cuda" if GPU available
    use_wandb=False
)

model.fit(X_train, y_train)
mean, lower, upper = model.predict(X_test)

plot_uncertainty_results(mean, lower, upper, "MC Dropout Regressor")
```
