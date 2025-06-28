from uqregressors.tuning.tuning import tune_hyperparams, interval_width
import numpy as np
import torch 
from uqregressors.utils.validate_dataset import clean_dataset, validate_dataset
import matplotlib.pyplot as plt
from uqregressors.metrics.metrics import compute_all_metrics
from uqregressors.conformal.conformal_split import ConformalQuantileRegressor 

if __name__ == "__main__": 
    seed = 42 

    np.random.seed(seed)
    torch.manual_seed(seed)

    rng = np.random.RandomState(seed)
    def true_function(x):
        return np.sin(3 * np.pi * x)

    X_test = np.linspace(0, 1, 1000).reshape(-1, 1)
    y_true = true_function(X_test)

    X_train = np.sort(rng.rand(100, 1))
    y_train = true_function(X_train).ravel() 

    X_train, y_train = clean_dataset(X_train, y_train)
    validate_dataset(X_train, y_train, name="Synthetic Sine")

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

    cqr = ConformalQuantileRegressor(hidden_sizes = [100, 100], 
                                    cal_size=0.2, 
                                    alpha=0.1, 
                                    dropout=None,
                                    epochs=1000, 
                                    learning_rate=1e-3, 
                                    device="cpu", 
                                    use_wandb=False 
                                    )

    cqr.fit(X_train, y_train)
    cqr_sol = cqr.predict(X_test)

    plot_uncertainty_results(*cqr_sol, "Split Conformal Quantile Regression")


    param_space = {
        "tau_lo": lambda trial: trial.suggest_float("tau_lo", 0.01, 0.1),
        "tau_hi": lambda trial: trial.suggest_float("tau_hi", 0.9, 0.99),
    }

    alpha = 0.1
    opt_cqr, opt_score, study = opt_model, opt_score, study = tune_hyperparams(
                                                                regressor=cqr,
                                                                param_space=param_space,
                                                                X=X_train,
                                                                y=y_train,
                                                                score_fn=interval_width,  # or log_likelihood
                                                                greater_is_better=False,
                                                                n_trials=1,
                                                                n_splits=3,
                                                                verbose=True,
                                                            )
    opt_cqr_sol = opt_cqr.predict(X_test)
    plot_uncertainty_results(*opt_cqr_sol, "Tuned Quantile Split Conformal Quantile Regression")

    metrics_unopt = compute_all_metrics(*cqr_sol, y_true, alpha)
    metrics_opt = compute_all_metrics(*opt_cqr_sol, y_true, alpha)

    print(f"unoptimized: {metrics_unopt}")
    print(f"optimized: {metrics_opt}")