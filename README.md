[![PyPI - Version](https://img.shields.io/pypi/v/uqregressors.svg)](https://pypi.org/project/uqregressors)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/uqregressors.svg)](https://pypi.org/project/uqregressors)
[![Python Tests](https://github.com/arjunrs3/uqregressors/actions/workflows/python-tests.yml/badge.svg)](https://github.com/arjunrs3/uqregressors/actions/workflows/python-tests.yml)
-----

# UQRegressors

**UQRegressors** is a Python library for regression models that provide **prediction intervals**, in addition to point estimates. It is meant for machine learning applications where quantifying uncertainty is important. 

It features **highly customizable** parameters for each model, an **easy to use** interface with built-in dataset validation, **GPU compatibility** with a PyTorch backend, **validated implementations** with comparisons to published results, **easy saving and loading** of created models, and a **wide variety of metrics and visualization** tools available to assess model quality. 

Please direct any questions or suggestions to the email arjunrs@stanford.edu.

---

## Key Capabilities

1. **Dataset Loading & Validation** — Utility functions to clean and validate your input data.
2. **Uncertainty‑Aware Regressors**  
   - Conformal: CQR, K‑Fold CQR, Ensemble‑based CQR  
   - Bayesian: Deep Ensembles, MC Dropout, Gaussian Processes (GP, BBMM GP)
3. **Hyperparameter Tuning** — Optuna‑based tuning with support for custom interval‑width objective functions.
4. **Uncertainty Metrics** — RMSE, coverage, interval width, interval score, NLL, correlation diagnostics, conditional coverage, RMSCD variants, and more.
5. **Visualization Tools** — Calibration curves, prediction-vs-true plots, model comparison bar charts.

---

## Installation
To install the core features of UQRegressors: 
```bash
pip install uqregressors
```
UQRegressors requires **PyTorch**, which you should install according to your setup:
- **CPU only**:  
  ```bash
  pip install torch torchvision torchaudio
  ```
- **CUDA GPU** (choose the version matching your GPU):
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```
- For other versions, see [PyTorch Install Guide](https://pytorch.org/get-started/locally).
---

## Getting Started (More detail in full docs)

```python
from uqregressors.bayesian.dropout import MCDropoutRegressor
from uqregressors.tuning.tuning import tune_hyperparams, interval_width
from uqregressors.plotting import plot_uncertainty_results

# Train an MC‑Dropout regressor
reg = MCDropoutRegressor(epochs=50, random_seed=42)
reg.fit(X_train, y_train)
mean, lower, upper = reg.predict(X_test)

# Visualize results
plot_uncertainty_results(mean, lower, upper, model_name="MC‑Dropout")

# Hyperparameter tuning example (e.g., tuning CQR)
from uqregressors.conformal.cqr import ConformalQuantileRegressor
cqr = ConformalQuantileRegressor(alpha=0.1, epochs=20, random_seed=42)

opt_cqr, best_score, study = tune_hyperparams(
    regressor=cqr,
    param_space={"tau_lo": lambda t: t.suggest_float("tau_lo", 0.01, 0.1),
                 "tau_hi": lambda t: t.suggest_float("tau_hi", 0.9, 0.99)},
    X=X_train, y=y_train,
    score_fn=interval_width,
    greater_is_better=False,
    n_trials=10,
    n_splits=3
)
mean_t, lo_t, hi_t = opt_cqr.predict(X_test)
plot_uncertainty_results(mean_t, lo_t, hi_t, model_name="Tuned CQR")
```

---

## Testing

Clone the repository and run:

```bash
pip install hatch
hatch env create
pip install pytest
hatch run pytest -v
```

---

## Documentation

See the complete API Documentation with complete examples:  
https://arjunrs3.github.io/UQRegressors/

---

## Contributing

Contributions, issues, and feature requests are welcome! Please:

1. Fork the repo  
2. Create a feature branch (`git checkout -b my-feature`)  
3. Commit your changes and push  
4. Open a Pull Request
5. Email arjunrs@stanford.edu with any questions

---

## License

[MIT License](LICENSE)

---
