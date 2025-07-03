# UQRegressors
UQRegressors is a Python package that provides machine learning regression models capable of generating prediction intervals for a user-specified confidence level. These models not only estimate the expected output but also quantify the uncertainty around each prediction. For instance, a model from UQRegressors trained to predict house prices could, given a new set of input features, return a 95% confidence interval—indicating that the true price is expected to lie between a predicted lower and upper bound with 95% certainty. Several models from Bayesian and Conformal Prediction literature are implemented and validated on a PyTorch backend, which can be easily applied to regression problems through a scikit-learn `fit`, `predict` interface. 

!!! note "Key Features"
    - **Highly customizable** parameters for each model
    - **Easy-to-use** interface with built in dataset validation
    - **GPU compatibility** with PyTorch backend 
    - **Validated implementations** with comparisons to published results 
    - **Easy saving and loading** of created models 
    - **Wide variety of metrics** available to assess quality of fit and prediction intervals

## Use Cases
There are five main capabilities of UQRegessors: 

1. **Dataset** Loading and Validation 
2. **Regression** using models of various types created with UQ capability
3. **Hyperparameter Tuning** using bayesian optimization (wrapper around Optuna)
4. **Metrics** for evaluating goodness of fit and quality of uncertainty intervals
5. **Visualization** of metrics, goodness of fit, and quality of uncertainty intervals

For a simple demonstration of how these features could be used for your problem, check the "Is UQRegressors right for me?" example. For a more holistic view of UQRegressors' cabilities, look at the "Getting Started" example. 

## Installation 
To install all core components of **UQRegressors**, run:
```
pip install UQRegressors 
``` 
### Installing PyTorch

UQRegressors requires **PyTorch**, which must be installed separately to match your system's configuration.

#### CPU-only:
```
pip install torch torchvision torchaudio
```

#### With CUDA support for GPU:
Choose the appropriate command based on your CUDA version:

- **CUDA 11.8:**
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
- **CUDA 12.1:** 
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

For other versions or platforms, check the [official PyTorch installation page](https://pytorch.org/get-started/locally/)

## What Next? 