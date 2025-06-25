import numpy as np 
import pandas as pd 

def clean_dataset(X, y): 
    X_df = pd.DataFrame(X)
    y_series = pd.Series(y) if isinstance(y, (np.ndarray, list)) else pd.Series(y.values)

    combined = pd.concat([X_df, y_series], axis=1)
    combined_clean = combined.dropna()

    X_clean = combined_clean.iloc[:, :-1].astype(np.float32).values
    y_clean = combined_clean.iloc[:, -1].astype(np.float32).values.reshape(-1, 1)

    return X_clean, y_clean

def validate_dataset(X, y, name="unnamed"): 
    print(f"Summary for: {name} dataset")
    print("=" * (21 + len(name)))

    if isinstance(X, pd.DataFrame): 
        X = X.values 
    if isinstance(y, (pd.Series, pd.DataFrame)): 
        y = y.values 

    if X.ndim != 2: 
        raise ValueError("X must be a 2D array (n_samples, n_features)")
    if y.ndim == 2 and y.shape[1] != 1: 
        raise ValueError("y must be 1D or a 2D column vector with shape (n_samples, 1)")
    if y.ndim > 2: 
        raise ValueError("y must be 1D or 2D with a single output")
    
    n_samples, n_features = X.shape 

    if y.shape[0] != n_samples: 
        raise ValueError("X and y must have the same number of samples")
    
    if np.isnan(X).any() or np.isnan(y).any(): 
        raise ValueError("Dataset contains NaNs or missing values.")
    
    if not np.issubdtype(X.dtype, np.floating):
        raise ValueError("X must contain only float values (use float32 or float64)")

    print(f"Number of samples: {n_samples}")
    print(f"Number of features: {n_features}")
    print(f"Output shape: {y.shape}")
    print("Dataset validation passed.\n") 