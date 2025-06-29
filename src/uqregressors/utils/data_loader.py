import os 
import pandas as pd 
import numpy as np
import torch

def load_unformatted_dataset(path, target_column=None, drop_columns=None):
    """
    Load and standardize a dataset from a file.

    Parameters:
    -----------
    path : str
        Path to the dataset file (CSV, XLSX, ARFF, etc.)
    target_column : str or int, optional
        Name or index of the target column.
    drop_columns : list of str or int, optional
        Columns to drop (e.g., indices, metadata).

    Returns:
    --------
    X : np.ndarray
        Input features (n_samples, n_features)
    y : np.ndarray
        Target values (n_samples,)
    """

    ext = os.path.splitext(path)[-1].lower()

    if ext == ".csv":
        try:
            df = pd.read_csv(path)
            if df.shape[1] <= 1:
                raise ValueError("Only one column detected; trying semicolon delimiter.")
        except Exception:
            df = pd.read_csv(path, sep=';')
    elif ext == ".xlsx" or ext == ".xls":
        df = pd.read_excel(path)
    elif ext == ".arff":
        data = load_arff(path)
        df = pd.DataFrame(data)
        # Decode bytes to str if needed
        for col in df.select_dtypes([object]):
            df[col] = df[col].apply(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
    elif ext == ".txt":
        # Try common delimiters: comma, tab, space
        for delim in [',', '\t', r'\s+']:
            try:
                df = pd.read_csv(path, sep=delim, engine='python', header=None)
                if df.shape[1] < 2:
                    continue  # unlikely to be valid
                break
            except Exception:
                continue
        else:
            raise ValueError(f"Could not parse .txt file: {path}")
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    df = df.dropna()

    if drop_columns:
        df.drop(columns=drop_columns, inplace=True)

    if target_column is None:
        target_column = df.columns[-1]  # default: last column

    y = df[target_column].values.astype(np.float32)
    X = df.drop(columns=[target_column]).values.astype(np.float32)

    return X, y

def load_arff(path):
    """
    Minimal ARFF file loader without external dependencies.

    Parameters:
    -----------
    path : str
        Path to the ARFF file.

    Returns:
    --------
    df : pd.DataFrame
        Parsed ARFF data as a DataFrame.
    """
    attributes = []
    data = []
    reading_data = False

    with open(path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('%'):
                continue
            if line.lower().startswith('@attribute'):
                # Example: @attribute age numeric
                parts = line.split()
                if len(parts) >= 2:
                    attributes.append(parts[1])
            elif line.lower() == '@data':
                reading_data = True
            elif reading_data:
                # Data line
                row = [x.strip().strip('"') for x in line.split(',')]
                data.append(row)

    df = pd.DataFrame(data, columns=attributes)
    df = df.apply(pd.to_numeric, errors='coerce')  # convert to floats where possible
    return df.dropna()


def validate_and_prepare_inputs(X, y, device="cpu", requires_grad=False):
    """
    Convert X and y into compatible torch.Tensors for training.
    
    Parameters
    ----------
    X : array-like
        Feature matrix. Supports np.ndarray, pd.DataFrame, list, or torch.Tensor.
    y : array-like
        Target vector. Supports np.ndarray, pd.Series, list, or torch.Tensor.
    device : str
        Device to place tensors on (e.g., 'cpu' or 'cuda').
    requires_grad : bool
        Whether the X tensor should require gradients (for gradient-based inference).
    
    Returns
    -------
    X_tensor : torch.Tensor of shape (n_samples, n_features)
    y_tensor : torch.Tensor of shape (n_samples, 1)
    """
    # --- Convert X ---
    if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
        X = X.values
    elif isinstance(X, list):
        X = np.array(X)
    elif isinstance(X, torch.Tensor):
        pass  # leave as is
    elif not isinstance(X, np.ndarray):
        raise TypeError(f"Unsupported type for X: {type(X)}")

    if isinstance(X, np.ndarray):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        elif X.ndim != 2:
            raise ValueError(f"X must be 2D. Got shape {X.shape}")
        X = torch.tensor(X, dtype=torch.float32)

    if not isinstance(X, torch.Tensor):
        raise TypeError("X could not be converted to a torch.Tensor")

    # --- Convert y ---
    if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
        y = y.values
    elif isinstance(y, list):
        y = np.array(y)
    elif isinstance(y, torch.Tensor):
        pass
    elif not isinstance(y, np.ndarray):
        raise TypeError(f"Unsupported type for y: {type(y)}")

    if isinstance(y, np.ndarray):
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        elif y.ndim == 2 and y.shape[1] != 1:
            raise ValueError("y must be 1D or 2D with shape (n, 1)")
        y = torch.tensor(y, dtype=torch.float32)

    if not isinstance(y, torch.Tensor):
        raise TypeError("y could not be converted to a torch.Tensor")

    # --- Final checks ---
    if y.ndim == 1:
        y = y.unsqueeze(1)
    elif y.ndim == 2 and y.shape[1] != 1:
        raise ValueError(f"Expected y to have shape (n_samples,) or (n_samples, 1), but got {y.shape}")

    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y must have the same number of samples. Got {X.shape[0]} and {y.shape[0]}")

    X = X.to(device)
    y = y.to(device)

    if requires_grad:
        X.requires_grad_()

    return X, y

def validate_X_input(X, input_dim = None, device="cpu", requires_grad=False):
    """
    Convert X to a torch.Tensor for inference.
    
    Parameters
    ----------
    X : array-like
        Input data to convert.
    device : str
        Target device ('cpu' or 'cuda').
    requires_grad : bool
        Whether the tensor should track gradients.
    
    Returns
    -------
    torch.Tensor of shape (n_samples, n_features)
    """
    if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
        X = X.values
    elif isinstance(X, list):
        X = np.array(X)
    elif isinstance(X, torch.Tensor):
        pass
    elif not isinstance(X, np.ndarray):
        raise TypeError(f"Unsupported type for X: {type(X)}")

    if isinstance(X, np.ndarray):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        elif X.ndim != 2:
            raise ValueError(f"X must be 2D. Got shape {X.shape}")
        X = torch.tensor(X, dtype=torch.float32)

    if not isinstance(X, torch.Tensor):
        raise TypeError("X could not be converted to a torch.Tensor")
    
    if input_dim is not None: 
        if X.shape[1] != input_dim: 
            raise ValueError(f"Based on the training samples, the number of features of X should be {input_dim}. Got {X.shape[1] = }")

    X = X.to(device)
    if requires_grad:
        X.requires_grad_()

    return X