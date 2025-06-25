import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from torch.serialization import safe_globals
import pandas as pd
import numpy as np
import os

BASE_SAVE_DIR = Path.home() / ".uqregressors" / "models"
BASE_SAVE_DIR.mkdir(parents=True, exist_ok=True)


def get_timestamped_path(model_class_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return BASE_SAVE_DIR / f"{model_class_name}_{timestamp}"

def get_named_path(name): 
    return BASE_SAVE_DIR / name

def _save_array(array, path):
    np.save(path, array)


def _load_array(path):
    return np.load(path)

def save_model(
    model,
    path=None,
    name=None,
    metrics=None,
    X_train=None,
    y_train=None,
    X_test=None,
    y_test=None
):
    """
    Save the model using model.save(), and optionally save metrics and datasets.
    """
    if path is None and name is None:
        path = get_timestamped_path(model.__class__.__name__)
    elif name is not None: 
        path = get_named_path("name")
    else:
        path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Save the model using its own method
    if not hasattr(model, "save") or not callable(model.save):
        raise AttributeError(f"{model.__class__.__name__} must implement `save(path)`")
    model.save(path)

    # Save metrics
    if metrics is not None:
        with open(path / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    # Save datasets if provided
    if X_train is not None:
        _save_array(np.array(X_train), path / "X_train.npy")
    if y_train is not None:
        _save_array(np.array(y_train), path / "y_train.npy")
    if X_test is not None:
        _save_array(np.array(X_test), path / "X_test.npy")
    if y_test is not None:
        _save_array(np.array(y_test), path / "y_test.npy")

    print(f"Model and additional artifacts saved to: {path}")
    return path


def load_model(model_class, path, device="cpu"):
    """
    Load the model using model_class.load(), and optionally load metrics and datasets.
    Returns:
        model, metrics, X_train, y_train, X_test, y_test
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Path {path} does not exist")

    if not hasattr(model_class, "load") or not callable(model_class.load):
        raise AttributeError(f"{model_class.__name__} must implement `load(path)`")

    with safe_globals([np._core.multiarray._reconstruct, np.ndarray, np.dtype]):
        model = model_class.load(path, device=device)

    # Load metrics if available
    metrics = None
    if (path / "metrics.json").exists():
        with open(path / "metrics.json") as f:
            metrics = json.load(f)

    # Load datasets if available
    def try_load(name):
        f = path / f"{name}.npy"
        return _load_array(f) if f.exists() else None

    X_train = try_load("X_train")
    y_train = try_load("y_train")
    X_test = try_load("X_test")
    y_test = try_load("y_test")

    return_dict = {"model": model, "metrics": metrics, "X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}
    return return_dict

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