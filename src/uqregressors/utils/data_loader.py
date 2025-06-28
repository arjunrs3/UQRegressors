import os 
import pandas as pd 
import numpy as np

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
