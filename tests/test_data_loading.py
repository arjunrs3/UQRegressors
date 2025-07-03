import os
import tempfile
import numpy as np
import pandas as pd
import pytest

from uqregressors.utils.data_loader import (
    load_unformatted_dataset,
    clean_dataset,
    validate_dataset,
    load_arff,
)

@pytest.fixture
def csv_path(tmp_path):
    data = "feat1,feat2,target\n1.0,2.0,3.0\n4.0,5.0,6.0\n"
    p = tmp_path / "test.csv"
    p.write_text(data)
    return str(p)

@pytest.fixture
def arff_path(tmp_path):
    content = """
    @relation test
    @attribute feat1 numeric
    @attribute feat2 numeric
    @attribute target numeric
    @data
    1.0,2.0,3.0
    4.0,5.0,6.0
    """
    p = tmp_path / "test.arff"
    p.write_text(content)
    return str(p)

def test_load_unformatted_dataset_csv(csv_path):
    X, y = load_unformatted_dataset(csv_path)
    assert X.shape == (2, 2)
    assert y.shape == (2,)
    np.testing.assert_allclose(X[0], [1.0, 2.0])
    np.testing.assert_allclose(y[1], 6.0)

def test_load_unformatted_dataset_with_target_and_drop(csv_path):
    X, y = load_unformatted_dataset(csv_path, target_column="target", drop_columns=["feat1"])
    assert X.shape == (2, 1)
    assert "feat1" not in pd.DataFrame(X).columns

def test_load_unformatted_dataset_arff(arff_path):
    df = load_arff(arff_path)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["feat1", "feat2", "target"]
    assert df.shape[0] == 2
    # Now test loading via main function
    X, y = load_unformatted_dataset(arff_path)
    assert X.shape == (2, 2)
    assert y.shape == (2,)

def test_clean_dataset_removes_nans():
    X = np.array([[1, 2], [np.nan, 3], [4, 5]])
    y = np.array([1, 2, np.nan])
    X_clean, y_clean = clean_dataset(X, y)
    assert X_clean.shape[0] == 1  # Only one row with no NaNs
    assert y_clean.shape == (1, 1)

def test_validate_dataset_accepts_valid_data():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([1.0, 2.0])
    # Should not raise
    validate_dataset(X, y, name="valid")

def test_validate_dataset_raises_on_bad_shapes():
    X = np.array([1.0, 2.0])  # 1D instead of 2D
    y = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        validate_dataset(X, y)

    X = np.array([[1.0], [2.0]])
    y = np.array([[1.0, 2.0]])  # 2D but wrong shape
    with pytest.raises(ValueError):
        validate_dataset(X, y)

def test_validate_dataset_raises_on_mismatched_samples():
    X = np.array([[1.0], [2.0]])
    y = np.array([1.0])
    with pytest.raises(ValueError):
        validate_dataset(X, y)

def test_validate_dataset_raises_on_nans():
    X = np.array([[np.nan, 2.0], [3.0, 4.0]])
    y = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        validate_dataset(X, y)

def test_validate_dataset_raises_on_nonfloat():
    X = np.array([[1, 2], [3, 4]], dtype=int)
    y = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        validate_dataset(X, y)

def test_load_unformatted_dataset_txt(tmp_path):
    # Create a txt file with comma delimiter
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("1.0,2.0,3.0\n4.0,5.0,6.0\n")
    X, y = load_unformatted_dataset(str(txt_file))
    assert X.shape == (2, 2)
    assert y.shape == (2,)

def test_load_unformatted_dataset_unsupported_extension(tmp_path):
    unsupported_file = tmp_path / "file.unsupported"
    unsupported_file.write_text("irrelevant content")
    with pytest.raises(ValueError):
        load_unformatted_dataset(str(unsupported_file))