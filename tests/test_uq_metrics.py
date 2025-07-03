import numpy as np
import torch
import pytest
from uqregressors.metrics.metrics import (
    rmse, coverage, average_interval_width, interval_score, nll_gaussian,
    error_width_corr, group_conditional_coverage, RMSCD, RMSCD_under,
    lowest_group_coverage, compute_all_metrics
)

@pytest.fixture
def synthetic_data():
    """Generate synthetic test data."""
    n = 100
    x = np.linspace(0, 1, n)
    y_true = np.sin(2 * np.pi * x) + np.random.normal(0, 0.1, size=n)
    mean = np.sin(2 * np.pi * x)
    lower = mean - 0.2
    upper = mean + 0.2
    alpha = 0.1
    return mean, lower, upper, y_true, alpha

def test_rmse(synthetic_data):
    mean, _, _, y_true, _ = synthetic_data
    result = rmse(mean, y_true)
    assert isinstance(result, float)
    assert result >= 0

def test_coverage(synthetic_data):
    _, lower, upper, y_true, _ = synthetic_data
    result = coverage(lower, upper, y_true)
    assert isinstance(result, float)
    assert 0 <= result <= 1

def test_average_interval_width(synthetic_data):
    _, lower, upper, _, _ = synthetic_data
    result = average_interval_width(lower, upper)
    assert isinstance(result, float)
    assert result >= 0

def test_interval_score(synthetic_data):
    _, lower, upper, y_true, alpha = synthetic_data
    result = interval_score(lower, upper, y_true, alpha)
    assert isinstance(result, float)

def test_nll_gaussian(synthetic_data):
    mean, lower, upper, y_true, alpha = synthetic_data
    result = nll_gaussian(mean, lower, upper, y_true, alpha)
    assert isinstance(result, float)

def test_error_width_corr(synthetic_data):
    mean, lower, upper, y_true, _ = synthetic_data
    result = error_width_corr(mean, lower, upper, y_true)
    assert isinstance(result, float)
    assert -1 <= result <= 1

def test_group_conditional_coverage(synthetic_data):
    _, lower, upper, y_true, _ = synthetic_data
    result = group_conditional_coverage(lower, upper, y_true, n_bins=5)
    assert isinstance(result, dict)
    assert "y_true_bin_means" in result
    assert "bin_coverages" in result
    assert len(result["y_true_bin_means"]) == 5
    assert len(result["bin_coverages"]) == 5

def test_RMSCD(synthetic_data):
    _, lower, upper, y_true, alpha = synthetic_data
    result = RMSCD(lower, upper, y_true, alpha, n_bins=5)
    assert isinstance(result, float)
    assert result >= 0

def test_RMSCD_under(synthetic_data):
    _, lower, upper, y_true, alpha = synthetic_data
    result = RMSCD_under(lower, upper, y_true, alpha, n_bins=5)
    assert isinstance(result, float)
    assert result >= 0

def test_lowest_group_coverage(synthetic_data):
    _, lower, upper, y_true, _ = synthetic_data
    result = lowest_group_coverage(lower, upper, y_true, n_bins=5)
    assert isinstance(result, float)
    assert 0 <= result <= 1

def test_compute_all_metrics(synthetic_data):
    mean, lower, upper, y_true, alpha = synthetic_data
    result = compute_all_metrics(mean, lower, upper, y_true, alpha, n_bins=5)
    assert isinstance(result, dict)
    required_keys = [
        "rmse", "coverage", "average_interval_width", "interval_score",
        "nll_gaussian", "error_width_corr", "RMSCD", "RMSCD_under", "lowest_group_coverage"
    ]
    for key in required_keys:
        assert key in result
        assert isinstance(result[key], float)
