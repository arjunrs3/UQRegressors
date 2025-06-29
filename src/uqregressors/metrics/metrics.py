import numpy as np
from scipy.stats import norm
import torch

def validate_inputs(mean, lower, upper, y_true, alpha=0.5):
    """Ensure inputs are converted to 1D numpy arrays and alpha is a float in (0, 1)."""

    def to_1d_numpy(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        x = np.asarray(x)
        if x.ndim != 1:
            x = x.flatten()
        return x

    mean = to_1d_numpy(mean)
    lower = to_1d_numpy(lower)
    upper = to_1d_numpy(upper)
    y_true = to_1d_numpy(y_true)

    if not (0 < float(alpha) < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    length = len(mean)
    if not (len(lower) == len(upper) == len(y_true) == length):
        raise ValueError("All input arrays must be of the same length.")

    return mean, lower, upper, y_true, float(alpha)


def rmse(mean, y_true, **kwargs):
    mean, _, _, y_true, _ = validate_inputs(mean, mean, mean, y_true)
    return np.sqrt(np.mean((mean - y_true) ** 2))


def coverage(lower, upper, y_true, **kwargs):
    _, lower, upper, y_true, _ = validate_inputs(lower, lower, upper, y_true)
    covered = (y_true >= lower) & (y_true <= upper)
    return np.mean(covered)


def average_interval_width(lower, upper, **kwargs):
    _, lower, upper, _, _ = validate_inputs(lower, lower, upper, lower)
    return np.mean(upper - lower)


def interval_score(lower, upper, y_true, alpha, **kwargs):
    _, lower, upper, y_true, alpha = validate_inputs(lower, lower, upper, y_true)
    width = upper - lower
    penalty_lower = (2 / alpha) * (lower - y_true) * (y_true < lower)
    penalty_upper = (2 / alpha) * (y_true - upper) * (y_true > upper)
    return np.mean(width + penalty_lower + penalty_upper)


def nll_gaussian(mean, lower, upper, y_true, alpha, **kwargs):
    mean, lower, upper, y_true, alpha = validate_inputs(mean, lower, upper, y_true, alpha)
    z = norm.ppf(1 - alpha / 2)
    std = (upper - lower) / (2 * z)
    std = np.clip(std, 1e-6, None)

    log_likelihoods = -0.5 * np.log(2 * np.pi * std**2) - 0.5 * ((y_true - mean) / std) ** 2
    return -np.mean(log_likelihoods)

def error_width_corr(mean, lower, upper, y_true, **kwargs): 
    mean, lower, upper, y_true, _ = validate_inputs(mean, lower, upper, y_true)
    width = upper - lower 
    res = np.abs(mean - y_true)
    corr = np.corrcoef(width, res)[0, 1]
    return corr

def group_conditional_coverage(lower, upper, y_true, n_bins = 10): 
    _, lower, upper, y_true, alpha = validate_inputs(lower, lower, upper, y_true)
    coverage_mask = (y_true > lower) & (y_true < upper)
    sort_ind = np.argsort(y_true)
    y_true_sort = y_true[sort_ind]
    coverage_mask_sort = coverage_mask[sort_ind]
    split_y_true = np.array_split(y_true_sort, n_bins)
    split_coverage_mask = np.array_split(coverage_mask_sort, n_bins)
    bin_means = [np.mean(bin) for bin in split_y_true]
    bin_coverages = [np.mean(bin) for bin in split_coverage_mask]
    return {"y_true_bin_means": np.array(bin_means), 
            "bin_coverages": np.array(bin_coverages)}

def RMSCD(lower, upper, y_true, alpha, n_bins=10): 
    _, lower, upper, y_true, alpha = validate_inputs(lower, lower, upper, y_true, alpha)
    gcc = group_conditional_coverage(lower, upper, y_true, n_bins)
    return np.sqrt(np.mean((gcc["bin_coverages"] - (1-alpha)) ** 2))

def RMSCD_under(lower, upper, y_true, alpha, n_bins=10):
    _, lower, upper, y_true, alpha = validate_inputs(lower, lower, upper, y_true, alpha)
    gcc = group_conditional_coverage(lower, upper, y_true, n_bins)
    miscovered_bins = gcc["bin_coverages"][gcc["bin_coverages"] < (1-alpha)]
    if len(miscovered_bins) == 0: 
        rmscd = 0.0
    else: 
        rmscd = np.sqrt(np.mean((miscovered_bins - (1-alpha)) ** 2))
    return rmscd

def lowest_group_coverage(lower, upper, y_true, n_bins=10): 
    _, lower, upper, y_true, alpha = validate_inputs(lower, lower, upper, y_true)
    gcc = group_conditional_coverage(lower, upper, y_true, n_bins)
    return np.min(gcc["bin_coverages"])

def compute_all_metrics(mean, lower, upper, y_true, alpha, n_bins=10, excluded_metrics=["group_conditional_coverage"]):
    """
    Compute all standard uncertainty quantification metrics and return as a dictionary.
    """
    mean, lower, upper, y_true, alpha = validate_inputs(mean, lower, upper, y_true, alpha)

    metrics_dict = {
        "rmse": rmse(mean, y_true, alpha=alpha),
        "coverage": coverage(lower, upper, y_true, alpha=alpha),
        "average_interval_width": average_interval_width(lower, upper, alpha=alpha),
        "interval_score": interval_score(lower, upper, y_true, alpha),
        "nll_gaussian": nll_gaussian(mean, lower, upper, y_true, alpha),
        "error_width_corr": error_width_corr(mean, lower, upper, y_true), 
        "group_conditional_coverage": group_conditional_coverage(lower, upper, y_true, n_bins),
        "RMSCD": RMSCD(lower, upper, y_true, alpha, n_bins),
        "RMSCD_under": RMSCD_under(lower, upper, y_true, alpha, n_bins),
        "lowest_group_coverage": lowest_group_coverage(lower, upper, y_true, n_bins)
    }

    return_dict = {}
    for metric, value in metrics_dict.items(): 
        if metric not in excluded_metrics: 
            return_dict[metric] = value 

    return return_dict