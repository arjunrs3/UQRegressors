import numpy as np
from scipy.stats import norm

def validate_inputs(mean, lower, upper, y_true, alpha=0.5):
    """Ensure inputs are 1D numpy arrays and alpha is a float in (0, 1)."""
    def to_1d_np(x):
        x = np.asarray(x)
        if x.ndim != 1:
            x = x.flatten()
        return x

    mean = to_1d_np(mean)
    lower = to_1d_np(lower)
    upper = to_1d_np(upper)
    y_true = to_1d_np(y_true)

    if not (0 < float(alpha) < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    if not (len(mean) == len(lower) == len(upper) == len(y_true)):
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

def compute_all_metrics(mean, lower, upper, y_true, alpha):
    """
    Compute all standard uncertainty quantification metrics and return as a dictionary.
    """
    mean, lower, upper, y_true, alpha = validate_inputs(mean, lower, upper, y_true, alpha)


    return {
        "rmse": rmse(mean, y_true, alpha=alpha),
        "coverage": coverage(lower, upper, y_true, alpha=alpha),
        "average_interval_width": average_interval_width(lower, upper, alpha=alpha),
        "interval_score": interval_score(lower, upper, y_true, alpha),
        "nll_gaussian": nll_gaussian(mean, lower, upper, y_true, alpha),
        "error_width_correlation": error_width_corr(mean, lower, upper, y_true)
    }