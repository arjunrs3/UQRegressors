import os
import tempfile
import numpy as np
import pytest
import matplotlib.pyplot as plt

from uqregressors.plotting.plotting import (
    generate_cal_curve, plot_cal_curve,
    plot_pred_vs_true, plot_metrics_comparisons
)

@pytest.fixture
def sample_predictions():
    n = 50
    x = np.linspace(0, 1, n)
    mean = np.sin(2 * np.pi * x)
    lower = mean - 0.2
    upper = mean + 0.2
    y_true = mean + np.random.normal(0, 0.1, size=n)
    return mean, lower, upper, y_true

def test_generate_cal_curve_runs(sample_predictions):
    mean, lower, upper, y_true = sample_predictions
    class DummyModel:
        def __init__(self):
            self.alpha = 0.1
        def predict(self, X):
            return mean, lower, upper
        def save(self, path=None): 
            pass 

        @classmethod
        def load(cls, path=None, device=None, load_logs=None): 
            return cls()
    model = DummyModel()
    
    desired_cov, cov, widths = generate_cal_curve(model, None, y_true, alphas=[0.1, 0.2], refit=False)
    assert len(desired_cov) == 2
    assert cov.shape == widths.shape == desired_cov.shape

def test_plot_cal_curve_no_save(tmp_path):
    desired_cov = np.array([0.9, 0.8])
    coverages = np.array([0.85, 0.78])
    
    # Should return None if no save_dir provided
    ret = plot_cal_curve(desired_cov, coverages, show=False, save_dir=None)
    assert ret is None

def test_plot_cal_curve_saves_file(tmp_path):
    desired_cov = np.array([0.9, 0.8])
    coverages = np.array([0.85, 0.78])
    
    save_path = plot_cal_curve(desired_cov, coverages, show=False, save_dir=str(tmp_path))
    assert os.path.exists(save_path)

def test_plot_pred_vs_true_runs_and_saves(tmp_path, sample_predictions):
    mean, lower, upper, y_true = sample_predictions
    
    # Without saving
    ret_none = plot_pred_vs_true(mean, lower, upper, y_true, show=False, save_dir=None)
    assert ret_none is None

    # With saving
    ret_path = plot_pred_vs_true(mean, lower, upper, y_true, show=False, save_dir=str(tmp_path))
    assert os.path.exists(ret_path)

def test_plot_metrics_comparisons_runs_and_saves(tmp_path, sample_predictions):
    mean, lower, upper, y_true = sample_predictions
    solution_dict = {"dummy_method": (mean, lower, upper)}

    ret_path = plot_metrics_comparisons(solution_dict, y_true, alpha=0.1, show=False, save_dir=str(tmp_path))
    assert os.path.exists(ret_path)

def test_plots_do_not_crash_with_minimal_inputs(tmp_path):
    mean = lower = upper = y_true = np.array([1.0, 1.0])
    solution_dict = {"method": (mean, lower, upper)}

    plot_cal_curve(np.array([0.9]), np.array([1.0]), show=False)
    plot_pred_vs_true(mean, lower, upper, y_true, show=False)
    plot_metrics_comparisons(solution_dict, y_true, alpha=0.1, show=False, save_dir=str(tmp_path))
