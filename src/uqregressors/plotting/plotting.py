"""
Plotting
--------
A collection of functions to visualize data generated by UQregressors. 

The supported types of plots are: 
    - Calibration curves 
    - Predicted values vs. true values 
    - Bar chart of model comparisons based on metrics
"""

from uqregressors.metrics.metrics import coverage, average_interval_width, compute_all_metrics
from uqregressors.utils.file_manager import FileManager
import tempfile
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd


def generate_cal_curve(model, X_test, y_test, alphas=np.linspace(0.7, 0.01, 10), refit=False, 
                       X_train=None, y_train=None):
    """
    Generate the data for a calibration curve, which can be plotted with plot_cal_curve. 

    Args: 
        model (BaseEstimator): The model for which to generate the calibration curve. 
        X_test (array-like): An array of testing features to generate the calibration curve for. 
        y_test (array-like): An array of testing targets to generate the calibration curve for. 
        alphas (array-like): The complement of the confidence intervals tested. If none, 10 alphas between 0.7 and 0.01 are linearly generated.
        refit (bool): Whether to re-fit the model for each alpha (useful for models like CQR where the underlying regressor depends on alpha).
        X_train (array-like): Training features if refit is True. 
        y_train (array-like): Training targets if refit is True.

    Returns: 
        Tuple(np.ndarray, np.ndarray, np.ndarray): The desired coverages, the empirical coverages, and the average interval widths for each alpha. 
    """
    if (refit == True) and (X_train is None or y_train is None): 
        raise ValueError("X_train and y_train must be given to generate a calibration curve with refit=True")
    alphas = np.array(alphas)
    desired_coverage = 1 - alphas 
    coverages = np.zeros_like(desired_coverage)
    avg_interval_widths = np.zeros_like(desired_coverage)

    for i, alpha in enumerate(alphas): 
        # Clone model: 
        with tempfile.TemporaryDirectory() as tmpdirname: 
            fm = FileManager(tmpdirname)
            saved_path = fm.save_model(model, name=None, path=None)
            cloned_model = fm.load_model(model.__class__, path=saved_path)["model"]
        cloned_model.alpha = alpha 
        if refit == True: 
            cloned_model.fit(X_train, y_train)

        mean, lower, upper = cloned_model.predict(X_test)
        coverages[i] = coverage(lower, upper, y_test)
        avg_interval_widths[i] = average_interval_width(lower, upper)

    return desired_coverage, coverages, avg_interval_widths 

def plot_cal_curve(desired_coverage, coverages, show=False, save_dir=None, filename="calibration_curve.png", title=None, figsize=(8, 5)): 
    """
    Plot a calibration curve with data generated from uqregressors.plotting.plotting.generate_cal_curve. 

    Args: 
        desired_coverage (array-like): An array of the desired coverages for which the model was evaluated.
        coverages (array-like): An array of the empirical coverages achieved by the model for each desired coverage. 
        show (bool): Whether to display the plot after generating it (True) or simply close (False).
        save_dir (str): If not None, the plot will be saved to the directory: save_dir/plots/filename. If associated with a model, 
                        it is recommended that this directory is the directory in which the model is saved. 
        filename (str): The filename, including extension, to which the plots will be saved. 
        title (str): The title included in the plot, if not None.
        figsize (tuple): The size of the figure to be generated.

    Returns: 
        (Union[str, None]): If save_dir is not none, the path to which the file was saved is returned. Otherwise None is returned. 
    """
    plt.figure(figsize=figsize)
    sns.set_theme(style='whitegrid')
    sns.lineplot(x=desired_coverage, y=coverages, marker='o', label='Empirical Coverage')
    plt.plot([0, 1], [0, 1], 'k--', label='Ideal (y = x)')
    plt.xlabel('Desired Coverage (1 - alpha)')
    plt.ylabel('Empirical Coverage')
    if title is not None: 
        plt.title(title)
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()

    if save_dir is not None: 
        fm = FileManager(save_dir)
        save_path = fm.save_plot(plt.gcf(), save_dir, filename, show=show)

        print (f"Saved calibration curve to {save_path}")
        return save_path
    
    else: 
        return None
    
def plot_pred_vs_true(mean, lower, upper, y_true, samples=None, include_confidence=True, show=False, save_dir=None, filename="pred_vs_true.png", title=None, alpha=None, figsize=(8, 8)):
    """
    Plot predicted vs true values with optional confidence intervals.

    Args:
        mean (array-like): Predicted mean values.
        lower (array-like): Lower bound of prediction intervals.
        upper (array-like): Upper bound of prediction intervals.
        y_true (array-like): True target values.
        samples (int): Number of samples to plot. Defaults to all.
        include_confidence (bool): Whether to plot error bars. Default: True.
        show (bool): Whether to display the plot. Default: False.
        save_dir (str): Directory to save the figure. If None, the figure is not saved.
        filename (str): File name for the plot. Default: "pred_vs_true.png".
        title (str): Title of the plot.
        alpha (float): Confidence level (e.g., 0.1 for 90% interval).
        figsize (tuple): Size of the figure to be generated. 

    Returns: 
        (Union[str, None]): The save path if the plot should be saved, otherwise None
    """
    mean = np.asarray(mean)
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    y_true = np.asarray(y_true)

    n = len(y_true)
    idx = np.arange(n)
    if samples is not None:
        samples = min(samples, n)
        idx = np.random.choice(n, samples, replace=False)

    fig, ax = plt.subplots(figsize=figsize)
    if include_confidence:
        ax.errorbar(y_true[idx], mean[idx], 
                    yerr=[mean[idx] - lower[idx], upper[idx] - mean[idx]], 
                    fmt='o', ecolor='gray', alpha=0.75, capsize=3, label=f"Predictions w/ CI")
    else:
        ax.scatter(y_true[idx], mean[idx], alpha=0.75, label="Predictions")

    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', label="y = x")
    ax.set_xlabel("True values")
    ax.set_ylabel("Predicted values")
    ax.legend()
    
    if alpha is not None:
        ax.text(0.05, 0.95, f"$\\alpha$ = {alpha}", transform=ax.transAxes, va='top')

    if title:
        ax.set_title(title)

    plt.tight_layout()
    if save_dir is not None: 
        fm = FileManager(save_dir)
        save_path = fm.save_plot(plt.gcf(), save_dir, filename, show=show)

        print (f"Saved calibration curve to {save_path}")
        return save_path
    
    else: 
        return None

def plot_metrics_comparisons(solution_dict, y_test, alpha, excluded_metrics=[], show=False, save_dir=None, filename=".png", log_metrics = ["rmse", "interval_score", "average_interval_width"], figsize=(8, 5)): 
    """
    Generate bar charts which compare several models on the basis of all available metrics. 

    Args: 
        solution_dict (dict[str: Tuple[np.ndarray, np.ndarray, np.ndarray]]): A dictionary containing the names of the methods to plot as the keys and 
            a tuple containing the mean, lower, and upper predictions of the model on the test set as the values. 
        y_test (array-like): The true values of the targets to compare against. 
        alpha (float): 1 - the confidence level of predictions. Should be a float between 0 and 1. 
        excluded_metrics (list[str]): The names of metrics to exclude. See uqregressors.metrics.metrics.compute_all_metrics for a list of possible keys 
        show (bool): Whether to display the plot. Default: False.
        save_dir (str, optional): Directory to save the figure. If None, the figure is not saved.
        filename (str): File name for the plot. Default: "pred_vs_true.png".
        log_metrics (list): A list containing the keys of metrics to display on a log scale. 
        figsize (tuple): Desired figure size. 

    Returns: 
        (Union[str, None]): The save path to the directory in which plots were saved if save_dir is True, otherwise None
    """

    better_direction = {
        "rmse": "lower is better",
        "nll_gaussian": "lower is better",
        "interval_score": "lower is better",
        "coverage": "closer to {:.2f} is better".format(1 - alpha),
        "average_interval_width": "lower is better",
        "error_width_corr":"higher is better", 
        "RMSCD": "lower is better", 
        "RMSCD_under": "lower is better", 
        "lowest_group_coverage": "higher is better"
    }

    rows = [] 
    for method, (mean, lower, upper) in solution_dict.items(): 
        metrics = compute_all_metrics(mean, lower, upper, y_test, alpha, excluded_metrics=excluded_metrics + ["group_conditional_coverage"])
        metrics["method"] = method 
        rows.append(metrics)
        rows.append(metrics)
    metrics_df = pd.DataFrame(rows)

    metrics = [col for col in metrics_df.columns if col!="method"]

    for metric in metrics: 
        plt.figure(figsize=figsize)
        sns.barplot(data=metrics_df, x="method", y=metric)
        plt.xticks(rotation=45)

        if metric in log_metrics:
            plt.yscale("log")

        if metric == "coverage": 
            plt.axhline(1 - alpha, color="red", linestyle="--", label="Nominal")
            plt.legend() 
        
        title = f"{metric} ({better_direction.get(metric, '')})"
        plt.title(title)
        plt.ylabel(metric)
        plt.xlabel("Method")
        plt.tight_layout()

        if save_dir is not None: 
            fm = FileManager(save_dir)
            save_path = fm.save_plot(plt.gcf(), save_dir, metric.strip() + "_" + filename, show=show)

            print (f"Saved model comparison to {save_path}")
    
    if save_dir is not None: 
        return save_dir
    else: 
        return None