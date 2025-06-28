from uqregressors.metrics.metrics import coverage, average_interval_width 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
from copy import deepcopy

def generate_cal_curve(model, X_test, y_test, alphas=None, refit=False, 
                       X_train=None, y_train=None, show=False, save_path=None):
    if (refit == True) and (X_train == None or y_train == None): 
        raise ValueError("X_train and y_train must be given to generate a calibration curve with refit=True")
    alphas = alphas or np.linspace(0.7, 0.01, 10)
    desired_coverage = 1 - alphas 
    coverages = np.zeros_like(desired_coverage)
    avg_interval_widths = np.zeros_like(desired_coverage)
    orig_alpha = deepcopy(model.alpha)
    for i, alpha in enumerate(alphas): 
        model.alpha = alpha 
        if refit == True: 
            model.fit(X_train, y_train)
        mean, lower, upper = model.predict(X_test)
        coverages[i] = coverage(lower, upper, y_test)
        avg_interval_widths[i] = average_interval_width(lower, upper)

    

