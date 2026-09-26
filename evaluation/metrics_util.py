"""
metrics_util.py - Statistical & Evaluation Metrics Utilities

Implements specialized epidemiological and machine learning evaluation metrics:
- Parametric Hanley–McNeil (1982) formulation for country AUROC Standard Errors and 95% CIs.
- Cross-national unweighted arithmetic mean and cross-national standard error of the mean.
- Expected Calibration Error (ECE) with equal-frequency or equal-width bins.
- Logistic calibration regression (Cox / Steyerberg formulation) for calibration slopes and intercepts.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression


def hanley_mcneil_ci(auroc: float, n_pos: int, n_neg: int) -> tuple[float, float, float]:
    """
    Computes the parametric Hanley-McNeil (1982) standard error and 95% confidence interval
    for the Area Under the Receiver Operating Characteristic (AUROC) curve.

    SE = sqrt((theta*(1-theta) + (n1-1)*(Q1-theta^2) + (n0-1)*(Q2-theta^2)) / (n1*n0))
    where Q1 = theta / (2 - theta), Q2 = 2*theta^2 / (1 + theta).
    """
    if n_pos <= 0 or n_neg <= 0:
        return auroc, auroc, 0.0

    theta = float(np.clip(auroc, 0.001, 0.999))
    q1 = theta / (2.0 - theta)
    q2 = (2.0 * (theta ** 2)) / (1.0 + theta)

    var = (
        theta * (1.0 - theta) +
        (n_pos - 1) * (q1 - theta ** 2) +
        (n_neg - 1) * (q2 - theta ** 2)
    ) / (n_pos * n_neg)

    se = float(np.sqrt(max(0.0, var)))
    ci_low = float(np.clip(theta - 1.96 * se, 0.0, 1.0))
    ci_high = float(np.clip(theta + 1.96 * se, 0.0, 1.0))
    return ci_low, ci_high, se


def cross_national_summary_ci(country_estimates: list[float]) -> tuple[float, float, float, float]:
    """
    Computes the unweighted cross-national arithmetic mean and its 95% confidence interval
    reflecting cross-national heterogeneity across distinct national health systems:
    Mean +/- 1.96 * (SD / sqrt(K))
    """
    arr = np.array(country_estimates, dtype=float)
    k = len(arr)
    mean_val = float(np.mean(arr))
    sd_val = float(np.std(arr, ddof=1)) if k > 1 else 0.0
    se_val = sd_val / np.sqrt(k) if k > 0 else 0.0
    ci_low = mean_val - 1.96 * se_val
    ci_high = mean_val + 1.96 * se_val
    return mean_val, ci_low, ci_high, sd_val


def calculate_calibration_slope_intercept(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """
    Calculates the Cox / Steyerberg logistic calibration intercept and slope:
    logit(P(Y=1)) = alpha + beta * logit(p_predicted)
    Ideal model: alpha = 0.0 (intercept), beta = 1.0 (slope).
    """
    eps = 1e-6
    y_prob_clipped = np.clip(y_prob, eps, 1.0 - eps)
    logit_p = np.log(y_prob_clipped / (1.0 - y_prob_clipped)).reshape(-1, 1)

    lr = LogisticRegression(solver='lbfgs', max_iter=1000)
    lr.fit(logit_p, y_true)
    slope = float(lr.coef_[0][0])
    intercept = float(lr.intercept_[0])
    return intercept, slope


def calculate_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Computes Expected Calibration Error (ECE) across uniform probability bins.
    """
    bin_limits = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total_n = len(y_true)
    if total_n == 0:
        return 0.0

    for i in range(n_bins):
        bin_mask = (y_prob >= bin_limits[i]) & (y_prob < bin_limits[i+1])
        n_bin = np.sum(bin_mask)
        if n_bin > 0:
            bin_acc = np.mean(y_true[bin_mask])
            bin_conf = np.mean(y_prob[bin_mask])
            ece += n_bin * np.abs(bin_acc - bin_conf)

    return float(ece / total_n)
