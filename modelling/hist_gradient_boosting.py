"""
hist_gradient_boosting.py - Histogram-Based Gradient Boosting Classifier

Implements scikit-learn's HistGradientBoostingClassifier utilizing integer histogram binning:
- max_iter = 150
- max_depth = 8
- min_samples_leaf = 30
- l2_regularization = 1.0
- Native NaN routing during split evaluation (no artificial imputation needed)
"""

from sklearn.ensemble import HistGradientBoostingClassifier
from modelling.hyperparameters import SELECTED_HYPERPARAMETERS


def get_hist_gradient_boosting_model(random_state: int = 42, **kwargs) -> HistGradientBoostingClassifier:
    """Builds the HistGradientBoostingClassifier with prespecified hyperparameters."""
    params = SELECTED_HYPERPARAMETERS["HistGradientBoosting"].copy()
    params["random_state"] = random_state
    params.update(kwargs)
    return HistGradientBoostingClassifier(**params)
