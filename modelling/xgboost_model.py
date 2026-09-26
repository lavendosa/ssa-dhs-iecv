"""
xgboost_model.py - Extreme Gradient Boosting (XGBoost) Classifier

Implements regularized gradient boosted decision trees utilizing second-order
Taylor expansion loss and column/subsampling:
- n_estimators = 150
- max_depth = 6
- learning_rate = 0.08
- subsample = 0.80
- colsample_bytree = 0.80
- Native NaN handling
"""

from xgboost import XGBClassifier
from modelling.hyperparameters import SELECTED_HYPERPARAMETERS


def get_xgboost_model(random_state: int = 42, **kwargs) -> XGBClassifier:
    """Builds the XGBClassifier with prespecified hyperparameters."""
    params = SELECTED_HYPERPARAMETERS["XGBoost"].copy()
    params["random_state"] = random_state
    params.update(kwargs)
    return XGBClassifier(**params)
