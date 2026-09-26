"""
lightgbm_model.py - LightGBM Classifier (Operational Study Algorithm)

Implements leaf-wise tree growth with Gradient-based One-Side Sampling (GOSS)
and Exclusive Feature Bundling (EFB):
- n_estimators = 150
- max_depth = 7
- learning_rate = 0.08
- subsample = 0.80
- colsample_bytree = 0.80
- min_child_samples = 50
- Native NaN routing during tree split optimization

Selected as the primary operational algorithm for the 38-country leave-one-country-out
IECV pipeline based on superior CPU runtime and memory efficiency.
"""

from lightgbm import LGBMClassifier
from modelling.hyperparameters import SELECTED_HYPERPARAMETERS


def get_lightgbm_model(random_state: int = 42, **kwargs) -> LGBMClassifier:
    """Builds the LightGBM Classifier with prespecified hyperparameters."""
    params = SELECTED_HYPERPARAMETERS["LightGBM"].copy()
    params["random_state"] = random_state
    params.update(kwargs)
    return LGBMClassifier(**params)
