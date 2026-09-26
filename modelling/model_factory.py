"""
model_factory.py - Unified Machine Learning Model Factory

Provides standardized instantiation and access to all five benchmarked
machine learning algorithms:
- ElasticNet_Logistic
- Random_Forest
- HistGradientBoosting
- XGBoost
- LightGBM
"""

from typing import Dict, Any
from modelling.elastic_net import get_elastic_net_pipeline
from modelling.random_forest import get_random_forest_pipeline
from modelling.hist_gradient_boosting import get_hist_gradient_boosting_model
from modelling.xgboost_model import get_xgboost_model
from modelling.lightgbm_model import get_lightgbm_model

MODEL_BUILDERS = {
    "ElasticNet_Logistic": get_elastic_net_pipeline,
    "Random_Forest": get_random_forest_pipeline,
    "HistGradientBoosting": get_hist_gradient_boosting_model,
    "XGBoost": get_xgboost_model,
    "LightGBM": get_lightgbm_model
}


def get_model(name: str, random_state: int = 42, **kwargs):
    """Instantiates a single model by name with prespecified hyperparameters."""
    if name not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model name '{name}'. Available: {list(MODEL_BUILDERS.keys())}")
    return MODEL_BUILDERS[name](random_state=random_state, **kwargs)


def get_all_benchmark_models(random_state: int = 42, fast_rf: bool = False) -> Dict[str, Any]:
    """Returns a dictionary containing all five instantiated benchmark algorithms."""
    rf_trees = 100 if fast_rf else 500
    return {
        "XGBoost": get_xgboost_model(random_state=random_state),
        "HistGradientBoosting": get_hist_gradient_boosting_model(random_state=random_state),
        "LightGBM": get_lightgbm_model(random_state=random_state),
        "Random_Forest": get_random_forest_pipeline(n_trees=rf_trees, random_state=random_state),
        "ElasticNet_Logistic": get_elastic_net_pipeline(random_state=random_state)
    }
