"""
hyperparameters.py - Prespecified Machine Learning Hyperparameters

Defines the prespecified hyperparameter search spaces and final selected
configurations for all five machine learning algorithms benchmarked in the study,
corresponding exactly to Table S5 of the Supplementary Appendix.
"""

from typing import Dict, Any

# Prespecified Selected Configurations (Table S5)
SELECTED_HYPERPARAMETERS: Dict[str, Dict[str, Any]] = {
    "ElasticNet_Logistic": {
        "penalty": "elasticnet",
        "solver": "saga",
        "C": 0.1,
        "l1_ratio": 0.5,
        "max_iter": 200,
        "random_state": 42
    },
    "Random_Forest": {
        "n_estimators": 500,
        "max_depth": 12,
        "min_samples_leaf": 20,
        "max_features": "sqrt",
        "n_jobs": -1,
        "random_state": 42
    },
    "HistGradientBoosting": {
        "max_iter": 150,
        "max_depth": 8,
        "min_samples_leaf": 30,
        "l2_regularization": 1.0,
        "random_state": 42
    },
    "XGBoost": {
        "n_estimators": 150,
        "max_depth": 6,
        "learning_rate": 0.08,
        "subsample": 0.80,
        "colsample_bytree": 0.80,
        "eval_metric": "logloss",
        "n_jobs": -1,
        "random_state": 42
    },
    "LightGBM": {
        "n_estimators": 150,
        "max_depth": 7,
        "learning_rate": 0.08,
        "subsample": 0.80,
        "colsample_bytree": 0.80,
        "min_child_samples": 50,
        "n_jobs": -1,
        "verbose": -1,
        "random_state": 42
    }
}

# Hyperparameter Search Spaces Catalogued in Protocol (Table S5)
SEARCH_SPACES: Dict[str, Dict[str, list]] = {
    "ElasticNet_Logistic": {
        "C": [0.001, 0.01, 0.1, 1.0, 10.0],
        "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        "solver": ["saga"]
    },
    "Random_Forest": {
        "n_estimators": [100, 200, 500],
        "max_depth": [8, 12, 16, None],
        "min_samples_leaf": [10, 20, 50]
    },
    "HistGradientBoosting": {
        "max_iter": [100, 150, 200],
        "max_depth": [6, 8, 10],
        "min_samples_leaf": [20, 30, 50],
        "l2_regularization": [0.1, 1.0, 5.0]
    },
    "XGBoost": {
        "n_estimators": [100, 150, 200],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.03, 0.08, 0.15],
        "subsample": [0.70, 0.80, 0.90],
        "colsample_bytree": [0.70, 0.80, 0.90]
    },
    "LightGBM": {
        "n_estimators": [100, 150, 200],
        "max_depth": [5, 7, 9],
        "learning_rate": [0.03, 0.08, 0.15],
        "subsample": [0.70, 0.80, 0.90],
        "colsample_bytree": [0.70, 0.80, 0.90],
        "min_child_samples": [20, 50, 100]
    }
}
