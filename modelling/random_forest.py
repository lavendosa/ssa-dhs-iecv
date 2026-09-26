"""
random_forest.py - Random Forest Ensemble Classifier

Implements ensemble bagging of 500 decorrelated classification trees:
- n_estimators = 500
- max_depth = 12
- min_samples_leaf = 20
- max_features = 'sqrt'
- Median imputation for missing numeric values inside pipeline
"""

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from modelling.hyperparameters import SELECTED_HYPERPARAMETERS


def get_random_forest_pipeline(n_trees: int = 500, random_state: int = 42, **kwargs) -> Pipeline:
    """Builds the Random Forest Pipeline with prespecified hyperparameters."""
    params = SELECTED_HYPERPARAMETERS["Random_Forest"].copy()
    params["n_estimators"] = n_trees
    params["random_state"] = random_state
    params.update(kwargs)

    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('clf', RandomForestClassifier(**params))
    ])
    return pipeline
