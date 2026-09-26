"""
elastic_net.py - ElasticNet Regularized Logistic Regression

Implements penalized linear classification with combined L1 (Lasso) and L2 (Ridge)
penalties. Strict in-fold preprocessing encapsulates:
- SimpleImputer(strategy='median')
- StandardScaler()
- LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, C=0.1, max_iter=200)

This prevents data leakage across cross-validation folds.
"""

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from modelling.hyperparameters import SELECTED_HYPERPARAMETERS


def get_elastic_net_pipeline(random_state: int = 42, **kwargs) -> Pipeline:
    """Builds the leak-free ElasticNet Pipeline."""
    params = SELECTED_HYPERPARAMETERS["ElasticNet_Logistic"].copy()
    params["random_state"] = random_state
    params.update(kwargs)

    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(**params))
    ])
    return pipeline
