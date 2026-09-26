#!/usr/bin/env python3
"""
04_missing_data_handler.py - Missing Data Management & Imputation

Implements the study's protocol for missing value management:
1. Native missing value handling for Tree Ensembles (LightGBM, XGBoost, HistGradientBoosting):
   Missing values (NaN) are preserved and routed natively during decision tree traversal
   based on maximum gain reduction, avoiding synthetic distortion of real health patterns.
2. In-fold median imputation for Linear Models (ElasticNet Logistic Regression):
   Missing numerical values are imputed using medians computed strictly within each training fold
   to prevent fold-to-fold data leakage, followed by standardisation.
3. Missingness audit and country-by-country diagnostic reporting (reproducing Table S4).

Usage:
    python 04_missing_data_handler.py --input-cohort analytic_cohort.parquet --output-report missingness_summary.csv
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PREDICTOR_NAMES = [
    'child_age_months', 'child_female', 'child_is_twin', 'birth_order', 'delivered_in_facility',
    'maternal_age', 'maternal_edu_years', 'marital_in_union', 'mother_employed',
    'wealth_quintile', 'wealth_score', 'residence_rural', 'household_size', 'u5_children_count',
    'has_electricity', 'has_radio', 'has_television', 'improved_water', 'improved_toilet', 'clean_cooking_fuel',
    'barrier_distance', 'barrier_money', 'barrier_permission', 'barrier_alone', 'has_insurance',
    'illness_fever', 'illness_diarrhea', 'illness_cough', 'illness_ari_rapid_breath', 'illness_ari_chest_problem',
    'illness_count'
]


class InFoldLinearPreprocessor(BaseEstimator, TransformerMixin):
    """
    Leak-free in-fold imputer and standardizer for linear/penalized models.
    Fits median values and standard scaling strictly on training partitions.
    """
    def __init__(self):
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        X_imp = self.imputer.fit_transform(X)
        self.scaler.fit(X_imp)
        return self

    def transform(self, X):
        X_imp = self.imputer.transform(X)
        return self.scaler.transform(X_imp)


class NativeTreePreprocessor(BaseEstimator, TransformerMixin):
    """
    Identity pass-through for tree-based models that natively handle NaNs.
    Ensures input is float formatted so NaNs are preserved cleanly.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.to_numpy(dtype=np.float32)
        return np.asarray(X, dtype=np.float32)


def generate_missingness_matrix(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Computes missingness percentage per feature by country (Table S4)."""
    rows = []
    countries = sorted(df['country_code'].unique())
    
    for c in countries:
        c_sub = df[df['country_code'] == c]
        cname = c_sub['country_name'].iloc[0] if 'country_name' in c_sub.columns else c
        row = {"country_code": c, "country_name": cname, "N": len(c_sub)}
        for f in features:
            if f in c_sub.columns:
                pct = (c_sub[f].isna().sum() / len(c_sub)) * 100
                row[f] = round(pct, 2)
            else:
                row[f] = 100.0
        rows.append(row)

    # Overall cohort row
    row_all = {"country_code": "ALL", "country_name": "Full Pooled Cohort", "N": len(df)}
    for f in features:
        if f in df.columns:
            pct = (df[f].isna().sum() / len(df)) * 100
            row_all[f] = round(pct, 2)
        else:
            row_all[f] = 100.0
    rows.append(row_all)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Audit predictor missingness and export summary matrix.")
    parser.add_argument("--input-cohort", type=str, required=True, help="Input analytic cohort path.")
    parser.add_argument("--output-report", type=str, default="missingness_matrix_by_country.csv", help="Output summary CSV.")
    args = parser.parse_args()

    input_p = Path(args.input_cohort)
    if input_p.suffix in ['.parquet', '.pq']:
        df = pd.read_parquet(input_p)
    else:
        df = pd.read_csv(input_p)

    df_miss = generate_missingness_matrix(df, PREDICTOR_NAMES)
    df_miss.to_csv(args.output_report, index=False)
    logger.info(f"Missingness audit complete across {len(df['country_code'].unique())} countries. Saved to {args.output_report}")

    overall_row = df_miss[df_miss['country_code'] == 'ALL'].iloc[0]
    high_missing = [f for f in PREDICTOR_NAMES if overall_row[f] > 5.0]
    logger.info(f"Features with >5% missingness overall: {high_missing}")


if __name__ == "__main__":
    main()
