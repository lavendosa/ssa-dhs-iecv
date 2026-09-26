#!/usr/bin/env python3
"""
sa5_exclude_high_missingness.py - Sensitivity Analysis 5: Excluding High Missingness Features

Evaluates model performance after dropping features with >20% missingness:
specifically `delivered_in_facility` (22.7% missingness overall).
Cohort size: N = 118,910; Unmet need events: 70,535 (59.3%).
Evaluates 5-fold stratified cross-validated AUROC and AUPRC on the remaining 30 features.
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from modelling.lightgbm_model import get_lightgbm_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Features excluding delivered_in_facility (30 features)
PREDICTOR_NAMES_SA5 = [
    'child_age_months', 'child_female', 'child_is_twin', 'birth_order',
    'maternal_age', 'maternal_edu_years', 'marital_in_union', 'mother_employed',
    'wealth_quintile', 'wealth_score', 'residence_rural', 'household_size', 'u5_children_count',
    'has_electricity', 'has_radio', 'has_television', 'improved_water', 'improved_toilet', 'clean_cooking_fuel',
    'barrier_distance', 'barrier_money', 'barrier_permission', 'barrier_alone', 'has_insurance',
    'illness_fever', 'illness_diarrhea', 'illness_cough', 'illness_ari_rapid_breath', 'illness_ari_chest_problem',
    'illness_count'
]
TARGET_COL = 'unmet_need'


def run_sa5(df: pd.DataFrame, random_seed: int = 42) -> dict:
    """Executes SA-5 excluding delivered_in_facility."""
    n_sub = len(df)
    events = int(df[TARGET_COL].sum())
    pct = round(events / n_sub * 100.0, 1)

    logger.info(f"Running SA-5 (Excl. high missingness): N = {n_sub:,}, Features = {len(PREDICTOR_NAMES_SA5)}")

    X = df[PREDICTOR_NAMES_SA5]
    y = df[TARGET_COL].values

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    oof_probs = np.zeros(n_sub)

    for train_idx, val_idx in cv.split(X, y):
        clf = get_lightgbm_model(random_state=random_seed)
        clf.fit(X.iloc[train_idx], y[train_idx])
        oof_probs[val_idx] = clf.predict_proba(X.iloc[val_idx])[:, 1]

    auroc = roc_auc_score(y, oof_probs)
    auprc = average_precision_score(y, oof_probs)

    res = {
        "Sensitivity_Analysis": "SA-5: Model excluding features with >20% missingness",
        "Cohort_N": n_sub,
        "Unmet_Events": events,
        "Unmet_Pct": pct,
        "AUROC": round(auroc, 3),
        "AUPRC": round(auprc, 3),
        "Evaluation_Scheme": "5-Fold Stratified Out-of-Fold (OOF)"
    }
    logger.info(f"SA-5 Complete: AUROC = {res['AUROC']:.3f}, AUPRC = {res['AUPRC']:.3f}")
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-cohort", type=str, required=True)
    args = parser.parse_args()
    df = pd.read_parquet(args.input_cohort) if args.input_cohort.endswith('.parquet') else pd.read_csv(args.input_cohort)
    print(run_sa5(df))


if __name__ == "__main__":
    main()
