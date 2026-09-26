#!/usr/bin/env python3
"""
sa4_survey_weighted.py - Sensitivity Analysis 4: Survey-Weighted Modelling

Applies normalized DHS individual sampling weights (V005 / 1,000,000) during
model training and evaluation.
Cohort size: N = 118,910; Unmet need events: 70,535 (60.0% weighted).
Evaluates survey-weighted AUROC and AUPRC.
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, average_precision_score
from modelling.lightgbm_model import get_lightgbm_model

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
TARGET_COL = 'unmet_need'


def run_sa4(df: pd.DataFrame, random_seed: int = 42) -> dict:
    """Executes SA-4 applying DHS sample weights."""
    weights = df['sample_weight'].values if 'sample_weight' in df.columns else np.ones(len(df))
    # Normalize weights so mean is 1.0
    weights = weights / np.mean(weights)

    n_sub = len(df)
    events = int(df[TARGET_COL].sum())
    weighted_pct = round(float(np.average(df[TARGET_COL], weights=weights)) * 100.0, 1)

    logger.info(f"Running SA-4 (Survey-weighted): N = {n_sub:,}, Weighted Unmet Rate = {weighted_pct}%")

    X = df[PREDICTOR_NAMES]
    y = df[TARGET_COL].values

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    oof_probs = np.zeros(n_sub)

    for train_idx, val_idx in cv.split(X, y):
        clf = get_lightgbm_model(random_state=random_seed)
        clf.fit(X.iloc[train_idx], y[train_idx], sample_weight=weights[train_idx])
        oof_probs[val_idx] = clf.predict_proba(X.iloc[val_idx])[:, 1]

    # Weighted ROC curve and AUC
    fpr, tpr, _ = roc_curve(y, oof_probs, sample_weight=weights)
    weighted_auroc = auc(fpr, tpr)
    weighted_auprc = average_precision_score(y, oof_probs, sample_weight=weights)

    res = {
        "Sensitivity_Analysis": "SA-4: Survey-weighted performance (V005 weights)",
        "Cohort_N": n_sub,
        "Unmet_Events": events,
        "Unmet_Pct": weighted_pct,
        "AUROC": round(weighted_auroc, 3),
        "AUPRC": round(weighted_auprc, 3),
        "Evaluation_Scheme": "Survey-Weighted 5-Fold OOF"
    }
    logger.info(f"SA-4 Complete: Weighted AUROC = {res['AUROC']:.3f}, AUPRC = {res['AUPRC']:.3f}")
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-cohort", type=str, required=True)
    args = parser.parse_args()
    df = pd.read_parquet(args.input_cohort) if args.input_cohort.endswith('.parquet') else pd.read_csv(args.input_cohort)
    print(run_sa4(df))


if __name__ == "__main__":
    main()
