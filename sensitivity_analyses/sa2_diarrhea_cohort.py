#!/usr/bin/env python3
"""
sa2_diarrhea_cohort.py - Sensitivity Analysis 2: Diarrhea-Present Episodes Only

Restricts analysis to children presenting with reported acute diarrhea (illness_diarrhea == 1).
Cohort size: N = 44,889; Unmet need events: 23,466 (52.3%).
Evaluates 5-fold stratified cross-validated AUROC and AUPRC.
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

PREDICTOR_NAMES = [
    'child_age_months', 'child_female', 'child_is_twin', 'birth_order', 'delivered_in_facility',
    'maternal_age', 'maternal_edu_years', 'marital_in_union', 'mother_employed',
    'wealth_quintile', 'wealth_score', 'residence_rural', 'household_size', 'u5_children_count',
    'has_electricity', 'has_radio', 'has_television', 'improved_water', 'improved_toilet', 'clean_cooking_fuel',
    'barrier_distance', 'barrier_money', 'barrier_permission', 'barrier_alone', 'has_insurance',
    'illness_fever', 'illness_cough', 'illness_ari_rapid_breath', 'illness_ari_chest_problem',
    'illness_count'
]
TARGET_COL = 'unmet_need'


def run_sa2(df: pd.DataFrame, random_seed: int = 42) -> dict:
    """Executes SA-2 on diarrhea-present illness episodes."""
    sub_df = df[df['illness_diarrhea'] == 1].copy().reset_index(drop=True)
    n_sub = len(sub_df)
    events = int(sub_df[TARGET_COL].sum())
    pct = round(events / n_sub * 100.0, 1)

    logger.info(f"Running SA-2 (Diarrhea cohort): N = {n_sub:,}, Unmet Events = {events:,} ({pct}%)")

    features = [f for f in PREDICTOR_NAMES if f in sub_df.columns]
    X = sub_df[features]
    y = sub_df[TARGET_COL].values

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    oof_probs = np.zeros(n_sub)

    for train_idx, val_idx in cv.split(X, y):
        clf = get_lightgbm_model(random_state=random_seed)
        clf.fit(X.iloc[train_idx], y[train_idx])
        oof_probs[val_idx] = clf.predict_proba(X.iloc[val_idx])[:, 1]

    auroc = roc_auc_score(y, oof_probs)
    auprc = average_precision_score(y, oof_probs)

    res = {
        "Sensitivity_Analysis": "SA-2: Diarrhea-present cohort",
        "Cohort_N": n_sub,
        "Unmet_Events": events,
        "Unmet_Pct": pct,
        "AUROC": round(auroc, 3),
        "AUPRC": round(auprc, 3),
        "Evaluation_Scheme": "5-Fold Stratified Out-of-Fold (OOF)"
    }
    logger.info(f"SA-2 Complete: AUROC = {res['AUROC']:.3f}, AUPRC = {res['AUPRC']:.3f}")
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-cohort", type=str, required=True)
    args = parser.parse_args()
    df = pd.read_parquet(args.input_cohort) if args.input_cohort.endswith('.parquet') else pd.read_csv(args.input_cohort)
    print(run_sa2(df))


if __name__ == "__main__":
    main()
