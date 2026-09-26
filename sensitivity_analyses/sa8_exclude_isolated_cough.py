#!/usr/bin/env python3
"""
sa8_exclude_isolated_cough.py - Sensitivity Analysis 8: Exclusion of Isolated Mild Cough

Excludes children presenting with uncomplicated, isolated mild cough:
defined as acute cough without fever, diarrhea, rapid breathing, or chest-related breathing difficulty:
(illness_cough == 1) & (illness_fever == 0) & (illness_diarrhea == 0) &
(illness_ari_rapid_breath == 0) & (illness_ari_chest_problem == 0).

Key Clinical Background:
- In the primary cohort (N = 118,910), 19,656 children presented with isolated mild cough.
- Caregivers in SSA rarely seek facility care for uncomplicated mild cough in the absence of
  systemic danger signs: empirical unmet healthcare need in this sub-cohort is 96.91% (19,048 / 19,656).
- Retained cohort size: N = 99,254; Unmet need events: 51,487 (51.9%).
- Eliminating this stratum isolates the acute illness episodes requiring triage and aligns
  discrimination with the fever (0.664) and diarrhea (0.684) sub-cohorts.

Evaluates:
- 5-Fold Stratified Cross-Validation on the retained cohort.
- 38-Country Leave-One-Country-Out IECV on the retained cohort.
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from modelling.lightgbm_model import get_lightgbm_model
from evaluation.metrics_util import cross_national_summary_ci, calculate_calibration_slope_intercept

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


def run_sa8(df: pd.DataFrame, run_iecv: bool = False, random_seed: int = 42) -> tuple[dict, dict | None]:
    """Executes SA-8 excluding isolated mild cough."""
    cond_isolated_cough = (
        (df['illness_cough'] == 1) &
        (df['illness_fever'] == 0) &
        (df['illness_diarrhea'] == 0) &
        (df['illness_ari_rapid_breath'] == 0) &
        (df['illness_ari_chest_problem'] == 0)
    )
    mask_retained = ~cond_isolated_cough
    df_retained = df[mask_retained].copy().reset_index(drop=True)

    n_total = len(df)
    n_isolated = int(cond_isolated_cough.sum())
    n_retained = len(df_retained)
    events_isolated = int(df.loc[cond_isolated_cough, TARGET_COL].sum())
    events_retained = int(df_retained[TARGET_COL].sum())
    pct_retained = round(events_retained / n_retained * 100.0, 1)

    logger.info("=" * 70)
    logger.info("SA-8: ISOLATED MILD COUGH AUDIT & EXCLUSION")
    logger.info(f"Total Cohort: N = {n_total:,}")
    logger.info(f"Isolated Mild Cough: N = {n_isolated:,} ({n_isolated/n_total*100:.2f}%)")
    logger.info(f" - Unmet Events in Isolated Cough: {events_isolated:,} ({events_isolated/n_isolated*100:.2f}%)")
    logger.info(f"Retained Analytic Cohort: N = {n_retained:,} ({n_retained/n_total*100:.2f}%)")
    logger.info(f" - Unmet Events in Retained Cohort: {events_retained:,} ({pct_retained}%)")
    logger.info("=" * 70)

    # 1. 5-Fold Stratified Cross-Validation on Retained Cohort
    X_sub = df_retained[PREDICTOR_NAMES]
    y_sub = df_retained[TARGET_COL].values

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    oof_probs = np.zeros(n_retained)

    for train_idx, val_idx in cv.split(X_sub, y_sub):
        clf = get_lightgbm_model(random_state=random_seed)
        clf.fit(X_sub.iloc[train_idx], y_sub[train_idx])
        oof_probs[val_idx] = clf.predict_proba(X_sub.iloc[val_idx])[:, 1]

    auroc_cv = roc_auc_score(y_sub, oof_probs)
    auprc_cv = average_precision_score(y_sub, oof_probs)

    summary_cv = {
        "Sensitivity_Analysis": "SA-8: Cohort excluding isolated mild cough (5-Fold CV)",
        "Cohort_N": n_retained,
        "Unmet_Events": events_retained,
        "Unmet_Pct": pct_retained,
        "AUROC": round(auroc_cv, 3),
        "AUPRC": round(auprc_cv, 3),
        "Evaluation_Scheme": "5-Fold Stratified Out-of-Fold (OOF)"
    }
    logger.info(f"SA-8 5-Fold CV Complete: AUROC = {auroc_cv:.3f}, AUPRC = {auprc_cv:.3f}")

    summary_iecv = None
    if run_iecv:
        logger.info(f"Executing 38-Country IECV on retained cohort (N = {n_retained:,})...")
        countries = sorted(df_retained['country_code'].unique())
        country_aurocs, country_auprcs = [], []

        for c_code in countries:
            c_mask = (df_retained['country_code'] == c_code)
            df_dev = df_retained[~c_mask]
            df_test = df_retained[c_mask]

            X_dev = df_dev[PREDICTOR_NAMES]
            y_dev = df_dev[TARGET_COL].values
            X_test = df_test[PREDICTOR_NAMES]
            y_test = df_test[TARGET_COL].values

            clf = get_lightgbm_model(random_state=random_seed)
            clf.fit(X_dev, y_dev)
            preds = clf.predict_proba(X_test)[:, 1]

            country_aurocs.append(roc_auc_score(y_test, preds))
            country_auprcs.append(average_precision_score(y_test, preds))

        mean_auc, ci_lo, ci_hi, _ = cross_national_summary_ci(country_aurocs)
        mean_pr, _, _, _ = cross_national_summary_ci(country_auprcs)

        summary_iecv = {
            "Sensitivity_Analysis": "SA-8: Cohort excluding isolated mild cough (38-Country IECV)",
            "Cohort_N": n_retained,
            "Unmet_Events": events_retained,
            "Unmet_Pct": pct_retained,
            "AUROC": round(mean_auc, 3),
            "AUPRC": round(mean_pr, 3),
            "Evaluation_Scheme": "38-Country Leave-One-Country-Out IECV"
        }
        logger.info(f"SA-8 38-Country IECV Complete: Mean AUROC = {mean_auc:.3f} (95% CI: {ci_lo:.3f} to {ci_hi:.3f})")

    return summary_cv, summary_iecv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-cohort", type=str, required=True)
    parser.add_argument("--run-iecv", action="store_true")
    args = parser.parse_args()
    df = pd.read_parquet(args.input_cohort) if args.input_cohort.endswith('.parquet') else pd.read_csv(args.input_cohort)
    s_cv, s_iecv = run_sa8(df, run_iecv=args.run_iecv)
    print(s_cv)
    if s_iecv:
        print(s_iecv)


if __name__ == "__main__":
    main()
