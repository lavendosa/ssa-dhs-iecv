#!/usr/bin/env python3
"""
iecv_validation.py - 38-Country Internal-External Cross-Validation (IECV)

Executes leave-one-country-out internal-external cross-validation across all 38 nations:
1. Iteratively holds out one complete national DHS survey as an unseen external validation set.
2. Trains the primary LightGBM model on the pooled data of the remaining 37 nations.
3. Generates out-of-sample predictions for the held-out nation.
4. Computes:
   - Country-specific AUROC with parametric Hanley-McNeil (1982) SEs and 95% CIs.
   - Country-specific AUPRC, Brier score, and Expected Calibration Error (ECE).
   - Logistic calibration intercept and slope (ideal = 0.0 and 1.0).
5. Computes the unweighted cross-national arithmetic mean AUROC and its 95% confidence interval
   reflecting cross-national health system heterogeneity:
   Mean AUROC = 0.6967 (95% CI: 0.6690 to 0.7245), Mean slope = 0.8860 (95% CI: 0.7670 to 1.0040).

Reproduces Table 3 and Table S6.

Usage:
    python iecv_validation.py --input-cohort analytic_cohort.parquet --output-table table3_iecv.csv
"""

import time
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np

from modelling.lightgbm_model import get_lightgbm_model
from evaluation.metrics_util import (
    hanley_mcneil_ci, cross_national_summary_ci,
    calculate_calibration_slope_intercept, calculate_ece
)
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

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


def run_38country_iecv(
    df: pd.DataFrame,
    features: list[str] = PREDICTOR_NAMES,
    target: str = TARGET_COL,
    random_seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Runs 38-country leave-one-country-out IECV using LightGBM."""
    countries = sorted(df['country_code'].unique())
    n_countries = len(countries)
    logger.info(f"Starting {n_countries}-country leave-one-country-out IECV on N = {len(df):,}...")

    results = []
    all_preds_list = []
    t_start = time.time()

    for idx, c_code in enumerate(countries, 1):
        t_fold = time.time()
        c_mask = (df['country_code'] == c_code)
        df_dev = df[~c_mask]
        df_test = df[c_mask]

        X_dev = df_dev[features]
        y_dev = df_dev[target].values
        X_test = df_test[features]
        y_test = df_test[target].values

        c_name = df_test['country_name'].iloc[0] if 'country_name' in df_test.columns else c_code
        c_wave = df_test['survey_wave'].iloc[0] if 'survey_wave' in df_test.columns else "Unknown"

        # Fit operational LightGBM model
        model = get_lightgbm_model(random_state=random_seed)
        model.fit(X_dev, y_dev)

        y_prob = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        n_test = len(y_test)
        n_unmet = int(np.sum(y_test))
        n_met = n_test - n_unmet
        prev_pct = round((n_unmet / n_test) * 100.0, 1)

        auroc = roc_auc_score(y_test, y_prob)
        auprc = average_precision_score(y_test, y_prob)
        brier = brier_score_loss(y_test, y_prob)
        ece = calculate_ece(y_test, y_prob)
        cal_int, cal_slope = calculate_calibration_slope_intercept(y_test, y_prob)
        ci_lo, ci_hi, se_auc = hanley_mcneil_ci(auroc, n_unmet, n_met)

        results.append({
            "ISO/DHS Code": c_code,
            "Country": c_name,
            "DHS Wave": c_wave,
            "Sample Size (N)": n_test,
            "Unmet Need Events": n_unmet,
            "Prevalence (%)": prev_pct,
            "Out-of-Country AUROC": round(auroc, 4),
            "AUROC_95CI": f"{auroc:.3f} ({ci_lo:.3f}–{ci_hi:.3f})",
            "Out-of-Country AUPRC": round(auprc, 4),
            "Brier Score": round(brier, 4),
            "Calibration Intercept": round(cal_int, 4),
            "Calibration Slope": round(cal_slope, 4),
            "ECE": round(ece, 4)
        })

        # Save predictions for plotting
        pred_sub = df_test[['country_code']].copy()
        pred_sub['y_true'] = y_test
        pred_sub['y_prob'] = y_prob
        all_preds_list.append(pred_sub)

        logger.info(
            f"[{idx:02d}/{n_countries}] {c_code} ({c_name}): N={n_test:,}, "
            f"Events={n_unmet:,} ({prev_pct}%), AUROC={auroc:.4f}, Slope={cal_slope:.3f} "
            f"({time.time()-t_fold:.1f}s)"
        )

    df_results = pd.DataFrame(results)
    df_all_preds = pd.concat(all_preds_list, ignore_index=True)

    # Compute unweighted cross-national summary row
    mean_auc, auc_lo, auc_hi, _ = cross_national_summary_ci(df_results["Out-of-Country AUROC"])
    mean_auprc, _, _, _ = cross_national_summary_ci(df_results["Out-of-Country AUPRC"])
    mean_brier, _, _, _ = cross_national_summary_ci(df_results["Brier Score"])
    mean_slope, slope_lo, slope_hi, _ = cross_national_summary_ci(df_results["Calibration Slope"])
    mean_int, int_lo, int_hi, _ = cross_national_summary_ci(df_results["Calibration Intercept"])
    mean_ece, _, _, _ = cross_national_summary_ci(df_results["ECE"])

    summary_row = {
        "ISO/DHS Code": "Summary",
        "Country": f"Unweighted Mean ({n_countries} Countries)",
        "DHS Wave": "—",
        "Sample Size (N)": int(df_results["Sample Size (N)"].sum()),
        "Unmet Need Events": int(df_results["Unmet Need Events"].sum()),
        "Prevalence (%)": round(df_results["Unmet Need Events"].sum() / df_results["Sample Size (N)"].sum() * 100, 1),
        "Out-of-Country AUROC": round(mean_auc, 4),
        "AUROC_95CI": f"{mean_auc:.4f} ({auc_lo:.4f}–{auc_hi:.4f})",
        "Out-of-Country AUPRC": round(mean_auprc, 4),
        "Brier Score": round(mean_brier, 4),
        "Calibration Intercept": round(mean_int, 4),
        "Calibration Slope": round(mean_slope, 4),
        "ECE": round(mean_ece, 4)
    }

    df_full_table = pd.concat([df_results, pd.DataFrame([summary_row])], ignore_index=True)
    total_time = time.time() - t_start

    logger.info("=" * 80)
    logger.info(f"38-COUNTRY IECV COMPLETED IN {total_time:.1f}s")
    logger.info(f"Unweighted Mean AUROC: {mean_auc:.4f} (95% CI: {auc_lo:.4f} to {auc_hi:.4f})")
    logger.info(f"Unweighted Mean Calibration Slope: {mean_slope:.4f} (95% CI: {slope_lo:.4f} to {slope_hi:.4f})")
    logger.info(f"Unweighted Mean Calibration Intercept: {mean_int:.4f} (95% CI: {int_lo:.4f} to {int_hi:.4f})")
    logger.info("=" * 80)

    return df_full_table, df_all_preds


def main():
    parser = argparse.ArgumentParser(description="Run 38-Country IECV Validation.")
    parser.add_argument("--input-cohort", type=str, required=True, help="Input cohort path.")
    parser.add_argument("--output-table", type=str, default="table3_country_iecv_breakdown.csv", help="Output summary table CSV.")
    parser.add_argument("--output-preds", type=str, default="iecv_all_predictions.parquet", help="Output out-of-sample predictions.")
    args = parser.parse_args()

    input_p = Path(args.input_cohort)
    if input_p.suffix in ['.parquet', '.pq']:
        df = pd.read_parquet(input_p)
    else:
        df = pd.read_csv(input_p)

    df_table, df_preds = run_38country_iecv(df)
    df_table.to_csv(args.output_table, index=False)
    df_preds.to_parquet(args.output_preds, index=False)
    logger.info(f"Saved IECV results to {args.output_table}")


if __name__ == "__main__":
    main()
