#!/usr/bin/env python3
"""
cross_validation.py - 5-Fold Stratified Cross-Validation Benchmark

Benchmarks five candidate machine learning algorithms on the pooled cohort (N = 118,910):
- XGBoost
- HistGradientBoosting
- LightGBM
- Random Forest
- ElasticNet Logistic Regression

Evaluates:
- Discrimination: AUROC and AUPRC (with 5-fold mean and SD).
- Calibration: Brier score, Expected Calibration Error (ECE).
- Classification metrics at p >= 0.50: Accuracy, Sensitivity/Recall, Specificity, PPV, NPV, F1 Score.
- Training runtimes in seconds.

Outputs:
- Benchmark summary table reproducing Table 2.
- Out-of-fold (OOF) predicted probability arrays.

Usage:
    python cross_validation.py --input-cohort analytic_cohort.parquet --output-table table2_benchmark.csv
"""

import time
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

from modelling.model_factory import get_all_benchmark_models
from evaluation.metrics_util import calculate_ece

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


def run_5fold_cross_validation(
    df: pd.DataFrame,
    features: list[str] = PREDICTOR_NAMES,
    target: str = TARGET_COL,
    fast_rf: bool = False,
    random_seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Executes 5-fold stratified cross-validation across all benchmark algorithms."""
    X = df[features].copy()
    y = df[target].values

    models = get_all_benchmark_models(random_state=random_seed, fast_rf=fast_rf)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)

    benchmark_rows = []
    oof_df = pd.DataFrame(index=df.index)

    logger.info(f"Starting 5-fold stratified cross-validation on N = {len(df):,} records...")

    for model_name, model in models.items():
        logger.info(f"--- Training {model_name} ---")
        t0 = time.time()

        auroc_folds = []
        auprc_folds = []
        brier_folds = []
        acc_folds = []
        rec_folds = []
        f1_folds = []
        spec_folds = []
        ppv_folds = []

        oof_probs = np.zeros(len(df))

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            model.fit(X_tr, y_tr)
            preds_prob = model.predict_proba(X_val)[:, 1]
            oof_probs[val_idx] = preds_prob

            preds_bin = (preds_prob >= 0.50).astype(int)

            auroc = roc_auc_score(y_val, preds_prob)
            auprc = average_precision_score(y_val, preds_prob)
            brier = brier_score_loss(y_val, preds_prob)
            acc = accuracy_score(y_val, preds_bin)
            rec = recall_score(y_val, preds_bin)
            f1 = f1_score(y_val, preds_bin)

            tn, fp, fn, tp = confusion_matrix(y_val, preds_bin).ravel()
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0

            auroc_folds.append(auroc)
            auprc_folds.append(auprc)
            brier_folds.append(brier)
            acc_folds.append(acc)
            rec_folds.append(rec)
            f1_folds.append(f1)
            spec_folds.append(spec)
            ppv_folds.append(ppv)

        elapsed = time.time() - t0
        oof_df[model_name] = oof_probs

        summary = {
            "Algorithm": model_name,
            "AUROC (Mean)": round(float(np.mean(auroc_folds)), 6),
            "AUROC (SD)": round(float(np.std(auroc_folds, ddof=1)), 6),
            "AUPRC (Mean)": round(float(np.mean(auprc_folds)), 6),
            "AUPRC (SD)": round(float(np.std(auprc_folds, ddof=1)), 6),
            "Brier Score": round(float(np.mean(brier_folds)), 6),
            "Accuracy": round(float(np.mean(acc_folds)), 6),
            "Sensitivity / Recall": round(float(np.mean(rec_folds)), 6),
            "Specificity": round(float(np.mean(spec_folds)), 6),
            "Positive Predictive Value": round(float(np.mean(ppv_folds)), 6),
            "F1 Score": round(float(np.mean(f1_folds)), 6),
            "Training Time (sec)": round(elapsed, 1)
        }
        benchmark_rows.append(summary)
        logger.info(
            f"Finished {model_name}: AUROC = {summary['AUROC (Mean)']:.4f} "
            f"(SD {summary['AUROC (SD)']:.4f}), AUPRC = {summary['AUPRC (Mean)']:.4f} "
            f"in {elapsed:.1f}s"
        )

    results_df = pd.DataFrame(benchmark_rows).sort_values("AUROC (Mean)", ascending=False)
    return results_df, oof_df


def main():
    parser = argparse.ArgumentParser(description="Run 5-Fold Cross-Validation on pooled cohort.")
    parser.add_argument("--input-cohort", type=str, required=True, help="Input cohort path.")
    parser.add_argument("--output-table", type=str, default="table2_model_comparison.csv", help="Output summary table CSV.")
    parser.add_argument("--output-oof", type=str, default="oof_predictions.parquet", help="Output OOF predictions Parquet.")
    parser.add_argument("--fast-rf", action="store_true", help="Use 100 trees for Random Forest to accelerate runtime.")
    args = parser.parse_args()

    input_p = Path(args.input_cohort)
    if input_p.suffix in ['.parquet', '.pq']:
        df = pd.read_parquet(input_p)
    else:
        df = pd.read_csv(input_p)

    results_df, oof_df = run_5fold_cross_validation(df, fast_rf=args.fast_rf)
    results_df.to_csv(args.output_table, index=False)
    oof_df.to_parquet(args.output_oof, index=False)

    logger.info("=" * 70)
    logger.info("5-FOLD STRATIFIED CROSS-VALIDATION SUMMARY (TABLE 2):")
    logger.info("=" * 70)
    print(results_df.to_string(index=False))
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
