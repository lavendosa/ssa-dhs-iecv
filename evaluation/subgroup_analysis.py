#!/usr/bin/env python3
"""
subgroup_analysis.py - Subgroup Discrimination and Healthcare Access Equity

Evaluates model discrimination and calibration across demographic, equity,
and clinical presentation subgroups (Table 4):
- Wealth Quintiles: Poorest (Q1) to Richest (Q5)
- Place of Residence: Urban vs Rural
- Maternal Education: No formal (0y), Primary (1-6y), Secondary or higher (7+y)
- Child Age Group: Infants (0-11m), Toddlers (12-23m), Young children (24-59m)
- Child Sex: Male vs Female
- Clinical Illness Presentation: Single illness, Multimorbidity (>=2 illnesses),
  Fever-present, Diarrhea-present, Acute cough, Isolated mild cough.

Usage:
    python subgroup_analysis.py --input-cohort analytic_cohort.parquet --input-oof oof_predictions.parquet --output-table table4_subgroup_analysis.csv
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

TARGET_COL = 'unmet_need'


def evaluate_subgroups(df: pd.DataFrame, pred_probs: np.ndarray, target: str = TARGET_COL) -> pd.DataFrame:
    """Evaluates discrimination metrics across all prespecified subgroups."""
    df = df.copy()
    df['pred_prob'] = pred_probs

    # Define subgroup partitions matching Table 4
    subgroups = {
        "Wealth Quintile": {
            "Poorest (Q1)": df['wealth_quintile'] == 1,
            "Poorer (Q2)": df['wealth_quintile'] == 2,
            "Middle (Q3)": df['wealth_quintile'] == 3,
            "Richer (Q4)": df['wealth_quintile'] == 4,
            "Richest (Q5)": df['wealth_quintile'] == 5,
        },
        "Place of Residence": {
            "Urban": df['residence_rural'] == 0,
            "Rural": df['residence_rural'] == 1,
        },
        "Maternal Education": {
            "No formal education (0 yrs)": df['maternal_edu_years'] == 0,
            "Primary (1-6 yrs)": (df['maternal_edu_years'] >= 1) & (df['maternal_edu_years'] <= 6),
            "Secondary or higher (7+ yrs)": df['maternal_edu_years'] >= 7,
        },
        "Child Age Group": {
            "Infants (0-11 months)": df['child_age_months'] < 12,
            "Toddlers (12-23 months)": (df['child_age_months'] >= 12) & (df['child_age_months'] < 24),
            "Young children (24-59 months)": df['child_age_months'] >= 24,
        },
        "Child Sex": {
            "Male": df['child_female'] == 0,
            "Female": df['child_female'] == 1,
        },
        "Illness Presentation": {
            "Single illness": df['illness_count'] == 1,
            "Multimorbidity (>=2 illnesses)": df['illness_count'] >= 2,
            "Fever-present episode": df['illness_fever'] == 1,
            "Diarrhea-present episode": df['illness_diarrhea'] == 1,
            "Acute cough episode": df['illness_cough'] == 1,
            "Isolated mild cough": (
                (df['illness_cough'] == 1) & (df['illness_fever'] == 0) &
                (df['illness_diarrhea'] == 0) & (df['illness_ari_rapid_breath'] == 0) &
                (df['illness_ari_chest_problem'] == 0)
            )
        }
    }

    records = []
    for domain, grps in subgroups.items():
        for sub_name, mask in grps.items():
            sub = df[mask].dropna(subset=[target, 'pred_prob'])
            n_sub = len(sub)
            if n_sub == 0:
                continue

            y_true = sub[target].values
            p_hat = sub['pred_prob'].values
            n_events = int(np.sum(y_true))
            unmet_rate = (n_events / n_sub) * 100.0 if n_sub > 0 else 0.0

            # Compute discrimination and calibration
            try:
                auroc = roc_auc_score(y_true, p_hat)
            except Exception:
                auroc = np.nan

            try:
                auprc = average_precision_score(y_true, p_hat)
            except Exception:
                auprc = np.nan

            brier = brier_score_loss(y_true, p_hat)

            records.append({
                "Domain": domain,
                "Subgroup": sub_name,
                "N": n_sub,
                "Unmet_Events": n_events,
                "Unmet_Rate_Pct": round(unmet_rate, 1),
                "AUROC": round(auroc, 3) if not np.isnan(auroc) else "—",
                "AUPRC": round(auprc, 3) if not np.isnan(auprc) else "—",
                "Brier_Score": round(brier, 3)
            })

    results_df = pd.DataFrame(records)
    return results_df


def main():
    parser = argparse.ArgumentParser(description="Evaluate subgroup performance across equity and clinical strata.")
    parser.add_argument("--input-cohort", type=str, required=True, help="Input cohort path.")
    parser.add_argument("--input-oof", type=str, required=True, help="Input OOF predictions path.")
    parser.add_argument("--output-table", type=str, default="table4_subgroup_analysis.csv", help="Output CSV path.")
    args = parser.parse_args()

    input_p = Path(args.input_cohort)
    oof_p = Path(args.input_oof)

    df = pd.read_parquet(input_p) if input_p.suffix in ['.parquet', '.pq'] else pd.read_csv(input_p)
    df_oof = pd.read_parquet(oof_p) if oof_p.suffix in ['.parquet', '.pq'] else pd.read_csv(oof_p)

    # Use LightGBM predictions if available, else first column
    prob_col = 'LightGBM' if 'LightGBM' in df_oof.columns else df_oof.columns[0]
    preds = df_oof[prob_col].values

    res = evaluate_subgroups(df, preds)
    res.to_csv(args.output_table, index=False)

    logger.info("=" * 70)
    logger.info("SUBGROUP DISCRIMINATION AND EQUITY BREAKDOWN (TABLE 4):")
    logger.info("=" * 70)
    print(res.to_string(index=False))
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
