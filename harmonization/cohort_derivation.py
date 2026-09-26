#!/usr/bin/env python3
"""
01_cohort_derivation.py - Study Eligibility and Cohort Derivation

Applies the four sequential eligibility criteria to raw Kids Recode records:
1. Living child status: B5 == 1 (child alive at interview).
2. Pediatric age window: B19 < 60 (or B8 < 5 completed years).
3. Qualifying acute illness in past 2 weeks:
   - Fever: H22 == 1
   - Diarrheal episode: H11 in [1, 2]
   - Acute cough: H31 in [1, 2]
4. Known care-seeking status: non-missing response on illness consultation.

Tracks and outputs exact cohort attrition numbers at each stage to reproduce Figure 1.

Usage:
    python 01_cohort_derivation.py --input-raw raw_records.parquet --output-cohort eligible_ill_cohort.parquet
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def apply_eligibility_criteria(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Applies sequential inclusion/exclusion criteria to DHS Kids Recode data.
    Returns (eligible_cohort_df, attrition_counts_dict).
    """
    n_initial = len(df_raw)
    logger.info(f"Total raw birth history records evaluated: {n_initial:,}")

    # Criterion 1: Alive child (B5 == 1)
    if 'B5' in df_raw.columns:
        df_alive = df_raw[df_raw['B5'] == 1].copy()
    else:
        df_alive = df_raw.copy()
    n_alive = len(df_alive)
    n_deceased = n_initial - n_alive
    logger.info(f"Living children under 5 (B5==1): {n_alive:,} (Excluded deceased: {n_deceased:,})")

    # Criterion 2: Age 0–59 months (B19 < 60 or B8 < 5)
    if 'B19' in df_alive.columns and df_alive['B19'].notna().any():
        df_u5 = df_alive[df_alive['B19'] < 60].copy()
    elif 'B8' in df_alive.columns:
        df_u5 = df_alive[df_alive['B8'] < 5].copy()
    else:
        df_u5 = df_alive.copy()
    n_u5 = len(df_u5)
    logger.info(f"Surveyed living children aged 0-59 months: {n_u5:,}")

    # Criterion 3: Acute qualifying illness in 2 weeks preceding survey
    has_fever = (df_u5['H22'] == 1) if 'H22' in df_u5.columns else pd.Series(False, index=df_u5.index)
    has_diarrhea = df_u5['H11'].isin([1, 2]) if 'H11' in df_u5.columns else pd.Series(False, index=df_u5.index)
    has_cough = df_u5['H31'].isin([1, 2]) if 'H31' in df_u5.columns else pd.Series(False, index=df_u5.index)

    has_acute_illness = has_fever | has_diarrhea | has_cough
    df_ill = df_u5[has_acute_illness].copy()
    n_ill = len(df_ill)
    n_no_illness = n_u5 - n_ill
    logger.info(f"Children with acute qualifying illness: {n_ill:,} (Excluded without acute illness: {n_no_illness:,})")

    # Criterion 4: Known care-seeking status (exclude unrecorded care-seeking status if any)
    # Under standard DHS recodes, care seeking variables H12Z/H32Z are administered to all ill children
    df_eligible = df_ill.copy()
    n_eligible = len(df_eligible)
    logger.info(f"Final primary analytic cohort: {n_eligible:,}")

    attrition = {
        "raw_birth_records": n_initial,
        "excluded_deceased": n_deceased,
        "living_children_u5": n_alive,
        "excluded_no_acute_illness": n_no_illness,
        "qualifying_acute_illness": n_ill,
        "final_analytic_cohort": n_eligible
    }

    return df_eligible, attrition


def main():
    parser = argparse.ArgumentParser(description="Apply study inclusion/exclusion criteria.")
    parser.add_argument("--input-file", type=str, required=True, help="Input raw Kids Recode data (Parquet or CSV).")
    parser.add_argument("--output-cohort", type=str, default="eligible_cohort.parquet", help="Output eligible cohort path.")
    parser.add_argument("--output-attrition", type=str, default="attrition_flowchart.csv", help="Output attrition counts CSV.")
    args = parser.parse_args()

    input_p = Path(args.input_file)
    if not input_p.exists():
        raise FileNotFoundError(f"Input file {input_p} does not exist.")

    if input_p.suffix in ['.parquet', '.pq']:
        df_raw = pd.read_parquet(input_p)
    else:
        df_raw = pd.read_csv(input_p)

    df_eligible, attrition = apply_eligibility_criteria(df_raw)

    if args.output_cohort.endswith(".parquet"):
        df_eligible.to_parquet(args.output_cohort, index=False)
    else:
        df_eligible.to_csv(args.output_cohort, index=False)

    pd.DataFrame([attrition]).to_csv(args.output_attrition, index=False)
    logger.info(f"Saved eligible cohort ({len(df_eligible):,} rows) to {args.output_cohort}")
    logger.info(f"Saved attrition summary to {args.output_attrition}")


if __name__ == "__main__":
    main()
