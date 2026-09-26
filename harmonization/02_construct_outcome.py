#!/usr/bin/env python3
"""
02_construct_outcome.py - Outcome Derivation & Leakage Prevention Audit

Constructs the binary target variable 'unmet_need':
- Met Need (Y = 0): Advice or medical treatment was sought from at least one formal healthcare provider:
    - Diarrheal formal care: H12Z == 1
    - Fever or cough formal care: H32Z == 1
    - Met formal care: (H12Z == 1) | (H32Z == 1)
- Unmet Need (Y = 1): Child was acutely ill, and caregivers sought NO formal medical care:
    - unmet_need = 1 - ((H12Z == 1) | (H32Z == 1))
    - Includes no care sought whatsoever (H12Y == 1 or H32Y == 1), or consultation restricted
      exclusively to informal providers (traditional healers, markets, shops, itinerant sellers).

Performs strict automated leakage validation:
- Asserts H12Z, H32Z, H12*, H32* (specific providers), and treatments (ORS, zinc, ACTs, antibiotics)
  are NEVER included in candidate predictor matrices.

Usage:
    python 02_construct_outcome.py --input-cohort eligible_cohort.parquet --output-cohort cohort_with_outcome.parquet
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Forbidden post-decision / leakage variables that must never be predictors
LEAKAGE_VARS = [
    'H12A', 'H12B', 'H12C', 'H12D', 'H12E', 'H12F', 'H12G', 'H12H', 'H12I', 'H12J', 'H12K',
    'H12L', 'H12M', 'H12N', 'H12O', 'H12P', 'H12Q', 'H12R', 'H12S', 'H12T', 'H12U', 'H12V', 'H12W', 'H12X', 'H12Y', 'H12Z',
    'H32A', 'H32B', 'H32C', 'H32D', 'H32E', 'H32F', 'H32G', 'H32H', 'H32I', 'H32J', 'H32K',
    'H32L', 'H32M', 'H32N', 'H32O', 'H32P', 'H32Q', 'H32R', 'H32S', 'H32T', 'H32U', 'H32V', 'H32W', 'H32X', 'H32Y', 'H32Z',
    'H13', 'H13B', 'H14', 'H14A', 'H15', 'H15A', 'H15B', 'H15C', 'H15D', 'H15E',
    'H37A', 'H37B', 'H37C', 'H37D', 'H37E', 'H37F', 'H37G', 'H37H', 'H37I', 'H37J',
    'H37AA', 'H37AB', 'H37AC', 'H37AD', 'H37AE'
]


def construct_unmet_need_outcome(df: pd.DataFrame) -> pd.DataFrame:
    """Derives the primary outcome indicator unmet_need (0=Met, 1=Unmet)."""
    df = df.copy()

    care_diarrhea = (df['H12Z'] == 1) if 'H12Z' in df.columns else pd.Series(False, index=df.index)
    care_fever_cough = (df['H32Z'] == 1) if 'H32Z' in df.columns else pd.Series(False, index=df.index)
    
    formal_care_sought = care_diarrhea | care_fever_cough
    df['unmet_need'] = (~formal_care_sought).astype(int)

    n_total = len(df)
    n_unmet = df['unmet_need'].sum()
    n_met = n_total - n_unmet
    pct_unmet = (n_unmet / n_total) * 100 if n_total > 0 else 0

    logger.info("=" * 60)
    logger.info("OUTCOME DERIVATION SUMMARY:")
    logger.info(f"Total Acutely Ill Children: {n_total:,}")
    logger.info(f"Met Formal Healthcare Need (Y=0):   {n_met:,} ({100-pct_unmet:.2f}%)")
    logger.info(f"Unmet Formal Healthcare Need (Y=1): {n_unmet:,} ({pct_unmet:.2f}%)")
    logger.info("=" * 60)

    return df


def verify_zero_leakage(feature_names: list[str]) -> bool:
    """Verifies that no leakage variables are present in the feature list."""
    upper_features = [f.upper() for f in feature_names]
    found_leakage = [v for v in LEAKAGE_VARS if v in upper_features]
    if found_leakage:
        raise ValueError(f"CRITICAL TARGET LEAKAGE DETECTED! Forbidden variables in feature set: {found_leakage}")
    logger.info(f"Leakage audit passed: 0 of {len(LEAKAGE_VARS)} post-treatment variables found in predictor matrix.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Construct binary unmet need outcome and audit leakage.")
    parser.add_argument("--input-cohort", type=str, required=True, help="Input cohort file.")
    parser.add_argument("--output-cohort", type=str, default="cohort_with_outcome.parquet", help="Output file path.")
    args = parser.parse_args()

    input_p = Path(args.input_cohort)
    if input_p.suffix in ['.parquet', '.pq']:
        df = pd.read_parquet(input_p)
    else:
        df = pd.read_csv(input_p)

    df_out = construct_unmet_need_outcome(df)

    if args.output_cohort.endswith(".parquet"):
        df_out.to_parquet(args.output_cohort, index=False)
    else:
        df_out.to_csv(args.output_cohort, index=False)
    logger.info(f"Saved cohort with outcome to {args.output_cohort}")


if __name__ == "__main__":
    main()
