#!/usr/bin/env python3
"""
03_engineer_predictors.py - Candidate Predictor Engineering

Engineers the 31 candidate predictors across five substantive domains
from standardized DHS questionnaire items:

1. Child Demographics (5):
   - child_age_months (B19 or B8*12+6)
   - child_female (B4 == 2)
   - child_is_twin (B0 > 0)
   - birth_order (BORD)
   - delivered_in_facility (M15 in facility codes)

2. Maternal Characteristics (4):
   - maternal_age (V012)
   - maternal_edu_years (V133)
   - marital_in_union (V502 == 1)
   - mother_employed (V714 == 1)

3. Household Socioeconomic & Environmental Factors (11):
   - wealth_quintile (V190)
   - wealth_score (V191 / 100,000)
   - residence_rural (V025 == 2)
   - household_size (V136)
   - u5_children_count (V137)
   - has_electricity (V119 == 1)
   - has_radio (V120 == 1)
   - has_television (V121 == 1)
   - improved_water (V113 in WHO/UNICEF JMP improved sources)
   - improved_toilet (V116 in WHO/UNICEF JMP improved facilities)
   - clean_cooking_fuel (V161 in clean fuels: electricity, gas, LPG, biogas)

4. Perceived Healthcare Access Barriers (5):
   - barrier_distance (V467B == 1)
   - barrier_money (V467C == 1)
   - barrier_permission (V467D == 1)
   - barrier_alone (V467E == 1)
   - has_insurance (V481 == 1)

5. Clinical Illness Presentation (6):
   - illness_fever (H22 == 1)
   - illness_diarrhea (H11 in [1, 2])
   - illness_cough (H31 in [1, 2])
   - illness_ari_rapid_breath (H31B == 1)
   - illness_ari_chest_problem (H31C in [1, 3])
   - illness_count (sum of fever, diarrhea, cough)

Usage:
    python 03_engineer_predictors.py --input-cohort cohort_with_outcome.parquet --output-features analytic_cohort.parquet
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Complete list of 31 candidate predictors
PREDICTOR_NAMES = [
    'child_age_months', 'child_female', 'child_is_twin', 'birth_order', 'delivered_in_facility',
    'maternal_age', 'maternal_edu_years', 'marital_in_union', 'mother_employed',
    'wealth_quintile', 'wealth_score', 'residence_rural', 'household_size', 'u5_children_count',
    'has_electricity', 'has_radio', 'has_television', 'improved_water', 'improved_toilet', 'clean_cooking_fuel',
    'barrier_distance', 'barrier_money', 'barrier_permission', 'barrier_alone', 'has_insurance',
    'illness_fever', 'illness_diarrhea', 'illness_cough', 'illness_ari_rapid_breath', 'illness_ari_chest_problem',
    'illness_count'
]


def engineer_predictors(df: pd.DataFrame) -> pd.DataFrame:
    """Derives the 31 pre-treatment predictors from raw/harmonized columns."""
    df_out = df.copy()

    # 1. Child Demographics
    if 'B19' in df.columns and df['B19'].notna().any():
        df_out['child_age_months'] = df['B19']
    elif 'child_age_months' not in df.columns:
        df_out['child_age_months'] = df['B8'] * 12 + 6 if 'B8' in df.columns else np.nan

    if 'B4' in df.columns:
        df_out['child_female'] = (df['B4'] == 2).astype(int)
    if 'B0' in df.columns:
        df_out['child_is_twin'] = (df['B0'] > 0).astype(int)
    if 'BORD' in df.columns:
        df_out['birth_order'] = df['BORD']

    # Facility delivery (M15)
    if 'M15' in df.columns:
        # Standard DHS codes: 20-39 represent hospital, health center, clinic, maternity
        df_out['delivered_in_facility'] = df['M15'].apply(
            lambda x: 1 if (20 <= x <= 39 or x in [21, 22, 23, 24, 25, 26, 31, 32, 33]) else (0 if pd.notna(x) else np.nan)
        )

    # 2. Maternal Characteristics
    if 'V012' in df.columns:
        df_out['maternal_age'] = df['V012']
    if 'V133' in df.columns:
        df_out['maternal_edu_years'] = df['V133']
    if 'V502' in df.columns:
        df_out['marital_in_union'] = (df['V502'] == 1).astype(int)
    if 'V714' in df.columns:
        df_out['mother_employed'] = (df['V714'] == 1).astype(int)

    # 3. Household Socioeconomic & Environmental Factors
    if 'V190' in df.columns:
        df_out['wealth_quintile'] = df['V190']
    if 'V191' in df.columns:
        df_out['wealth_score'] = df['V191'] / 100000.0
    if 'V025' in df.columns:
        df_out['residence_rural'] = (df['V025'] == 2).astype(int)
    if 'V136' in df.columns:
        df_out['household_size'] = df['V136']
    if 'V137' in df.columns:
        df_out['u5_children_count'] = df['V137']
    if 'V119' in df.columns:
        df_out['has_electricity'] = (df['V119'] == 1).astype(int)
    if 'V120' in df.columns:
        df_out['has_radio'] = (df['V120'] == 1).astype(int)
    if 'V121' in df.columns:
        df_out['has_television'] = (df['V121'] == 1).astype(int)

    # Improved water (V113) per WHO/UNICEF JMP
    if 'V113' in df.columns:
        df_out['improved_water'] = df['V113'].apply(
            lambda x: 1 if (10 <= x <= 70 and x not in [32, 42, 43]) else (0 if pd.notna(x) else np.nan)
        )

    # Improved sanitation (V116) per WHO/UNICEF JMP
    if 'V116' in df.columns:
        df_out['improved_toilet'] = df['V116'].apply(
            lambda x: 1 if (10 <= x <= 23 or x in [11, 12, 13, 14, 15, 21, 22]) else (0 if pd.notna(x) else np.nan)
        )

    # Clean cooking fuel (V161)
    if 'V161' in df.columns:
        df_out['clean_cooking_fuel'] = df['V161'].apply(
            lambda x: 1 if x in [1, 2, 3, 4] else (0 if pd.notna(x) else np.nan)
        )

    # 4. Perceived Healthcare Access Barriers
    if 'V467B' in df.columns:
        df_out['barrier_distance'] = (df['V467B'] == 1).astype(int)
    if 'V467C' in df.columns:
        df_out['barrier_money'] = (df['V467C'] == 1).astype(int)
    if 'V467D' in df.columns:
        df_out['barrier_permission'] = (df['V467D'] == 1).astype(int)
    if 'V467E' in df.columns:
        df_out['barrier_alone'] = (df['V467E'] == 1).astype(int)
    if 'V481' in df.columns:
        df_out['has_insurance'] = (df['V481'] == 1).astype(int)

    # 5. Clinical Illness Presentation
    if 'H22' in df.columns:
        df_out['illness_fever'] = (df['H22'] == 1).astype(int)
    if 'H11' in df.columns:
        df_out['illness_diarrhea'] = (df['H11'].isin([1, 2])).astype(int)
    if 'H31' in df.columns:
        df_out['illness_cough'] = (df['H31'].isin([1, 2])).astype(int)
    if 'H31B' in df.columns:
        df_out['illness_ari_rapid_breath'] = (df['H31B'] == 1).astype(int)
    if 'H31C' in df.columns:
        df_out['illness_ari_chest_problem'] = (df['H31C'].isin([1, 3])).astype(int)
        
    df_out['illness_count'] = (
        df_out['illness_fever'].fillna(0) +
        df_out['illness_diarrhea'].fillna(0) +
        df_out['illness_cough'].fillna(0)
    ).astype(int)

    # Validate presence of all 31 candidate predictors
    missing_preds = [p for p in PREDICTOR_NAMES if p not in df_out.columns]
    if missing_preds:
        logger.warning(f"Note: Predictors not found in raw extract: {missing_preds}")

    logger.info(f"Engineered {len(PREDICTOR_NAMES)} candidate predictors.")
    return df_out


def main():
    parser = argparse.ArgumentParser(description="Engineer 31 pre-treatment candidate predictors.")
    parser.add_argument("--input-cohort", type=str, required=True, help="Input cohort path.")
    parser.add_argument("--output-cohort", type=str, default="analytic_cohort.parquet", help="Output path.")
    args = parser.parse_args()

    input_p = Path(args.input_cohort)
    if input_p.suffix in ['.parquet', '.pq']:
        df = pd.read_parquet(input_p)
    else:
        df = pd.read_csv(input_p)

    df_feats = engineer_predictors(df)

    if args.output_cohort.endswith(".parquet"):
        df_feats.to_parquet(args.output_cohort, index=False)
    else:
        df_feats.to_csv(args.output_cohort, index=False)

    logger.info(f"Saved analytic cohort ({len(df_feats):,} rows, {len(df_feats.columns)} columns) to {args.output_cohort}")


if __name__ == "__main__":
    main()
