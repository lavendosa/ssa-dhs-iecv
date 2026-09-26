#!/usr/bin/env python3
"""
run_all_sensitivities.py - Sensitivity Analyses Orchestrator

Executes all eight prespecified sensitivity analyses:
1. SA-1: Fever-present illness episodes only.
2. SA-2: Diarrhea-present illness episodes only.
3. SA-3: Complete-case cohort (zero missing predictors).
4. SA-4: Survey-weighted modelling (applying DHS individual weights).
5. SA-5: Model excluding predictors with >20% missingness (delivered_in_facility).
6. SA-6: Restriction to 34 countries with >= 500 unmet need events.
7. SA-7: 38-Country IECV excluding 3 structurally missing features.
8. SA-8: Exclusion of children with isolated mild cough.

Assembles and exports the complete sensitivity analysis summary (reproducing Table S2).

Usage:
    python run_all_sensitivities.py --input-cohort analytic_cohort.parquet --output-table Table_S2.csv
"""

import argparse
import logging
from pathlib import Path
import pandas as pd

from sensitivity_analyses.sa1_fever_cohort import run_sa1
from sensitivity_analyses.sa2_diarrhea_cohort import run_sa2
from sensitivity_analyses.sa3_complete_case import run_sa3
from sensitivity_analyses.sa4_survey_weighted import run_sa4
from sensitivity_analyses.sa5_exclude_high_missingness import run_sa5
from sensitivity_analyses.sa6_large_event_countries import run_sa6
from sensitivity_analyses.sa7_structural_missingness_iecv import run_sa7
from sensitivity_analyses.sa8_exclude_isolated_cough import run_sa8

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def execute_all_sensitivities(df: pd.DataFrame, run_full_iecv: bool = False, random_seed: int = 42) -> pd.DataFrame:
    """Executes all 8 sensitivity analyses and compiles summary table."""
    records = []

    logger.info("Executing SA-1: Fever-present cohort...")
    records.append(run_sa1(df, random_seed=random_seed))

    logger.info("Executing SA-2: Diarrhea-present cohort...")
    records.append(run_sa2(df, random_seed=random_seed))

    logger.info("Executing SA-3: Complete-case cohort...")
    records.append(run_sa3(df, random_seed=random_seed))

    logger.info("Executing SA-4: Survey-weighted modelling...")
    records.append(run_sa4(df, random_seed=random_seed))

    logger.info("Executing SA-5: Excluding features with >20% missingness...")
    records.append(run_sa5(df, random_seed=random_seed))

    logger.info("Executing SA-6: Restricting to countries with >= 500 events...")
    records.append(run_sa6(df, random_seed=random_seed))

    logger.info("Executing SA-7: Structural-missingness IECV...")
    sa7_res, _ = run_sa7(df, random_seed=random_seed)
    records.append(sa7_res)

    logger.info("Executing SA-8: Excluding isolated mild cough...")
    sa8_cv, sa8_iecv = run_sa8(df, run_iecv=run_full_iecv, random_seed=random_seed)
    records.append(sa8_cv)
    if sa8_iecv:
        records.append(sa8_iecv)

    df_table = pd.DataFrame(records)
    return df_table


def main():
    parser = argparse.ArgumentParser(description="Run all 8 prespecified sensitivity analyses.")
    parser.add_argument("--input-cohort", type=str, required=True, help="Input cohort path.")
    parser.add_argument("--output-table", type=str, default="Table_S2_sensitivity_analyses.csv", help="Output CSV path.")
    parser.add_argument("--run-full-iecv", action="store_true", help="Also run full 38-country IECV for SA-8.")
    args = parser.parse_args()

    input_p = Path(args.input_cohort)
    df = pd.read_parquet(input_p) if input_p.suffix in ['.parquet', '.pq'] else pd.read_csv(input_p)

    df_summary = execute_all_sensitivities(df, run_full_iecv=args.run_full_iecv)
    df_summary.to_csv(args.output_table, index=False)

    logger.info("=" * 80)
    logger.info("PRE-SPECIFIED SENSITIVITY ANALYSES SUMMARY (TABLE S2):")
    logger.info("=" * 80)
    print(df_summary.to_string(index=False))
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
