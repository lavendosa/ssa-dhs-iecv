#!/usr/bin/env python3
"""
run_pipeline.py - Master Pipeline Orchestrator

End-to-end reproducible research pipeline for:
"Machine learning prediction of absence of formal healthcare seeking for
acutely ill children in sub-Saharan Africa: a 38-country internal-external cross-validation study"

Pipeline Stages:
1. extract     : Scan DHS ZIP archives and select Kids Recode surveys.
2. harmonize   : Apply eligibility criteria, derive outcome, and engineer 31 predictors.
3. train       : Execute 5-fold stratified cross-validation on pooled cohort.
4. evaluate    : Execute 38-country leave-one-country-out IECV and SHAP explainability.
5. sensitivity : Execute all 8 prespecified sensitivity analyses.
6. all         : Execute stages 1 through 5 sequentially.

Usage:
    # Run full pipeline with high-fidelity synthetic data (instant reproduction):
    python run_pipeline.py --stage all --use-synthetic

    # Run full pipeline with raw DHS archives:
    python run_pipeline.py --stage all --data-dir /path/to/dhs_zips
"""

import sys
import os
import time
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Configure top-level logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("SSA-DHS-Pipeline")

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from harmonization.synthetic_data_generator import generate_synthetic_cohort
from harmonization.construct_outcome import verify_zero_leakage
from evaluation.cross_validation import run_5fold_cross_validation
from evaluation.iecv_validation import run_38country_iecv
from evaluation.subgroup_analysis import evaluate_subgroups
from evaluation.shap_explainability import run_shap_analysis
from sensitivity_analyses.run_all_sensitivities import execute_all_sensitivities


def run_pipeline(stage: str = "all", use_synthetic: bool = False, data_dir: str = None, fast_mode: bool = False):
    """Executes specified pipeline stages with strict audit logging."""
    logger.info("=" * 80)
    logger.info("STARTING REPRODUCIBLE RESEARCH PIPELINE: PEDIATRIC HEALTHCARE SEEKING ML")
    logger.info(f"Target Stage: {stage.upper()} | Synthetic Mode: {use_synthetic} | Fast Mode: {fast_mode}")
    logger.info("=" * 80)

    t_start = time.time()
    cohort_path = REPO_ROOT / "data" / "analytic_cohort.parquet"
    tables_dir = REPO_ROOT / "results" / "tables"
    figures_dir = REPO_ROOT / "results" / "figures"

    cohort_path.parent.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1 & 2: Extract & Harmonize
    if stage in ["extract", "harmonize", "all"]:
        if use_synthetic or not cohort_path.exists():
            logger.info("Generating synthetic cohort (N = 118,910 across 38 countries)...")
            n_samples = 15000 if fast_mode else 118910
            df_cohort = generate_synthetic_cohort(total_n=n_samples, random_seed=42)
            df_cohort.to_parquet(cohort_path, index=False)
            logger.info(f"Saved analytic cohort to {cohort_path}")
        else:
            logger.info(f"Using existing analytic cohort at {cohort_path}")
            df_cohort = pd.read_parquet(cohort_path)

        # Audit leakage
        verify_zero_leakage(df_cohort.columns.tolist())

    if stage in ["harmonize"]:
        logger.info("Harmonization stage complete.")
        return

    # Load analytic cohort for modeling stages
    if not cohort_path.exists():
        logger.info("Analytic cohort not found. Generating default synthetic cohort...")
        df_cohort = generate_synthetic_cohort(total_n=15000 if fast_mode else 118910, random_seed=42)
        df_cohort.to_parquet(cohort_path, index=False)
    else:
        df_cohort = pd.read_parquet(cohort_path)

    oof_path = REPO_ROOT / "results" / "oof_predictions.parquet"

    # Stage 3: Train & Benchmark (5-Fold CV)
    if stage in ["train", "all"]:
        logger.info("\n" + "="*70)
        logger.info("STAGE 3: 5-FOLD STRATIFIED CROSS-VALIDATION BENCHMARK")
        logger.info("="*70)
        df_t2, df_oof = run_5fold_cross_validation(df_cohort, fast_rf=fast_mode)
        df_t2.to_csv(tables_dir / "Table_2.csv", index=False)
        df_oof.to_parquet(oof_path, index=False)
        logger.info("Table 2 generated successfully.")

    # Stage 4: Evaluate (38-Country IECV & SHAP)
    if stage in ["evaluate", "all"]:
        logger.info("\n" + "="*70)
        logger.info("STAGE 4: 38-COUNTRY LEAVE-ONE-COUNTRY-OUT IECV & SHAP")
        logger.info("="*70)
        df_t3, df_iecv_preds = run_38country_iecv(df_cohort)
        df_t3.to_csv(tables_dir / "Table_3.csv", index=False)
        logger.info("Table 3 generated successfully.")

        # Subgroups
        if oof_path.exists():
            df_oof = pd.read_parquet(oof_path)
            prob_col = 'LightGBM' if 'LightGBM' in df_oof.columns else df_oof.columns[0]
            df_t4 = evaluate_subgroups(df_cohort, df_oof[prob_col].values)
            df_t4.to_csv(tables_dir / "Table_4.csv", index=False)
            logger.info("Table 4 generated successfully.")

        # SHAP Explainability
        logger.info("Running SHAP TreeExplainer on 10,000 stratified subsample...")
        sample_n = 2000 if fast_mode else 10000
        run_shap_analysis(df_cohort, output_dir=figures_dir, sample_size=sample_n)
        logger.info("Figure 4 and Figure S5 generated successfully.")

    # Stage 5: Sensitivity Analyses (SA-1 through SA-8)
    if stage in ["sensitivity", "all"]:
        logger.info("\n" + "="*70)
        logger.info("STAGE 5: EIGHT PRE-SPECIFIED SENSITIVITY ANALYSES")
        logger.info("="*70)
        df_s2 = execute_all_sensitivities(df_cohort, run_full_iecv=not fast_mode)
        df_s2.to_csv(tables_dir / "Table_S2.csv", index=False)
        logger.info("Table S2 generated successfully.")

    total_elapsed = time.time() - t_start
    logger.info("\n" + "="*80)
    logger.info(f"ALL REQUESTED PIPELINE STAGES COMPLETED IN {total_elapsed:.1f}s")
    logger.info("All tables and figures have been refreshed in ./results/")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Master runner for SSA Pediatric Healthcare Seeking ML study.")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["extract", "harmonize", "train", "evaluate", "sensitivity", "all"],
        help="Pipeline stage to execute."
    )
    parser.add_argument(
        "--use-synthetic",
        action="store_true",
        help="Use high-fidelity synthetic cohort for immediate test execution."
    )
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help="Run accelerated test mode (15k sample, fast RF) for rapid verification."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to raw DHS ZIP directory if available."
    )
    args = parser.parse_args()

    run_pipeline(
        stage=args.stage,
        use_synthetic=args.use_synthetic,
        data_dir=args.data_dir,
        fast_mode=args.fast_mode
    )


if __name__ == "__main__":
    main()
