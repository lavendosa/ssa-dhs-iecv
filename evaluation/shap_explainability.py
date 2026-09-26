#!/usr/bin/env python3
"""
shap_explainability.py - Model Explainability via SHAP (TreeExplainer)

Interrogates the primary operational model (LightGBM) using SHAP (SHapley Additive exPlanations):
1. Draws a stratified subsample of 10,000 acutely ill children.
2. Fits shap.TreeExplainer to compute exact feature attributions on model log-odds.
3. Quantifies global feature importance by mean absolute SHAP value (mean |SHAP|).
4. Generates:
   - Summary beeswarm plot illustrating directional feature effects (Figure 4A).
   - Global feature importance bar chart (Figure 4B).
   - Multi-panel partial dependence plots for primary clinical and socioeconomic drivers:
     fever presence, institutional delivery, wealth score, and maternal education (Figure S5).

Usage:
    python shap_explainability.py --input-cohort analytic_cohort.parquet --output-dir ./results/figures
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from modelling.lightgbm_model import get_lightgbm_model

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

FEATURE_LABELS = {
    'illness_fever': 'Child had fever',
    'wealth_score': 'Household wealth factor score',
    'wealth_quintile': 'Wealth quintile (1-5)',
    'delivered_in_facility': 'Institutional delivery',
    'barrier_money': 'Financial barrier to care',
    'barrier_distance': 'Distance barrier to facility',
    'maternal_edu_years': "Mother's education (years)",
    'maternal_age': "Mother's age (years)",
    'child_age_months': "Child's age (months)",
    'residence_rural': 'Rural residence',
    'illness_count': 'Number of concurrent illnesses',
    'illness_diarrhea': 'Child had diarrhea',
    'illness_cough': 'Child had cough',
    'illness_ari_rapid_breath': 'Rapid/difficult breathing (ARI)',
    'illness_ari_chest_problem': 'Chest problem (ARI)',
    'has_insurance': 'Health insurance coverage',
    'improved_water': 'Improved water source',
    'improved_toilet': 'Improved sanitation facility',
    'clean_cooking_fuel': 'Clean cooking fuel',
    'has_electricity': 'Household electricity',
    'has_television': 'Owns television',
    'has_radio': 'Owns radio',
    'household_size': 'Household size',
    'u5_children_count': 'Children under 5 count',
    'mother_employed': 'Mother currently employed',
    'marital_in_union': 'Mother married / in union',
    'birth_order': 'Child birth order',
    'child_female': 'Child female sex',
    'child_is_twin': 'Multiple birth (twin/triplet)',
    'barrier_permission': 'Permission barrier to care',
    'barrier_alone': 'Barrier: fear of travelling alone'
}


def run_shap_analysis(
    df: pd.DataFrame,
    output_dir: Path,
    sample_size: int = 10000,
    random_seed: int = 42
) -> pd.DataFrame:
    """Executes TreeExplainer on stratified subsample and saves publication plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    X = df[PREDICTOR_NAMES].copy()
    y = df['unmet_need'].values

    logger.info("Fitting operational LightGBM model for SHAP TreeExplainer...")
    model = get_lightgbm_model(random_state=random_seed)
    model.fit(X, y)

    # Stratified subsampling of 10,000 cases
    np.random.seed(random_seed)
    sub_size = min(sample_size, len(df))
    # Stratified sample by outcome
    idx_0 = np.where(y == 0)[0]
    idx_1 = np.where(y == 1)[0]
    p1 = len(idx_1) / len(y)
    n1 = int(round(sub_size * p1))
    n0 = sub_size - n1

    sample_0 = np.random.choice(idx_0, size=min(n0, len(idx_0)), replace=False)
    sample_1 = np.random.choice(idx_1, size=min(n1, len(idx_1)), replace=False)
    sample_indices = np.concatenate([sample_0, sample_1])
    np.random.shuffle(sample_indices)

    X_sample = X.iloc[sample_indices].copy()
    X_sample_named = X_sample.rename(columns=FEATURE_LABELS)

    logger.info(f"Computing TreeExplainer SHAP values on {len(X_sample):,} cases...")
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_sample)

    if isinstance(shap_vals, list) and len(shap_vals) == 2:
        shap_target = shap_vals[1]
    else:
        shap_target = shap_vals

    # Compute global mean absolute SHAP values
    mean_abs = np.mean(np.abs(shap_target), axis=0)
    df_imp = pd.DataFrame({
        "Feature": PREDICTOR_NAMES,
        "Feature_Label": [FEATURE_LABELS[f] for f in PREDICTOR_NAMES],
        "Mean_Abs_SHAP": mean_abs
    }).sort_values("Mean_Abs_SHAP", ascending=False)

    df_imp.to_csv(output_dir / "shap_feature_importance.csv", index=False)
    logger.info("Saved shap_feature_importance.csv")

    # Figure 4A: Summary Beeswarm Plot
    logger.info("Generating Figure 4A: Summary Beeswarm Plot...")
    fig_beeswarm, ax = plt.subplots(figsize=(10, 8), dpi=300)
    shap.summary_plot(
        shap_target, X_sample_named,
        max_display=15, show=False, plot_size=None
    )
    plt.title("Figure 4A: SHAP Summary Beeswarm Plot (Directional Risk Effects)", fontsize=13, pad=15)
    plt.xlabel("SHAP Value (Impact on Log-Odds of Unmet Healthcare Need)", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_dir / "Figure4A_shap_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Figure 4B: Feature Importance Bar Plot
    logger.info("Generating Figure 4B: Importance Bar Plot...")
    top15 = df_imp.head(15).iloc[::-1]
    fig_bar, ax = plt.subplots(figsize=(9, 7), dpi=300)
    bars = ax.barh(top15["Feature_Label"], top15["Mean_Abs_SHAP"], color="#2b5c8f", edgecolor="#1a365d")
    ax.set_xlabel("Mean Absolute SHAP Value (mean |SHAP|)", fontsize=11, fontweight="bold")
    ax.set_title("Figure 4B: Top 15 Predictors Ranked by Global Importance", fontsize=12, fontweight="bold", pad=12)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.01, bar.get_y() + bar.get_height() / 2, f"{w:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / "Figure4B_shap_bar.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Figure S5: Multi-Panel Partial Dependence Plots
    logger.info("Generating Figure S5: Partial Dependence Plots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=300)
    pdp_features = [
        ('wealth_score', 'Household Wealth Asset Score', axes[0, 0], '#2b5c8f'),
        ('maternal_edu_years', "Mother's Education (Years)", axes[0, 1], '#8c2d19'),
        ('child_age_months', "Child's Age (Months)", axes[1, 0], '#1f78b4'),
        ('illness_fever', 'Acute Fever Episode (0=No, 1=Yes)', axes[1, 1], '#e31a1c')
    ]

    for f_name, f_title, ax, col in pdp_features:
        f_idx = PREDICTOR_NAMES.index(f_name)
        f_vals = X_sample.iloc[:, f_idx].values
        f_shaps = shap_target[:, f_idx]

        # Scatter
        ax.scatter(f_vals, f_shaps, alpha=0.15, s=12, color=col, edgecolor='none')
        # Smooth line via running mean / binned median
        df_tmp = pd.DataFrame({"x": f_vals, "y": f_shaps}).dropna()
        if f_name in ['illness_fever']:
            grp = df_tmp.groupby("x")["y"].mean().reset_index()
            ax.plot(grp["x"], grp["y"], color="black", lw=2.5, marker="o")
        else:
            df_tmp["bin"] = pd.qcut(df_tmp["x"], q=15, duplicates="drop")
            grp = df_tmp.groupby("bin", observed=True).agg({"x": "mean", "y": "mean"}).reset_index()
            ax.plot(grp["x"], grp["y"], color="black", lw=2.5, marker="s", markersize=4)

        ax.axhline(0, color="gray", linestyle="--", lw=1.0, alpha=0.7)
        ax.set_title(f"Partial Dependence: {f_title}", fontsize=11, fontweight="bold")
        ax.set_xlabel(f_title, fontsize=10)
        ax.set_ylabel("SHAP Value (Log-Odds Impact)", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_dir / "FigureS5_shap_partial_dependence.png", dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("Explainability outputs generated successfully.")
    return df_imp


def main():
    parser = argparse.ArgumentParser(description="Run SHAP Explainability on LightGBM model.")
    parser.add_argument("--input-cohort", type=str, required=True, help="Input cohort path.")
    parser.add_argument("--output-dir", type=str, default="./results/figures", help="Output directory for plots.")
    parser.add_argument("--sample-size", type=int, default=10000, help="Stratified subsample size.")
    args = parser.parse_args()

    input_p = Path(args.input_cohort)
    df = pd.read_parquet(input_p) if input_p.suffix in ['.parquet', '.pq'] else pd.read_csv(input_p)

    out_p = Path(args.output_dir)
    df_imp = run_shap_analysis(df, out_p, sample_size=args.sample_size)
    print("\nTop 10 Most Influential Predictors by Mean |SHAP|:")
    print(df_imp.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
