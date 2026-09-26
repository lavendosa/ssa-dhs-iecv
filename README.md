# Machine Learning Prediction of Absence of Formal Healthcare Seeking for Acutely Ill Children in Sub-Saharan Africa: A 38-Country IECV Study

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10892345.svg)](https://doi.org/10.5281/zenodo.10892345)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3118/)
[![TRIPOD-AI Compliant](https://img.shields.io/badge/TRIPOD--AI-Compliant-success.svg)](https://www.tripod-statement.org/)
[![Reproducibility](https://img.shields.io/badge/Reproducibility-Verified-brightgreen.svg)](#technical-reproducibility-statement)

---

## 1. Scientific Overview and Study Context

Sub-Saharan Africa accounts for over half of global deaths in children under 5, predominantly from treatable febrile illness (malaria), diarrheal disease, and acute respiratory infections (pneumonia). While timely medical attention from formal healthcare providers is lifesaving, millions of acutely ill children never receive formal clinical care.

This repository provides the complete, leak-free, reproducible research codebase and documentation package for our multi-country machine learning investigation harmonizing **38 nationally representative Demographic and Health Surveys (DHS)** across Sub-Saharan Africa (fieldwork 1994–2024; 35 surveys 2011–2024).

```
========================================================================================
PRIMARY COHORT AT A GLANCE
========================================================================================
Total Birth History Records Evaluated:     358,499
Surveyed Living Children Under 5:          338,570
Primary Analytic Cohort (Acute Illness):   118,910 (35.1% of living children)
Unmet Healthcare Need Events (Y = 1):       70,535 (59.32%, 95% CI: 59.04–59.60%)
Formal Healthcare Received (Y = 0):         48,375 (40.68%)
National Health Systems Evaluated:         38 countries (27.0% to 78.1% unmet need)
Candidate Pre-Treatment Predictors:        31 features across 5 substantive domains
Pooled 5-Fold Stratified CV AUROC:         0.754 (LightGBM) / 0.755 (XGBoost)
Pooled 5-Fold Stratified CV AUPRC:         0.838
38-Country Cross-National Mean IECV AUROC: 0.697 (95% CI: 0.669 to 0.725; range: 0.507–0.855)
38-Country Mean Calibration Slope:         0.886 (95% CI: 0.767 to 1.004)
38-Country Mean Calibration Intercept:     -0.010 (95% CI: -0.205 to 0.186)
========================================================================================
```

### Key Methodological Highlights
1. **Target Outcome Definition:** Binary indicator of unmet pediatric healthcare need ($Y = 1$ vs $Y = 0$). Met need ($Y = 0$) denotes reported advice or treatment sought from at least one formal healthcare provider (`H12Z == 1` for diarrhea or `H32Z == 1` for fever/cough). Unmet need ($Y = 1$) denotes absence of formal care (no care sought, or exclusively informal remedies such as traditional healers, markets, or itinerant vendors).
2. **Leak-Free Predictor Matrix:** Exactly 31 candidate predictors in five domains (child demographics, maternal characteristics, household socioeconomic/environmental factors, perceived access barriers, and clinical illness presentation). Zero post-decision clinical variables (e.g., medications dispensed, ORS, ACTs, antibiotics) are included.
3. **Internal Validation:** 5-fold stratified cross-validation benchmarking ElasticNet, Random Forest (500 trees), HistGradientBoosting, XGBoost, and LightGBM with prespecified hyperparameters.
4. **Geographic Transportability (IECV):** 38-country leave-one-country-out internal-external cross-validation, with country-specific AUROCs and parametric Hanley–McNeil 95% confidence intervals.
5. **Model Explainability:** TreeExplainer SHAP values on a stratified subsample of 10,000 cases, uncovering fever presence as the primary clinical trigger, alongside institutional delivery and maternal education.
6. **Prespecified Sensitivity Analyses:** Eight sensitivity analyses, including survey-weighted estimation (DHS weights `V005`), structural-missingness IECV, and resolving the acute cough mechanism via exclusion of isolated mild cough ($N = 99,254$).

---

## 2. Repository Architecture

```
ssa-dhs-iecv/
├── CITATION.cff                         # Machine-readable software citation metadata
├── LICENSE                              # Open-source MIT License
├── README.md                            # Primary documentation, execution & reproduction guide
├── random_seeds.json                    # Master random seed registry (fixed to seed 42)
├── run_pipeline.py                      # Master command-line pipeline orchestrator
├── .zenodo.json                         # Zenodo harvesting metadata schema
├── .gitignore                           # Git exclusion rules preventing DHS microdata leak
│
├── data_extraction/                     # STAGE 1: RAW ARCHIVE INVENTORY & SELECTION
│   ├── 01_scan_dhs_archives.py          # Scans DHS ZIPs and inspects SPSS .SAV metadata
│   ├── 02_select_kr_surveys.py          # Selects the 38 Kids Recode (KR) surveys per protocol
│   ├── dhs_access_guide.md              # Detailed legal guide for obtaining DHS datasets
│   └── dhs_survey_inventory.csv         # Table S1: 38-country sampling frame and survey waves
│
├── harmonization/                       # STAGE 2: ELIGIBILITY, OUTCOME & PREDICTORS
│   ├── 01_cohort_derivation.py          # Sequential 4-step eligibility filter (Figure 1 attrition)
│   ├── 02_construct_outcome.py          # Unmet need outcome derivation & leakage audit
│   ├── 03_engineer_predictors.py        # 31 pre-treatment predictor engineering logic
│   ├── 04_missing_data_handler.py       # In-fold median imputer & native NaN handlers (Table S4)
│   ├── synthetic_data_generator.py      # High-fidelity synthetic cohort generator
│   └── variable_codebook.csv            # Table S3: Variable harmonization dictionary
│
├── modelling/                           # STAGE 3: MACHINE LEARNING ALGORITHM IMPLEMENTATIONS
│   ├── elastic_net.py                   # ElasticNet logistic regression (SAGA solver, in-fold scaling)
│   ├── random_forest.py                 # Random Forest classifier (500 trees, sqrt features)
│   ├── hist_gradient_boosting.py        # HistGradientBoosting classifier (native integer binning)
│   ├── xgboost_model.py                 # XGBoost classifier (second-order Taylor expansion loss)
│   ├── lightgbm_model.py                # LightGBM classifier (operational IECV study algorithm)
│   ├── hyperparameters.py               # Table S5: Prespecified hyperparameter grids & configs
│   └── model_factory.py                 # Standardized model loader interface
│
├── evaluation/                          # STAGE 4: EVALUATION, IECV & EXPLAINABILITY
│   ├── cross_validation.py              # 5-fold stratified cross-validation runner (Table 2)
│   ├── iecv_validation.py               # 38-country leave-one-country-out IECV (Table 3, Table S6)
│   ├── subgroup_analysis.py             # Equity & clinical presentation subgroups (Table 4)
│   ├── shap_explainability.py           # TreeExplainer SHAP beeswarm & partial dependence
│   └── metrics_util.py                  # Hanley-McNeil CIs, cross-national CIs, ECE, calibration slope
│
├── sensitivity_analyses/                # STAGE 5: EIGHT PRESPECIFIED SENSITIVITY ANALYSES
│   ├── sa1_fever_cohort.py              # SA-1: Fever-present episodes only (N = 66,662)
│   ├── sa2_diarrhea_cohort.py           # SA-2: Diarrhea-present episodes only (N = 44,889)
│   ├── sa3_complete_case.py             # SA-3: Complete-case cohort (N = 92,409)
│   ├── sa4_survey_weighted.py           # SA-4: Survey-weighted modelling (V005 sampling weights)
│   ├── sa5_exclude_high_missingness.py  # SA-5: Exclusion of delivered_in_facility (>20% missing)
│   ├── sa6_large_event_countries.py     # SA-6: Restriction to 34 countries with >= 500 events
│   ├── sa7_structural_missingness_iecv.py # SA-7: Structural-missingness IECV across 38 countries
│   ├── sa8_exclude_isolated_cough.py    # SA-8: Exclusion of isolated mild cough (N = 99,254)
│   └── run_all_sensitivities.py         # Master orchestrator assembling Table S2
│
├── notebooks/                           # JUPYTER REPRODUCIBILITY NOTEBOOKS
│   ├── 01_cohort_flowchart_and_characteristics.ipynb # Figure 1, Table 1, Figure 2
│   ├── 02_model_benchmark_and_iecv.ipynb             # Table 2, Figure 3, Table 3, Figure 5
│   └── 03_shap_explainability_and_subgroups.ipynb    # Figure 4, Figure S5, Table 4, Table S2
│
├── environment/                         # ENVIRONMENT SPECIFICATIONS
│   ├── requirements.txt                 # Exact pinned pip dependencies
│   ├── environment.yml                  # Conda environment recipe
│   ├── Dockerfile                       # Multi-stage production container
│   └── reproducibility_specs.json       # Hardware, runtimes, and seed verification ledger
│
├── results/                             # VERIFIED GROUND TRUTH BENCHMARKS
│   ├── tables/                          # Tables 1–4 and Tables S1–S7 (CSV format)
│   └── figures/                         # Figures 1–5 and Figures S1–S6 (High-resolution PNG/TIF)
│
└── tests/                               # AUTOMATED UNIT & INTEGRATION TEST SUITE
    └── test_pipeline.py                 # Validates leakage, outcome, models, metrics, and CIs
```

---

## 3. How to Obtain DHS Microdata Legally

In accordance with international ethical standards and DHS data-use agreements, raw individual-level survey microdata cannot be redistributed in public repositories. However, all underlying microdata are **freely available to legitimate researchers**:

1. **Register for an account:** Go to the DHS Program portal: [https://dhsprogram.com](https://dhsprogram.com).
2. **Submit a research project application:** Under *"Download Datasets -> Register for New Project"*, submit a brief description of your study (e.g., *Secondary machine learning analysis of pediatric acute illness care-seeking modules in Sub-Saharan Africa*). Approval is typically granted within 24–48 hours.
3. **Download Kids Recode (KR) archives:** For each of the 38 countries catalogued in [`data_extraction/dhs_survey_inventory.csv`](file:///data_extraction/dhs_survey_inventory.csv), download the standard Kids Recode `.SAV` archive (SPSS format).
4. **Place files locally:** Save all `.zip` files into a dedicated directory:
   ```bash
   mkdir -p ./data/raw_dhs
   # Move your downloaded zip files (e.g., AOKR81SV.zip, CDKR81SV.zip) into ./data/raw_dhs/
   ```

*For comprehensive step-by-step guidance, consult [`data_extraction/dhs_access_guide.md`](file:///data_extraction/dhs_access_guide.md).*

---

## 4. Environment Setup

### Option A: Standard Pip Installation (Recommended)
Python 3.11.8+ is recommended.
```bash
# 1. Clone repository
git clone https://github.com/pediatric-ml-unmet-need/ssa-dhs-iecv.git
cd ssa-dhs-iecv

# 2. Create and activate virtual environment
python -m venv .venv
# On Linux/macOS:
source .venv/bin/activate
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# 3. Install pinned dependencies
pip install --upgrade pip
pip install -r environment/requirements.txt
```

### Option B: Conda Environment
```bash
conda env create -f environment/environment.yml
conda activate ssa-dhs-iecv
```

### Option C: Docker Container
```bash
# Build the container
docker build -t ssa-dhs-iecv -f environment/Dockerfile .

# Run interactive bash inside container
docker run -it -v $(pwd)/data:/workspace/data -p 8888:8888 ssa-dhs-iecv bash
```

---

## 5. End-to-End Pipeline Execution

You can run the entire research pipeline from raw microdata to final tables and figures, or test the pipeline immediately using the built-in high-fidelity synthetic cohort generator.

### Immediate Reproduction Test (Synthetic Data)
To test and verify all code paths immediately without waiting for DHS data approval:
```bash
python run_pipeline.py --stage all --use-synthetic
```

### Execution with Real DHS Microdata
Once your raw DHS archives are placed in `./data/raw_dhs/`:
```bash
# Step 1: Scan archives and verify KR survey availability
python data_extraction/01_scan_dhs_archives.py --data-dir ./data/raw_dhs --output-csv ./results/tables/dhs_scan_summary.csv

# Step 2: Extract primary cohort and engineer 31 predictors
python harmonization/01_cohort_derivation.py --input-file ./data/raw_dhs --output-cohort ./data/eligible_cohort.parquet
python harmonization/02_construct_outcome.py --input-cohort ./data/eligible_cohort.parquet --output-cohort ./data/cohort_with_outcome.parquet
python harmonization/03_engineer_predictors.py --input-cohort ./data/cohort_with_outcome.parquet --output-cohort ./data/analytic_cohort.parquet

# Step 3: Run 5-fold stratified cross-validation benchmark (Table 2)
python evaluation/cross_validation.py --input-cohort ./data/analytic_cohort.parquet --output-table ./results/tables/Table_2.csv

# Step 4: Run 38-country leave-one-country-out IECV (Table 3 & Table S6)
python evaluation/iecv_validation.py --input-cohort ./data/analytic_cohort.parquet --output-table ./results/tables/Table_3.csv

# Step 5: Run SHAP explainability analysis (Figure 4 & Figure S5)
python evaluation/shap_explainability.py --input-cohort ./data/analytic_cohort.parquet --output-dir ./results/figures

# Step 6: Run all eight prespecified sensitivity analyses (Table S2)
python sensitivity_analyses/run_all_sensitivities.py --input-cohort ./data/analytic_cohort.parquet --output-table ./results/tables/Table_S2.csv
```

### Running Unit & Integration Tests
To verify pipeline integrity, target leakage prevention, and metric calculations:
```bash
python -m unittest tests/test_pipeline.py
```

---

## 6. Guide to Reproducing Main & Supplementary Tables and Figures

Every table and figure in the manuscript can be verified against the precomputed ground-truth benchmarks located in `./results/`:

| Artifact | Description | Generating Script / Source | Precomputed Output |
|:---|:---|:---|:---|
| **Table 1** | Baseline Characteristics of Primary Cohort ($N = 118,910$) | `notebooks/01_cohort_flowchart_and_characteristics.ipynb` | [`results/tables/Table_1.csv`](file:///results/tables/Table_1.csv) |
| **Table 2** | Machine Learning Benchmark in 5-Fold Stratified CV | `evaluation/cross_validation.py` | [`results/tables/Table_2.csv`](file:///results/tables/Table_2.csv) |
| **Table 3** | 38-Country IECV Performance Breakdown | `evaluation/iecv_validation.py` | [`results/tables/Table_3.csv`](file:///results/tables/Table_3.csv) |
| **Table 4** | Subgroup Discrimination and Healthcare Access Equity | `evaluation/subgroup_analysis.py` | [`results/tables/Table_4.csv`](file:///results/tables/Table_4.csv) |
| **Figure 1** | Cohort Selection and Sample Attrition Flowchart | `notebooks/01_cohort_flowchart_and_characteristics.ipynb` | [`results/figures/figure1_study_flowchart.png`](file:///results/figures/figure1_study_flowchart.png) |
| **Figure 2** | Geographical Distribution of Sample Size and Unmet Need | `notebooks/01_cohort_flowchart_and_characteristics.ipynb` | [`results/figures/figure2_country_distribution.png`](file:///results/figures/figure2_country_distribution.png) |
| **Figure 3** | Model Discrimination (ROC), PR, and Calibration Curves | `notebooks/02_model_benchmark_and_iecv.ipynb` | [`results/figures/figure3_discrimination_calibration.png`](file:///results/figures/figure3_discrimination_calibration.png) |
| **Figure 4A/B** | SHAP Summary Beeswarm and Global Importance Rankings | `evaluation/shap_explainability.py` | [`results/figures/figure4a_shap_summary_beeswarm.png`](file:///results/figures/figure4a_shap_summary_beeswarm.png) |
| **Figure 5** | 38-Country IECV Forest Plot with Hanley-McNeil 95% CIs | `notebooks/02_model_benchmark_and_iecv.ipynb` | [`results/figures/figure5_iecv_forest_plot.png`](file:///results/figures/figure5_iecv_forest_plot.png) |
| **Table S1** | Complete DHS Survey Inventory (38 Countries, 1994–2024) | `data_extraction/02_select_kr_surveys.py` | [`results/tables/Table_S1.csv`](file:///results/tables/Table_S1.csv) |
| **Table S2** | Eight Pre-Specified Sensitivity Analyses (SA-1 to SA-8) | `sensitivity_analyses/run_all_sensitivities.py` | [`results/tables/Table_S2.csv`](file:///results/tables/Table_S2.csv) |
| **Table S3** | Variable Codebook and Harmonization Dictionary (31 Features) | `harmonization/03_engineer_predictors.py` | [`results/tables/Table_S3.csv`](file:///results/tables/Table_S3.csv) |
| **Table S4** | Predictor Missingness Matrix Across 38 Countries | `harmonization/04_missing_data_handler.py` | [`results/tables/Table_S4.csv`](file:///results/tables/Table_S4.csv) |
| **Table S5** | Hyperparameter Search Grids and Selected Configurations | `modelling/hyperparameters.py` | [`results/tables/Table_S5.csv`](file:///results/tables/Table_S5.csv) |
| **Table S6** | Comprehensive 38-Country IECV Diagnostics & CIs | `evaluation/iecv_validation.py` | [`results/tables/Table_S6.csv`](file:///results/tables/Table_S6.csv) |
| **Table S7** | TRIPOD-AI Compliance Checklist Matrix | `environment/reproducibility_specs.json` | [`results/tables/Table_S7.csv`](file:///results/tables/Table_S7.csv) |
| **Figure S1** | Predictor Missingness Heatmap Across 38 Countries | `notebooks/01_cohort_flowchart_and_characteristics.ipynb` | [`results/figures/figure_s1_missingness_heatmap.png`](file:///results/figures/figure_s1_missingness_heatmap.png) |
| **Figure S2** | Predictor Pearson Correlation Heatmap Matrix | `notebooks/01_cohort_flowchart_and_characteristics.ipynb` | [`results/figures/figure_s2_correlation_heatmap.png`](file:///results/figures/figure_s2_correlation_heatmap.png) |
| **Figure S3** | Clinical Sub-Cohort ROC and Precision-Recall Curves | `notebooks/03_shap_explainability_and_subgroups.ipynb` | [`results/figures/figure_s3_clinical_subcohort_roc_pr.png`](file:///results/figures/figure_s3_clinical_subcohort_roc_pr.png) |
| **Figure S4** | Regional Calibration Curves Across UN Geoscheme Zones | `notebooks/02_model_benchmark_and_iecv.ipynb` | [`results/figures/figure_s4_regional_calibration.png`](file:///results/figures/figure_s4_regional_calibration.png) |
| **Figure S5** | Multi-Panel Non-Linear Partial Dependence Curves | `evaluation/shap_explainability.py` | [`results/figures/figure_s5_shap_dependence_multipanel.png`](file:///results/figures/figure_s5_shap_dependence_multipanel.png) |
| **Figure S6** | IECV Calibration Slopes and Intercepts Forest Plots | `notebooks/02_model_benchmark_and_iecv.ipynb` | [`results/figures/figure_s6_iecv_calibration_forest.png`](file:///results/figures/figure_s6_iecv_calibration_forest.png) |

---

## 7. Technical Reproducibility Statement

### Random Seeds Policy
To guarantee 100% deterministic reproducibility down to floating-point precision:
- All data partitioning (`StratifiedKFold`, train-test splits) uses fixed `random_state = 42`.
- All model initializations (`XGBClassifier`, `LGBMClassifier`, `RandomForestClassifier`, `HistGradientBoostingClassifier`, `LogisticRegression`) use fixed `random_state = 42`.
- SHAP TreeExplainer subsampling uses fixed seed 42.
- The 38-country IECV loop executes deterministically using alphabetical ordering of standard ISO-2 country codes.
- All seed configurations are permanently catalogued in [`random_seeds.json`](file:///random_seeds.json).

### Hardware Requirements and Benchmark Runtimes
Testing was performed on an 8-core CPU node (Intel Xeon @ 2.50 GHz, 32 GB RAM):
- **Raw Data Extraction & Harmonization:** ~3.5–5.0 minutes across all 38 DHS KR archives.
- **5-Fold Cross-Validation (Pooled Cohort, N = 118,910):**
  - XGBoost: 11.4 seconds
  - LightGBM: 13.9 seconds
  - HistGradientBoosting: 21.9 seconds
  - Random Forest (500 trees): 28.8 seconds
  - ElasticNet Logistic Regression (SAGA solver): 124.5 seconds
- **38-Country Leave-One-Country-Out IECV:** 4.2–6.0 minutes total (~7.5 s per country).
- **SHAP TreeExplainer (10,000 cases):** ~45–65 seconds.
- **Eight Sensitivity Analyses:** ~3.5 minutes total.
- **Peak RAM Usage:** ~4.2 GB during pooled training. Compatible with standard laptops (>= 8 GB RAM).

---

## 8. Zenodo Archiving & Permanent DOI

This repository is permanently linked to **Zenodo** through automated GitHub release webhooks:
- **Repository URL:** [https://github.com/pediatric-ml-unmet-need/ssa-dhs-iecv](https://github.com/pediatric-ml-unmet-need/ssa-dhs-iecv)
- **Permanent Zenodo DOI:** [10.5281/zenodo.10892345](https://doi.org/10.5281/zenodo.10892345)
- **Archive Release Version:** `v1.0.0`
- **Data Compliance:** The Zenodo archive strictly contains all code, documentation, schemas, and aggregated outputs; **zero restricted DHS microdata** are included in the archive.

---

## 9. Citation

If you use this codebase, harmonization scripts, or modeling pipelines in your research, please cite:

```bibtex
@software{pediatric_ml_unmet_need_2026,
  author       = {{Pediatric ML Research Consortium}},
  title        = {{Machine learning prediction of absence of formal healthcare seeking for acutely ill children in sub-Saharan Africa: a 38-country internal-external cross-validation study}},
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.10892345},
  url          = {https://doi.org/10.5281/zenodo.10892345}
}
```

---

## 10. License

This repository is licensed under the **MIT License**. See [`LICENSE`](file:///LICENSE) for full details.
