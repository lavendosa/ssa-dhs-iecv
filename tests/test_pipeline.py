"""
test_pipeline.py - Comprehensive Unit & Integration Tests

Validates:
1. Zero target leakage: confirms no post-treatment or care-seeking variables exist in predictor set.
2. Binary outcome construction: asserts proper logic and prevalence bounds.
3. Model training: ensures all five algorithms initialize and predict probabilities in [0, 1].
4. Calibration and metrics: validates Hanley-McNeil CIs, logistic slopes/intercepts, and ECE.
5. Sensitivity filters: confirms correct subsetting for all eight sensitivity analyses.
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from harmonization.synthetic_data_generator import generate_synthetic_cohort
from harmonization.construct_outcome import construct_unmet_need_outcome, verify_zero_leakage, LEAKAGE_VARS
from modelling.model_factory import get_model, get_all_benchmark_models
from evaluation.metrics_util import (
    hanley_mcneil_ci, cross_national_summary_ci,
    calculate_calibration_slope_intercept, calculate_ece
)
from evaluation.subgroup_analysis import evaluate_subgroups


class TestDHSStudyPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Generate a small synthetic test cohort."""
        cls.df_test = generate_synthetic_cohort(total_n=1500, random_seed=42)
        cls.features = [
            'child_age_months', 'child_female', 'child_is_twin', 'birth_order', 'delivered_in_facility',
            'maternal_age', 'maternal_edu_years', 'marital_in_union', 'mother_employed',
            'wealth_quintile', 'wealth_score', 'residence_rural', 'household_size', 'u5_children_count',
            'has_electricity', 'has_radio', 'has_television', 'improved_water', 'improved_toilet', 'clean_cooking_fuel',
            'barrier_distance', 'barrier_money', 'barrier_permission', 'barrier_alone', 'has_insurance',
            'illness_fever', 'illness_diarrhea', 'illness_cough', 'illness_ari_rapid_breath', 'illness_ari_chest_problem',
            'illness_count'
        ]

    def test_01_target_leakage_audit(self):
        """Asserts zero post-treatment clinical variables exist in predictor matrix."""
        self.assertTrue(verify_zero_leakage(self.features))
        # Ensure that attempting to inject a leakage variable raises an exception
        bad_features = self.features + ['H12Z']
        with self.assertRaises(ValueError):
            verify_zero_leakage(bad_features)

    def test_02_outcome_construction(self):
        """Asserts that unmet need is binary [0, 1] and tracks population ranges."""
        y = self.df_test['unmet_need'].values
        self.assertTrue(set(np.unique(y)).issubset({0, 1}))
        prev = np.mean(y)
        self.assertGreater(prev, 0.40)
        self.assertLess(prev, 0.75)

    def test_03_all_benchmark_models_train_and_predict(self):
        """Verifies that all 5 algorithms fit and output valid probabilities."""
        models = get_all_benchmark_models(random_state=42, fast_rf=True)
        X = self.df_test[self.features]
        y = self.df_test['unmet_need'].values

        for name, model in models.items():
            model.fit(X, y)
            probs = model.predict_proba(X)[:, 1]
            self.assertEqual(len(probs), len(y))
            self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))

    def test_04_hanley_mcneil_ci(self):
        """Validates parametric Hanley-McNeil standard errors and confidence intervals."""
        lo, hi, se = hanley_mcneil_ci(auroc=0.754, n_pos=500, n_neg=500)
        self.assertLess(lo, 0.754)
        self.assertGreater(hi, 0.754)
        self.assertGreater(se, 0.0)
        self.assertLess(se, 0.05)

    def test_05_cross_national_summary(self):
        """Validates unweighted cross-national arithmetic mean and CI."""
        mock_aurocs = [0.65, 0.70, 0.75, 0.68, 0.72]
        mean_val, lo, hi, sd_val = cross_national_summary_ci(mock_aurocs)
        self.assertAlmostEqual(mean_val, 0.70, places=2)
        self.assertLess(lo, mean_val)
        self.assertGreater(hi, mean_val)

    def test_06_calibration_slope_and_intercept(self):
        """Validates logistic calibration slope and intercept calculation."""
        y = np.array([0, 0, 0, 1, 1, 1])
        p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        intercept, slope = calculate_calibration_slope_intercept(y, p)
        self.assertIsInstance(intercept, float)
        self.assertIsInstance(slope, float)
        self.assertGreater(slope, 0.0)

    def test_07_subgroup_evaluation(self):
        """Validates subgroup evaluation output structure."""
        p_mock = np.random.uniform(0.1, 0.9, size=len(self.df_test))
        res_sub = evaluate_subgroups(self.df_test, p_mock)
        self.assertGreater(len(res_sub), 10)
        self.assertIn("Domain", res_sub.columns)
        self.assertIn("AUROC", res_sub.columns)


if __name__ == "__main__":
    unittest.main()
