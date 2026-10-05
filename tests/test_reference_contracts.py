import unittest

import pandas as pd
import numpy as np

from analysis.aggregate_metrics import paired_wilcoxon_holm, select_best_lora_per_task
from analysis.external_adaptation import external_adaptation_gains
from data.preprocess import (
    merge_sites_into_combined,
    nested_stratified_subject_subsets,
    stratified_subject_train_validation_split,
    truncate_sequences_to_landmark,
)
from eval.metrics import (
    apply_logistic_recalibration,
    brier_skill_score,
    decision_curve,
    fit_logistic_recalibration,
    select_validation_thresholds,
    threshold_metrics,
)


class ReferenceContractTests(unittest.TestCase):
    def test_fixed_landmark_preserves_alignment(self):
        frame = pd.DataFrame({
            "Events": [["d0", "d1", "d2"]],
            "Type": [[1, 1, 1]],
            "Time": [[0, 1, 2]],
        })
        result = truncate_sequences_to_landmark(frame, landmark_day=1)
        self.assertEqual(result.loc[0, "Events"], ["d0", "d1"])
        self.assertEqual(result.loc[0, "Type"], [1, 1])
        self.assertEqual(result.loc[0, "Time"], [0, 1])

    def test_combined_cohort_retains_source_site(self):
        columns = {"SUBJECT_ID": [1], "HADM_ID": [1], "Events": [[]], "Type": [[]], "Time": [[]]}
        result = merge_sites_into_combined({"A": pd.DataFrame(columns), "B": pd.DataFrame(columns)})
        self.assertEqual(set(result["SOURCE_SITE"]), {"A", "B"})
        self.assertEqual(len(result), 2)

    def test_low_resource_subsets_are_nested_and_class_preserving(self):
        frame = pd.DataFrame({
            "SUBJECT_ID": np.arange(100),
            "outcome": [1] * 20 + [0] * 80,
        })
        subsets = nested_stratified_subject_subsets(
            frame,
            label_col="outcome",
            fractions=(0.1, 0.2, 0.5),
            seed=0,
        )
        subject_sets = [set(subsets[fraction].SUBJECT_ID) for fraction in (0.1, 0.2, 0.5)]
        self.assertTrue(subject_sets[0] <= subject_sets[1] <= subject_sets[2])
        for subset in subsets.values():
            training, validation = stratified_subject_train_validation_split(
                subset,
                label_col="outcome",
                validation_fraction=0.2,
                seed=1,
            )
            self.assertEqual(set(training.outcome), {0, 1})
            self.assertEqual(set(validation.outcome), {0, 1})
            self.assertFalse(set(training.SUBJECT_ID) & set(validation.SUBJECT_ID))

    def test_lora_selection_uses_mean_validation_auprc(self):
        records = []
        for seed, score in enumerate((0.9, 0.1)):
            records.append({
                "site": "ALL", "task": "task", "config": "A",
                "seed": seed, "val_pr_auc": score,
            })
        for seed, score in enumerate((0.6, 0.6)):
            records.append({
                "site": "ALL", "task": "task", "config": "B",
                "seed": seed, "val_pr_auc": score,
            })
        selected = select_best_lora_per_task(records)
        self.assertEqual(len(selected), 2)
        self.assertEqual({row["config"] for row in selected}, {"B"})

    def test_internal_lora_selection_is_fixed_for_external_records(self):
        records = [
            {"site": "internal", "task": "task", "config": "A", "seed": 0, "val_pr_auc": 0.7},
            {"site": "internal", "task": "task", "config": "B", "seed": 0, "val_pr_auc": 0.6},
            {"site": "external", "task": "task", "config": "A", "seed": 0, "val_pr_auc": 0.4},
            {"site": "external", "task": "task", "config": "B", "seed": 0, "val_pr_auc": 0.8},
        ]
        selected = select_best_lora_per_task(records, selection_site="internal")
        self.assertEqual({row["config"] for row in selected}, {"A"})
        self.assertEqual({row["site"] for row in selected}, {"internal", "external"})

    def test_operating_points_are_selected_from_validation_predictions(self):
        validation_y = np.array([0, 0, 1, 1])
        validation_probability = np.array([0.1, 0.4, 0.6, 0.9])
        thresholds = select_validation_thresholds(validation_y, validation_probability)
        self.assertEqual(set(thresholds), {
            "max_validation_f1", "max_validation_youden", "validation_sensitivity_0.80",
        })
        result = threshold_metrics(
            np.array([0, 1]), np.array([0.2, 0.8]),
            threshold=thresholds["max_validation_f1"],
        )
        self.assertEqual((result["tp"], result["tn"]), (1, 1))

    def test_brier_skill_and_decision_curve_contracts(self):
        y = np.array([0, 0, 1, 1])
        probability = y.astype(float)
        self.assertAlmostEqual(brier_skill_score(y, probability, reference_probability=0.5), 1.0)
        curve = decision_curve(y, probability, thresholds=np.array([0.1, 0.5]))
        self.assertEqual(len(curve), 2)
        self.assertEqual({row["threshold"] for row in curve}, {0.1, 0.5})

    def test_validation_recalibration_can_be_applied_to_test_predictions(self):
        fitted = fit_logistic_recalibration(
            np.array([0, 0, 1, 1]), np.array([0.2, 0.4, 0.6, 0.8])
        )
        recalibrated = apply_logistic_recalibration(
            np.array([0.25, 0.75]),
            intercept=fitted["intercept"],
            slope=fitted["slope"],
        )
        self.assertEqual(recalibrated.shape, (2,))
        self.assertTrue(np.all((recalibrated > 0) & (recalibrated < 1)))

    def test_external_adaptation_gain_uses_matched_freeze_all_runs(self):
        records = [
            {"task": "task", "method": "freeze_all", "seed": 0, "pr_auc": 0.20},
            {"task": "task", "method": "lora", "seed": 0, "pr_auc": 0.25},
            {"task": "task", "method": "freeze_all", "seed": 1, "pr_auc": 0.40},
            {"task": "task", "method": "lora", "seed": 1, "pr_auc": 0.44},
        ]
        result = external_adaptation_gains(records, metrics=["pr_auc"])
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(float(result.loc[0, "gain_mean"]), 17.5)

    def test_paired_comparisons_use_holm_correction(self):
        records = []
        for run, (reference, candidate) in enumerate(((0.20, 0.25), (0.22, 0.27))):
            records.extend([
                {"task": "task", "method": "lora", "seed": run, "pr_auc": reference},
                {"task": "task", "method": "full", "seed": run, "pr_auc": candidate},
            ])
        result = paired_wilcoxon_holm(
            records, metrics=["pr_auc"], reference_method="lora"
        )
        self.assertEqual(len(result), 1)
        self.assertIn("p_value_holm", result)
        self.assertAlmostEqual(float(result.loc[0, "mean_difference"]), 0.05)


if __name__ == "__main__":
    unittest.main()
