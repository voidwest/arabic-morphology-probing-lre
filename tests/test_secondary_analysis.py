from __future__ import annotations

import unittest

import numpy as np

from research_stack.revision.secondary_analysis import (
    evaluate_alpha_grid,
    paired_cluster_bootstrap_deltas,
    paired_bootstrap_deltas,
    stimulus_cluster_id,
    stratified_bootstrap_samples,
    validate_and_pair_predictions,
)


def prediction_row(
    *,
    location: str,
    example_id: str,
    gold: str,
    prediction: str,
) -> dict[str, str]:
    return {
        "model": "model",
        "task": "pos",
        "split": "lemma-heldout",
        "representation_location": location,
        "example_id": example_id,
        "gold_label": gold,
        "predicted_label": prediction,
    }


class TestPredictionPairing(unittest.TestCase):
    def test_stimulus_cluster_id_uses_exact_prompt_fields(self):
        row = {
            "surface_dediac": "كتب",
            "surface": "كَتَبَ",
            "lemma": "كَتَبَ",
            "root": "ك.ت.ب",
            "abstract_pattern": "1َ2َ3َ",
            "gender": "masc",
        }
        baseline = stimulus_cluster_id(row)
        row["gender"] = "fem"
        self.assertEqual(stimulus_cluster_id(row), baseline)
        row["lemma"] = "كِتَاب"
        self.assertNotEqual(stimulus_cluster_id(row), baseline)

    def test_exact_pairing(self):
        rows = [
            prediction_row(
                location="prompt_final", example_id="b", gold="N", prediction="V"
            ),
            prediction_row(
                location="target_final_subtoken", example_id="a",
                gold="V", prediction="V",
            ),
            prediction_row(
                location="target_final_subtoken", example_id="b",
                gold="N", prediction="N",
            ),
            prediction_row(
                location="prompt_final", example_id="a", gold="V", prediction="N"
            ),
        ]
        paired = validate_and_pair_predictions(rows)
        self.assertEqual([row["example_id"] for row in paired], ["a", "b"])
        self.assertEqual(paired[1]["prompt_prediction"], "V")
        self.assertEqual(paired[1]["target_prediction"], "N")

    def test_missing_example_detection(self):
        rows = [
            prediction_row(
                location="prompt_final", example_id="a", gold="N", prediction="N"
            )
        ]
        with self.assertRaisesRegex(ValueError, "pairing mismatch"):
            validate_and_pair_predictions(rows)

    def test_duplicate_example_detection(self):
        row = prediction_row(
            location="prompt_final", example_id="a", gold="N", prediction="N"
        )
        with self.assertRaisesRegex(ValueError, "duplicate prediction"):
            validate_and_pair_predictions([row, dict(row)])

    def test_mismatched_gold_label_detection(self):
        rows = [
            prediction_row(
                location="prompt_final", example_id="a", gold="N", prediction="N"
            ),
            prediction_row(
                location="target_final_subtoken", example_id="a",
                gold="V", prediction="V",
            ),
        ]
        with self.assertRaisesRegex(ValueError, "mismatched gold"):
            validate_and_pair_predictions(rows)


class TestPairedBootstrap(unittest.TestCase):
    def test_stratification_preserves_class_sizes(self):
        gold = ["a", "a", "a", "b", "b", "c"]
        samples = stratified_bootstrap_samples(gold, n_bootstrap=50, seed=9)
        labels = np.asarray(gold, dtype=object)
        for sample in samples:
            values, counts = np.unique(labels[sample], return_counts=True)
            self.assertEqual(dict(zip(values, counts, strict=True)), {
                "a": 3, "b": 2, "c": 1,
            })

    def test_deterministic_output(self):
        paired = [
            {
                "gold_label": gold,
                "prompt_prediction": prompt,
                "target_prediction": target,
            }
            for gold, prompt, target in [
                ("a", "a", "a"),
                ("a", "b", "a"),
                ("b", "b", "a"),
                ("b", "a", "b"),
            ]
        ]
        first = paired_bootstrap_deltas(paired, n_bootstrap=200, seed=17)
        second = paired_bootstrap_deltas(paired, n_bootstrap=200, seed=17)
        self.assertEqual(first, second)

    def test_synthetic_no_difference(self):
        paired = [
            {
                "gold_label": gold,
                "prompt_prediction": prediction,
                "target_prediction": prediction,
            }
            for gold, prediction in [
                ("a", "a"), ("a", "b"), ("b", "b"), ("b", "a")
            ]
        ]
        for row in paired_bootstrap_deltas(paired, n_bootstrap=200, seed=3):
            self.assertEqual(row["delta"], 0.0)
            self.assertEqual(row["ci_lower"], 0.0)
            self.assertEqual(row["ci_upper"], 0.0)

    def test_synthetic_known_direction(self):
        paired = [
            {
                "gold_label": gold,
                "prompt_prediction": "b" if gold == "a" else "a",
                "target_prediction": gold,
            }
            for gold in ["a"] * 10 + ["b"] * 10
        ]
        for row in paired_bootstrap_deltas(paired, n_bootstrap=200, seed=5):
            self.assertGreater(row["delta"], 0.0)
            self.assertGreater(row["ci_lower"], 0.0)
            self.assertEqual(row["proportion_delta_gt_zero"], 1.0)

    def test_cluster_bootstrap_is_deterministic_and_reports_clusters(self):
        paired = [
            {
                "example_id": f"e{index}",
                "stimulus_cluster_id": cluster,
                "gold_label": gold,
                "prompt_prediction": prompt,
                "target_prediction": target,
            }
            for index, (cluster, gold, prompt, target) in enumerate([
                ("a1", "a", "a", "a"),
                ("a1", "a", "a", "a"),
                ("a2", "a", "b", "a"),
                ("b1", "b", "b", "a"),
                ("b2", "b", "a", "b"),
            ])
        ]
        first = paired_cluster_bootstrap_deltas(
            paired, n_bootstrap=200, seed=19
        )
        second = paired_cluster_bootstrap_deltas(
            paired, n_bootstrap=200, seed=19
        )
        self.assertEqual(first, second)
        self.assertTrue(all(row["n_clusters"] == 4 for row in first))
        self.assertTrue(all(row["duplicate_rows"] == 1 for row in first))
        self.assertTrue(all(row["max_cluster_size"] == 2 for row in first))

    def test_cluster_bootstrap_rejects_conflicting_gold_within_cluster(self):
        paired = [
            {
                "example_id": f"e{index}",
                "stimulus_cluster_id": "same",
                "gold_label": gold,
                "prompt_prediction": gold,
                "target_prediction": gold,
            }
            for index, gold in enumerate(["a", "b"])
        ]
        with self.assertRaisesRegex(ValueError, "conflicting gold"):
            paired_cluster_bootstrap_deltas(paired, n_bootstrap=10, seed=1)

    def test_cluster_bootstrap_requires_cluster_ids(self):
        paired = [{
            "example_id": "e0",
            "gold_label": "a",
            "prompt_prediction": "a",
            "target_prediction": "a",
        }]
        with self.assertRaisesRegex(ValueError, "missing stimulus_cluster_id"):
            paired_cluster_bootstrap_deltas(paired, n_bootstrap=10, seed=1)


class TestAlphaGrid(unittest.TestCase):
    def test_alpha_grid_is_deterministic_and_returns_predictions(self):
        rng = np.random.default_rng(4)
        features = rng.normal(size=(30, 3, 5))
        labels = ["a", "b"] * 15
        kwargs = {
            "train_indices": list(range(16)),
            "dev_indices": list(range(16, 22)),
            "test_indices": list(range(22, 30)),
            "alphas": [0.1, 1.0, 10.0, 100.0],
        }
        first = evaluate_alpha_grid(features, labels, **kwargs)
        second = evaluate_alpha_grid(features, labels, **kwargs)
        self.assertEqual(first, second)
        self.assertEqual([item.alpha for item in first], kwargs["alphas"])
        self.assertTrue(all(len(item.test_predictions) == 8 for item in first))

    def test_alpha_grid_does_not_fit_label_vocabulary_on_test(self):
        features = np.zeros((9, 2, 2), dtype=float)
        labels = ["a", "b", "a", "b", "a", "b", "a", "b", "c"]
        fits = evaluate_alpha_grid(
            features,
            labels,
            train_indices=[0, 1, 2, 3],
            dev_indices=[4, 5],
            test_indices=[6, 7, 8],
            alphas=[1.0],
        )
        self.assertEqual(len(fits), 1)
        self.assertNotIn("c", fits[0].test_predictions)


if __name__ == "__main__":
    unittest.main()
