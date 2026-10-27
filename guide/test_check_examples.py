"""Test float tolerances and exact output requirements."""

import unittest

from check_examples import numeric_line_matches, output_matches


class OutputComparisonTests(unittest.TestCase):
    def test_rounding_boundary(self):
        for actual, expected in [("MSE: 0.334\n", "MSE: 0.333\n"),
                                 ("MSE: 0.333\n", "MSE: 0.334\n")]:
            with self.subTest(actual=actual):
                self.assertTrue(output_matches(actual, expected, "evaluation.md"))

    def test_absolute_tolerance_near_zero(self):
        self.assertTrue(output_matches(
            "test MAE: 0.000001\n", "test MAE: 0.000000\n", "first-model.md"))
        self.assertFalse(output_matches(
            "test MAE: 0.000002\n", "test MAE: 0.000000\n", "first-model.md"))

    def test_relative_tolerance(self):
        self.assertTrue(numeric_line_matches("value: 1000000.5\n", "value: 1000000.0\n", 1e-6))
        self.assertFalse(numeric_line_matches("value: 1000002.0\n", "value: 1000000.0\n", 1e-6))

    def test_signed_and_scientific_values(self):
        self.assertTrue(numeric_line_matches("value: -1.000001e-3\n", "value: -0.001\n", 1e-6))
        self.assertTrue(numeric_line_matches("value: -0.000000\n", "value: 0.000000\n", 1e-6))

    def test_large_numeric_errors_fail(self):
        self.assertFalse(output_matches("MSE: 0.335\n", "MSE: 0.333\n", "evaluation.md"))
        self.assertFalse(output_matches("MSE: -0.333\n", "MSE: 0.333\n", "evaluation.md"))
        self.assertFalse(output_matches("prediction: 7.1\n", "prediction: 7.0\n", "installation.md"))

    def test_every_float_on_selected_line(self):
        expected = "precision: 0.667, recall: 1.000, F1: 0.800\n"
        self.assertTrue(output_matches(
            "precision: 0.668, recall: 0.999, F1: 0.801\n", expected, "evaluation.md"))
        self.assertFalse(output_matches(
            "precision: 0.668, recall: 0.999, F1: 0.803\n", expected, "evaluation.md"))

    def test_nonfinite_values_fail(self):
        for value in ["NaN", "nan", "inf", "-inf", "Infinity"]:
            with self.subTest(value=value):
                self.assertFalse(output_matches(f"MSE: {value}\n", "MSE: 0.333\n", "evaluation.md"))
                self.assertFalse(output_matches(f"MSE: {value}\n", f"MSE: {value}\n", "evaluation.md"))

    def test_labels_remain_exact(self):
        for actual, expected in [
            ("binary labels: [0.000001, 1.0]\n", "binary labels: [0.0, 1.0]\n"),
            ("classes: [0, 2]\n", "classes: [0, 1]\n"),
            ("first row: [0.999999, 0.0]\n", "first row: [1.0, 0.0]\n"),
        ]:
            with self.subTest(actual=actual):
                self.assertFalse(output_matches(actual, expected, "supervised.md"))

    def test_shapes_counts_and_booleans_remain_exact(self):
        for actual, expected in [
            ("prediction shape: [4, 2]\n", "prediction shape: [4, 1]\n"),
            ("epochs: 201\n", "epochs: 200\n"),
            ("loss decreased: false\n", "loss decreased: true\n"),
        ]:
            with self.subTest(actual=actual):
                self.assertFalse(output_matches(actual, expected, "neural-networks.md"))
        self.assertFalse(numeric_line_matches("step 2: 0.333", "step 1: 0.333", 1e-3))
        self.assertFalse(numeric_line_matches("step 1.0: 0.333", "step 1: 0.333", 1e-3))

    def test_text_spacing_and_line_structure_remain_exact(self):
        for actual in ["MAE: 0.333\n", "MSE:  0.333\n", "MSE: 0.333", "MSE: 0.333\n\n",
                       "MSE: 0.333\nextra\n", "MSE: 0.333, 0.333\n", "MSE: 0.333oops\n"]:
            with self.subTest(actual=actual):
                self.assertFalse(output_matches(actual, "MSE: 0.333\n", "evaluation.md"))

    def test_float_syntax_is_required(self):
        self.assertFalse(output_matches("R2: 1\n", "R2: 1.000\n", "evaluation.md"))

    def test_unselected_lines_remain_exact(self):
        self.assertTrue(output_matches("value: 0.333\n", "value: 0.333\n", "other.md"))
        self.assertFalse(output_matches("value: 0.334\n", "value: 0.333\n", "other.md"))
        self.assertFalse(output_matches("MSE: 0.334\n", "MSE: 0.333\n", "other.md"))


if __name__ == "__main__":
    unittest.main()
