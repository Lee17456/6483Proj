"""Offline checks for split and metric invariants without the optional torch runtime."""

import ast
import unittest
from pathlib import Path

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "experiment.py"
TREE = ast.parse(SOURCE.read_text(encoding="utf-8"))
FUNCTIONS = {node.name: node for node in TREE.body if isinstance(node, ast.FunctionDef)}
NAMESPACE = {"np": np}
for name in ("stratified_split", "long_tail_subset", "score_rows"):
    one_function = ast.Module(body=[FUNCTIONS[name]], type_ignores=[])
    exec(compile(one_function, str(SOURCE), "exec"), NAMESPACE)


class PureExperimentTests(unittest.TestCase):
    def test_split_is_stratified_and_disjoint(self):
        labels = [0] * 100 + [1] * 100
        train, val = NAMESPACE["stratified_split"](labels, 0.1, 42)
        self.assertEqual(len(train), 180)
        self.assertEqual(len(val), 20)
        self.assertFalse(set(train) & set(val))
        self.assertEqual(sum(labels[i] == 0 for i in val), 10)

    def test_long_tail_changes_only_training_indices(self):
        labels = [cls for cls in range(10) for _ in range(100)]
        train = [i for i in range(1000) if i % 100 < 90]
        reduced = NAMESPACE["long_tail_subset"](train, labels, 0.1, 42, 10)
        self.assertTrue(set(reduced) <= set(train))
        self.assertEqual(sum(labels[i] == 0 for i in reduced), 90)
        self.assertEqual(sum(labels[i] == 9 for i in reduced), 9)
        self.assertFalse(any(i % 100 >= 90 for i in reduced))

    def test_metrics_use_true_rows_and_macro_average(self):
        rows = [
            {"true_label": 0, "pred_label": 0},
            {"true_label": 0, "pred_label": 1},
            {"true_label": 1, "pred_label": 1},
        ]
        result = NAMESPACE["score_rows"](rows, ["cat", "dog"])
        self.assertEqual(result["confusion_matrix_true_rows_pred_columns"], [[1, 1], [0, 1]])
        self.assertAlmostEqual(result["accuracy"], 2 / 3)
        self.assertAlmostEqual(result["balanced_accuracy"], 0.75)
        self.assertAlmostEqual(result["macro_f1"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
