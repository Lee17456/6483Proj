"""Aggregation must never combine different source manifests or duplicate seeds."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from experiment import summarize


def make_run(root: Path, name: str, fingerprint: str, seed: int):
    folder = root / name
    folder.mkdir()
    config = {"dataset": "dogs_cats", "seed": seed, "val_fraction": 0.2,
              "architecture": "resnet18", "pretrained": True, "augmentation": "basic",
              "imbalance_ratio": 1.0, "remedy": "none"}
    split = {"split_seed": 2026, "data_fingerprint_sha256": fingerprint}
    summary = {"selected_epoch": 1, "val": {"labeled": True, "accuracy": 0.8,
               "balanced_accuracy": 0.8, "macro_f1": 0.8}, "test": None}
    for filename, value in (("config.json", config), ("split.json", split), ("summary.json", summary)):
        (folder / filename).write_text(json.dumps(value), encoding="utf-8")
    return folder


class SummarizeTests(unittest.TestCase):
    def test_different_manifests_remain_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = make_run(root, "a", "kaggle", 42)
            b = make_run(root, "b", "microsoft", 42)
            output = root / "runs.csv"
            summarize(SimpleNamespace(runs=[str(a), str(b)], output=str(output)))
            with (root / "runs_aggregate.csv").open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            self.assertEqual({row["data_fingerprint_sha256"] for row in rows}, {"kaggle", "microsoft"})

    def test_duplicate_seed_in_one_group_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = make_run(root, "a", "same", 42)
            b = make_run(root, "b", "same", 42)
            with self.assertRaisesRegex(ValueError, "Duplicate training seeds"):
                summarize(SimpleNamespace(runs=[str(a), str(b)], output=str(root / "runs.csv")))


if __name__ == "__main__":
    unittest.main()
