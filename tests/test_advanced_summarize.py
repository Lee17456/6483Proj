"""Ensure tail-group summaries use train counts and keep seed runs separate."""

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


class ClassSummaryTests(unittest.TestCase):
    def test_tail_group_and_seed_aggregation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = []
            for seed, tail_recall in ((42, 0.4), (43, 0.6)):
                folder = root / f"icl2_seed{seed}"
                folder.mkdir()
                (folder / "config.json").write_text(json.dumps({"seed": seed, "classes": CLASSES}))
                (folder / "split.json").write_text(json.dumps({"train_counts":
                    {str(i): 1000 - i * 100 for i in range(10)}}))
                recalls = {name: (tail_recall if i >= 7 else 0.9)
                           for i, name in enumerate(CLASSES)}
                metrics = {"per_class_recall": recalls, "per_class_f1": recalls}
                for split in ("val", "test"):
                    (folder / f"{split}_metrics.json").write_text(json.dumps(metrics))
                runs.append(folder)
            script = Path(__file__).resolve().parents[1] / "scripts" / "summarize_advanced_classes.py"
            output = root / "classes.csv"
            subprocess.run([sys.executable, str(script), "--runs", *(str(p) for p in runs),
                            "--output", str(output)], check=True, capture_output=True)
            with (root / "classes_aggregate.csv").open(newline="", encoding="utf-8-sig") as stream:
                rows = list(csv.DictReader(stream))
            tail = next(row for row in rows if row["method"] == "icl2" and row["split"] == "test"
                        and row["metric"] == "recall" and row["label"] == "tail_mean")
            self.assertEqual(tail["runs"], "2")
            self.assertAlmostEqual(float(tail["mean"]), 0.5)
            self.assertEqual(tail["train_count"], "600")


if __name__ == "__main__":
    unittest.main()
