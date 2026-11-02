"""Exercise both CIFAR imbalance remedies without downloading benchmark data."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from experiment import build_parser


class FakeCIFAR:
    def __init__(self, root, train=True, download=False):
        self.train = train
        self.targets = [i % 10 for i in range(30 if train else 10)]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        rng = np.random.default_rng(index + (0 if self.train else 1000))
        image = Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8))
        return image, self.targets[index]


class CIFARPipelineTests(unittest.TestCase):
    def test_weighting_and_sampling_export_complete_metrics(self):
        with tempfile.TemporaryDirectory() as temp, patch("experiment.datasets.CIFAR10", FakeCIFAR):
            for remedy in ("class_weight", "sampler"):
                with self.subTest(remedy=remedy):
                    output = Path(temp) / remedy
                    args = build_parser().parse_args([
                        "train", "--dataset", "cifar10", "--data-dir", temp,
                        "--output-dir", str(output), "--architecture", "small_cnn",
                        "--augmentation", "strong", "--label-smoothing", "0.1",
                        "--imbalance-ratio", "0.5", "--remedy", remedy,
                        "--val-fraction", "0.34", "--epochs", "1", "--batch-size", "8",
                        "--workers", "0", "--device", "cpu", "--seed", "42",
                    ])
                    args.func(args)
                    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
                    split = json.loads((output / "split.json").read_text(encoding="utf-8"))
                    self.assertEqual(summary["val"]["count"], 10)
                    self.assertEqual(summary["test"]["count"], 10)
                    self.assertEqual(len(split["train_counts"]), 10)
                    self.assertEqual(len(summary["test"]["per_class_recall"]), 10)
                    self.assertTrue((output / "best.pt").is_file())


if __name__ == "__main__":
    unittest.main()
