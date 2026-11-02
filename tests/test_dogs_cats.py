"""Verify the large cat/dog manifest and split control with small local fixtures."""

import tempfile
import unittest
import io
import zipfile
from pathlib import Path

from PIL import Image

from experiment import build_parser, load_data
from prepare_dogs_cats import prepare


class DogsCatsTests(unittest.TestCase):
    def test_microsoft_zip_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with zipfile.ZipFile(root / "kagglecatsanddogs_5340.zip", "w") as archive:
                for cls in ("Cat", "Dog"):
                    buffer = io.BytesIO()
                    Image.new("RGB", (16, 16), "yellow").save(buffer, format="JPEG")
                    archive.writestr(f"PetImages/{cls}/0.jpg", buffer.getvalue())
            metadata = prepare(root)
            self.assertEqual(metadata["source"], "microsoft")
            self.assertEqual(metadata["class_counts"], {"cat": 1, "dog": 1})

    def test_kaggle_zip_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with zipfile.ZipFile(root / "train.zip", "w") as archive:
                for cls in ("cat", "dog"):
                    buffer = io.BytesIO()
                    Image.new("RGB", (16, 16), "green").save(buffer, format="JPEG")
                    archive.writestr(f"train/{cls}.0.jpg", buffer.getvalue())
                archive.writestr("train/README.txt", "ignored")
            metadata = prepare(root)
            self.assertEqual(metadata["source"], "kaggle")
            self.assertEqual(metadata["class_counts"], {"cat": 1, "dog": 1})
            self.assertTrue((root / "train" / "cat.0.jpg").is_file())

    def test_fixed_balanced_split_and_corrupt_filter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train"
            train.mkdir()
            for cls, amount in (("cat", 12), ("dog", 10)):
                for index in range(amount):
                    Image.new("RGB", (32, 32), (index, 0, 0)).save(train / f"{cls}.{index}.jpg")
            (train / "cat.99.jpg").write_bytes(b"invalid image")
            metadata = prepare(root)
            self.assertEqual(metadata["candidate_count"], 23)
            self.assertEqual(metadata["valid_count"], 22)
            self.assertEqual(len(metadata["rejected"]), 1)
            splits = []
            for seed in (42, 43):
                args = build_parser().parse_args([
                    "train", "--dataset", "dogs_cats", "--data-dir", directory,
                    "--output-dir", str(root / f"run{seed}"), "--min-class-images", "1",
                    "--seed", str(seed), "--split-seed", "2026", "--val-fraction", "0.2",
                ])
                _, _, _, _, split = load_data(args)
                self.assertEqual(split["train_counts"], {0: 8, 1: 8})
                self.assertEqual(split["val_counts"], {0: 2, 1: 2})
                splits.append((split["train_indices"], split["val_indices"]))
            self.assertEqual(splits[0], splits[1])
            hidden = root / "test1"
            hidden.mkdir()
            Image.new("RGB", (32, 32), "red").save(hidden / "10.jpg")
            Image.new("RGB", (32, 32), "blue").save(hidden / "2.jpg")
            args = build_parser().parse_args([
                "train", "--dataset", "dogs_cats", "--data-dir", directory,
                "--output-dir", str(root / "complete_run"), "--min-class-images", "1",
                "--architecture", "small_cnn", "--image-size", "32", "--epochs", "1",
                "--batch-size", "4", "--workers", "0", "--device", "cpu",
            ])
            args.func(args)
            submission = (root / "complete_run" / "submission.csv").read_text(encoding="utf-8-sig")
            self.assertEqual([line.split(",")[0] for line in submission.splitlines()], ["id", "2", "10"])
            Image.new("RGB", (32, 32), "white").save(train / "cat.0.jpg")
            with self.assertRaisesRegex(ValueError, "changed after manifest"):
                load_data(args)


if __name__ == "__main__":
    unittest.main()
