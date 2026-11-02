"""Validate a large public Dogs vs Cats dataset and freeze its file manifest.

Supported inputs below --data-dir:
  train.zip                 Kaggle competition training archive (extracted once)
  train/cat.0.jpg ...       Extracted Kaggle training images
  kagglecatsanddogs_5340.zip  Microsoft public archive (extracted once)
  PetImages/Cat/*.jpg ...   Microsoft Cats and Dogs images
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image, ImageFile


KAGGLE_NAME = re.compile(r"^(cat|dog)\.\d+\.(?:jpg|jpeg|png)$", re.IGNORECASE)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
ImageFile.LOAD_TRUNCATED_IMAGES = False


def extract_kaggle_train(archive: Path, root: Path) -> None:
    target = root / "train"
    target.mkdir(parents=True, exist_ok=True)
    seen = set()
    with zipfile.ZipFile(archive) as source:
        for info in source.infolist():
            if info.is_dir():
                continue
            filename = Path(info.filename).name
            if not KAGGLE_NAME.fullmatch(filename):
                continue
            if filename.lower() in seen:
                raise ValueError(f"Duplicate image name in archive: {filename}")
            seen.add(filename.lower())
            destination = target / filename
            if destination.exists() and destination.stat().st_size == info.file_size:
                continue
            destination.write_bytes(source.read(info))


def extract_microsoft(archive: Path, root: Path) -> None:
    with zipfile.ZipFile(archive) as source:
        for info in source.infolist():
            if info.is_dir():
                continue
            parts = Path(info.filename.replace("\\", "/")).parts
            if len(parts) != 3 or parts[0] != "PetImages" or parts[1] not in ("Cat", "Dog"):
                continue
            if Path(parts[2]).suffix.lower() not in IMAGE_SUFFIXES:
                continue
            destination = root / "PetImages" / parts[1] / parts[2]
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and destination.stat().st_size == info.file_size:
                continue
            destination.write_bytes(source.read(info))


def discover(root: Path) -> tuple[str, list[tuple[Path, int]]]:
    flat = root / "train"
    microsoft = root / "PetImages"
    if flat.is_dir() and microsoft.is_dir():
        raise ValueError("Do not mix Kaggle train and Microsoft PetImages in one data directory")
    if flat.is_dir():
        paths = sorted(p for p in flat.iterdir() if p.is_file() and KAGGLE_NAME.fullmatch(p.name))
        return "kaggle", [(p, 0 if p.name.lower().startswith("cat.") else 1) for p in paths]
    if (microsoft / "Cat").is_dir() and (microsoft / "Dog").is_dir():
        records = []
        for cls, label in (("Cat", 0), ("Dog", 1)):
            records.extend((p, label) for p in sorted((microsoft / cls).iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        return "microsoft", records
    raise FileNotFoundError("Expected train.zip, train/cat.N.jpg and train/dog.N.jpg, or PetImages/Cat and PetImages/Dog")


def prepare(root: Path) -> dict:
    root = root.expanduser().resolve()
    microsoft_zip = root / "kagglecatsanddogs_5340.zip"
    kaggle_zip = root / "train.zip"
    if (kaggle_zip.is_file() or (root / "train").is_dir()) and (
        microsoft_zip.is_file() or (root / "PetImages").is_dir()
    ):
        raise ValueError("Do not mix Kaggle and Microsoft image sources in one data directory")
    if microsoft_zip.is_file():
        extract_microsoft(microsoft_zip, root)
    if kaggle_zip.is_file():
        extract_kaggle_train(kaggle_zip, root)
    source, files = discover(root)
    if not files:
        raise ValueError("No matching cat/dog images found")
    rows, rejected = [], []
    fingerprint = hashlib.sha256()
    for path, label in files:
        relative = path.relative_to(root).as_posix()
        try:
            contents = path.read_bytes()
            with Image.open(io.BytesIO(contents)) as image:
                image.convert("RGB").load()
            digest = hashlib.sha256(contents).hexdigest()
        except (OSError, ValueError, SyntaxError) as error:
            rejected.append({"path": relative, "error": type(error).__name__})
            continue
        rows.append({"path": relative, "label": label, "bytes": len(contents), "sha256": digest})
        fingerprint.update(f"{relative}\t{label}\t{digest}\n".encode("utf-8"))
    if not rows:
        raise ValueError("All candidate images were rejected")
    with (root / "dogs_cats_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "label", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "source": source, "candidate_count": len(files), "valid_count": len(rows),
        "class_counts": {"cat": Counter(row["label"] for row in rows)[0],
                         "dog": Counter(row["label"] for row in rows)[1]},
        "rejected": rejected, "fingerprint_sha256": fingerprint.hexdigest(),
    }
    (root / "dogs_cats_manifest.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()
    result = prepare(Path(args.data_dir))
    print(json.dumps({key: value for key, value in result.items() if key != "rejected"}, ensure_ascii=False, indent=2))
    print(f"Rejected images: {len(result['rejected'])}")
