"""Package plot-ready numeric results without large model checkpoints."""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.results_dir).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not root.is_dir():
        parser.error(f"Missing result directory: {root}")
    if output.is_relative_to(root):
        parser.error("Place the archive outside the result directory")
    files = sorted(path for path in root.rglob("*") if path.is_file()
                   and path.suffix.lower() in {".csv", ".json"})
    if not files:
        parser.error(f"No CSV or JSON result files in {root}")
    if not (root / "advanced_runs_aggregate.csv").is_file():
        parser.error("Missing final aggregate; finish all training jobs first")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        for file in files:
            archive.add(file, arcname=str(Path(root.name) / file.relative_to(root)))
    print(f"Packed {len(files)} numeric files into {output}")


if __name__ == "__main__":
    main()
