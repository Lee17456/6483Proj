"""Summarize class and head/mid/tail metrics across controlled runs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def write_rows(path: Path, rows: list[dict], columns: list[str]):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = []
    reference_counts = None
    for name in args.runs:
        folder = Path(name).expanduser().resolve()
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        split = json.loads((folder / "split.json").read_text(encoding="utf-8"))
        counts = {int(key): int(value) for key, value in split["train_counts"].items()}
        if reference_counts is None:
            reference_counts = counts
        elif counts != reference_counts:
            raise ValueError(f"Different class counts in {folder}; refuse pooled class metrics")
        ordered = sorted(counts, key=lambda cls: (-counts[cls], cls))
        groups = {"head": ordered[:3], "middle": ordered[3:7], "tail": ordered[7:]}
        classes = config.get("classes") or ["airplane", "automobile", "bird", "cat", "deer",
                                             "dog", "frog", "horse", "ship", "truck"]
        method = folder.name.rsplit("_seed", 1)[0]
        for evaluated in ("val", "test"):
            metrics = json.loads((folder / f"{evaluated}_metrics.json").read_text(encoding="utf-8"))
            for metric_name, field in (("recall", "per_class_recall"), ("f1", "per_class_f1")):
                values = [metrics[field][name] for name in classes]
                for cls, value in enumerate(values):
                    rows.append({"run": str(folder), "method": method, "seed": config["seed"],
                                 "split": evaluated, "metric": metric_name,
                                 "group": "class", "label": classes[cls], "value": value,
                                 "train_count": counts[cls]})
                for group, indices in groups.items():
                    rows.append({"run": str(folder), "method": method, "seed": config["seed"],
                                 "split": evaluated, "metric": metric_name,
                                 "group": group, "label": f"{group}_mean",
                                 "value": statistics.mean(values[i] for i in indices),
                                 "train_count": sum(counts[i] for i in indices)})
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = ["run", "method", "seed", "split", "metric", "group", "label", "value", "train_count"]
    write_rows(output, rows, columns)
    grouped = defaultdict(list)
    for row in rows:
        key = tuple(row[field] for field in ("method", "split", "metric", "group", "label", "train_count"))
        grouped[key].append(row)
    aggregate = []
    for key, members in grouped.items():
        if len({row["seed"] for row in members}) != len(members):
            raise ValueError(f"Duplicate seed for {key}")
        values = [float(row["value"]) for row in members]
        record = dict(zip(("method", "split", "metric", "group", "label", "train_count"), key))
        record.update({"runs": len(values), "mean": statistics.mean(values),
                       "sd": statistics.stdev(values) if len(values) > 1 else 0.0})
        aggregate.append(record)
    aggregate_path = output.with_name(output.stem + "_aggregate.csv")
    write_rows(aggregate_path, aggregate,
               ["method", "split", "metric", "group", "label", "train_count", "runs", "mean", "sd"])
    print(f"Wrote {output} and {aggregate_path}")


if __name__ == "__main__":
    main()
