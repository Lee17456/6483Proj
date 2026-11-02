"""Run comparable CIFAR-10-LT CE, RDR and ICL experiments by seed."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from run_suite import run_parallel_gpu_jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--reference-split")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--methods", nargs="+", choices=["ce", "rdr", "icl2", "icl3"],
                        default=["ce", "rdr", "icl2"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--icl-finetune-epochs", type=int, default=0)
    parser.add_argument("--icl-ad-weight", type=float, default=0.05)
    parser.add_argument("--icl-ccl-weight", type=float, default=0.1)
    parser.add_argument("--icl-kd-weight", type=float, default=1.0)
    parser.add_argument("--icl-topk", type=int, default=3)
    parser.add_argument("--icl-temperature", type=float, default=0.1)
    parser.add_argument("--rdr-ridge", type=float, default=1e-4)
    parser.add_argument("--rdr-temperature", type=float, default=1.0)
    parser.add_argument("--gpu-ids", nargs="+", type=int,
                        help="Logical GPU indices visible to this Conda session")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Two-epoch pipeline smoke test")
    args = parser.parse_args()
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must be nonempty and unique")
    if len(set(args.methods)) != len(args.methods):
        parser.error("--methods must be unique")
    if args.gpu_ids and (args.device != "cuda" or len(set(args.gpu_ids)) != len(args.gpu_ids)
                         or any(i < 0 for i in args.gpu_ids)):
        parser.error("--gpu-ids needs distinct nonnegative CUDA indices")
    if args.epochs < 2:
        parser.error("--epochs must be at least 2")
    root = Path(__file__).resolve().parent
    output = Path(args.results_dir).expanduser().resolve()
    data = Path(args.data_dir).expanduser().resolve()
    seeds = args.seeds[:1] if args.quick else args.seeds
    epochs = 2 if args.quick else args.epochs
    jobs, runs = [], []
    for method_name in args.methods:
        for seed in seeds:
            run_dir = output / f"{method_name}_seed{seed}"
            method = "icl" if method_name.startswith("icl") else method_name
            gpu = args.gpu_ids[len(jobs) % len(args.gpu_ids)] if args.gpu_ids else 0
            command = [sys.executable, str(root / "advanced_longtail.py"),
                       "--method", method, "--data-dir", str(data),
                       "--output-dir", str(run_dir), "--seed", str(seed),
                       "--split-seed", str(args.split_seed), "--epochs", str(epochs),
                       "--batch-size", str(args.batch_size), "--workers", str(args.workers),
                       "--lr", "0.001", "--weight-decay", "0.0001",
                       "--augmentation", "strong", "--imbalance-ratio", "0.1",
                       "--label-smoothing", "0.1", "--device", args.device,
                       "--cuda-index", str(gpu)]
            if args.device == "cuda":
                command.append("--amp")
            if args.reference_split:
                command += ["--reference-split", str(Path(args.reference_split).expanduser().resolve())]
            if method == "rdr":
                command += ["--rdr-warmup", str(1 if args.quick else round(epochs * 0.8)),
                            "--rdr-ridge", str(args.rdr_ridge),
                            "--rdr-temperature", str(args.rdr_temperature)]
            if method == "icl":
                command += ["--icl-experts", method_name[-1],
                            "--icl-ad-until", str(1 if args.quick else round(epochs / 6)),
                            "--icl-ccl-after", str(1 if args.quick else round(epochs * 2 / 3)),
                            "--icl-finetune-epochs", str(0 if args.quick else args.icl_finetune_epochs),
                            "--icl-ad-weight", str(args.icl_ad_weight),
                            "--icl-ccl-weight", str(args.icl_ccl_weight),
                            "--icl-kd-weight", str(args.icl_kd_weight),
                            "--icl-topk", str(args.icl_topk),
                            "--icl-temperature", str(args.icl_temperature)]
            jobs.append((f"{method_name}_seed{seed}", run_dir, command))
            runs.append(str(run_dir))
    if args.dry_run:
        for name, _, command in jobs:
            print(name, " ".join(command))
        return
    if args.gpu_ids:
        import torch
        visible = torch.cuda.device_count()
        if any(i >= visible for i in args.gpu_ids):
            parser.error(f"Requested {args.gpu_ids}, but only {visible} GPUs are visible")
    elif args.device == "cuda":
        import torch
        if torch.cuda.device_count() < 1:
            parser.error("No GPU visible to PyTorch")
    output.mkdir(parents=True, exist_ok=True)
    if args.gpu_ids:
        run_parallel_gpu_jobs(jobs, args.gpu_ids)
    else:
        for name, _, command in jobs:
            print("Running", name, flush=True)
            subprocess.run(command, check=True)
    subprocess.run([sys.executable, str(root / "experiment.py"), "summarize", "--runs",
                    *runs, "--output", str(output / "advanced_runs.csv")], check=True)
    subprocess.run([sys.executable, str(root / "scripts" / "summarize_advanced_classes.py"),
                    "--runs", *runs, "--output", str(output / "advanced_class_metrics.csv")],
                   check=True)


if __name__ == "__main__":
    main()
