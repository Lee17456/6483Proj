"""Run the controlled EE6483 experiment matrix and combine metrics."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path


def prefetch_shared_downloads(data_root: Path, catdog_dataset: str) -> None:
    """Avoid concurrent workers extracting the same datasets or model weights."""
    from torchvision import datasets
    from torchvision.models import ResNet18_Weights

    datasets.CIFAR10(data_root, train=True, download=True)
    datasets.CIFAR10(data_root, train=False, download=True)
    if catdog_dataset == "pet":
        datasets.OxfordIIITPet(data_root, split="trainval", target_types="binary-category", download=True)
        datasets.OxfordIIITPet(data_root, split="test", target_types="binary-category", download=True)
    ResNet18_Weights.DEFAULT.get_state_dict(progress=True)


def run_parallel_gpu_jobs(jobs: list[tuple[str, Path, list[str]]], gpu_ids: list[int],
                          poll_interval: float = 0.5) -> None:
    """Run one independent training process per GPU; keep outputs separate."""
    queues = {gpu: deque(jobs[index::len(gpu_ids)]) for index, gpu in enumerate(gpu_ids)}
    active = {}
    try:
        while any(queues.values()) or active:
            for gpu in gpu_ids:
                if gpu in active or not queues[gpu]:
                    continue
                name, run_dir, command = queues[gpu].popleft()
                run_dir.mkdir(parents=True, exist_ok=True)
                log_path = run_dir / "console.log"
                log_stream = log_path.open("w", encoding="utf-8")
                child_env = os.environ.copy()
                child_env["PYTHONUNBUFFERED"] = "1"
                try:
                    process = subprocess.Popen(command, stdout=log_stream,
                                               stderr=subprocess.STDOUT, env=child_env)
                except BaseException:
                    log_stream.close()
                    raise
                active[gpu] = (process, log_stream, name, log_path)
                print(f"Started {name} on visible GPU {gpu}; log: {log_path}", flush=True)
            time.sleep(poll_interval)
            for gpu in list(active):
                process, log_stream, name, log_path = active[gpu]
                return_code = process.poll()
                if return_code is None:
                    continue
                log_stream.close()
                del active[gpu]
                if return_code != 0:
                    raise RuntimeError(f"{name} failed on GPU {gpu} (exit {return_code}); see {log_path}")
                print(f"Finished {name} on GPU {gpu}", flush=True)
    except BaseException:
        for process, log_stream, _, _ in active.values():
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            log_stream.close()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--catdog-dataset", choices=["dogs_cats", "assignment", "pet"], default="dogs_cats")
    parser.add_argument("--catdog-dir", default=None,
                        help="Defaults to data/dogs_cats, data/assignment, or data for the chosen source")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--quick", action="store_true", help="One seed, one epoch: pipeline check only")
    parser.add_argument("--dry-run", action="store_true", help="Print the full command matrix without training")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--gpu-ids", nargs="+", type=int, default=None,
                        help="Run one job per logical GPU ID visible in the current session")
    parser.add_argument("--pet-batch", type=int, default=16)
    parser.add_argument("--cifar-batch", type=int, default=128)
    args = parser.parse_args()
    if args.gpu_ids is not None:
        if args.device == "cpu" or len(set(args.gpu_ids)) != len(args.gpu_ids) or any(gpu < 0 for gpu in args.gpu_ids):
            parser.error("--gpu-ids requires unique non-negative IDs and a CUDA device")
    root = Path(__file__).resolve().parent
    output = Path(args.results_dir).expanduser().resolve()
    if not args.dry_run:
        output.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_dir).expanduser().resolve()
    catdog_root = Path(args.catdog_dir).expanduser().resolve() if args.catdog_dir else (
        data_root if args.catdog_dataset == "pet" else data_root / args.catdog_dataset
    )
    seeds = args.seeds[:1] if args.quick else args.seeds
    designs = [
        ("catdog_resnet", args.catdog_dataset, "resnet18", "basic", 1.0, "none", True),
        ("catdog_resnet_scratch", args.catdog_dataset, "resnet18", "basic", 1.0, "none", False),
        ("catdog_smallcnn", args.catdog_dataset, "small_cnn", "basic", 1.0, "none", False),
        ("cifar_basic", "cifar10", "resnet18", "basic", 1.0, "none", False),
        ("cifar_strong", "cifar10", "resnet18", "strong", 1.0, "none", False),
        ("cifar_longtail", "cifar10", "resnet18", "strong", 0.1, "none", False),
        ("cifar_weight", "cifar10", "resnet18", "strong", 0.1, "class_weight", False),
        ("cifar_sampler", "cifar10", "resnet18", "strong", 0.1, "sampler", False),
    ]
    runs = []
    jobs = []
    for name, dataset, architecture, augmentation, ratio, remedy, pretrained in designs:
        for seed in seeds:
            run_dir = output / f"{name}_seed{seed}"
            command = [sys.executable, str(root / "experiment.py"), "train", "--dataset", dataset,
                       "--data-dir", str(data_root if dataset == "cifar10" else catdog_root),
                       "--output-dir", str(run_dir), "--architecture", architecture,
                       "--augmentation", augmentation, "--imbalance-ratio", str(ratio),
                       "--remedy", remedy, "--seed", str(seed), "--split-seed", str(args.split_seed),
                       "--workers", str(args.workers),
                       "--device", "cuda" if args.gpu_ids is not None else args.device,
                       "--batch-size", str(args.pet_batch if dataset != "cifar10" else args.cifar_batch),
                       "--epochs", str(1 if args.quick else 20 if dataset != "cifar10" else 100),
                       "--lr", "0.0003" if dataset != "cifar10" else "0.001", "--amp"]
            if dataset != "cifar10":
                command += ["--val-fraction", "0.2" if dataset == "dogs_cats" else "0.1"]
            if dataset in ("cifar10", "pet"):
                command.append("--download")
            if pretrained:
                command.append("--pretrained")
            if dataset == "cifar10":
                command += ["--label-smoothing", "0.1"]
            if args.gpu_ids is not None:
                command += ["--cuda-index", str(args.gpu_ids[len(jobs) % len(args.gpu_ids)])]
            jobs.append((f"{name}_seed{seed}", run_dir, command))
            runs.append(str(run_dir))
    if args.dry_run:
        for index, (_, _, command) in enumerate(jobs):
            assignment = f"GPU {args.gpu_ids[index % len(args.gpu_ids)]}" if args.gpu_ids else "sequential"
            print(f"Planned [{assignment}]:", " ".join(command))
        return
    if args.gpu_ids is not None:
        import torch

        visible_count = torch.cuda.device_count()
        if any(gpu >= visible_count for gpu in args.gpu_ids):
            parser.error(f"Requested GPU indices {args.gpu_ids}, but only {visible_count} GPUs are visible. "
                         "Check CUDA_VISIBLE_DEVICES and the server's GPU allocation.")
        if args.catdog_dataset == "dogs_cats" and not (catdog_root / "dogs_cats_manifest.json").is_file():
            parser.error(f"Missing verified cat/dog manifest in {catdog_root}; prepare that dataset first")
        print("Checking shared downloads before launching parallel GPU jobs", flush=True)
        prefetch_shared_downloads(data_root, args.catdog_dataset)
        run_parallel_gpu_jobs(jobs, args.gpu_ids)
    else:
        for name, _, command in jobs:
            print(f"Running {name}:", " ".join(command), flush=True)
            subprocess.run(command, check=True)
    subprocess.run([sys.executable, str(root / "experiment.py"), "summarize", "--runs", *runs,
                    "--output", str(output / "all_runs.csv")], check=True)


if __name__ == "__main__":
    main()
