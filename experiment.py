"""Reproducible EE6483 image-classification experiments.

Run ``python experiment.py --help`` for the command line interface.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
    from torchvision import datasets, models, transforms
    from torchvision.models import ResNet18_Weights
except ImportError as exc:
    raise SystemExit(
        "Missing PyTorch dependencies. Install requirements.txt with the Python "
        "environment used to run this script."
    ) from exc


PET_CLASSES = ["cat", "dog"]
CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def seed_everything(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        torch.use_deterministic_algorithms(True)


def worker_init(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def stratified_split(labels: list[int], val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)
    train, val = [], []
    for cls in sorted(set(labels)):
        indices = np.flatnonzero(np.asarray(labels) == cls)
        rng.shuffle(indices)
        n_val = max(1, round(len(indices) * val_fraction))
        if n_val >= len(indices):
            raise ValueError(f"Class {cls} has too few samples for a validation split")
        val.extend(indices[:n_val].tolist())
        train.extend(indices[n_val:].tolist())
    return sorted(train), sorted(val)


def long_tail_subset(indices: list[int], labels: list[int], ratio: float, seed: int, classes: int) -> list[int]:
    if not 0 < ratio <= 1:
        raise ValueError("--imbalance-ratio must be in (0, 1]")
    rng = np.random.default_rng(seed)
    by_class = [[i for i in indices if labels[i] == cls] for cls in range(classes)]
    maximum = min(map(len, by_class))
    chosen = []
    for cls, candidates in enumerate(by_class):
        desired = max(1, round(maximum * ratio ** (cls / (classes - 1))))
        chosen.extend(rng.choice(candidates, size=min(desired, len(candidates)), replace=False).tolist())
    return sorted(chosen)


class IndexedImages(Dataset):
    def __init__(self, base: Dataset, indices: list[int], labels: list[int], transform, ids: list[str]):
        self.base, self.indices, self.labels, self.transform, self.ids = base, indices, labels, transform, ids

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int):
        index = self.indices[position]
        image, _ = self.base[index]
        return self.transform(image), self.labels[index], self.ids[index]

    @property
    def targets(self) -> list[int]:
        return [self.labels[i] for i in self.indices]


class ManifestImages(Dataset):
    def __init__(self, root: Path):
        manifest = root / "dogs_cats_manifest.csv"
        metadata = root / "dogs_cats_manifest.json"
        if not manifest.is_file() or not metadata.is_file():
            raise FileNotFoundError(
                f"Missing verified manifest in {root}. Run: python prepare_dogs_cats.py --data-dir {root}"
            )
        with manifest.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.paths = [(root / row["path"]).resolve() for row in rows]
        if any(not path.is_relative_to(root) for path in self.paths):
            raise ValueError("Manifest contains a path outside its data directory")
        self.targets = [int(row["label"]) for row in rows]
        self.ids = [path.stem for path in self.paths]
        self.metadata = json.loads(metadata.read_text(encoding="utf-8"))
        if len(self.paths) != self.metadata["valid_count"]:
            raise ValueError("Manifest row count does not match metadata; regenerate the manifest")
        fingerprint = hashlib.sha256()
        for path, row in zip(self.paths, rows):
            if not path.is_file():
                raise FileNotFoundError(f"Manifest image missing: {path}")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != row["sha256"]:
                raise ValueError(f"Image changed after manifest creation: {path}")
            fingerprint.update(f"{row['path']}\t{row['label']}\t{digest}\n".encode("utf-8"))
        if fingerprint.hexdigest() != self.metadata["fingerprint_sha256"]:
            raise ValueError("Manifest fingerprint changed; regenerate the manifest")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        with Image.open(self.paths[index]) as image:
            return image.convert("RGB"), self.targets[index]


class UnlabeledImages(Dataset):
    def __init__(self, folder: Path, transform):
        self.paths = sorted((p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS),
                            key=lambda p: (0, int(p.stem)) if p.stem.isdecimal() else (1, p.name.lower()))
        if not self.paths:
            raise ValueError(f"No test images found in {folder}")
        if len({p.stem for p in self.paths}) != len(self.paths):
            raise ValueError("Test image stems must be unique for submission IDs")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB")
            return self.transform(image), -1, self.paths[index].stem


def image_transform(dataset: str, train: bool, size: int, augmentation: str):
    if dataset == "cifar10":
        steps = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()] if train else []
        if train and augmentation == "strong":
            steps.append(transforms.RandAugment(num_ops=2, magnitude=9))
        steps += [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]
    else:
        steps = [transforms.RandomResizedCrop(size, scale=(0.70, 1.0)), transforms.RandomHorizontalFlip()] if train else [transforms.Resize(round(size * 1.14)), transforms.CenterCrop(size)]
        if train and augmentation == "strong":
            steps.append(transforms.RandAugment(num_ops=2, magnitude=9))
        steps += [transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    return transforms.Compose(steps)


def load_data(args):
    data_root = Path(args.data_dir).expanduser().resolve()
    train_tf = image_transform(args.dataset, True, args.image_size, args.augmentation)
    eval_tf = image_transform(args.dataset, False, args.image_size, args.augmentation)
    if args.dataset == "pet":
        source = datasets.OxfordIIITPet(data_root, split="trainval", target_types="binary-category", download=args.download)
        test_source = datasets.OxfordIIITPet(data_root, split="test", target_types="binary-category", download=args.download)
        labels = list(source._bin_labels)
        train_idx, val_idx = stratified_split(labels, args.val_fraction, args.split_seed)
        ids = [p.stem for p in source._images]
        test_labels = list(test_source._bin_labels)
        test_ids = [p.stem for p in test_source._images]
        val = IndexedImages(source, val_idx, labels, eval_tf, ids)
        test = IndexedImages(test_source, list(range(len(test_source))), test_labels, eval_tf, test_ids)
        classes = PET_CLASSES
    elif args.dataset == "dogs_cats":
        source = ManifestImages(data_root)
        labels = list(source.targets)
        counts = Counter(labels)
        if min(counts[0], counts[1]) < args.min_class_images:
            raise ValueError(f"Large cat/dog experiment needs >= {args.min_class_images} valid images per class; got {dict(counts)}")
        equal_count = min(counts[0], counts[1])
        split_rng = np.random.default_rng(args.split_seed)
        balanced_indices = []
        for cls in (0, 1):
            candidates = np.flatnonzero(np.asarray(labels) == cls)
            balanced_indices.extend(split_rng.choice(candidates, size=equal_count, replace=False).tolist())
        balanced_indices.sort()
        local_train, local_val = stratified_split([labels[i] for i in balanced_indices], args.val_fraction, args.split_seed)
        train_idx = [balanced_indices[i] for i in local_train]
        val_idx = [balanced_indices[i] for i in local_val]
        ids = list(source.ids)
        val = IndexedImages(source, val_idx, labels, eval_tf, ids)
        test = UnlabeledImages(data_root / "test1", eval_tf) if (data_root / "test1").is_dir() else None
        classes = PET_CLASSES
    elif args.dataset == "cifar10":
        source = datasets.CIFAR10(data_root, train=True, download=args.download)
        test_source = datasets.CIFAR10(data_root, train=False, download=args.download)
        labels = list(source.targets)
        train_idx, val_idx = stratified_split(labels, args.val_fraction, args.split_seed)
        ids = [str(i) for i in range(len(source))]
        val = IndexedImages(source, val_idx, labels, eval_tf, ids)
        test = IndexedImages(test_source, list(range(len(test_source))), list(test_source.targets), eval_tf, [str(i) for i in range(len(test_source))])
        classes = CIFAR_CLASSES
    else:
        train_path, val_path = data_root / "train", data_root / "val"
        if not train_path.is_dir() or not val_path.is_dir():
            raise ValueError("Assignment data requires data/train/{cat,dog} and data/val/{cat,dog}")
        source, val_source = datasets.ImageFolder(train_path), datasets.ImageFolder(val_path)
        if source.classes != PET_CLASSES or val_source.classes != PET_CLASSES:
            raise ValueError("Both assignment splits must contain exactly cat and dog folders")
        labels = list(source.targets)
        train_idx = list(range(len(source)))
        val = IndexedImages(val_source, list(range(len(val_source))), list(val_source.targets), eval_tf, [Path(p).stem for p, _ in val_source.samples])
        test = UnlabeledImages(data_root / "test", eval_tf) if (data_root / "test").is_dir() else None
        ids = [Path(p).stem for p, _ in source.samples]
        classes = PET_CLASSES
    original_train_idx = list(train_idx)
    if args.dataset == "cifar10" and args.imbalance_ratio < 1:
        train_idx = long_tail_subset(train_idx, labels, args.imbalance_ratio, args.split_seed, len(classes))
    elif args.imbalance_ratio < 1:
        raise ValueError("Synthetic long-tail imbalance is supported only for CIFAR-10")
    train = IndexedImages(source, train_idx, labels, train_tf, ids)
    split = {
        "dataset": args.dataset, "seed": args.seed, "split_seed": args.split_seed,
        "data_fingerprint_sha256": source.metadata["fingerprint_sha256"] if args.dataset == "dogs_cats" else None,
        "train_indices_before_imbalance": original_train_idx,
        "train_indices": train_idx, "val_indices": val.indices,
        "train_counts": dict(sorted(Counter(train.targets).items())),
        "val_counts": dict(sorted(Counter(val.targets).items())),
        "source_counts": dict(sorted(Counter(labels).items())),
        "test_count": len(test) if test is not None else 0,
    }
    return train, val, test, classes, split


class SmallCNN(nn.Module):
    def __init__(self, classes: int):
        super().__init__()
        layers = []
        channels = [3, 32, 64, 128, 256]
        for incoming, outgoing in zip(channels[:-1], channels[1:]):
            layers.extend([nn.Conv2d(incoming, outgoing, 3, padding=1), nn.BatchNorm2d(outgoing), nn.ReLU(inplace=True), nn.MaxPool2d(2)])
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(256, classes))

    def forward(self, x):
        return self.head(self.features(x))


def build_model(dataset: str, architecture: str, classes: int, pretrained: bool):
    if architecture == "small_cnn":
        if pretrained:
            raise ValueError("Pretrained weights are available only for ResNet18")
        return SmallCNN(classes)
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    if dataset == "cifar10":
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, classes)
    return model


def class_balanced_weights(targets: list[int], classes: int, beta: float):
    counts = np.bincount(targets, minlength=classes)
    if np.any(counts == 0):
        raise ValueError("Every class needs at least one training sample")
    effective = (1 - np.power(beta, counts)) / (1 - beta)
    weights = 1 / effective
    weights *= classes / weights.sum()
    return torch.tensor(weights, dtype=torch.float32)


def make_loader(dataset, args, train: bool, sampler=None):
    generator = torch.Generator().manual_seed(args.seed)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=train and sampler is None,
                      sampler=sampler, num_workers=args.workers, pin_memory=torch.cuda.is_available(),
                      worker_init_fn=worker_init, generator=generator)


def score_rows(rows: list[dict], classes: list[str]) -> dict:
    labeled = [r for r in rows if r["true_label"] != -1]
    if not labeled:
        return {"count": len(rows), "labeled": False}
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for row in labeled:
        matrix[int(row["true_label"]), int(row["pred_label"])] += 1
    support = matrix.sum(axis=1)
    recall = np.divide(np.diag(matrix), support, out=np.zeros(len(classes), dtype=float), where=support > 0)
    precision = np.divide(np.diag(matrix), matrix.sum(axis=0), out=np.zeros(len(classes), dtype=float), where=matrix.sum(axis=0) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(len(classes), dtype=float), where=precision + recall > 0)
    return {
        "count": len(labeled), "labeled": True, "accuracy": float(np.trace(matrix) / len(labeled)),
        "balanced_accuracy": float(recall.mean()), "macro_f1": float(f1.mean()),
        "per_class_recall": {name: float(recall[i]) for i, name in enumerate(classes)},
        "per_class_f1": {name: float(f1[i]) for i, name in enumerate(classes)},
        "confusion_matrix_true_rows_pred_columns": matrix.tolist(),
    }


@torch.inference_mode()
def predict_rows(model, loader, device, classes):
    model.eval()
    rows = []
    for images, targets, ids in loader:
        probabilities = torch.softmax(model(images.to(device)), dim=1).cpu().numpy()
        for i, probability in enumerate(probabilities):
            pred = int(probability.argmax())
            row = {"id": ids[i], "true_label": int(targets[i]), "pred_label": pred,
                   "pred_name": classes[pred], "confidence": float(probability[pred])}
            row.update({f"prob_{name}": float(probability[k]) for k, name in enumerate(classes)})
            rows.append(row)
    return rows


def export_evaluation(output: Path, split_name: str, rows: list[dict], classes: list[str]):
    columns = ["id", "true_label", "pred_label", "pred_name", "confidence"] + [f"prob_{name}" for name in classes]
    write_csv(output / f"{split_name}_predictions.csv", rows, columns)
    metrics = score_rows(rows, classes)
    write_json(output / f"{split_name}_metrics.json", metrics)
    if metrics["labeled"]:
        wrong = sorted((r for r in rows if r["true_label"] != r["pred_label"]), key=lambda r: -r["confidence"])
        correct = sorted((r for r in rows if r["true_label"] == r["pred_label"]), key=lambda r: r["confidence"])
        write_csv(output / f"{split_name}_errors.csv", wrong, columns)
        write_csv(output / f"{split_name}_correct_low_confidence.csv", correct[:50], columns)
    return metrics


def train(args):
    if args.val_fraction is None:
        args.val_fraction = 0.2 if args.dataset == "dogs_cats" else 0.1
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        raise ValueError("epochs, batch size and learning rate must be positive")
    if not 0 < args.val_fraction < 1:
        raise ValueError("--val-fraction must be in (0, 1)")
    if not 0 <= args.label_smoothing < 1:
        raise ValueError("--label-smoothing must be in [0, 1)")
    if not 0 <= args.beta < 1:
        raise ValueError("--beta must be in [0, 1)")
    if args.remedy != "none" and args.imbalance_ratio == 1:
        raise ValueError("Choose --imbalance-ratio below 1 when using an imbalance remedy")
    if args.dataset == "cifar10" and args.pretrained:
        raise ValueError("The CIFAR-specific 3x3 stem is trained from scratch; omit --pretrained")
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed, deterministic=not args.fast)
    use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    if use_cuda:
        visible_count = torch.cuda.device_count()
        if args.cuda_index < 0 or args.cuda_index >= visible_count:
            raise ValueError(f"CUDA device {args.cuda_index} requested, but only {visible_count} visible")
        device = torch.device(f"cuda:{args.cuda_index}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    train_data, val_data, test_data, classes, split = load_data(args)
    write_json(output / "split.json", split)
    model = build_model(args.dataset, args.architecture, len(classes), args.pretrained).to(device)
    weights = class_balanced_weights(train_data.targets, len(classes), args.beta).to(device) if args.remedy == "class_weight" else None
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    sampler = None
    if args.remedy == "sampler":
        counts = Counter(train_data.targets)
        sample_weights = torch.tensor([1 / counts[t] for t in train_data.targets], dtype=torch.double)
        sampler = WeightedRandomSampler(sample_weights, len(train_data), replacement=True,
                                        generator=torch.Generator().manual_seed(args.seed))
    train_loader = make_loader(train_data, args, True, sampler)
    val_loader = make_loader(val_data, args, False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.amp)
    history, best_score = [], -math.inf
    config = {key: value for key, value in vars(args).items() if key != "func"}
    config.update({"resolved_data_dir": str(Path(args.data_dir).resolve()), "classes": classes,
                   "torch_version": torch.__version__, "torchvision_version": __import__("torchvision").__version__,
                   "device_used": str(device), "train_count": len(train_data), "val_count": len(val_data),
                   "test_count": len(test_data) if test_data is not None else 0,
                   "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                   "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None})
    write_json(output / "config.json", config)
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, seen = 0.0, 0
        for images, targets, _ in train_loader:
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda" and args.amp):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach()) * len(targets)
            seen += len(targets)
        val_rows = predict_rows(model, val_loader, device, classes)
        metrics = score_rows(val_rows, classes)
        score = metrics["macro_f1"]
        row = {"epoch": epoch, "train_loss": running_loss / seen, "val_accuracy": metrics["accuracy"],
               "val_balanced_accuracy": metrics["balanced_accuracy"], "val_macro_f1": score,
               "lr": optimizer.param_groups[0]["lr"]}
        history.append(row)
        print(json.dumps(row), flush=True)
        if score > best_score:
            best_score = score
            torch.save({"model": model.state_dict(), "epoch": epoch, "best_val_macro_f1": score,
                        "dataset": args.dataset, "architecture": args.architecture, "classes": classes,
                        "image_size": args.image_size}, output / "best.pt")
        scheduler.step()
        write_csv(output / "history.csv", history, list(history[0]))
    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    val_metrics = export_evaluation(output, "val", predict_rows(model, val_loader, device, classes), classes)
    test_metrics = None
    if test_data is not None:
        test_rows = predict_rows(model, make_loader(test_data, args, False), device, classes)
        test_metrics = export_evaluation(output, "test", test_rows, classes)
        if args.dataset in ("assignment", "dogs_cats"):
            write_csv(output / "submission.csv", [{"id": r["id"], "label": r["pred_label"]} for r in test_rows], ["id", "label"])
    write_json(output / "summary.json", {"selected_epoch": checkpoint["epoch"], "selection_metric": "val_macro_f1",
                                          "val": val_metrics, "test": test_metrics})
    print(f"Saved best checkpoint and results to {output}")


def summarize(args):
    records = []
    for folder in [Path(p) for p in args.runs]:
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        split = json.loads((folder / "split.json").read_text(encoding="utf-8"))
        row = {"run": str(folder), "dataset": config["dataset"], "seed": config["seed"],
               "split_seed": split["split_seed"], "val_fraction": config["val_fraction"],
               "data_fingerprint_sha256": split["data_fingerprint_sha256"],
               "architecture": config["architecture"], "pretrained": config["pretrained"],
               "augmentation": config["augmentation"],
               "imbalance_ratio": config["imbalance_ratio"], "remedy": config["remedy"],
               "selected_epoch": summary["selected_epoch"],
               "parameters_total": config.get("parameters_total"),
               "run_seconds": summary.get("run_seconds"),
               "peak_cuda_memory_bytes": summary.get("peak_cuda_memory_bytes")}
        # Preserve advanced-method settings so 2/3-expert or two-stage runs
        # cannot be silently pooled into one reported mean.
        for optional in ("epochs", "batch_size", "lr", "weight_decay", "label_smoothing",
                         "icl_experts", "icl_finetune_epochs", "icl_finetune_lr",
                         "icl_ad_until", "icl_ccl_after", "icl_kd_weight",
                         "icl_ad_weight", "icl_ccl_weight", "icl_momentum", "icl_topk",
                         "icl_temperature", "rdr_warmup", "rdr_momentum", "rdr_ridge",
                         "rdr_temperature"):
            row[optional] = config.get(optional)
        for split in ("val", "test"):
            metrics = summary.get(split)
            if metrics and metrics.get("labeled"):
                for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
                    row[f"{split}_{metric}"] = metrics[metric]
        records.append(row)
    if not records:
        raise ValueError("Provide at least one run directory")
    columns = sorted({key for row in records for key in row})
    output = Path(args.output).expanduser().resolve()
    write_csv(output, records, columns)
    groups = {}
    group_columns = ("dataset", "data_fingerprint_sha256", "split_seed", "val_fraction",
                     "architecture", "pretrained", "augmentation", "imbalance_ratio", "remedy",
                     "parameters_total",
                     "epochs", "batch_size", "lr", "weight_decay", "label_smoothing",
                     "icl_experts", "icl_finetune_epochs", "icl_finetune_lr",
                     "icl_ad_until", "icl_ccl_after", "icl_kd_weight",
                     "icl_ad_weight", "icl_ccl_weight", "icl_momentum", "icl_topk",
                     "icl_temperature", "rdr_warmup", "rdr_momentum", "rdr_ridge",
                     "rdr_temperature")
    for row in records:
        key = tuple(row[k] for k in group_columns)
        groups.setdefault(key, []).append(row)
    aggregate = []
    for key, group in groups.items():
        if len({row["seed"] for row in group}) != len(group):
            raise ValueError("Duplicate training seeds in one aggregate group")
        item = dict(zip(group_columns, key))
        item["runs"] = len(group)
        for metric in ("val_accuracy", "val_balanced_accuracy", "val_macro_f1", "test_accuracy",
                       "test_balanced_accuracy", "test_macro_f1", "run_seconds",
                       "peak_cuda_memory_bytes"):
            values = [row[metric] for row in group if row.get(metric) is not None]
            if values:
                item[f"{metric}_mean"] = statistics.mean(values)
                item[f"{metric}_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
        aggregate.append(item)
    write_csv(output.with_name(output.stem + "_aggregate.csv"), aggregate,
              sorted({key for row in aggregate for key in row}))
    print(f"Wrote {output} and {output.with_name(output.stem + '_aggregate.csv')}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    training = sub.add_parser("train", help="Train, select by validation macro F1, and evaluate once on test")
    training.add_argument("--dataset", choices=["dogs_cats", "pet", "cifar10", "assignment"], required=True)
    training.add_argument("--data-dir", default="data", help="Dataset root, separate from the old data_set directory")
    training.add_argument("--output-dir", required=True)
    training.add_argument("--download", action="store_true", help="Download Oxford-IIIT Pet or CIFAR-10 if absent")
    training.add_argument("--architecture", choices=["resnet18", "small_cnn"], default="resnet18")
    training.add_argument("--pretrained", action="store_true", help="ImageNet weights for pet/assignment only")
    training.add_argument("--image-size", type=int, default=224, help="Used by pet/assignment; CIFAR-10 stays at 32")
    training.add_argument("--augmentation", choices=["basic", "strong"], default="basic")
    training.add_argument("--epochs", type=int, default=30)
    training.add_argument("--batch-size", type=int, default=64)
    training.add_argument("--lr", type=float, default=0.0003)
    training.add_argument("--weight-decay", type=float, default=0.0001)
    training.add_argument("--label-smoothing", type=float, default=0.0)
    training.add_argument("--val-fraction", type=float, default=None,
                          help="Defaults to 0.2 for dogs_cats and 0.1 for CIFAR-10/Pet")
    training.add_argument("--split-seed", type=int, default=2026,
                          help="Fixed dataset partition seed, independent of training seed")
    training.add_argument("--min-class-images", type=int, default=10000,
                          help="Minimum verified cat and dog images for the large-data experiment")
    training.add_argument("--imbalance-ratio", type=float, default=1.0,
                          help="CIFAR-10 long-tail minority/majority count; only training is reduced")
    training.add_argument("--remedy", choices=["none", "class_weight", "sampler"], default="none")
    training.add_argument("--beta", type=float, default=0.9999, help="Effective-number weight beta")
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--workers", type=int, default=4)
    training.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    training.add_argument("--cuda-index", type=int, default=0,
                          help="Logical CUDA index within this session's visible GPU allocation")
    training.add_argument("--amp", action="store_true", help="Mixed precision on CUDA")
    training.add_argument("--fast", action="store_true", help="Disable deterministic algorithms for speed")
    training.set_defaults(func=train)
    combining = sub.add_parser("summarize", help="Merge real run results into plot-ready CSV files")
    combining.add_argument("--runs", nargs="+", required=True)
    combining.add_argument("--output", required=True)
    combining.set_defaults(func=summarize)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    arguments.func(arguments)
