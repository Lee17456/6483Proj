"""Controlled CIFAR-10-LT experiments: RDR and an ICL adaptation.

This module reuses experiment.py's exact split, augmentation and evaluation.
It is an independent implementation, not a reproduction of the papers' reported
ResNet-32 results. See ADVANCED_LONGTAIL_SERVER.md for the protocol.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiment import (
    build_model,
    export_evaluation,
    load_data,
    make_loader,
    predict_rows,
    score_rows,
    seed_everything,
    write_csv,
    write_json,
)


class ResNet18Features(nn.Module):
    """Expose the penultimate ResNet18 features required by RDR."""

    def __init__(self, classes: int):
        super().__init__()
        self.net = build_model("cifar10", "resnet18", classes, False)

    def forward_features(self, x):
        net = self.net
        x = net.maxpool(net.relu(net.bn1(net.conv1(x))))
        x = net.layer4(net.layer3(net.layer2(net.layer1(x))))
        return torch.flatten(net.avgpool(x), 1)

    def forward(self, x):
        return self.net.fc(self.forward_features(x))


class ICLResNet18(nn.Module):
    """Shared CIFAR stem/layers 1-2, independent layers 3-4 and classifiers."""

    def __init__(self, classes: int, experts: int):
        super().__init__()
        if experts < 2:
            raise ValueError("ICL needs at least two experts")
        base = build_model("cifar10", "resnet18", classes, False)
        self.shared = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool,
                                    base.layer1, base.layer2)
        self.branches = nn.ModuleList([
            nn.Sequential(copy.deepcopy(base.layer3), copy.deepcopy(base.layer4),
                          nn.AdaptiveAvgPool2d(1), nn.Flatten())
            for _ in range(experts)
        ])
        self.classifiers = nn.ModuleList([nn.Linear(512, classes) for _ in range(experts)])

    def expert_forward(self, x):
        shared = self.shared(x)
        features = torch.stack([branch(shared) for branch in self.branches], dim=0)
        logits = torch.stack([head(features[i]) for i, head in enumerate(self.classifiers)], dim=0)
        return logits, features

    def forward(self, x):
        logits, _ = self.expert_forward(x)
        # Average probabilities, as described in the ICL paper. log(p) has the
        # same argmax and gives predict_rows() a valid softmax distribution.
        return torch.log_softmax(logits, dim=-1).logsumexp(dim=0) - math.log(len(self.branches))


class RDRState:
    """Class prototypes and per-sample density-ratio weights (RDR Eq. 7-8)."""

    def __init__(self, counts: torch.Tensor, feature_dim: int, momentum: float):
        self.counts = counts.float()
        self.prototypes = counts.new_zeros((len(counts), feature_dim), dtype=torch.float32)
        self.seen = counts.new_zeros(len(counts), dtype=torch.bool)
        self.momentum = momentum

    @torch.no_grad()
    def weights(self, features: torch.Tensor, labels: torch.Tensor, ridge: float,
                temperature: float) -> torch.Tensor:
        z = features.detach().float()
        weights = torch.ones(len(labels), device=z.device, dtype=torch.float32)
        for cls in labels.unique().tolist():
            mask = labels == cls
            if not bool(self.seen[cls]):
                continue
            x = z[mask]
            with torch.autocast(device_type=x.device.type, enabled=False):
                gram = x @ x.T
                # A small ridge stabilizes singular minibatch Gram matrices.
                # The authors use inverse with pseudoinverse fallback; this is
                # a documented numerical adaptation for AMP/ResNet18 features.
                scale = gram.diagonal().mean().clamp_min(1e-12)
                gram = gram + ridge * scale * torch.eye(len(x), device=x.device)
                rhs = x @ self.prototypes[cls]
                ratio = len(x) * torch.linalg.solve(gram, rhs)
            weights[mask] = ratio.clamp(1e-9, 1e6)
        inverse_frequency = self.counts.sum() / self.counts[labels]
        weights = (weights * inverse_frequency).clamp_min(1e-12).pow(temperature)
        return weights * (len(labels) / weights.sum().clamp_min(1e-12))

    @torch.no_grad()
    def update(self, features: torch.Tensor, labels: torch.Tensor):
        z = features.detach().float()
        for cls in labels.unique().tolist():
            mean = z[labels == cls].mean(dim=0)
            if bool(self.seen[cls]):
                self.prototypes[cls].mul_(self.momentum).add_(mean, alpha=1 - self.momentum)
            else:
                self.prototypes[cls].copy_(mean)
                self.seen[cls] = True


class ICLState:
    """Per-expert prototype and confusion statistics for CCL."""

    def __init__(self, experts: int, classes: int, feature_dim: int,
                 device: torch.device, momentum: float):
        self.prototypes = torch.zeros(experts, classes, feature_dim, device=device)
        self.seen = torch.zeros(experts, classes, dtype=torch.bool, device=device)
        self.confusion = torch.zeros(experts, classes, classes, device=device)
        self.momentum = momentum

    @torch.no_grad()
    def update(self, features: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor):
        for expert in range(len(features)):
            prediction = logits[expert].argmax(dim=1)
            classes = self.confusion.shape[-1]
            batch_confusion = F.one_hot(labels * classes + prediction,
                                        num_classes=classes * classes).sum(dim=0)
            batch_confusion = batch_confusion.reshape(classes, classes).float()
            batch_confusion.fill_diagonal_(0)
            self.confusion[expert].mul_(self.momentum).add_(batch_confusion,
                                                              alpha=1 - self.momentum)
            for cls in labels.unique().tolist():
                mean = features[expert, labels == cls].detach().float().mean(dim=0)
                if bool(self.seen[expert, cls]):
                    self.prototypes[expert, cls].mul_(self.momentum).add_(
                        mean, alpha=1 - self.momentum)
                else:
                    self.prototypes[expert, cls].copy_(mean)
                    self.seen[expert, cls] = True

    def confusion_contrastive(self, features: torch.Tensor, labels: torch.Tensor,
                              topk: int, temperature: float) -> torch.Tensor:
        losses = []
        classes = self.prototypes.shape[1]
        k = min(topk, classes - 1)
        for expert in range(len(features)):
            if not bool(self.seen[expert].all()):
                continue
            ranking = self.confusion[expert].clone()
            ranking.fill_diagonal_(-float("inf"))
            negatives = ranking.topk(k, dim=1).indices[labels]
            candidates = torch.cat((labels[:, None], negatives), dim=1)
            prototypes = F.normalize(self.prototypes[expert][candidates].detach(), dim=-1)
            embedding = F.normalize(features[expert].float(), dim=-1)
            similarities = torch.einsum("bd,bkd->bk", embedding, prototypes) / temperature
            losses.append(F.cross_entropy(similarities, torch.zeros_like(labels)))
        return torch.stack(losses).mean() if losses else features.sum() * 0


def icl_loss(logits: torch.Tensor, features: torch.Tensor, labels: torch.Tensor,
             state: ICLState, epoch: int, args) -> tuple[torch.Tensor, dict]:
    expert_ce = torch.stack([F.cross_entropy(branch.float(), labels,
                                            label_smoothing=args.label_smoothing)
                             for branch in logits])
    parallel = 1 / torch.reciprocal(expert_ce + 1e-8).sum()
    # Pairwise bidirectional KL, with the teacher detached to avoid the
    # experts collapsing into one another through both sides of the loss.
    distill_terms = []
    for i in range(len(logits)):
        for j in range(i + 1, len(logits)):
            for student, teacher in ((i, j), (j, i)):
                distill_terms.append(F.kl_div(
                    F.log_softmax(logits[student].float(), dim=1),
                    F.softmax(logits[teacher].detach().float(), dim=1),
                    reduction="batchmean"))
    distill = torch.stack(distill_terms).mean()
    diversity = features.sum() * 0
    if epoch <= args.icl_ad_until:
        normalized = F.normalize(features.float(), dim=-1)
        similarities = torch.einsum("ebd,fbd->efb", normalized, normalized)
        mask = ~torch.eye(len(features), dtype=torch.bool, device=features.device)
        diversity = similarities[mask].mean()
    contrastive = features.sum() * 0
    if epoch > args.icl_ccl_after:
        contrastive = state.confusion_contrastive(features, labels,
                                                   args.icl_topk, args.icl_temperature)
    total = parallel + args.icl_kd_weight * distill + args.icl_ad_weight * diversity \
        + args.icl_ccl_weight * contrastive
    return total, {"parallel": float(parallel.detach()), "distill": float(distill.detach()),
                   "diversity": float(diversity.detach()), "contrastive": float(contrastive.detach())}


def checked_split(split: dict, reference: str | None):
    if not reference:
        return
    old = json.loads(Path(reference).expanduser().read_text(encoding="utf-8"))
    keys = ("dataset", "split_seed", "train_indices", "val_indices", "train_counts", "val_counts")
    mismatched = [key for key in keys if old.get(key) != split.get(key)]
    if mismatched:
        raise ValueError(f"Split differs from reference ({', '.join(mismatched)}): {reference}")


def train(args):
    if not 0 < args.imbalance_ratio < 1:
        raise ValueError("The advanced methods require --imbalance-ratio in (0, 1)")
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        raise ValueError("epochs, batch size and learning rate must be positive")
    if not 0 <= args.label_smoothing < 1:
        raise ValueError("label smoothing must be in [0, 1)")
    if not 0 <= args.icl_momentum < 1 or not 0 <= args.rdr_momentum < 1:
        raise ValueError("prototype momentum must be in [0, 1)")
    if args.icl_finetune_epochs < 0 or args.icl_finetune_lr <= 0:
        raise ValueError("ICL classifier training settings must be nonnegative/positive")
    if args.method == "icl" and args.icl_ccl_after >= args.epochs and args.icl_ccl_weight:
        raise ValueError("ICL CCL starts after training ends; reduce --icl-ccl-after")
    if args.method == "rdr" and args.rdr_warmup >= args.epochs:
        raise ValueError("RDR warmup must be shorter than training")
    if args.icl_topk < 1 or args.icl_topk >= 10 or args.icl_temperature <= 0:
        raise ValueError("ICL topk must be 1-9 and temperature positive")
    if min(args.icl_kd_weight, args.icl_ad_weight, args.icl_ccl_weight) < 0:
        raise ValueError("ICL loss coefficients must be nonnegative")
    if args.rdr_ridge <= 0 or args.rdr_temperature <= 0:
        raise ValueError("RDR ridge and temperature must be positive")
    seed_everything(args.seed, deterministic=not args.fast)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    if use_cuda and not 0 <= args.cuda_index < torch.cuda.device_count():
        raise ValueError(f"CUDA index {args.cuda_index} is outside this session's visible GPUs")
    device = torch.device(f"cuda:{args.cuda_index}" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(device)
    # These attributes match experiment.load_data / make_loader exactly.
    args.dataset = "cifar10"
    args.image_size = 32
    args.val_fraction = 0.1
    args.download = False
    train_data, val_data, test_data, classes, split = load_data(args)
    checked_split(split, args.reference_split)
    output = Path(args.output_dir).expanduser().resolve()
    # The parallel scheduler creates console.log before launching this process.
    existing = [path for path in output.iterdir() if path.name != "console.log"] if output.exists() else []
    if existing and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output}; choose a new path")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "split.json", split)
    train_loader = make_loader(train_data, args, True)
    val_loader = make_loader(val_data, args, False)
    counts = torch.bincount(torch.tensor(train_data.targets), minlength=10).to(device)
    model = ICLResNet18(10, args.icl_experts) if args.method == "icl" else ResNet18Features(10)
    model = model.to(device)
    rdr = RDRState(counts, 512, args.rdr_momentum) if args.method == "rdr" else None
    icl = ICLState(args.icl_experts, 10, 512, device, args.icl_momentum) if args.method == "icl" else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda and args.amp)
    config = vars(args).copy()
    config.update({"architecture": "resnet18", "remedy": args.method,
                   "pretrained": False, "data_fingerprint_sha256": None, "classes": classes,
                   "parameters_total": sum(parameter.numel() for parameter in model.parameters()),
                   "train_count": len(train_data), "val_count": len(val_data),
                   "torch_version": torch.__version__,
                   "torchvision_version": __import__("torchvision").__version__,
                   "device_used": str(device), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                   "cuda_device_name": torch.cuda.get_device_name(device) if use_cuda else None,
                   "paper_protocol_reproduction": False})
    write_json(output / "config.json", config)
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    run_started = time.perf_counter()
    history, best_score = [], -math.inf
    weight_history = []

    def save_best(epoch: int, stage: str, metrics: dict):
        nonlocal best_score
        if metrics["macro_f1"] > best_score:
            best_score = metrics["macro_f1"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "stage": stage,
                        "best_val_macro_f1": best_score, "method": args.method,
                        "classes": classes}, output / "best.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, seen = 0.0, 0
        components = {"parallel": 0.0, "distill": 0.0, "diversity": 0.0, "contrastive": 0.0}
        class_weight_totals = torch.zeros(10, device=device)
        class_weight_counts = torch.zeros(10, device=device)
        for images, labels, _ in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_cuda and args.amp):
                if args.method == "icl":
                    branch_logits, features = model.expert_forward(images)
                    loss, parts = icl_loss(branch_logits, features, labels, icl, epoch, args)
                else:
                    features = model.forward_features(images)
                    logits = model.net.fc(features)
                    per_item = F.cross_entropy(logits.float(), labels, reduction="none",
                                               label_smoothing=args.label_smoothing)
                    if args.method == "rdr" and epoch > args.rdr_warmup:
                        weights = rdr.weights(features, labels, args.rdr_ridge,
                                              args.rdr_temperature)
                    else:
                        weights = torch.ones_like(per_item)
                    loss = (per_item * weights).mean()
                    parts = None
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if rdr is not None:
                rdr.update(features, labels)
                for cls in labels.unique().tolist():
                    mask = labels == cls
                    class_weight_totals[cls] += weights.detach()[mask].sum()
                    class_weight_counts[cls] += mask.sum()
            if icl is not None:
                icl.update(features, branch_logits, labels)
                for key, value in parts.items():
                    components[key] += value * len(labels)
            total_loss += float(loss.detach()) * len(labels)
            seen += len(labels)
        metrics = score_rows(predict_rows(model, val_loader, device, classes), classes)
        row = {"epoch": epoch, "stage": "feature", "train_loss": total_loss / seen,
               "val_accuracy": metrics["accuracy"], "val_balanced_accuracy": metrics["balanced_accuracy"],
               "val_macro_f1": metrics["macro_f1"], "lr": optimizer.param_groups[0]["lr"]}
        if icl is not None:
            row.update({f"loss_{key}": value / seen for key, value in components.items()})
        history.append(row)
        print(json.dumps(row), flush=True)
        save_best(epoch, "feature", metrics)
        if rdr is not None:
            weight_row = {"epoch": epoch}
            means = class_weight_totals / class_weight_counts.clamp_min(1)
            weight_row.update({name: float(means[i]) for i, name in enumerate(classes)})
            weight_history.append(weight_row)
            write_csv(output / "rdr_weight_history.csv", weight_history,
                      ["epoch", *classes])
        scheduler.step()
        write_csv(output / "history.csv", history, sorted({key for item in history for key in item}))

    # Optional ICL second stage: freeze all feature layers, retrain expert
    # classifiers with logit adjustment using only the imbalanced train split.
    if icl is not None and args.icl_finetune_epochs:
        best_feature = torch.load(output / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(best_feature["model"])
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for classifier in model.classifiers:
            for parameter in classifier.parameters():
                parameter.requires_grad_(True)
        classifier_optimizer = torch.optim.AdamW(model.classifiers.parameters(), lr=args.icl_finetune_lr,
                                                 weight_decay=args.weight_decay)
        classifier_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            classifier_optimizer, T_max=args.icl_finetune_epochs)
        log_prior = counts.float().log()
        for stage_epoch in range(1, args.icl_finetune_epochs + 1):
            model.eval()
            for classifier in model.classifiers:
                classifier.train()
            total_loss, seen = 0.0, 0
            for images, labels, _ in train_loader:
                images, labels = images.to(device), labels.to(device)
                classifier_optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    shared = model.shared(images)
                    features = [branch(shared) for branch in model.branches]
                logits = [head(features[i]) for i, head in enumerate(model.classifiers)]
                loss = torch.stack([F.cross_entropy(branch.float() + log_prior, labels)
                                    for branch in logits]).mean()
                loss.backward()
                classifier_optimizer.step()
                total_loss += float(loss.detach()) * len(labels)
                seen += len(labels)
            metrics = score_rows(predict_rows(model, val_loader, device, classes), classes)
            epoch = args.epochs + stage_epoch
            row = {"epoch": epoch, "stage": "classifier", "train_loss": total_loss / seen,
                   "val_accuracy": metrics["accuracy"],
                   "val_balanced_accuracy": metrics["balanced_accuracy"],
                   "val_macro_f1": metrics["macro_f1"],
                   "lr": classifier_optimizer.param_groups[0]["lr"]}
            history.append(row)
            print(json.dumps(row), flush=True)
            save_best(epoch, "classifier", metrics)
            classifier_scheduler.step()
            write_csv(output / "history.csv", history, sorted({key for item in history for key in item}))

    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    val_metrics = export_evaluation(output, "val", predict_rows(model, val_loader, device, classes), classes)
    test_metrics = export_evaluation(output, "test", predict_rows(model,
                                      make_loader(test_data, args, False), device, classes), classes)
    write_json(output / "summary.json", {"selected_epoch": checkpoint["epoch"],
               "selected_stage": checkpoint["stage"], "selection_metric": "val_macro_f1",
               "run_seconds": time.perf_counter() - run_started,
               "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if use_cuda else None,
               "val": val_metrics, "test": test_metrics})
    print(f"Saved {args.method} results to {output}", flush=True)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["ce", "rdr", "icl"], required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-split", help="Existing long-tail split.json to verify identical indices")
    parser.add_argument("--overwrite", action="store_true", help="Permit replacing files in output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--imbalance-ratio", type=float, default=0.1)
    parser.add_argument("--augmentation", choices=["basic", "strong"], default="strong")
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--cuda-index", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--fast", action="store_true", help="Disable deterministic algorithms")
    parser.add_argument("--rdr-warmup", type=int, default=80)
    parser.add_argument("--rdr-momentum", type=float, default=0.9)
    parser.add_argument("--rdr-ridge", type=float, default=1e-4)
    parser.add_argument("--rdr-temperature", type=float, default=1.0)
    parser.add_argument("--icl-experts", type=int, choices=[2, 3], default=2)
    parser.add_argument("--icl-momentum", type=float, default=0.1)
    parser.add_argument("--icl-topk", type=int, default=3)
    parser.add_argument("--icl-temperature", type=float, default=0.1)
    parser.add_argument("--icl-ad-until", type=int, default=17,
                        help="First 17/100 epochs approximate paper's 30/180 AD schedule")
    parser.add_argument("--icl-ccl-after", type=int, default=67,
                        help="After 67/100 epochs approximate paper's 120/180 CCL schedule")
    parser.add_argument("--icl-kd-weight", type=float, default=1.0)
    parser.add_argument("--icl-ad-weight", type=float, default=0.05)
    parser.add_argument("--icl-ccl-weight", type=float, default=0.1)
    parser.add_argument("--icl-finetune-epochs", type=int, default=0)
    parser.add_argument("--icl-finetune-lr", type=float, default=0.0001)
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
