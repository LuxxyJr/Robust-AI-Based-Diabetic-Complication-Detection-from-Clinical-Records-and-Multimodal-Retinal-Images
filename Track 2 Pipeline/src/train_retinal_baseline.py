#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train reproducible retinal image baselines for MMRDR.

Tasks
-----
- ``cfp_dr``: CFP 5-class DR grading.
- ``uwf_dr``: UWF 5-class DR grading.
- ``oct_dme``: 2D OCT 3-class DME classification.

The official test set is inferred from ``ts*.jpg`` names. Validation rows are
stratified from ``tr*.jpg`` only.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    cohen_kappa_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms


LOGGER = logging.getLogger("retinal_train")

TASKS = {
    "cfp_dr": {"folder": "MMRDR-CFP", "csv": "FP.csv", "classes": 5, "label": "grade"},
    "uwf_dr": {"folder": "MMRDR-UWF", "csv": "UWF.csv", "classes": 5, "label": "grade"},
    "oct_dme": {"folder": "MMRDR-OCT", "csv": "OCT.csv", "classes": 3, "label": "grade"},
}


@dataclass
class RunConfig:
    data_root: Path
    out_dir: Path
    task: str
    model_name: str
    seed: int
    epochs: int
    batch_size: int
    image_size: int
    learning_rate: float
    weight_decay: float
    num_workers: int
    use_weighted_sampler: bool
    gradcam_examples: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def split_from_image_path(path: str) -> str:
    name = Path(path).name.lower()
    if name.startswith("tr"):
        return "train"
    if name.startswith("ts"):
        return "test"
    return "unknown"


def load_manifest(cfg: RunConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    spec = TASKS[cfg.task]
    base = cfg.data_root / spec["folder"]
    df = pd.read_csv(base / spec["csv"])
    df["image_path"] = df["image"].map(lambda p: str(base / p))
    df["split"] = df["image"].map(split_from_image_path)
    df["label"] = pd.to_numeric(df[spec["label"]], errors="coerce").astype(int)
    df["exists"] = df["image_path"].map(lambda p: Path(p).exists())
    missing = df[~df["exists"]]
    unknown_split = df[df["split"] == "unknown"]
    if not missing.empty:
        raise FileNotFoundError(f"{len(missing)} image paths are missing. Run audit_dataset.py first.")
    if not unknown_split.empty:
        raise ValueError(f"{len(unknown_split)} rows have unknown split prefixes.")

    train_all = df[df["split"] == "train"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    train_df, val_df = train_test_split(
        train_all,
        test_size=0.15,
        random_state=cfg.seed,
        stratify=train_all["label"],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df


class RetinalDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        return self.transform(image), int(row["label"]), row["image_path"]


def build_transforms(image_size: int):
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_tf, eval_tf


def build_model(model_name: str, n_classes: int) -> nn.Module:
    weights = None
    if model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, n_classes)
        return model
    if model_name == "mobilenet_v3":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT
        model = models.mobilenet_v3_large(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, n_classes)
        return model
    if model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, n_classes)
        return model
    raise ValueError(f"Unsupported model: {model_name}")


def class_weights(labels: pd.Series, n_classes: int) -> torch.Tensor:
    counts = labels.value_counts().reindex(range(n_classes), fill_value=0).to_numpy(dtype=float)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def make_loaders(cfg: RunConfig, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    train_tf, eval_tf = build_transforms(cfg.image_size)
    train_ds = RetinalDataset(train_df, train_tf)
    val_ds = RetinalDataset(val_df, eval_tf)
    test_ds = RetinalDataset(test_df, eval_tf)

    if cfg.use_weighted_sampler:
        weights = class_weights(train_df["label"], TASKS[cfg.task]["classes"]).numpy()
        sample_weights = train_df["label"].map(lambda y: weights[int(y)]).to_numpy(dtype=float)
        sampler = WeightedRandomSampler(torch.from_numpy(sample_weights), num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler, num_workers=cfg.num_workers)
    else:
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    return train_loader, val_loader, test_loader


def run_epoch(model, loader, criterion, optimizer, device, train: bool) -> tuple[float, list[int], list[int], list[list[float]], list[str]]:
    model.train(train)
    total = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[list[float]] = []
    paths: list[str] = []
    for images, labels, batch_paths in loader:
        images = images.to(device)
        labels = labels.to(device)
        if train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()
        probs = torch.softmax(logits.detach(), dim=1)
        total += float(loss.item()) * images.size(0)
        y_true.extend(labels.cpu().numpy().astype(int).tolist())
        y_pred.extend(torch.argmax(probs, dim=1).cpu().numpy().astype(int).tolist())
        y_prob.extend(probs.cpu().numpy().tolist())
        paths.extend(list(batch_paths))
    return total / max(len(loader.dataset), 1), y_true, y_pred, y_prob, paths


def grading_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "quadratic_weighted_kappa": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
    }


def save_confusion_matrix(y_true: list[int], y_pred: list[int], n_classes: int, out_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    pd.DataFrame(cm, index=[f"true_{i}" for i in range(n_classes)], columns=[f"pred_{i}" for i in range(n_classes)]).to_csv(out_path)


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.forward_hook = target_layer.register_forward_hook(self._save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def close(self):
        self.forward_hook.remove()
        self.backward_hook.remove()

    def __call__(self, image_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image_tensor)
        score = logits[:, class_idx].sum()
        score.backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(cam, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def target_layer_for(model: nn.Module, model_name: str) -> nn.Module:
    if model_name == "resnet50":
        return model.layer4[-1]
    if model_name == "mobilenet_v3":
        return model.features[-1]
    if model_name == "efficientnet_b0":
        return model.features[-1]
    raise ValueError(model_name)


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(3, 1, 1)
    image = (tensor * std + mean).clamp(0, 1)
    return (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def save_gradcam_examples(model, model_name: str, loader, device, out_dir: Path, max_examples: int) -> None:
    if max_examples <= 0:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    cam = GradCAM(model, target_layer_for(model, model_name))
    saved_correct = 0
    saved_incorrect = 0
    try:
        for images, labels, paths in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            for i in range(images.size(0)):
                is_correct = bool(preds[i] == labels[i])
                if is_correct and saved_correct >= max_examples:
                    continue
                if (not is_correct) and saved_incorrect >= max_examples:
                    continue
                heat = cam(images[i : i + 1], int(preds[i].item()))
                base = denormalize(images[i])
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.imshow(base)
                ax.imshow(heat, cmap="jet", alpha=0.38)
                ax.axis("off")
                kind = "correct" if is_correct else "incorrect"
                ax.set_title(f"{kind}: true={int(labels[i])}, pred={int(preds[i])}")
                filename = f"{kind}_{saved_correct if is_correct else saved_incorrect}_{Path(paths[i]).stem}.png"
                fig.tight_layout()
                fig.savefig(out_dir / filename, dpi=150)
                plt.close(fig)
                if is_correct:
                    saved_correct += 1
                else:
                    saved_incorrect += 1
                if saved_correct >= max_examples and saved_incorrect >= max_examples:
                    return
    finally:
        cam.close()


def train(cfg: RunConfig) -> None:
    set_seed(cfg.seed)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    spec = TASKS[cfg.task]
    n_classes = spec["classes"]
    train_df, val_df, test_df = load_manifest(cfg)
    train_df.to_csv(cfg.out_dir / "train_manifest.csv", index=False)
    val_df.to_csv(cfg.out_dir / "val_manifest.csv", index=False)
    test_df.to_csv(cfg.out_dir / "test_manifest.csv", index=False)

    train_loader, val_loader, test_loader = make_loaders(cfg, train_df, val_df, test_df)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg.model_name, n_classes).to(device)
    weights = class_weights(train_df["label"], n_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)

    best_macro_f1 = -1.0
    history = []
    best_path = cfg.out_dir / "best_model.pt"
    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_true, train_pred, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_true, val_pred, _, _ = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        train_metrics = grading_metrics(train_true, train_pred)
        val_metrics = grading_metrics(val_true, val_pred)
        scheduler.step(val_metrics["macro_f1"])
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        LOGGER.info("epoch %03d train_loss=%.4f val_loss=%.4f val_macro_f1=%.4f", epoch, train_loss, val_loss, val_metrics["macro_f1"])
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            torch.save({"model_state": model.state_dict(), "config": cfg.__dict__, "epoch": epoch, "val_macro_f1": best_macro_f1}, best_path)

    pd.DataFrame(history).to_csv(cfg.out_dir / "history.csv", index=False)
    try:
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    test_loss, y_true, y_pred, y_prob, paths = run_epoch(model, test_loader, criterion, optimizer, device, train=False)
    metrics = grading_metrics(y_true, y_pred)
    per_class = precision_recall_fscore_support(y_true, y_pred, labels=list(range(n_classes)), zero_division=0)
    metrics["test_loss"] = float(test_loss)
    (cfg.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_confusion_matrix(y_true, y_pred, n_classes, cfg.out_dir / "confusion_matrix.csv")
    pd.DataFrame(
        {
            "image_path": paths,
            "y_true": y_true,
            "y_pred": y_pred,
            **{f"prob_class_{i}": [p[i] for p in y_prob] for i in range(n_classes)},
        }
    ).to_csv(cfg.out_dir / "test_predictions.csv", index=False)
    pd.DataFrame(
        {
            "class": list(range(n_classes)),
            "precision": per_class[0],
            "recall": per_class[1],
            "f1": per_class[2],
            "support": per_class[3],
        }
    ).to_csv(cfg.out_dir / "per_class_metrics.csv", index=False)
    save_gradcam_examples(model, cfg.model_name, test_loader, device, cfg.out_dir / "gradcam", cfg.gradcam_examples)
    LOGGER.info("Training complete. Outputs saved to %s", cfg.out_dir)


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description="Train MMRDR retinal image baseline.")
    parser.add_argument("--data-root", default="../Datasets/Paper 2")
    parser.add_argument("--out-root", default="outputs")
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--model", dest="model_name", choices=["resnet50", "mobilenet_v3", "efficientnet_b0"], default="resnet50")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-weighted-sampler", action="store_true")
    parser.add_argument("--gradcam-examples", type=int, default=3)
    args = parser.parse_args()
    out_dir = Path(args.out_root) / args.task / args.model_name / f"seed_{args.seed}"
    return RunConfig(
        data_root=Path(args.data_root),
        out_dir=out_dir,
        task=args.task,
        model_name=args.model_name,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        use_weighted_sampler=not args.no_weighted_sampler,
        gradcam_examples=args.gradcam_examples,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    cfg = parse_args()
    train(cfg)


if __name__ == "__main__":
    main()
