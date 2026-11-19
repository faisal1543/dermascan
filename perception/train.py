"""DERM-14: fine-tune a CNN on the frozen split and save validation outputs (contract C1).

Kaggle (GPU, internet on):
    python -m perception.train --images /kaggle/input --run-name resnet18-baseline

Ground rules followed here:
  1/2  reads splits/splits.csv; uses train and val only, never loads test images
  3    class weights from the training split only; model chosen on validation
  4    fixed class order (perception.labels)
  6    reports balanced accuracy + per-class recall with Wilson CIs
  7    writes a ready-made row for experiments/log.csv
  8    augmentation on training images only, no colour jitter
  9    weights are saved under --out (not committed to git)
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from perception.labels import CLASS_TO_INDEX, CLASSES
from perception.metrics import summarise
from perception.predictions import c1_frame

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------- data
def index_images(root: Path) -> dict[str, Path]:
    """image_id -> path. Kaggle's HAM10000 copy has every image twice; keep the first."""
    index: dict[str, Path] = {}
    for p in sorted(root.rglob("ISIC_*.jpg")):
        index.setdefault(p.stem, p)
    return index


def load_table(images_root: Path, splits_csv: Path) -> pd.DataFrame:
    """Frozen split joined with metadata and image paths. Test rows are dropped here."""
    splits = pd.read_csv(splits_csv)
    meta_path = next(images_root.rglob("HAM10000_metadata.csv"))
    meta = pd.read_csv(meta_path)[["image_id", "age", "sex", "localization"]]
    df = splits.merge(meta, on="image_id", how="left", validate="one_to_one")
    df = df[df["split"].isin(["train", "val"])].reset_index(drop=True)  # Ground rule 2
    index = index_images(images_root)
    df["path"] = df["image_id"].map(index)
    missing = df["path"].isna().sum()
    if missing:
        raise FileNotFoundError(f"{missing} train/val images not found under {images_root}")
    df["label"] = df["dx"].map(CLASS_TO_INDEX)
    return df


class LesionDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform):
        self.paths = frame["path"].tolist()
        self.labels = frame["label"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as img:
            x = self.transform(img.convert("RGB"))
        return x, self.labels[i]


def build_transforms(size: int):
    train = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])  # no colour jitter: lesion colour is diagnostic (Ground rule 8)
    evaluate = transforms.Compose([
        transforms.Resize(int(size * 1.14)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train, evaluate


# ---------------------------------------------------------------- model
def build_model(backbone: str, dropout: float) -> nn.Module:
    if backbone == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, len(CLASSES)))
        return m
    import timm  # other backbones (e.g. efficientnet_b0, convnext_tiny) for DERM-18

    return timm.create_model(backbone, pretrained=True, num_classes=len(CLASSES), drop_rate=dropout)


@torch.no_grad()
def predict(model, loader, device) -> np.ndarray:
    model.eval()
    out = []
    for x, _ in loader:
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(x.to(device, non_blocking=True))
        out.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    return np.concatenate(out)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description="Fine-tune a CNN on the frozen DermaScan split")
    ap.add_argument("--images", type=Path, required=True, help="folder holding HAM10000 images + metadata")
    ap.add_argument("--splits", type=Path, default=Path("splits/splits.csv"))
    ap.add_argument("--out", type=Path, default=Path("runs"))
    ap.add_argument("--run-name", default="resnet18-baseline")
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--patience", type=int, default=6, help="early stop after this many epochs without val gain")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--member", default="Alfaisal")
    ap.add_argument("--machine", default="kaggle-t4")
    ap.add_argument("--limit", type=int, default=0, help="debug: use only N train and N val images")
    args = ap.parse_args(argv)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = args.out / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    df = load_table(args.images, args.splits)
    train_df, val_df = df[df.split == "train"], df[df.split == "val"]
    if args.limit:
        train_df, val_df = train_df.head(args.limit), val_df.head(args.limit)
    print(f"device={device} train={len(train_df)} val={len(val_df)}")

    # Class weights from TRAIN only (Ground rule 3)
    present = np.unique(train_df.label)
    w = compute_class_weight("balanced", classes=present, y=train_df.label)
    weights = np.ones(len(CLASSES))
    weights[present] = w
    print("class weights:", dict(zip(CLASSES, weights.round(2))))

    t_train, t_eval = build_transforms(args.size)
    loader_kw = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=device.type == "cuda")
    train_loader = DataLoader(LesionDataset(train_df, t_train), shuffle=True, drop_last=True, **loader_kw)
    val_loader = DataLoader(LesionDataset(val_df, t_eval), shuffle=False, **loader_kw)

    model = build_model(args.backbone, args.dropout).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    best, best_epoch, history = -1.0, -1, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total, n = total + loss.item() * len(y), n + len(y)
        scheduler.step()

        val_probs = predict(model, val_loader, device)
        s = summarise(val_df.label.to_numpy(), val_probs)
        row = {"epoch": epoch, "train_loss": round(total / max(n, 1), 4),
               "val_balanced_acc": round(s["balanced_accuracy"], 4), "val_mel_recall": round(s["mel_recall"], 4),
               "seconds": round(time.time() - t0, 1)}
        history.append(row)
        print(row, flush=True)

        if s["balanced_accuracy"] > best:
            best, best_epoch = s["balanced_accuracy"], epoch
            torch.save({"model": model.state_dict(), "backbone": args.backbone, "classes": CLASSES,
                        "epoch": epoch, "args": vars(args) | {"images": str(args.images), "splits": str(args.splits),
                                                               "out": str(args.out)}},
                       run_dir / "best.pt")
        elif epoch - best_epoch >= args.patience:
            print(f"early stop: no val gain for {args.patience} epochs")
            break

    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

    # Re-score the best checkpoint on validation and save C1 rows for the reasoning layer
    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    val_probs = predict(model, val_loader, device)
    model_version = f"cnn-{args.backbone}-{args.run_name}"
    c1_frame(val_df, val_probs, model_version).to_csv(run_dir / "val_predictions.csv", index=False)
    summary = summarise(val_df.label.to_numpy(), val_probs) | {"best_epoch": best_epoch, "model_version": model_version}
    (run_dir / "metrics_val.json").write_text(json.dumps(summary, indent=2))

    config = f"lr={args.lr} wd={args.weight_decay} drop={args.dropout} bs={args.batch_size} size={args.size} " \
             f"epochs={len(history)} best={best_epoch} weighted-CE seed={args.seed}"
    with open(run_dir / "log_row.csv", "w", newline="") as f:
        csv.writer(f).writerow([date.today().isoformat(), args.member, args.machine, args.run_name, args.backbone,
                                config, round(summary["balanced_accuracy"], 4), round(summary["mel_recall"], 4),
                                f"mel CI {summary['mel_recall_ci'][0]:.2f}-{summary['mel_recall_ci'][1]:.2f}"])

    print("\nVALIDATION (best epoch %d)" % best_epoch)
    print(f"balanced accuracy {summary['balanced_accuracy']:.3f} | melanoma recall {summary['mel_recall']:.3f} "
          f"(95% CI {summary['mel_recall_ci'][0]:.2f}-{summary['mel_recall_ci'][1]:.2f})")
    for r in summary["per_class"]:
        print(f"  {r['class']:6s} n={r['n']:4d} recall={r['recall']:.3f} "
              f"[{r['recall_ci_low']:.2f}-{r['recall_ci_high']:.2f}] precision={r['precision']:.3f}")


if __name__ == "__main__":
    main()
