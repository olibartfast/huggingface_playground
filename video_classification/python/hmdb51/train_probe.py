"""Train and evaluate a probe on cached HMDB-51 embeddings.

The backbone stays frozen; this only fits a head on the vectors produced by
extract_features.py, so it runs in seconds even on CPU.

    python train_probe.py --features features/vjepa2 --split 1
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn


def load_subset(feature_dir: str, subset: str, split: int):
    stem = f"{subset}_split{split}"
    features = np.load(os.path.join(feature_dir, f"{stem}_features.npy"))
    labels = np.load(os.path.join(feature_dir, f"{stem}_labels.npy"))
    done = np.load(os.path.join(feature_dir, f"{stem}_done.npy"))

    if not done.all():
        missing = int((~done).sum())
        print(f"[warn] {subset}: dropping {missing} clips with no cached features")
        features, labels = features[done], labels[done]
    return features.astype(np.float32), labels.astype(np.int64)


def build_head(kind: str, dim: int, num_classes: int, hidden: int) -> nn.Module:
    if kind == "linear":
        return nn.Linear(dim, num_classes)
    if kind == "mlp":
        return nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(0.2), nn.Linear(hidden, num_classes)
        )
    raise ValueError(f"unknown head {kind!r}")


def accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int = 1) -> float:
    topk = logits.topk(k, dim=-1).indices
    hits = (topk == targets.unsqueeze(-1)).any(dim=-1).float()
    return hits.mean().item() * 100


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, help="features/<model> directory")
    parser.add_argument("--split", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--head", choices=["linear", "mlp"], default="linear")
    parser.add_argument("--hidden", type=int, default=1024, help="mlp hidden width")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--no-normalize", action="store_true", help="skip feature standardisation")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--report", default=None, help="write metrics JSON here")
    args = parser.parse_args()

    x_train, y_train = load_subset(args.features, "train", args.split)
    x_test, y_test = load_subset(args.features, "test", args.split)
    num_classes = int(max(y_train.max(), y_test.max())) + 1
    print(
        f"train {x_train.shape} | test {x_test.shape} | "
        f"{num_classes} classes | head={args.head}"
    )

    if not args.no_normalize:
        # Standardise on train statistics; probes are sensitive to feature scale.
        mean = x_train.mean(axis=0, keepdims=True)
        std = x_train.std(axis=0, keepdims=True) + 1e-6
        x_train = (x_train - mean) / std
        x_test = (x_test - mean) / std

    device = args.device
    xtr = torch.from_numpy(x_train).to(device)
    ytr = torch.from_numpy(y_train).to(device)
    xte = torch.from_numpy(x_test).to(device)
    yte = torch.from_numpy(y_test).to(device)

    head = build_head(args.head, xtr.shape[1], num_classes, args.hidden).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        head.train()
        perm = torch.randperm(xtr.shape[0], device=device)
        total_loss = 0.0
        for start in range(0, xtr.shape[0], args.batch_size):
            idx = perm[start : start + args.batch_size]
            optimizer.zero_grad()
            loss = criterion(head(xtr[idx]), ytr[idx])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * idx.numel()
        scheduler.step()

        if (epoch + 1) % 20 == 0 or epoch == args.epochs - 1:
            head.eval()
            with torch.no_grad():
                logits = head(xte)
                top1, top5 = accuracy(logits, yte, 1), accuracy(logits, yte, 5)
            print(
                f"  epoch {epoch + 1:3d} | loss {total_loss / xtr.shape[0]:.4f} | "
                f"top-1 {top1:.2f}% | top-5 {top5:.2f}%"
            )

    head.eval()
    with torch.no_grad():
        logits = head(xte)
        top1, top5 = accuracy(logits, yte, 1), accuracy(logits, yte, 5)

    model_name = os.path.basename(os.path.normpath(args.features))
    print(f"\n{model_name} split{args.split} {args.head}: top-1 {top1:.2f}% | top-5 {top5:.2f}%")

    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "model": model_name,
                    "split": args.split,
                    "head": args.head,
                    "top1": round(top1, 2),
                    "top5": round(top5, 2),
                    "train_clips": int(xtr.shape[0]),
                    "test_clips": int(xte.shape[0]),
                    "feature_dim": int(xtr.shape[1]),
                },
                fh,
                indent=2,
            )
        print(f"metrics -> {args.report}")


if __name__ == "__main__":
    main()
