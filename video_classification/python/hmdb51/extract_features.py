"""Cache frozen-encoder embeddings for an HMDB-51 split.

Decoding is the CPU bottleneck, so clips are decoded by DataLoader workers
while the GPU runs the encoder. Output is a memmapped .npy plus a boolean
`done` mask, which makes the run resumable: re-running skips finished clips.

    python extract_features.py --config configs/vjepa2.json \
        --data-root /data/hmdb51 --out features/ --subset train
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from hmdb51 import Hmdb51Index, decode_video
from models import ModelConfig, build_adapter


class ClipDataset(Dataset):
    """Decodes one clip per item. Returns (index, frames, label, ok)."""

    def __init__(self, clips, num_frames: int):
        self.clips = clips
        self.num_frames = num_frames

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, i: int):
        clip = self.clips[i]
        try:
            frames = decode_video(clip.path, self.num_frames)
            return i, frames, clip.label, True
        except Exception as exc:  # a handful of HMDB clips are genuinely corrupt
            print(f"[warn] decode failed for {clip.path}: {exc}")
            dummy = np.zeros((self.num_frames, 240, 320, 3), dtype=np.uint8)
            return i, dummy, clip.label, False


def collate(batch):
    indices, frames, labels, oks = zip(*batch)
    return list(indices), list(frames), list(labels), list(oks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to a configs/*.json")
    parser.add_argument("--data-root", required=True, help="HMDB-51 root directory")
    parser.add_argument("--out", default="features", help="output directory")
    parser.add_argument("--subset", choices=["train", "test"], required=True)
    parser.add_argument("--split", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=None, help="override config")
    parser.add_argument("--num-workers", type=int, default=8, help="decoder processes")
    parser.add_argument("--limit", type=int, default=None, help="debug: first N clips")
    args = parser.parse_args()

    config = ModelConfig.from_json(args.config)
    batch_size = args.batch_size or config.batch_size

    index = Hmdb51Index(args.data_root, split=args.split)
    clips = index.clips(args.subset)
    if args.limit:
        clips = clips[: args.limit]
    print(f"{config.name}: {len(clips)} clips in {args.subset} split {args.split}")

    out_dir = os.path.join(args.out, config.name)
    os.makedirs(out_dir, exist_ok=True)
    stem = f"{args.subset}_split{args.split}"
    feat_path = os.path.join(out_dir, f"{stem}_features.npy")
    done_path = os.path.join(out_dir, f"{stem}_done.npy")
    meta_path = os.path.join(out_dir, f"{stem}_meta.json")

    print(f"loading {config.repo_id} ...")
    adapter = build_adapter(config)
    adapter.load(args.device, config.dtype)

    # One warm-up clip tells us the feature width before allocating the memmap.
    probe_frames = decode_video(clips[0].path, adapter.num_frames)
    dim = adapter.encode([probe_frames]).shape[1]
    print(f"feature dim: {dim}")

    if os.path.exists(feat_path) and os.path.exists(done_path):
        features = np.lib.format.open_memmap(feat_path, mode="r+")
        done = np.load(done_path)
        if features.shape != (len(clips), dim):
            raise SystemExit(
                f"Existing cache at {feat_path} has shape {features.shape}, expected "
                f"{(len(clips), dim)}. Delete it to re-extract."
            )
        print(f"resuming: {int(done.sum())}/{len(clips)} already cached")
    else:
        features = np.lib.format.open_memmap(
            feat_path, mode="w+", dtype=np.float32, shape=(len(clips), dim)
        )
        done = np.zeros(len(clips), dtype=bool)

    labels = np.array([c.label for c in clips], dtype=np.int64)
    todo = [i for i in range(len(clips)) if not done[i]]
    if not todo:
        print("nothing to do - cache is complete")
        return

    loader = DataLoader(
        ClipDataset([clips[i] for i in todo], adapter.num_frames),
        batch_size=batch_size,
        num_workers=args.num_workers,
        collate_fn=collate,
        shuffle=False,
    )

    failures: list[str] = []
    processed = 0
    start = time.time()

    for local_indices, frames, _, oks in loader:
        global_indices = [todo[i] for i in local_indices]
        keep = [j for j, ok in enumerate(oks) if ok]
        for j, ok in enumerate(oks):
            if not ok:
                failures.append(clips[global_indices[j]].path)
        if keep:
            embeddings = adapter.encode([frames[j] for j in keep])
            for slot, j in enumerate(keep):
                features[global_indices[j]] = embeddings[slot]
                done[global_indices[j]] = True

        processed += len(local_indices)
        if processed % (batch_size * 10) < batch_size:
            elapsed = time.time() - start
            rate = processed / max(elapsed, 1e-6)
            remaining = (len(todo) - processed) / max(rate, 1e-6)
            print(
                f"  {processed}/{len(todo)} clips | {rate:.2f} clip/s | "
                f"eta {remaining / 60:.1f} min",
                flush=True,
            )
            features.flush()
            np.save(done_path, done)

    features.flush()
    np.save(done_path, done)
    np.save(os.path.join(out_dir, f"{stem}_labels.npy"), labels)

    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "model": config.name,
                "repo_id": config.repo_id,
                "num_frames": adapter.num_frames,
                "pooling": config.pooling,
                "dim": int(dim),
                "subset": args.subset,
                "split": args.split,
                "num_clips": len(clips),
                "num_cached": int(done.sum()),
                "classes": index.classes,
                "failures": failures,
            },
            fh,
            indent=2,
        )

    print(
        f"done: {int(done.sum())}/{len(clips)} cached in "
        f"{(time.time() - start) / 60:.1f} min -> {feat_path}"
    )
    if failures:
        print(f"[warn] {len(failures)} clips failed to decode; see {meta_path}")


if __name__ == "__main__":
    main()
