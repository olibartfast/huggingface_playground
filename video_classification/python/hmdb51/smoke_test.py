"""Validate each backbone adapter before committing to a long extraction run.

Loads every config, pushes one synthetic (or real) clip through the encoder,
and reports feature dim, latency and peak VRAM. Run this first on a freshly
rented machine - it turns a 2-hour failure into a 2-minute one.

    python smoke_test.py                       # all configs, synthetic clip
    python smoke_test.py --video /data/x.avi   # decode a real clip too
"""

from __future__ import annotations

import argparse
import glob
import os
import time
import traceback

import numpy as np
import torch

from models import ModelConfig, build_adapter


def run_one(config_path: str, video: str | None, device: str) -> dict:
    config = ModelConfig.from_json(config_path)
    print(f"\n=== {config.name} ({config.repo_id}) ===")
    result = {"name": config.name, "config": config_path, "ok": False}

    try:
        adapter = build_adapter(config)

        t0 = time.time()
        adapter.load(device, config.dtype)
        load_s = time.time() - t0
        print(f"  loaded in {load_s:.1f}s")

        if video:
            from hmdb51 import decode_video

            clip = decode_video(video, adapter.num_frames)
            print(f"  decoded {video} -> {clip.shape}")
        else:
            # 240x320 is a typical HMDB frame size; the processor resizes anyway.
            clip = np.random.randint(
                0, 256, (adapter.num_frames, 240, 320, 3), dtype=np.uint8
            )
            print(f"  synthetic clip {clip.shape}")

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        t0 = time.time()
        embeddings = adapter.encode([clip])
        encode_s = time.time() - t0

        peak_gb = (
            torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else 0.0
        )
        print(
            f"  OK  dim={embeddings.shape[1]}  {encode_s:.2f}s/clip  "
            f"peak VRAM {peak_gb:.2f} GB"
        )
        if not np.isfinite(embeddings).all():
            print("  [warn] embeddings contain NaN/Inf - try dtype=float32")

        result.update(
            ok=True,
            dim=int(embeddings.shape[1]),
            seconds_per_clip=round(encode_s, 3),
            peak_vram_gb=round(peak_gb, 2),
        )

        del adapter
        if device == "cuda":
            torch.cuda.empty_cache()

    except Exception as exc:
        print(f"  FAILED: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", default="configs", help="config file or directory")
    parser.add_argument("--video", default=None, help="optional real clip to decode")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if os.path.isdir(args.configs):
        paths = sorted(glob.glob(os.path.join(args.configs, "*.json")))
    else:
        paths = [args.configs]
    if not paths:
        raise SystemExit(f"no configs found at {args.configs}")

    print(f"device: {args.device}")
    if args.device == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"gpu: {props.name} ({props.total_memory / 1024**3:.1f} GB)")

    results = [run_one(p, args.video, args.device) for p in paths]

    print("\n=== summary ===")
    for r in results:
        status = "ok" if r["ok"] else "FAILED"
        detail = (
            f"dim={r['dim']} {r['seconds_per_clip']}s/clip {r['peak_vram_gb']}GB"
            if r["ok"]
            else r.get("error", "")
        )
        print(f"  {r['name']:12s} {status:7s} {detail}")

    if not all(r["ok"] for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
