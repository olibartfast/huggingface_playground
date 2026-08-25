# HMDB-51 Frozen-Probe Benchmark

Benchmarks three recent Hugging Face video backbones on HMDB-51 by freezing the
encoder, caching clip embeddings once, and training a small classifier head on
top. Designed to be cloned onto a rented GPU box and run unattended.

| Backbone | Checkpoint | Frames | Dim | Notes |
|---|---|---|---|---|
| V-JEPA 2 | `facebook/vjepa2-vitl-fpc64-256` | 64 @ 256px | 1024 | ViT-L, self-supervised encoder |
| VideoPrism | `google/videoprism-base-f16r288` | 16 @ 288px | 768 | needs `revision="refs/pr/4"` (see below) |
| PE Video | `facebook/pe-av-large-16-frame` | 16 @ 336px | 1792 | audio-visual; we keep only the video tower |

## Results

HMDB-51 **split 1**, official protocol (3570 train / 1530 test), frozen encoder,
mean-pooled (PE Video: CLS) features, linear probe, 200 epochs. Mean ± std over
5 probe runs on identical cached features, on an RTX 3060 Laptop:

| Backbone | Dim | top-1 | top-5 | Extraction (5100 clips) |
|---|---|---|---|---|
| **PE Video** (`pe-av-large-16-frame`) | 1792 | **77.23 ± 0.09** | 96.08 | 44 min |
| VideoPrism base | 768 | 63.18 ± 0.18 | 90.92 | 12 min |
| V-JEPA 2 ViT-L | 1024 | 62.84 ± 0.29 | 90.07 | 83 min |

PE Video wins by **14 points** — far outside the ±0.3 run-to-run noise. VideoPrism
and V-JEPA 2 differ by 0.34 pp against a pooled std of 0.23, i.e. **effectively
tied** under this protocol.

Read the V-JEPA 2 row with care. Mean-pooling ~8192 patch tokens into one vector
suits it badly: its own `VJEPA2ForVideoClassification` uses an *attentive* pooler,
and the published probe numbers use attentive probing. VideoPrism is factorised
spatio-temporal with far fewer tokens, so mean-pooling costs it less; PE Video is
contrastively trained with a real CLS token, so its pooled representation is
exactly what the model was optimised to produce. **This table ranks
mean-pooled linear probing, not the backbones' capability.**

The probe does not set a random seed, hence the ± figures. Treat any gap under
~0.5 pp as noise.

Raw measurements, including the individual per-run top-1 values behind each ±,
are committed in [`benchmarks/split1/`](benchmarks/split1/). `results/` is where
a fresh run writes its own output and is gitignored.

## Why a probe, and not just inference

**None of these three ship an HMDB-51 classification head**, so there is no
"load model → get label" path:

- **V-JEPA 2** has a `VJEPA2ForVideoClassification` class, but the released
  finetuned checkpoints target Something-Something-v2 and Diving48.
- **VideoPrism** has a `VideoPrismForVideoClassification` class whose head is
  **randomly initialised** — no classifier weights were published.
- **PE Video** has no classification head class at all. It is a video↔text
  contrastive model.

The frozen-probe protocol used here is what these papers actually report, and
it separates the expensive part (encoding, inference-only, done once) from the
cheap part (fitting a head, seconds on CPU).

## Hardware

The only heavy step is **inference**, so this does not need a big GPU. Measured
at batch 1, fp16, on an RTX 3060 Laptop (5.7 GB usable):

| Backbone | Peak VRAM | First clip (incl. warm-up) |
|---|---|---|
| VideoPrism base | 0.38 GB | 1.25 s |
| V-JEPA 2 ViT-L (64 frames) | 0.83 GB | 1.80 s |
| PE Video small (16 frames) | 0.87 GB | 1.71 s |

All three fit in 6 GB with room to raise `batch_size` well above the defaults.
`pe-av-large-16-frame` has a 1792-wide tower, so budget more than the small
variant measured here. Probe training is 3,570 × ~1,024 floats — trivial anywhere.

A pod mainly buys a faster extraction pass and more headroom to batch. It becomes
genuinely necessary only if you **unfreeze the backbone** and fully finetune. Get
probe numbers first so you know what finetuning has to beat.

## Quickstart on a fresh machine

```bash
cd video_classification/python/hmdb51

./setup.sh --venv              # deps + environment report
python download_hmdb51.py /data/hmdb51   # ~2.1 GB, resumable
python smoke_test.py           # validate all 3 backbones in ~2 min
./run_all.sh /data/hmdb51      # extract + probe + summary table
```

`run_all.sh` prints a ranked table at the end and writes per-model JSON to
`results/`.

**Run `smoke_test.py` before `run_all.sh`.** It loads each backbone, pushes one
clip through, and prints feature dim, latency and peak VRAM. A config or API
mismatch surfaces in two minutes instead of two hours into extraction.

## Running one model at a time

```bash
python extract_features.py --config configs/vjepa2.json \
    --data-root /data/hmdb51 --out features --subset train --split 1
python extract_features.py --config configs/vjepa2.json \
    --data-root /data/hmdb51 --out features --subset test --split 1
python train_probe.py --features features/vjepa2 --split 1 --head linear
```

Extraction is **resumable**: features go into a memmapped `.npy` alongside a
boolean `done` mask, so re-running after an interruption (or a spot-instance
eviction) picks up where it stopped. Clips that fail to decode are logged,
skipped, and listed in the `_meta.json`.

## Tuning

- `--batch-size` overrides the config; lower it first if you hit OOM.
- `--num-workers` controls decode parallelism. Decoding is the CPU bottleneck —
  on a box with many cores, raise it until the GPU stays busy.
- `--head mlp` fits a one-hidden-layer head instead of a linear one.
- `--split {1,2,3}` selects the official split. Papers report the mean of all
  three; `run_all.sh` does one split per invocation.

## Config format

`configs/*.json` drives everything:

```json
{
  "name": "vjepa2",
  "repo_id": "facebook/vjepa2-vitl-fpc64-256",
  "adapter": "vjepa2",
  "num_frames": 64,
  "pooling": "mean",
  "batch_size": 4,
  "dtype": "float16"
}
```

`pooling` is `"mean"` (average over tokens) or `"pooler"` (the CLS-style output,
used for PE Video). Add `"revision"` to pin a checkpoint revision — VideoPrism
was briefly served from a PR revision, so set it if `from_pretrained` fails.

To add a backbone, write an adapter in `models.py` exposing
`load()` / `encode()` and register it in `ADAPTERS`.

## Dataset sourcing

The canonical Brown University host is **dead** — `serre-lab.clps.brown.edu`
301s to the lab homepage and its `wp-content` paths 404. `download_hmdb51.py`
therefore pulls **videos** from the `innat/HMDB51` Hub mirror (a zip of
`hmdb51_org`, so the outer archive needs no RAR tool and the download resumes)
and the **official splits** from a Wayback snapshot of `test_train_splits.rar`.

Nested archives are unpacked with `libarchive`, which reads RAR3 from a normal
pip install — no non-free `unrar` binary and no `sudo`. Verified split-1 tag
counts are 3570 train / 1530 test / 1666 unused, matching the published protocol.

Avoid the other Hub mirrors for benchmarking: `jxie/hmdb51` and
`yogesh-dev/hmdb51-dataset` are parquet re-packagings, `divm/hmdb51` is a custom
train/val/test split, and `kiyoonkim/hmdb51-gulprgb` ships only 100 of the 153
official split files. None reproduce the official protocol.

## Checkpoint gotchas

Three things that are not obvious from the model docs, all found by running them:

- **VideoPrism** ships only a Flax `.npz` on `main`. The transformers conversion
  lives on `refs/pr/4`, which `configs/videoprism.json` pins via `revision`. If
  that PR merges, drop the field. `MHRDYN7/videoprism-base-f16r288` is a mirror
  with the same weights on `main`.
- **PE Video** has no standalone `pe_video` checkpoint. Every released PE
  checkpoint is a `PeAudioVideoModel` with the video tower nested at
  `.video_model.video_encoder`; the adapter lifts it out and drops the audio and
  text towers. We use `pe-av-large-16-frame` rather than `pe-av-large` because
  the latter has `num_frames: null` (fps-based sampling), ambiguous for a
  fixed benchmark.
- **PE Video needs `timm`** — its vision backbone loads through `TimmWrapper`.

## Requirements

`transformers>=5.15` — **VideoPrism is not in 5.12** and will fail to import on
older installs. `setup.sh` reports which of the three model modules are present.

## What is verified

Tested on an RTX 3060 Laptop:

- Split indexer against the real dataset — 3570 train / 1530 test, 70 train
  clips in each of the 51 classes, matching the official split-1 protocol.
- Frame sampler and all three decoder backends, including the 101 real HMDB
  clips whose container frame count overstates what is decodable.

  torchcodec samples by **timestamp** over the stream span, because many HMDB
  `.avi` files report a `num_frames` 1-2 higher than the decoder can actually
  reach and index-based sampling walks off the end. PyAV and OpenCV sample by
  index over the frames they actually decoded. All three return uniform
  coverage, but they are *not* frame-identical — clips that fall back to a
  secondary backend may land on slightly different frames.
- Resumable extraction: interrupt, re-run, and it skips finished clips.
- Both probe heads, including the partial-cache path.
- All three backbone adapters loaded with **real weights** and produced finite
  embeddings of the expected width.
- `download_hmdb51.py` end-to-end: 51 classes, 6766 clips, 153 split files.

Run `smoke_test.py` first on any new machine — it catches a config or API
mismatch in two minutes instead of two hours into extraction.

## Limitation

Only pooled clip embeddings are cached, so `train_probe.py` fits linear and MLP
heads. True *attentive* probing needs token-level features, which are ~2,000×
larger per clip and would need a different storage strategy.
