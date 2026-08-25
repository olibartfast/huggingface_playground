"""HMDB-51 split indexing and frame sampling.

Expects the official dataset layout produced by ``download_hmdb51.sh``::

    <root>/videos/<class_name>/<clip>.avi
    <root>/splits/<class_name>_test_split<N>.txt

Each split file lists ``<filename> <tag>`` where tag 1 = train, 2 = test,
0 = unused (the official protocol discards those clips).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

TRAIN_TAG = "1"
TEST_TAG = "2"


@dataclass(frozen=True)
class Clip:
    path: str
    label: int
    class_name: str


class Hmdb51Index:
    """Resolves the official train/test splits into flat lists of clips."""

    def __init__(self, root: str, split: int = 1):
        self.root = root
        self.split = split
        self.videos_dir = os.path.join(root, "videos")
        self.splits_dir = os.path.join(root, "splits")

        if not os.path.isdir(self.videos_dir):
            raise FileNotFoundError(
                f"No videos directory at {self.videos_dir}. Run download_hmdb51.sh first."
            )
        if not os.path.isdir(self.splits_dir):
            raise FileNotFoundError(
                f"No splits directory at {self.splits_dir}. Run download_hmdb51.sh first."
            )

        self.classes = sorted(
            d for d in os.listdir(self.videos_dir)
            if os.path.isdir(os.path.join(self.videos_dir, d))
        )
        if len(self.classes) != 51:
            raise ValueError(
                f"Expected 51 class directories under {self.videos_dir}, found "
                f"{len(self.classes)}. The download is probably incomplete."
            )
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def _split_file(self, class_name: str) -> str:
        return os.path.join(self.splits_dir, f"{class_name}_test_split{self.split}.txt")

    def clips(self, subset: str) -> list[Clip]:
        """Return the clips for ``subset`` ("train" or "test")."""
        if subset not in ("train", "test"):
            raise ValueError(f"subset must be 'train' or 'test', got {subset!r}")
        wanted = TRAIN_TAG if subset == "train" else TEST_TAG

        out: list[Clip] = []
        for class_name in self.classes:
            split_file = self._split_file(class_name)
            if not os.path.isfile(split_file):
                raise FileNotFoundError(f"Missing split file: {split_file}")

            with open(split_file, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    parts = line.split()
                    if len(parts) < 2 or parts[-1] != wanted:
                        continue
                    filename = " ".join(parts[:-1])
                    path = os.path.join(self.videos_dir, class_name, filename)
                    if not os.path.isfile(path):
                        # The .rar extraction occasionally mangles unusual filenames;
                        # skipping loudly beats failing 2 hours into extraction.
                        print(f"[warn] listed but missing on disk: {path}")
                        continue
                    out.append(Clip(path, self.class_to_idx[class_name], class_name))
        return out


def sample_indices(total_frames: int, num_frames: int) -> np.ndarray:
    """Uniformly sample ``num_frames`` frame indices across a clip.

    Short clips repeat frames rather than being dropped, which matches the
    padding behaviour of the reference implementations.
    """
    if total_frames <= 0:
        raise ValueError("clip has no frames")
    return np.linspace(0, total_frames - 1, num_frames).round().astype(np.int64)


def decode_video(path: str, num_frames: int) -> np.ndarray:
    """Decode ``num_frames`` uniformly spaced RGB frames as uint8 (T, H, W, C).

    Tries torchcodec (fastest, what the HF docs use), then PyAV, then OpenCV,
    so the harness runs on whatever the target machine happens to have.
    """
    errors = []
    for backend in (_decode_torchcodec, _decode_av, _decode_cv2):
        try:
            return backend(path, num_frames)
        except ImportError:
            continue
        except Exception as exc:
            # A backend that is present but chokes on this file must not sink the
            # clip - hand it to the next one.
            errors.append(f"{backend.__name__}: {type(exc).__name__}: {exc}")
    if errors:
        raise RuntimeError(f"all backends failed for {path}: " + " | ".join(errors))
    raise RuntimeError(
        "No usable video backend. Install one of: torchcodec, av, opencv-python-headless."
    )


def _decode_torchcodec(path: str, num_frames: int) -> np.ndarray:
    from torchcodec.decoders import VideoDecoder

    decoder = VideoDecoder(path)
    metadata = decoder.metadata

    # Sample by timestamp, not frame index. Many HMDB .avi files report a
    # num_frames that overstates what is actually decodable (e.g. 212 vs 210),
    # so index-based sampling walks off the end of the stream; the stream span
    # is reliable where the frame count is not.
    start = metadata.begin_stream_seconds
    end = metadata.end_stream_seconds
    if start is None or end is None or end <= start:
        raise RuntimeError(f"torchcodec reported no stream span for {path}")

    seconds = np.linspace(start, end, num_frames, endpoint=False)
    frames = decoder.get_frames_played_at(seconds=seconds.tolist()).data
    return frames.permute(0, 2, 3, 1).contiguous().numpy()  # (T, H, W, C)


def _decode_av(path: str, num_frames: int) -> np.ndarray:
    import av

    with av.open(path) as container:
        stream = container.streams.video[0]
        total = stream.frames
        decoded = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]

    if not decoded:
        raise RuntimeError(f"PyAV decoded 0 frames from {path}")
    total = total or len(decoded)
    indices = sample_indices(min(total, len(decoded)), num_frames)
    return np.stack([decoded[i] for i in indices])


def _decode_cv2(path: str, num_frames: int) -> np.ndarray:
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {path}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            # Some HMDB .avi files report no frame count; fall back to a full read.
            frames = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not frames:
                raise RuntimeError(f"OpenCV decoded 0 frames from {path}")
            indices = sample_indices(len(frames), num_frames)
            return np.stack([frames[i] for i in indices])

        indices = sample_indices(total, num_frames)
        out = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                if not out:
                    raise RuntimeError(f"OpenCV failed to read frame {idx} of {path}")
                frame_rgb = out[-1]
            else:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out.append(frame_rgb)
        return np.stack(out)
    finally:
        cap.release()
