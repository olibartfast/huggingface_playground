"""Frozen-encoder adapters for the video backbones we probe on HMDB-51.

Every adapter exposes the same contract:

    adapter.num_frames                  -> frames to decode per clip
    adapter.load(device, dtype)         -> materialise weights
    adapter.encode(list[np.ndarray])    -> (B, D) float32 clip embeddings

None of these checkpoints carry an HMDB-51 head, so we only ever run the
encoder and cache pooled embeddings; the classifier is trained separately by
train_probe.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import torch

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


@dataclass
class ModelConfig:
    name: str
    repo_id: str
    adapter: str
    num_frames: int
    pooling: str = "mean"          # "mean" over tokens, or "pooler" for the CLS-style output
    batch_size: int = 4
    dtype: str = "float16"
    revision: str | None = None    # some checkpoints still live on a PR revision
    processor_kwargs: dict = field(default_factory=dict)

    @classmethod
    def from_json(cls, path: str) -> "ModelConfig":
        with open(path, encoding="utf-8") as fh:
            return cls(**json.load(fh))


class BaseAdapter:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.processor = None
        self.device = "cpu"
        self.torch_dtype = torch.float32

    @property
    def num_frames(self) -> int:
        return self.config.num_frames

    def load(self, device: str, dtype: str) -> None:
        raise NotImplementedError

    def _prepare(self, videos: list[np.ndarray]) -> dict:
        """Run the HF video processor and move tensors onto the device."""
        inputs = self.processor(
            list(videos), return_tensors="pt", **self.config.processor_kwargs
        )
        return {
            k: v.to(self.device, dtype=self.torch_dtype if v.is_floating_point() else None)
            for k, v in inputs.items()
            if isinstance(v, torch.Tensor)
        }

    def _pool(self, outputs) -> torch.Tensor:
        if self.config.pooling == "pooler":
            pooled = getattr(outputs, "pooler_output", None)
            if pooled is None:
                raise RuntimeError(
                    f"{self.config.name}: pooling='pooler' but the model returned no "
                    "pooler_output. Use pooling='mean'."
                )
            return pooled
        return outputs.last_hidden_state.mean(dim=1)

    @torch.no_grad()
    def encode(self, videos: list[np.ndarray]) -> np.ndarray:
        raise NotImplementedError


class VJepa2Adapter(BaseAdapter):
    """facebook/vjepa2-* - self-supervised encoder, no released HMDB-51 head."""

    def load(self, device: str, dtype: str) -> None:
        from transformers import AutoModel, AutoVideoProcessor

        self.device, self.torch_dtype = device, DTYPES[dtype]
        self.processor = AutoVideoProcessor.from_pretrained(
            self.config.repo_id, revision=self.config.revision
        )
        self.model = AutoModel.from_pretrained(
            self.config.repo_id,
            revision=self.config.revision,
            dtype=self.torch_dtype,
            attn_implementation="sdpa",
        ).to(device).eval()

    @torch.no_grad()
    def encode(self, videos: list[np.ndarray]) -> np.ndarray:
        inputs = self._prepare(videos)
        # skip_predictor=True: we only want encoder features, not the JEPA predictor.
        features = self.model.get_vision_features(inputs["pixel_values_videos"])
        return features.mean(dim=1).float().cpu().numpy()


class VideoPrismAdapter(BaseAdapter):
    """google/videoprism-* - factorised spatio-temporal encoder, head is untrained."""

    def load(self, device: str, dtype: str) -> None:
        from transformers import AutoModel, AutoVideoProcessor

        self.device, self.torch_dtype = device, DTYPES[dtype]
        self.processor = AutoVideoProcessor.from_pretrained(
            self.config.repo_id, revision=self.config.revision
        )
        self.model = AutoModel.from_pretrained(
            self.config.repo_id, revision=self.config.revision, dtype=self.torch_dtype
        ).to(device).eval()

    @torch.no_grad()
    def encode(self, videos: list[np.ndarray]) -> np.ndarray:
        inputs = self._prepare(videos)
        outputs = self.model(inputs["pixel_values_videos"])
        return self._pool(outputs).float().cpu().numpy()


class PeVideoAdapter(BaseAdapter):
    """facebook/pe-av-* - Perception Encoder; we keep only the video tower.

    There is no standalone ``pe_video`` checkpoint on the Hub: every released
    PE checkpoint is a ``PeAudioVideoModel`` whose video tower is nested at
    ``.video_model.video_encoder``. We load the composite model, lift out that
    encoder, and drop the audio and text towers before touching the GPU.

    PeVideoEncoder returns pooler_output = hidden_states[:, 0], the CLS-style
    clip token, which is the natural probe feature. We read it *before* the
    contrastive projection head, since pre-projection features probe better
    than the retrieval embedding.
    """

    # Checked in order; the first that resolves wins.
    ENCODER_PATHS = (
        ("video_model", "video_encoder"),          # PeAudioVideoModel
        ("video_encoder",),                        # PeVideoModel
        ("audio_video_encoder", "embedder", "video_encoder"),
    )

    @staticmethod
    def _resolve_encoder(model):
        for path in PeVideoAdapter.ENCODER_PATHS:
            node = model
            for attr in path:
                node = getattr(node, attr, None)
                if node is None:
                    break
            if node is not None:
                return node, ".".join(path)
        raise RuntimeError(
            "Could not locate the video encoder on "
            f"{type(model).__name__}. Top-level modules: "
            f"{[n for n, _ in model.named_children()]}"
        )

    def load(self, device: str, dtype: str) -> None:
        from transformers import AutoModel, AutoProcessor

        self.device, self.torch_dtype = device, DTYPES[dtype]
        processor = AutoProcessor.from_pretrained(
            self.config.repo_id, revision=self.config.revision
        )
        self.processor = processor.video_processor

        full_model = AutoModel.from_pretrained(
            self.config.repo_id, revision=self.config.revision, dtype=self.torch_dtype
        )
        encoder, path = self._resolve_encoder(full_model)
        print(f"  [pe_video] using encoder at {type(full_model).__name__}.{path}")
        self.model = encoder.to(device).eval()
        del full_model

    @torch.no_grad()
    def encode(self, videos: list[np.ndarray]) -> np.ndarray:
        inputs = self._prepare(videos)
        outputs = self.model(
            pixel_values_videos=inputs["pixel_values_videos"],
            padding_mask_videos=inputs.get("padding_mask_videos"),
        )
        return self._pool(outputs).float().cpu().numpy()


ADAPTERS = {
    "vjepa2": VJepa2Adapter,
    "videoprism": VideoPrismAdapter,
    "pe_video": PeVideoAdapter,
}


def build_adapter(config: ModelConfig) -> BaseAdapter:
    if config.adapter not in ADAPTERS:
        raise ValueError(
            f"Unknown adapter {config.adapter!r}. Available: {sorted(ADAPTERS)}"
        )
    return ADAPTERS[config.adapter](config)
