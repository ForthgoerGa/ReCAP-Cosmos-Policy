"""Shared, explicit configuration for index construction and online retrieval."""
from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path

import yaml


class RetrievalError(RuntimeError):
    """A retrieval failure invalidates an experiment; never silently fall back."""


@dataclass
class RetrievalConfig:
    strategy: str = "qwen_rerank"
    model_path: str = ""
    model_revision: str = ""
    index_path: str = ""
    worker_python: str = ""
    candidate_k: int = 32
    image_size: int = 224
    embedding_dim: int = 2048
    batch_size: int = 8
    reranker_model_path: str = ""
    reranker_model_revision: str = ""
    reranker_batch_size: int = 30
    reranker_instruction: str = "Retrieve the PushT demonstration image whose pusher position, T-shaped block position and orientation, and target region are most similar to the query image."
    timeout_seconds: float = 300.0
    alpha: float = 0.5
    beta: float = 0.5
    history_length: int = 8
    history_weight_power: float = 1.0
    instruction: str = "Represent the spatial configuration of the pusher, T-shaped block, and target region for matching PushT demonstrations."

    def validate(self):
        if self.strategy in ("wan_vae_image", "wan_vae_video"):
            temporal = 1 if self.strategy == "wan_vae_image" else 3
            if self.image_size != 224 or self.embedding_dim != 16 * temporal * 28 * 28:
                raise ValueError("Wan VAE requires 224px and the complete flattened latent")
            if self.candidate_k != 1 or self.batch_size < 1:
                raise ValueError("Wan VAE requires top-1 and positive batch size")
            if not Path(self.model_path).is_file() or not self.index_path:
                raise ValueError("Wan VAE requires a local checkpoint and index path")
            if len(self.model_revision) != 64 or any(c not in "0123456789abcdef" for c in self.model_revision):
                raise ValueError("Wan VAE model_revision must pin the checkpoint SHA256")
            return
        if self.strategy not in ("standard", "qwen_rerank", "qwen_full", "qwen_two_stage", "qwen_state_text", "qwen_history_state", "qwen_late_fusion", "qwen_video", "qwen_video_late_fusion", "qwen_video_agent_state"):
            raise ValueError(f"Unsupported strategy: {self.strategy}")
        if not all(math.isfinite(x) for x in (self.alpha, self.beta, self.history_weight_power)) or self.alpha < 0 or self.beta < 0 or self.alpha + self.beta <= 0:
            raise ValueError("alpha and beta must be nonnegative with positive sum")
        if not 1 <= self.history_length <= 8 or self.history_weight_power <= 0:
            raise ValueError("history settings must be positive")
        if self.candidate_k < 1 or self.batch_size < 1 or self.timeout_seconds <= 0:
            raise ValueError("K, batch size and timeout must be positive")
        if self.strategy in ("qwen_full", "qwen_state_text", "qwen_history_state", "qwen_late_fusion", "qwen_video", "qwen_video_late_fusion", "qwen_video_agent_state") and self.candidate_k != 1:
            raise ValueError("qwen_full searches the entire index and returns top-1; candidate_k must be 1")
        if self.image_size != 224 or self.embedding_dim not in (2048, 4096):
            raise ValueError("Qwen3-VL retrieval uses 224px images and complete 2048-dim (2B) or 4096-dim (8B) embeddings")
        if self.strategy in ("qwen_rerank", "qwen_full", "qwen_two_stage", "qwen_state_text", "qwen_history_state", "qwen_late_fusion", "qwen_video", "qwen_video_late_fusion", "qwen_video_agent_state"):
            for field in ("model_path", "model_revision", "worker_python", "index_path"):
                if not getattr(self, field):
                    raise ValueError(f"Missing {field}")
            if not Path(self.worker_python).is_file() or not Path(self.model_path).is_dir():
                raise ValueError("worker_python and local model_path must exist")
        if self.strategy == "qwen_two_stage":
            if not self.reranker_model_path or not self.reranker_model_revision or not Path(self.reranker_model_path).is_dir():
                raise ValueError("A local reranker model and pinned revision are required")
            if self.reranker_batch_size < 1 or not self.reranker_instruction.strip():
                raise ValueError("Reranker batch size and instruction must be positive/nonempty")

    def encoder_signature(self):
        if self.strategy in ("wan_vae_image", "wan_vae_video"):
            video = self.strategy == "wan_vae_video"
            return dict(model_revision=self.model_revision, image_size=self.image_size,
                        embedding_dim=self.embedding_dim, modality="video" if video else "image",
                        dtype="float32_weights_bfloat16_autocast", normalization="cosmos_channel_scale_then_fp32_l2",
                        resize="PIL_BICUBIC_full_rgb", input_range="[-1,1]",
                        reduction="flatten_CTHW", temporal_window=8 if video else 1,
                        early_padding="repeat_first_left" if video else "none",
                        vae_input_frames=9 if video else 1,
                        vae_alignment_padding="one_first_frame_left" if video else "none",
                        latent_shape=[16, 3 if video else 1, 28, 28], adapter_version=1)
        signature = {k: getattr(self, k) for k in
                ("model_revision", "image_size", "embedding_dim", "instruction")} | {
                    "dtype": "bfloat16", "normalization": "fp32_l2",
                    "resize": "PIL_BICUBIC_full_rgb", "modality": "image",
                    "adapter_version": 1,
                }

        if self.strategy in ("qwen_state_text", "qwen_history_state"):
            signature.update(modality="image_text", adapter_version=2, state_format="pusht_normalized_state10_v1")
        if self.strategy in ("qwen_video", "qwen_video_late_fusion"):
            signature.update(modality="video", adapter_version=1, video_window=8, video_stride=1)
        if self.strategy == "qwen_video_agent_state":
            signature.update(modality="video_text", adapter_version=3, video_window=8, video_stride=1, state_format="pusht_agent_pos_vel_v1")
        return signature

    def to_dict(self):
        return asdict(self)


def load_config(path):
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    for key, value in raw.items():
        if isinstance(value, str):
            raw[key] = os.path.expanduser(os.path.expandvars(value))
            if "${" in raw[key]:
                raise ValueError(f"Unresolved environment variable in {key}")
    cfg = RetrievalConfig(**raw)
    cfg.validate()
    return cfg
