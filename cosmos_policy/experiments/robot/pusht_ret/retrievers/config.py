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
    timeout_seconds: float = 300.0
    alpha: float = 0.5
    beta: float = 0.5
    history_length: int = 8
    history_weight_power: float = 1.0
    instruction: str = "Represent the spatial configuration of the pusher, T-shaped block, and target region for matching PushT demonstrations."

    def validate(self):
        if self.strategy not in ("standard", "qwen_rerank", "qwen_full", "qwen_state_text", "qwen_history_state", "qwen_late_fusion"):
            raise ValueError(f"Unsupported strategy: {self.strategy}")
        if not all(math.isfinite(x) for x in (self.alpha, self.beta, self.history_weight_power)) or self.alpha < 0 or self.beta < 0 or self.alpha + self.beta <= 0:
            raise ValueError("alpha and beta must be nonnegative with positive sum")
        if not 1 <= self.history_length <= 8 or self.history_weight_power <= 0:
            raise ValueError("history settings must be positive")
        if self.candidate_k < 1 or self.batch_size < 1 or self.timeout_seconds <= 0:
            raise ValueError("K, batch size and timeout must be positive")
        if self.strategy in ("qwen_full", "qwen_state_text", "qwen_history_state", "qwen_late_fusion") and self.candidate_k != 1:
            raise ValueError("qwen_full searches the entire index and returns top-1; candidate_k must be 1")
        if self.image_size != 224 or self.embedding_dim != 2048:
            raise ValueError("v1 uses 224px images and complete 2048-dim 2B embeddings")
        if self.strategy in ("qwen_rerank", "qwen_full", "qwen_state_text", "qwen_history_state", "qwen_late_fusion"):
            for field in ("model_path", "model_revision", "worker_python", "index_path"):
                if not getattr(self, field):
                    raise ValueError(f"Missing {field}")
            if not Path(self.worker_python).is_file() or not Path(self.model_path).is_dir():
                raise ValueError("worker_python and local model_path must exist")

    def encoder_signature(self):
        signature = {k: getattr(self, k) for k in
                ("model_revision", "image_size", "embedding_dim", "instruction")} | {
                    "dtype": "bfloat16", "normalization": "fp32_l2",
                    "resize": "PIL_BICUBIC_full_rgb", "modality": "image",
                    "adapter_version": 1,
                }

        if self.strategy in ("qwen_state_text", "qwen_history_state"):
            signature.update(modality="image_text", adapter_version=2, state_format="pusht_normalized_state10_v1")
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
