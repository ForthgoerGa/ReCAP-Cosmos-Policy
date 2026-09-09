"""Shared full-pool infrastructure for independent multimodal ablations."""
import json
from pathlib import Path
import time

import numpy as np

from ..retrieval import PushTRetrieval
from .config import RetrievalError, load_config
from .index import pool_manifest, sha256, validate_vectors
from .qwen_client import QwenClient
from .qwen_encoder import implementation_hash as image_hash
from .qwen_joint_encoder import implementation_hash as joint_hash
from .state_features import history_embedding, representation_signature, state_feature, state_text


def validate_index(obj, root, cfg):
    manifest = json.loads((root / "manifest.json").read_text())
    for key, value in pool_manifest(obj).items():
        if manifest[key] != value:
            raise ValueError(f"Pool mismatch: {key}")
    joint = cfg.strategy in ("qwen_state_text", "qwen_history_state")
    expected_hash = joint_hash() if joint else image_hash()
    if manifest["encoder"] != cfg.encoder_signature() or manifest["implementation_hash"] != expected_hash:
        raise ValueError("Index encoder mismatch; rebuild index")
    if joint and manifest.get("representation") != representation_signature(cfg):
        raise ValueError("State/history representation mismatch; rebuild index")
    if manifest["embeddings_sha256"] != sha256(root / "embeddings.npy"):
        raise ValueError("Index checksum mismatch")
    obj.embeddings = np.load(root / "embeddings.npy", mmap_mode="r", allow_pickle=False)
    validate_vectors(obj.embeddings, len(obj._subframes), cfg.embedding_dim)
    obj.client = QwenClient(cfg)
    try:
        for key in ("implementation_hash", "model_hashes", "versions"):
            expected = manifest["implementation_hash"] if key == "implementation_hash" else manifest["worker"][key]
            if obj.client.metadata[key] != expected:
                raise ValueError(f"Worker/index mismatch: {key}")
    except Exception:
        obj.client.close()
        raise


class QwenMultimodalBase(PushTRetrieval):
    mode = "qwen_state_text"
    history = False

    def __init__(self, *args, retrieval_config, **kwargs):
        self.qwen_cfg = load_config(retrieval_config)
        if self.qwen_cfg.strategy != self.mode:
            raise RetrievalError(f"Requires strategy={self.mode}")
        if (kwargs.get("chunk_size", 8) != 8 or kwargs.get("block_rel", False)
                or kwargs.get("ret_context_multiplier", 1) != 1 or kwargs.get("ret_image_subsample", 1) != 1):
            raise RetrievalError("Requires absolute states and the original 8-step payload")
        super().__init__(*args, **kwargs)
        try:
            validate_index(self, Path(self.qwen_cfg.index_path), self.qwen_cfg)
        except Exception as exc:
            raise RetrievalError(f"Cannot initialize {self.mode}: {exc}") from exc

    def get_retrieved_data(self, agent_pos=None, block_pos=None, block_angle=None,
                           block_pos_history=None, agent_pos_history=None, *,
                           primary_image, state_history, primary_images=None):
        started = time.perf_counter()
        try:
            states = np.asarray(state_history, dtype=np.float32)
            state_feature(states)  # Validate before starting IPC.
            if self.history:
                if primary_images is None or not len(primary_images):
                    raise ValueError("Historical retrieval requires causal image history")
                images = list(primary_images[-self.qwen_cfg.history_length:])
                if len(images) > len(states) or not np.array_equal(images[-1], primary_image):
                    raise ValueError("Image/state history must end at the current observation")
                # States contain two extra old steps for the earliest image's velocity.
                ends = range(len(states) - len(images) + 1, len(states) + 1)
                texts = [state_text(state_feature(states[max(0, end - 3):end])) for end in ends]
            else:
                images = [primary_image]
                texts = [state_text(state_feature(states))]
            vectors, metadata = self.client.encode(images, texts)
            query = history_embedding(vectors, self.qwen_cfg.history_weight_power) if self.history else vectors[0]
            scores = self.embeddings @ query
            if not np.isfinite(scores).all():
                raise ValueError("Nonfinite cosine scores")
            selected = int(np.argmax(scores))
            result = self.get_candidate_data(selected)
            self.last_result = {
                "strategy": self.mode, "selected_id": self.candidate_id(selected),
                "selected_index": selected, "search_scope": "full_pool",
                "candidate_count": len(self._subframes), "qwen_score": float(scores[selected]),
                "query_state_texts": texts, "history_frames": len(images),
                "history_weights": (np.arange(1, len(images) + 1, dtype=float) ** self.qwen_cfg.history_weight_power).tolist(),
                "retrieval_seconds": time.perf_counter() - started, **metadata,
            }
            return result
        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError(f"{self.mode} failed: {exc}") from exc

    def close(self):
        self.client.close()


class QwenStateTextRetrieval(QwenMultimodalBase):
    mode = "qwen_state_text"


class QwenHistoryStateRetrieval(QwenMultimodalBase):
    mode = "qwen_history_state"
    history = True
