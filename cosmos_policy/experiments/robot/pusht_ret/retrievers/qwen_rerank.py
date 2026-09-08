"""State top-K followed by frozen Qwen image cosine ranking."""
import json
from pathlib import Path
import time
import numpy as np

from ..retrieval import PushTRetrieval
from .config import RetrievalError, load_config
from .index import pool_manifest, sha256, validate_vectors
from .qwen_client import QwenClient
from .qwen_encoder import implementation_hash


def cosine_order(query, candidates):
    scores = candidates @ query
    if not np.isfinite(scores).all():
        raise RetrievalError("Nonfinite cosine scores")
    return np.argsort(-scores, kind="stable"), scores


class QwenRerankRetrieval(PushTRetrieval):
    def __init__(self, *args, retrieval_config, **kwargs):
        self.qwen_cfg = load_config(retrieval_config)
        if kwargs.get("ret_context_multiplier", 1) != 1 or kwargs.get("ret_image_subsample", 1) != 1:
            raise RetrievalError("v1 uses original 8-frame context without subsampling")
        if kwargs.get("chunk_size", 8) != 8:
            raise RetrievalError("v1 requires the checkpoint-compatible chunk size 8")
        super().__init__(*args, **kwargs)
        root = Path(self.qwen_cfg.index_path)
        try:
            manifest = json.loads((root / "manifest.json").read_text())
            actual = pool_manifest(self)
            for k, v in actual.items():
                if manifest[k] != v:
                    raise ValueError(f"Pool mismatch: {k}")
            if manifest["encoder"] != self.qwen_cfg.encoder_signature():
                raise ValueError("Encoder configuration mismatch")
            if manifest["implementation_hash"] != implementation_hash():
                raise ValueError("Encoder implementation mismatch; rebuild index")
            if manifest["embeddings_sha256"] != sha256(root / "embeddings.npy"):
                raise ValueError("Index checksum mismatch")
            self.embeddings = np.load(root / "embeddings.npy", mmap_mode="r", allow_pickle=False)
            validate_vectors(self.embeddings, len(self._subframes), self.qwen_cfg.embedding_dim)
            self.client = QwenClient(self.qwen_cfg)
            if (self.client.metadata["implementation_hash"] != manifest["implementation_hash"]
                    or self.client.metadata["model_hashes"] != manifest["worker"]["model_hashes"]
                    or self.client.metadata["versions"] != manifest["worker"]["versions"]):
                self.client.close()
                raise ValueError("Worker/index implementation mismatch")
        except Exception as e:
            raise RetrievalError(f"Cannot initialize Qwen retrieval: {e}") from e

    def get_retrieved_data(self, agent_pos, block_pos, block_angle,
                           block_pos_history=None, agent_pos_history=None, *, primary_image):
        started = time.perf_counter()
        try:
            indices, distances = self.get_state_candidates(
                agent_pos, block_pos, block_angle, block_pos_history, agent_pos_history,
                k=self.qwen_cfg.candidate_k)
            query, metadata = self.client.encode([primary_image])
            order, scores = cosine_order(query[0], self.embeddings[indices])
            rank = int(order[0])
            selected = int(indices[rank])
            result = self.get_candidate_data(selected)
            self.last_result = {
                "strategy": "qwen_rerank", "selected_id": self.candidate_id(selected),
                "selected_state_rank": rank + 1,
                "candidate_ids": [self.candidate_id(i) for i in indices],
                "state_distances": distances.tolist(), "qwen_scores": scores.tolist(),
                "retrieval_seconds": time.perf_counter() - started, **metadata,
            }
            return result
        except RetrievalError:
            raise
        except Exception as e:
            raise RetrievalError(f"Qwen retrieval failed: {e}") from e

    def close(self):
        self.client.close()
