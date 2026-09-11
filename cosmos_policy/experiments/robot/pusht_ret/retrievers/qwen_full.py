"""Full-pool Qwen3-VL image embedding retrieval (top-1)."""
import json
from pathlib import Path
import time
import numpy as np

from ..retrieval import PushTRetrieval
from .config import RetrievalError, load_config
from .index import pool_manifest, sha256, validate_vectors
from .qwen_client import QwenClient
from .qwen_encoder import implementation_hash


def cosine_top1(query, candidates):
    scores = candidates @ query
    if not np.isfinite(scores).all():
        raise RetrievalError("Nonfinite cosine scores")
    # np.argmax picks the first index row on exact ties, deterministically.
    return int(np.argmax(scores)), scores


class QwenFullRetrieval(PushTRetrieval):
    """Retrieve the single best demo frame from the complete Qwen image index."""
    def __init__(self, *args, retrieval_config, **kwargs):
        self.qwen_cfg = load_config(retrieval_config)
        if self.qwen_cfg.strategy not in ("qwen_full", "qwen_two_stage"):
            raise RetrievalError("QwenFullRetrieval requires a full-pool image strategy")
        if kwargs.get("ret_context_multiplier", 1) != 1 or kwargs.get("ret_image_subsample", 1) != 1:
            raise RetrievalError("qwen_full uses the original 8-frame context without subsampling")
        if kwargs.get("chunk_size", 8) != 8:
            raise RetrievalError("qwen_full requires checkpoint-compatible chunk size 8")
        super().__init__(*args, **kwargs)
        root = Path(self.qwen_cfg.index_path)
        try:
            manifest = json.loads((root / "manifest.json").read_text())
            actual = pool_manifest(self)
            for key, value in actual.items():
                if manifest[key] != value:
                    raise ValueError(f"Pool mismatch: {key}")
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
        except Exception as exc:
            raise RetrievalError(f"Cannot initialize full Qwen retrieval: {exc}") from exc

    def get_retrieved_data(self, agent_pos=None, block_pos=None, block_angle=None,
                           block_pos_history=None, agent_pos_history=None, *, primary_image):
        started = time.perf_counter()
        try:
            query, metadata = self.client.encode([primary_image])
            selected, scores = cosine_top1(query[0], self.embeddings)
            result = self.get_candidate_data(selected)
            self.last_result = {
                "strategy": "qwen_full", "selected_id": self.candidate_id(selected),
                "selected_rank": 1, "selected_index": selected,
                "search_scope": "full_pool", "candidate_count": len(self._subframes),
                "qwen_score": float(scores[selected]),
                "retrieval_seconds": time.perf_counter() - started, **metadata,
            }
            return result
        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError(f"Full Qwen retrieval failed: {exc}") from exc

    def close(self):
        self.client.close()
