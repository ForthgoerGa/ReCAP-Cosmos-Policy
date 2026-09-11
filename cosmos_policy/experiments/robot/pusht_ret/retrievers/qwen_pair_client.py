"""Persistent local IPC for query/document cross-encoder reranking."""
import base64
import io
import json
import time

import numpy as np
from PIL import Image

from .config import RetrievalError
from .qwen_client import QwenClient


class QwenPairClient(QwenClient):
    def __init__(self, cfg):
        super().__init__(cfg, "cosmos_policy.experiments.robot.pusht_ret.retrievers.qwen_pair_worker")

    def rerank(self, query, candidates):
        started = time.perf_counter()
        try:
            images = []
            for frame in [query, *candidates]:
                if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3:
                    raise ValueError("Expected uint8 RGB images")
                buf = io.BytesIO()
                Image.fromarray(frame).save(buf, format="PNG")
                images.append(base64.b64encode(buf.getvalue()).decode())
            self.counter += 1
            self.process.stdin.write(json.dumps({"id": self.counter, "images": images}) + "\n")
            self.process.stdin.flush()
            result = self._receive()
            logits = np.asarray(result.pop("logits"), dtype=np.float32)
            scores = np.asarray(result.pop("scores"), dtype=np.float32)
            if (result["id"] != self.counter or logits.shape != (len(candidates),)
                    or scores.shape != logits.shape or not np.isfinite(logits).all()
                    or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any()):
                raise ValueError("Invalid reranker response")
            result["roundtrip_seconds"] = time.perf_counter() - started
            return logits, scores, result
        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError(f"Reranker IPC failed: {exc}") from exc
