"""Full image embedding top-K followed by a query/document cross-encoder."""
import time

import numpy as np

from .config import RetrievalError
from .qwen_full import QwenFullRetrieval
from .qwen_pair_client import QwenPairClient


def shortlist(query, embeddings, k):
    scores = embeddings @ query
    if k < 1 or k > len(scores) or not np.isfinite(scores).all():
        raise RetrievalError("Invalid full-pool shortlist")
    # Stable ties are resolved by the original index order.
    indices = np.argsort(-scores, kind="stable")[:k]
    return indices, scores[indices]


class QwenTwoStageRetrieval(QwenFullRetrieval):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.qwen_cfg.strategy != "qwen_two_stage":
            self.client.close()
            raise RetrievalError("QwenTwoStageRetrieval requires qwen_two_stage")
        try:
            self.reranker_client = QwenPairClient(self.qwen_cfg)
        except Exception:
            self.client.close()
            raise

    def get_retrieved_data(self, agent_pos=None, block_pos=None, block_angle=None,
                           block_pos_history=None, agent_pos_history=None, *, primary_image):
        started = time.perf_counter()
        try:
            query, embedding_meta = self.client.encode([primary_image])
            search_started = time.perf_counter()
            indices, coarse_scores = shortlist(query[0], self.embeddings, self.qwen_cfg.candidate_k)
            search_seconds = time.perf_counter() - search_started
            coarse_seconds = time.perf_counter() - started
            candidates_started = time.perf_counter()
            candidate_frames = []
            for index in indices:
                sf = self._subframes[int(index)]
                candidate_frames.append(self._base_data[(sf['split'], sf['demo'])]['images'][sf['t_last']])
            candidates_seconds = time.perf_counter() - candidates_started
            logits, scores, reranker_meta = self.reranker_client.rerank(primary_image, candidate_frames)
            # Logits preserve ordering when sigmoid probabilities saturate at 1.
            rank = int(np.argmax(logits))
            selected = int(indices[rank])
            materialize_started = time.perf_counter()
            result = self.get_candidate_data(selected)
            materialize_seconds = time.perf_counter() - materialize_started
            self.last_result = {
                "strategy": "qwen_two_stage", "selected_id": self.candidate_id(selected),
                "selected_index": selected, "selected_coarse_rank": rank + 1,
                "coarse_top1_id": self.candidate_id(indices[0]),
                "search_scope": "full_pool", "candidate_count": len(self._subframes),
                "rerank_count": len(indices), "candidate_ids": [self.candidate_id(i) for i in indices],
                "coarse_scores": coarse_scores.tolist(), "reranker_logits": logits.tolist(),
                "reranker_scores": scores.tolist(), "qwen_score": float(coarse_scores[rank]),
                "reranker_score": float(scores[rank]), "reranker_logit": float(logits[rank]),
                "coarse_retrieval_seconds": coarse_seconds, "coarse_search_seconds": search_seconds,
                "candidate_materialization_seconds": candidates_seconds,
                "rerank_seconds": reranker_meta["roundtrip_seconds"],
                "payload_materialization_seconds": materialize_seconds,
                "embedding_metrics": embedding_meta, "reranker_metrics": reranker_meta,
                "retrieval_seconds": time.perf_counter() - started,
            }
            return result
        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError(f"Two-stage retrieval failed: {exc}") from exc

    def close(self):
        if hasattr(self, 'reranker_client'):
            self.reranker_client.close()
        super().close()
