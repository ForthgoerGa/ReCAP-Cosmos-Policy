"""Full-pool image-only cosine and weighted geometric-distance late fusion."""
import time

import numpy as np

from .config import RetrievalError
from .qwen_multimodal import QwenMultimodalBase
from .state_features import FEATURE_WEIGHTS, state_feature


def minmax_similarity(values, distance=False):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Expected a finite, nonempty score vector")
    span = float(values.max() - values.min())
    if span <= 1e-8:
        return np.zeros_like(values)  # Constant evidence cannot change ranking.
    result = (values - values.min()) / span
    return 1.0 - result if distance else result


def fuse_scores(cosines, distances, alpha, beta):
    visual = minmax_similarity(cosines)
    state = minmax_similarity(distances, distance=True)
    return (alpha * visual + beta * state) / (alpha + beta), visual, state


class QwenLateFusionRetrieval(QwenMultimodalBase):
    mode = "qwen_late_fusion"

    def get_retrieved_data(self, agent_pos=None, block_pos=None, block_angle=None,
                           block_pos_history=None, agent_pos_history=None, *,
                           primary_image, state_history):
        started = time.perf_counter()
        try:
            query_state = state_feature(state_history) * FEATURE_WEIGHTS
            query, metadata = self.client.encode([primary_image])
            cosines = self.embeddings @ query[0]
            # Direct full-pool distances: no old initial-position filtering or Top-K.
            distances = ((self._feat - query_state) ** 2).sum(axis=1)
            final, visual, state = fuse_scores(cosines, distances, self.qwen_cfg.alpha, self.qwen_cfg.beta)
            if not np.isfinite(final).all():
                raise ValueError("Nonfinite late-fusion scores")
            selected = int(np.argmax(final))
            result = self.get_candidate_data(selected)
            self.last_result = {
                "strategy": self.mode, "selected_id": self.candidate_id(selected),
                "selected_index": selected, "search_scope": "full_pool",
                "candidate_count": len(self._subframes),
                "visual_cosine": float(cosines[selected]), "visual_score": float(visual[selected]),
                "state_distance": float(distances[selected]), "state_score": float(state[selected]),
                "final_score": float(final[selected]), "alpha": self.qwen_cfg.alpha, "beta": self.qwen_cfg.beta,
                "visual_minmax": [float(cosines.min()), float(cosines.max())],
                "distance_minmax": [float(distances.min()), float(distances.max())],
                "visual_top1_index": int(np.argmax(cosines)), "state_top1_index": int(np.argmin(distances)),
                "retrieval_seconds": time.perf_counter() - started, **metadata,
            }
            return result
        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError(f"Late fusion failed: {exc}") from exc
