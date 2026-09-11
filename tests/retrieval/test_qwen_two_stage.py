"""The visual shortlist must constrain reranking without GT-state gating."""
from types import SimpleNamespace
import numpy as np
import pytest

from test_qwen_rerank import pool
from cosmos_policy.experiments.robot.pusht_ret.retrievers.config import RetrievalError
from cosmos_policy.experiments.robot.pusht_ret.retrievers.qwen_two_stage import QwenTwoStageRetrieval, shortlist


def test_shortlist_matches_full_pool_sort_and_keeps_ties_stable():
    vectors = np.array([[.1, 0], [.9, 0], [.9, 0], [.3, 0]], np.float32)
    ids, scores = shortlist(np.array([1, 0]), vectors, 3)
    assert ids.tolist() == [1, 2, 3]
    np.testing.assert_allclose(scores, [.9, .9, .3])
    with pytest.raises(RetrievalError):
        shortlist(np.array([np.nan, 0]), vectors, 3)
    with pytest.raises(RetrievalError):
        shortlist(np.array([1, 0]), vectors, 5)


def test_cross_encoder_selects_only_from_visual_shortlist_and_preserves_payload():
    obj = pool(QwenTwoStageRetrieval)
    obj.qwen_cfg = SimpleNamespace(candidate_k=2)
    obj.embeddings = np.eye(4, dtype=np.float32)
    query = np.array([[.1, .2, .7, .6]], np.float32)
    image = np.full((8, 8, 3), 27, np.uint8)
    obj.client = SimpleNamespace(encode=lambda frames: (query, {}))
    seen = []

    def forbidden(*args, **kwargs):
        raise AssertionError('Privileged state must not gate visual retrieval')

    def rerank(q, candidates):
        np.testing.assert_array_equal(q, image)
        seen.extend(candidates)
        # Equal saturated probabilities still have different ranking logits.
        return np.array([20, 21], np.float32), np.ones(2, np.float32), {'roundtrip_seconds': .01}

    obj.get_state_candidates = forbidden
    obj.reranker_client = SimpleNamespace(rerank=rerank)
    expected = obj.get_candidate_data(3)
    result = obj.get_retrieved_data(object(), object(), object(), primary_image=image)
    for actual, original in zip(result, expected):
        np.testing.assert_array_equal(actual, original)
    assert obj.last_result['selected_index'] == 3
    assert obj.last_result['selected_coarse_rank'] == 2
    assert obj.last_result['rerank_count'] == 2
    assert obj.last_result['candidate_count'] == 4
    for frame, index in zip(seen, [2, 3]):
        np.testing.assert_array_equal(frame, obj.get_candidate_data(index)[0][0])


def test_cross_encoder_failure_invalidates_evaluation():
    obj = pool(QwenTwoStageRetrieval)
    obj.qwen_cfg = SimpleNamespace(candidate_k=2)
    obj.embeddings = np.eye(4, dtype=np.float32)
    obj.client = SimpleNamespace(encode=lambda frames: (np.ones((1, 4), np.float32), {}))
    def fail(*args):
        raise RuntimeError('Reranker stopped')
    obj.reranker_client = SimpleNamespace(rerank=fail)
    with pytest.raises(RetrievalError, match='Reranker stopped'):
        obj.get_retrieved_data(primary_image=np.zeros((8, 8, 3), np.uint8))
