"""Full visual search must neither gate by state nor alter demo payloads."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from test_qwen_rerank import pool
from cosmos_policy.experiments.robot.pusht_ret.retrievers.config import RetrievalConfig, RetrievalError
from cosmos_policy.experiments.robot.pusht_ret.retrievers.qwen_full import QwenFullRetrieval, cosine_top1


def test_full_search_ignores_state_and_keeps_selected_payload():
    obj = pool(QwenFullRetrieval)
    obj.embeddings = np.eye(4, dtype=np.float32)
    query = np.array([[0., 0., 0., 1.]], dtype=np.float32)
    obj.client = SimpleNamespace(encode=lambda images: (query, {}))

    def forbidden(*args, **kwargs):
        raise AssertionError("State candidate generation must not run")

    obj.get_state_candidates = forbidden
    obj._feat = obj._demo_init_block_pos = None
    expected = obj.get_candidate_data(3)
    # Image-only and eval-compatible APIs must both ignore query state.
    for kwargs in ({}, dict(agent_pos=object(), block_pos=object(), block_angle=object(),
                           block_pos_history=object(), agent_pos_history=object())):
        result = obj.get_retrieved_data(primary_image=np.zeros((8, 8, 3), np.uint8), **kwargs)
        for actual, original in zip(result, expected):
            np.testing.assert_array_equal(actual, original)
        assert obj.last_result["selected_index"] == 3
        assert obj.last_result["candidate_count"] == 4
        assert "selected_state_rank" not in obj.last_result


def test_global_cosine_matches_brute_force_and_stable_ties():
    rng = np.random.default_rng(87)
    vectors = rng.normal(size=(113, 16)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    query = vectors[112]
    selected, scores = cosine_top1(query, vectors)
    expected = max(range(len(vectors)), key=lambda i: float(np.dot(vectors[i], query)))
    assert selected == expected == 112
    assert cosine_top1(query, np.stack([query, query]))[0] == 0
    with pytest.raises(RetrievalError):
        cosine_top1(query, np.full_like(vectors, np.nan))


def test_full_config_requires_model_and_top1():
    with pytest.raises(ValueError, match="candidate_k"):
        RetrievalConfig(strategy="qwen_full", candidate_k=32).validate()
    with pytest.raises(ValueError, match="Missing model_path"):
        RetrievalConfig(strategy="qwen_full", candidate_k=1).validate()


def test_full_worker_failure_is_fatal():
    obj = pool(QwenFullRetrieval)

    def fail(images):
        raise RuntimeError("worker stopped")

    obj.client = SimpleNamespace(encode=fail)
    with pytest.raises(RetrievalError, match="worker stopped"):
        obj.get_retrieved_data(primary_image=np.zeros((8, 8, 3), np.uint8))


def test_full_trace_summary_has_no_state_rank(tmp_path):
    from cosmos_policy.scripts.retrieval.summarize_ablation import summarize

    out = tmp_path / "full"
    out.mkdir()
    (out / "manifest.json").write_text(json.dumps({"command": [], "seed": 42, "status": "complete"}))
    (out / "console.log").write_text("")
    (out / "episodes.jsonl").write_text(json.dumps({
        "seed": 42, "success": False, "terminal_coverage": 0.1}))
    (out / "retrieval_trace.jsonl").write_text(json.dumps({
        "strategy": "qwen_full", "total_retrieval_seconds": 0.1}))
    result = summarize(tmp_path)[0]
    assert result["retrieval_queries"] == 1
    assert result["changed_top1_fraction"] is None
