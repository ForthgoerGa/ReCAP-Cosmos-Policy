"""Causal state/image alignment, late fusion endpoints, and legacy compatibility."""
from types import SimpleNamespace

import numpy as np
import pytest

from test_qwen_rerank import pool
from cosmos_policy.experiments.robot.pusht_ret.retrieval import PushTRetrieval
from cosmos_policy.experiments.robot.pusht_ret.retrievers.config import RetrievalConfig, RetrievalError
from cosmos_policy.experiments.robot.pusht_ret.retrievers.qwen_late_fusion import (
    QwenLateFusionRetrieval, fuse_scores,
)
from cosmos_policy.experiments.robot.pusht_ret.retrievers.qwen_multimodal import (
    QwenHistoryStateRetrieval, QwenStateTextRetrieval,
)
from cosmos_policy.experiments.robot.pusht_ret.retrievers.state_features import (
    FEATURE_WEIGHTS, history_embedding, representation_signature, state_feature, state_text,
)


def sequence():
    t = np.arange(15, dtype=np.float32)
    return np.stack([100 + t, 200 - t, 256 + 2 * t, 128 + 3 * t, t / 10], axis=1)


def test_causal_features_match_original_pool_and_text():
    states = sequence()
    for t in range(len(states)):
        start = max(0, t - 7)
        original = PushTRetrieval._build_feature(
            states[:, 2:4], states[:, :2], states[:, 4], start, t - start + 1,
            512, 2, 2.5, 1.5, 1, 1)
        online = state_feature(states[max(0, t - 2):t + 1])
        np.testing.assert_allclose(online * FEATURE_WEIGHTS, original, atol=1e-7)
        assert state_text(online) == state_text(state_feature(states[:t + 1]))
    np.testing.assert_array_equal(state_feature(states[:1])[6:], np.zeros(4))


def test_history_weights_and_available_prefix():
    values = np.eye(3, dtype=np.float32)
    expected = np.array([1, 2, 3], np.float32)
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(history_embedding(values), expected, atol=1e-7)
    np.testing.assert_array_equal(history_embedding(values[:1]), values[0])
    assert not np.allclose(history_embedding(values), history_embedding(values[::-1]))


def forbid(*args, **kwargs):
    raise AssertionError("Old state candidate filtering must not run")


@pytest.mark.parametrize("length", [1, 3, 8])
def test_history_online_texts_match_each_causal_demo_frame(length):
    obj = pool(QwenHistoryStateRetrieval)
    obj.qwen_cfg = RetrievalConfig(strategy="qwen_history_state", candidate_k=1)
    obj.embeddings = np.eye(4, dtype=np.float32)
    obj.get_state_candidates = forbid
    states = sequence()[:length + 2] if length == 8 else sequence()[:length]
    images = [np.full((8, 8, 3), i, np.uint8) for i in range(length)]
    calls = []
    def encode(frames, texts):
        calls.append((frames, texts))
        return np.tile([0, 0, 0, 1], (len(frames), 1)).astype(np.float32), {}
    obj.client = SimpleNamespace(encode=encode)
    result = obj.get_retrieved_data(primary_image=images[-1], primary_images=images, state_history=states)
    expected = [state_text(state_feature(states[:end])) for end in range(len(states) - length + 1, len(states) + 1)]
    assert calls[0][1] == expected
    assert obj.last_result["history_frames"] == length
    for a, b in zip(result, obj.get_candidate_data(3)):
        np.testing.assert_array_equal(a, b)


def test_missing_history_is_fatal():
    obj = pool(QwenHistoryStateRetrieval)
    obj.qwen_cfg = RetrievalConfig(strategy="qwen_history_state", candidate_k=1)
    with pytest.raises(RetrievalError, match="requires causal image history"):
        obj.get_retrieved_data(primary_image=np.zeros((8, 8, 3), np.uint8), state_history=sequence())


def test_single_joint_passes_content_text():
    obj = pool(QwenStateTextRetrieval)
    obj.qwen_cfg = RetrievalConfig(strategy="qwen_state_text", candidate_k=1)
    obj.embeddings = np.eye(4, dtype=np.float32)
    obj.get_state_candidates = forbid
    calls = []
    def encode(frames, texts):
        calls.append(texts)
        return np.array([[0, 1, 0, 0]], np.float32), {}
    obj.client = SimpleNamespace(encode=encode)
    obj.get_retrieved_data(primary_image=np.zeros((8, 8, 3), np.uint8), state_history=sequence())
    assert calls == [[state_text(state_feature(sequence()))]]
    assert obj.last_result["selected_index"] == 1


def test_fusion_endpoints_and_distance_direction():
    visual = np.array([0.90, 0.95, 0.99], np.float32)
    distance = np.array([0, 2, 5], np.float32)
    assert np.argmax(fuse_scores(visual, distance, 1, 0)[0]) == 2
    assert np.argmax(fuse_scores(visual, distance, 0, 1)[0]) == 0
    scores, v, s = fuse_scores(visual, distance, 0.5, 0.5)
    np.testing.assert_allclose(scores, (v + s) / 2)
    assert np.isfinite(fuse_scores(np.ones(3), np.zeros(3), 1, 1)[0]).all()
    with pytest.raises(ValueError):
        fuse_scores(np.array([np.nan]), np.zeros(1), 1, 1)


def test_late_fusion_image_branch_never_receives_text():
    obj = pool(QwenLateFusionRetrieval)
    obj.qwen_cfg = RetrievalConfig(strategy="qwen_late_fusion", candidate_k=1, alpha=1, beta=0)
    obj.embeddings = np.eye(4, dtype=np.float32)
    obj.get_state_candidates = forbid
    # A one-argument client would fail if state text entered the image branch.
    obj.client = SimpleNamespace(encode=lambda images: (np.array([[0, 0, 1, 0]], np.float32), {}))
    obj.get_retrieved_data(primary_image=np.zeros((8, 8, 3), np.uint8), state_history=sequence())
    assert obj.last_result["selected_index"] == 2
    assert obj.last_result["candidate_count"] == 4


def test_signatures_preserve_image_indexes_and_separate_joint_history():
    image = RetrievalConfig(strategy="qwen_full", candidate_k=1)
    late = RetrievalConfig(strategy="qwen_late_fusion", candidate_k=1)
    joint = RetrievalConfig(strategy="qwen_state_text", candidate_k=1)
    hist = RetrievalConfig(strategy="qwen_history_state", candidate_k=1)
    assert image.encoder_signature() == late.encoder_signature()
    assert joint.encoder_signature() == hist.encoder_signature() != image.encoder_signature()
    assert representation_signature(joint) != representation_signature(hist)
    hist.history_weight_power = 2
    assert representation_signature(hist)["history_weight_power"] == 2
