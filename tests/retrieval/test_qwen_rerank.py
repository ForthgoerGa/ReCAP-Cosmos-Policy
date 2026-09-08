"""Meaningful CPU invariants; real-model smoke testing is a separate step."""
import json
import subprocess
import types

import numpy as np
import pytest

from cosmos_policy.experiments.robot.pusht_ret.retrieval import PushTRetrieval
from cosmos_policy.experiments.robot.pusht_ret.retrievers.config import RetrievalConfig
from cosmos_policy.experiments.robot.pusht_ret.retrievers.index import validate_vectors
from cosmos_policy.experiments.robot.pusht_ret.retrievers.qwen_rerank import cosine_order


def pool(cls=PushTRetrieval):
    obj = cls.__new__(cls)
    obj.chunk_size = 8
    obj.ret_context_multiplier = obj.ret_image_subsample = 1
    obj.block_rel = False
    obj._demo_keys = [("base_0", "demo_0"), ("base_0", "demo_1")]
    obj._demo_init_block_pos = np.array([[0.1, 0.1], [0.2, 0.2]])
    obj._demo_indices = {obj._demo_keys[0]: [0, 1], obj._demo_keys[1]: [2, 3]}
    obj._feat = np.zeros((4, 10), dtype=np.float32)
    obj._feat[1] = 0.5
    obj._feat[3] = 2
    obj._subframes = [
        {"split": "base_0", "demo": f"demo_{i // 2}", "t_last": 2 + i}
        for i in range(4)]
    obj._source_files = {k: "base_0/demo.hdf5" for k in obj._demo_keys}
    obj._base_data = {}
    for di, key in enumerate(obj._demo_keys):
        obj._base_data[key] = {
            "images": np.stack([np.full((8, 8, 3), i + di * 10, np.uint8) for i in range(7)]),
            "actions": np.arange(14, dtype=np.float32).reshape(7, 2) + di * 100,
            "proprio": np.arange(14, dtype=np.float32).reshape(7, 2) + di * 200}
    return obj


def test_original_top1_and_payload():
    # Compare to actual original source, not another copy of the new algorithm.
    source = subprocess.check_output([
        "git", "show", "f8ec232:cosmos_policy/experiments/robot/pusht_ret/retrieval.py"], text=True)
    module = types.ModuleType("original_retrieval")
    exec(compile(source, "original_retrieval.py", "exec"), module.__dict__)
    old, new = pool(module.PushTRetrieval), pool()
    rng = np.random.default_rng(42)
    for _ in range(20):
        query = dict(agent_pos=rng.uniform(0, 512, 2).astype(np.float32),
                     block_pos=rng.uniform(0, 512, 2).astype(np.float32),
                     block_angle=float(rng.uniform(-3, 3)),
                     block_pos_history=rng.uniform(0, 512, (3, 2)).astype(np.float32),
                     agent_pos_history=rng.uniform(0, 512, (3, 2)).astype(np.float32))
        for a, b in zip(old.get_retrieved_data(**query), new.get_retrieved_data(**query)):
            np.testing.assert_array_equal(a, b)
        ids, _ = new.get_state_candidates(**query, k=99)
        assert len(ids) == 4
        for a, b in zip(new.get_candidate_data(ids[0]), new.get_retrieved_data(**query)):
            np.testing.assert_array_equal(a, b)


def test_materialization_alignment_and_tail_padding():
    obj = pool()
    frames, acts, prop = obj.get_candidate_data(3)
    np.testing.assert_array_equal(frames[:, 0, 0, 0], [15, 16, 16, 16, 16, 16, 16, 16])
    np.testing.assert_array_equal(acts[:, 0], [110, 112, 112, 112, 112, 112, 112, 112])
    np.testing.assert_array_equal(prop, [210, 211])
    assert obj.candidate_id(3) == "base_0/demo.hdf5::demo_1::5"


def test_cosine_changes_selection_and_stable_ties():
    candidates = np.array([[0., 1.], [1., 0.], [1., 0.]], np.float32)
    order, scores = cosine_order(np.array([1., 0.], np.float32), candidates)
    np.testing.assert_array_equal(order, [1, 2, 0])
    one, _ = cosine_order(np.array([1., 0.], np.float32), candidates[:1])
    assert one.tolist() == [0]


def test_invalid_vectors_and_config_fail():
    with pytest.raises(ValueError):
        validate_vectors(np.zeros((2, 3)), 2, 3)
    with pytest.raises(ValueError):
        validate_vectors(np.full((2, 3), np.nan), 2, 3)
    with pytest.raises(ValueError):
        RetrievalConfig(candidate_k=0).validate()
    with pytest.raises(ValueError):
        RetrievalConfig().validate()


def test_missing_visual_input_is_fatal():
    from cosmos_policy.experiments.robot.pusht_ret.retrievers.qwen_rerank import QwenRerankRetrieval
    from cosmos_policy.experiments.robot.pusht_ret.retrievers.config import RetrievalError
    obj = pool(QwenRerankRetrieval)
    obj.qwen_cfg = RetrievalConfig()
    class BadClient:
        def encode(self, frames):
            raise ValueError("missing pixels")
    obj.client = BadClient()
    with pytest.raises(RetrievalError, match="missing pixels"):
        obj.get_retrieved_data(np.zeros(2), np.zeros(2), 0, primary_image=np.zeros((8, 8, 3), np.uint8))


def test_summary_handles_duplicate_console_logger_lines(tmp_path):
    from cosmos_policy.scripts.retrieval.summarize_ablation import summarize
    out = tmp_path / "standard"
    out.mkdir()
    (out / "manifest.json").write_text(json.dumps({"command": [], "seed": 42, "status": "complete"}))
    (out / "console.log").write_text(
        "  t=0 coverage=0.850\nINFO:logger:  t=0 coverage=0.850\n"
        "Episode 1: SUCCESS\nINFO:logger:Episode 1: SUCCESS\n")
    result = summarize(tmp_path)[0]
    assert result["episodes"] == [{"seed": 42, "success": True, "terminal_coverage": 0.85, "steps": 1}]


def test_worker_crash_propagates_without_fallback(tmp_path):
    from cosmos_policy.experiments.robot.pusht_ret.retrievers.qwen_client import QwenClient
    from cosmos_policy.experiments.robot.pusht_ret.retrievers.config import RetrievalError
    executable = tmp_path / "dead_worker"
    executable.write_text("#!/bin/sh\necho simulated_worker_failure >&2\nexit 1\n")
    executable.chmod(0o700)
    cfg = RetrievalConfig(worker_python=str(executable), timeout_seconds=5)
    with pytest.raises(RetrievalError):
        QwenClient(cfg)
