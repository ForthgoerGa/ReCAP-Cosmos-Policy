import ast
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cosmos_policy/experiments/robot/pusht"))
from gym_pusht.envs.pusht import PushTEnv
from cosmos_policy.experiments.robot.pusht_ret.retrievers.config import RetrievalError
from cosmos_policy.experiments.robot.pusht_ret.retrievers.query_visual import retrieval_query_frame


def make_env(size=224):
    return PushTEnv(obs_type="pixels_agent_pos", render_mode="rgb_array",
                    observation_width=size, observation_height=size)


@pytest.mark.parametrize("size", [96, 224, 512])
@pytest.mark.parametrize("position", [(200, 400), (256, 300), (5, 5)])
def test_query_matches_demo_renderer_without_mutating_scene(size, position):
    triangle, circle = make_env(size), make_env(size)
    try:
        triangle.reset(seed=42, options={"agent_shape": "triangle"})
        circle.reset(seed=42, options={})
        for env in (triangle, circle):
            env.agent.position = position
            env.block.position = (256, 300)
            env.block.angle = 0.3
            env.space.reindex_shapes_for_body(env.agent)
            env.space.reindex_shapes_for_body(env.block)
        raw = triangle.get_obs()["pixels"]
        original = raw.copy()
        shapes = tuple(triangle.agent.shapes)
        query = retrieval_query_frame(triangle, raw, "blue_circle")
        assert not np.shares_memory(raw, query)
        assert np.array_equal(raw, original)
        assert np.array_equal(triangle.get_obs()["pixels"], original)
        assert tuple(triangle.agent.shapes) == shapes
        assert tuple(triangle.agent.position) == position
        assert np.array_equal(query, circle.get_obs()["pixels"])
    finally:
        triangle.close()
        circle.close()


def test_off_is_identity_and_circle_is_idempotent():
    env = make_env()
    try:
        raw = env.reset(seed=42)[0]["pixels"]
        assert retrieval_query_frame(env, raw, "none") is raw
        assert np.array_equal(retrieval_query_frame(env, raw, "blue_circle"), raw)
        with pytest.raises(ValueError, match="Unknown"):
            retrieval_query_frame(env, raw, "typo")
    finally:
        env.close()


def test_rendering_does_not_change_trajectory():
    env, reference = make_env(), make_env()
    try:
        for e in (env, reference):
            e.reset(seed=42, options={"agent_shape": "triangle"})
        for action in [(256, 300), (300, 280), (320, 300), (200, 300)]:
            retrieval_query_frame(env, env.get_obs()["pixels"], "blue_circle")
            a, b = env.step(action), reference.step(action)
            assert np.array_equal(a[0]["pixels"], b[0]["pixels"])
            assert np.array_equal(a[0]["agent_pos"], b[0]["agent_pos"])
            assert a[1:4] == b[1:4]
    finally:
        env.close()
        reference.close()


@pytest.mark.parametrize("strategy", ["qwen_full", "qwen_video", "qwen_video_agent_state", "wan_vae_image", "wan_vae_video"])
@pytest.mark.parametrize("mode", ["none", "blue_circle"])
def test_episode_routes_only_query_pixels_and_history(strategy, mode, tmp_path):
    # Execute the real episode function with a recording policy, without loading a GPU model.
    tree = ast.parse((ROOT / "cosmos_policy/experiments/robot/pusht_ret/run_eval.py").read_text())
    func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_episode")
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), func], type_ignores=[])
    env = make_env()
    policies, queries, histories, expected_history = [], [], [], []
    ret_frames = np.zeros((8, 224, 224, 3), np.uint8)
    ret_actions = np.full((8, 2), 256, np.float32)
    ret_proprio = np.zeros((8, 2), np.float32)

    class Retriever:
        WINDOW_SIZE = 8
        last_result = {}

        def get_retrieved_data(self, **kwargs):
            queries.append(kwargs["primary_image"].copy())
            histories.append(kwargs.get("primary_images"))
            return ret_frames, ret_actions, ret_proprio

    def policy(cfg, model, stats, obs, task, **kwargs):
        raw = env.get_obs()["pixels"]
        assert np.array_equal(obs["primary_image"], raw)
        assert obs["retrieved_frames"] is ret_frames
        assert obs["retrieved_actions"] is ret_actions
        assert obs["retrieved_proprio"] is ret_proprio
        policies.append(raw.copy())
        expected = retrieval_query_frame(env, raw, mode)
        assert np.array_equal(queries[-1], expected)
        expected_history.append(expected)
        if strategy not in ("qwen_full", "wan_vae_image"):
            assert len(histories[-1]) == len(expected_history)
            assert all(np.array_equal(a, b) for a, b in zip(histories[-1], expected_history))
        return {"actions": [np.array([256, 300], np.float32)]}

    cfg = SimpleNamespace(seed=42, retrieval_query_visual=mode, retrieval_strategy=strategy,
                          retrieval_trace=True, use_residual_actions=False, randomize_seed=False,
                          num_denoising_steps_action=5, predict_future_states=True,
                          num_open_loop_steps=1, local_log_dir=str(tmp_path), fail_on_episode_error=True)
    scope = dict(np=np, deque=deque, time=time, os=os, hashlib=hashlib, json=json,
                 MAX_STEPS=3, SUCCESS_THRESHOLD=2, RetrievalError=RetrievalError,
                 retrieval_query_frame=retrieval_query_frame, TASK_DESCRIPTION="test",
                 prepare_observation=lambda obs: {"primary_image": obs["pixels"], "proprio": obs["agent_pos"]},
                 get_action=policy, log_message=lambda *args: None)
    exec(compile(ast.fix_missing_locations(module), "run_episode", "exec"), scope)
    try:
        result = scope["run_episode"](cfg, env, {"agent_shape": "triangle"}, None, None, {}, 0, Retriever())
        assert len(policies) == 3
        assert all(np.array_equal(a, b) for a, b in zip(policies, result[2]))
        records = [json.loads(line) for line in (tmp_path / "retrieval_trace.jsonl").read_text().splitlines()]
        assert len(records) == 3
        for row in records:
            assert row["retrieval_query_visual"] == mode
            assert (row["policy_frame_sha256"] == row["query_frame_sha256"]) == (mode == "none")
    finally:
        env.close()
