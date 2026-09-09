"""Causal, identical demo/query state features and canonical text formatting."""
import hashlib
from pathlib import Path

import numpy as np

STATE_FORMAT = "pusht_normalized_state10_v1"
FEATURE_WEIGHTS = np.array([2, 2, 2.5, 2.5, 1.5, 1.5, 1, 1, 1, 1], np.float32)


def state_feature(states):
    """Rows are [agent_x, agent_y, block_x, block_y, block_yaw_radians]."""
    states = np.asarray(states, dtype=np.float32)
    if states.ndim != 2 or states.shape[1] != 5 or not len(states) or not np.isfinite(states).all():
        raise ValueError("Expected a nonempty finite (T, 5) causal state history")
    current = states[-1]
    velocity = np.diff(states[-3:, :4] / 512.0, axis=0).mean(axis=0) if len(states) > 1 else np.zeros(4)
    return np.concatenate([current[2:4] / 512.0, current[:2] / 512.0,
                           [np.sin(current[4]), np.cos(current[4])],
                           velocity[2:4], velocity[:2]]).astype(np.float32)


def state_text(feature):
    feature = np.asarray(feature, dtype=np.float32)
    if feature.shape != (10,) or not np.isfinite(feature).all():
        raise ValueError("Expected finite state feature of shape (10,)")
    def pair(start):
        return "(" + ", ".join(f"{0.0 if abs(float(x)) < 0.5e-6 else float(x):.6f}" for x in feature[start:start + 2]) + ")"
    return (f"PushT state. Agent position xy={pair(2)}; block position xy={pair(0)}; "
            f"block orientation sin_cos={pair(4)}; agent velocity xy={pair(8)}; "
            f"block velocity xy={pair(6)}. Positions are simulator coordinates divided by 512; "
            "velocities are mean displacement over the last two available steps divided by 512.")


def history_embedding(vectors, power=1.0):
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or not len(vectors) or not np.isfinite(vectors).all():
        raise ValueError("Expected a nonempty matrix of historical vectors")
    weights = np.arange(1, len(vectors) + 1, dtype=np.float32) ** power
    pooled = (vectors * (weights / weights.sum())[:, None]).sum(axis=0)
    norm = np.linalg.norm(pooled)
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError("Degenerate pooled embedding")
    return (pooled / norm).astype(np.float32)


def representation_signature(cfg):
    return {"mode": cfg.strategy, "state_format": STATE_FORMAT,
            "history_length": cfg.history_length if cfg.strategy == "qwen_history_state" else 1,
            "history_weight_power": cfg.history_weight_power if cfg.strategy == "qwen_history_state" else None,
            "history_state": "per_frame_causal", "early_history": "available_only",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
