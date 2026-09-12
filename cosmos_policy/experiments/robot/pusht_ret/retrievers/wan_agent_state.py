"""Causal four-dimensional agent state and block-scaled latent injection."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .config import RetrievalError
from .index import sha256

STATE_SPEC = dict(version=1, fields=["px", "py", "vx", "vy"],
                  velocity="mean_last_up_to_two_real_displacements_pixel_per_observation_step",
                  first_velocity="zero", position_scale=512.0,
                  velocity_percentile=99, percentile_method="linear",
                  velocity_scale_floor=1.0, latent_shape=[16, 1, 28, 28])

def implementation_hash():
    return sha256(Path(__file__))

def causal_states(positions):
    p = np.asarray(positions, dtype=np.float32)
    if p.ndim != 2 or p.shape[1] != 2 or not len(p) or not np.isfinite(p).all():
        raise RetrievalError("Expected nonempty finite agent positions (T,2)")
    velocity = np.zeros_like(p)
    if len(p) > 1:
        velocity[1:] = np.diff(p, axis=0)
    if len(p) > 2:
        velocity[2:] = (p[2:] - p[:-2]) / 2
    return np.concatenate([p, velocity], axis=1)

def query_state(position, history):
    p = np.asarray(position, dtype=np.float32)
    if p.shape != (2,) or not np.isfinite(p).all():
        raise RetrievalError("Expected finite current agent position")
    h = p[None] if history is None else np.asarray(history, dtype=np.float32)
    states = causal_states(h)
    if not np.array_equal(h[-1], p):
        raise RetrievalError("Agent history must end at the current position")
    return states[-1]

def normalize_states(raw, velocity_scale):
    raw = np.asarray(raw, dtype=np.float32)
    scale = np.asarray(velocity_scale, dtype=np.float32)
    if raw.shape[-1:] != (4,) or not np.isfinite(raw).all():
        raise RetrievalError("Expected finite agent state (...,4)")
    if scale.shape != (2,) or not np.isfinite(scale).all() or np.any(scale < 1):
        raise RetrievalError("Invalid frozen velocity scale")
    unbounded = np.concatenate([raw[..., :2] * (2 / 512.) - 1,
                                raw[..., 2:] / scale], axis=-1)
    return np.clip(unbounded, -1, 1).astype(np.float32), np.abs(unbounded) > 1

def validate_sidecar(manifest, visual_manifest):
    if manifest.get("spec") != STATE_SPEC or manifest.get("implementation_hash") != implementation_hash():
        raise RetrievalError("Agent sidecar implementation/spec mismatch")
    for key in ("candidate_ids", "files", "embeddings_sha256"):
        if manifest.get("visual_index", {}).get(key) != visual_manifest.get(key):
            raise RetrievalError(f"Agent sidecar visual identity mismatch: {key}")

def load_sidecar(root, visual_manifest):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    validate_sidecar(manifest, visual_manifest)
    if sha256(root / "states.npz") != manifest["states_sha256"]:
        raise RetrievalError("Agent sidecar checksum mismatch")
    with np.load(root / "states.npz", allow_pickle=False) as a:
        raw, states, scale = a["raw"].copy(), a["normalized"].copy(), a["velocity_scale"].copy()
    count = len(visual_manifest["candidate_ids"])
    if raw.shape != (count, 4) or states.shape != (count, 4):
        raise RetrievalError("Agent sidecar candidate count/shape mismatch")
    expected, clipped = normalize_states(raw, scale)
    if not np.array_equal(expected, states):
        raise RetrievalError("Agent sidecar normalization mismatch")
    return raw, states, scale, manifest

def injection_distances(cosines, candidate_states, current_state, weight, gap):
    """Exactly squared L2 of [sqrt(1-w)*z/2, sqrt(w)*tile(u)/(2*sqrt(M))]."""
    if not np.isfinite(weight) or not 0 <= weight <= 1 or (gap is not None and
            (not np.isfinite(gap) or not 0 <= gap <= 2)):
        raise RetrievalError("Invalid injection weight or cosine guard")
    if cosines.ndim != 1 or not len(cosines) or candidate_states.shape != (len(cosines), 4) or current_state.shape != (4,):
        raise RetrievalError("Invalid injection tensor shapes")
    if not all(torch.isfinite(a).all() for a in (cosines, candidate_states, current_state)):
        raise RetrievalError("Nonfinite injection input")
    if (candidate_states.abs() > 1).any() or (current_state.abs() > 1).any():
        raise RetrievalError("State outside calibrated [-1,1] range")
    visual = (1 - cosines.clamp(-1, 1)) / 2
    differences = (candidate_states - current_state).square() / 4
    position, velocity = differences[:, :2].mean(1), differences[:, 2:].mean(1)
    state = (position + velocity) / 2
    eligible = torch.ones_like(cosines, dtype=torch.bool) if gap is None else cosines >= cosines.max() - gap
    distances = (1 - weight) * visual + weight * state
    # Preserve original FP32 argmax and first-index tie behavior at weight zero.
    selected = torch.argmax(cosines) if weight == 0 else torch.argmin(distances.masked_fill(~eligible, float("inf")))
    return int(selected.item()), dict(visual=visual, state=state, position=position,
                                     velocity=velocity, distance=distances, eligible=eligible)
