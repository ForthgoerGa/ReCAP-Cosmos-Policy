"""Wan visual latents plus a calibrated independent agent-state latent slot."""
import json
from pathlib import Path
import time
import numpy as np
import torch
from .config import RetrievalError
from .wan_vae import WanVAERetrieval
from .wan_vae_encoder import full_precision_search
from .wan_agent_state import load_sidecar, query_state, normalize_states, injection_distances

class WanAgentInjectionRetrieval(WanVAERetrieval):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.video or self.vae_cfg.agent_state_weight is None:
            raise RetrievalError("Agent latent injection requires configured Wan video")
        visual = json.loads((Path(self.vae_cfg.index_path) / "manifest.json").read_text())
        raw, states, self.velocity_scale, self.state_manifest = load_sidecar(
            self.vae_cfg.agent_state_index_path, visual)
        self.raw_agent_states = raw
        self.agent_states = torch.from_numpy(states).to(self.embeddings.device)
        self.agent_state_weight = self.vae_cfg.agent_state_weight
        self.visual_cosine_gap = self.vae_cfg.visual_cosine_gap

    def get_retrieved_data(self, agent_pos=None, block_pos=None, block_angle=None,
                           block_pos_history=None, agent_pos_history=None, *,
                           primary_image, primary_images=None, **kwargs):
        started = time.perf_counter()
        frames = list(primary_images)[-8:] if primary_images is not None else [primary_image]
        if not frames or not np.array_equal(frames[-1], primary_image):
            raise RetrievalError("Video query must end at the current image")
        raw = query_state(agent_pos, agent_pos_history)
        normalized, clipped = normalize_states(raw, self.velocity_scale)
        current_state = torch.from_numpy(normalized).to(self.embeddings.device)
        query = self.encoder.encode([frames])
        search_started = time.perf_counter()
        with torch.inference_mode(), full_precision_search():
            cosines = self.embeddings @ query[0]
            selected, details = injection_distances(cosines, self.agent_states, current_state,
                                                   self.agent_state_weight, self.visual_cosine_gap)
            original = int(torch.argmax(cosines).item())
            gap = float((cosines[original] - cosines[selected]).item())
            if self.visual_cosine_gap is not None and gap > self.visual_cosine_gap + 1e-6:
                raise RetrievalError("Visual dominance guard violated")
            score = float(cosines[selected].item())
            rank = int((cosines > cosines[selected]).sum().item()) + 1
            scalar = {key: float(details[key][selected].item())
                      for key in ("visual", "state", "position", "velocity", "distance")}
            eligible_count = int(details["eligible"].sum().item())
        search_seconds = time.perf_counter() - search_started
        result = self.get_candidate_data(selected)
        self.last_result = dict(
            strategy="wan_vae_video", retrieval_variant="agent_latent_injection_v2",
            selected_id=self.candidate_id(selected), selected_index=selected, selected_rank=1,
            cosine_score=score, search_scope="full_pool" if self.visual_cosine_gap is None else "full_pool_visual_guard",
            candidate_count=len(self._subframes), history_frames=len(frames),
            history_padding_frames=8-len(frames), search_seconds=search_seconds,
            retrieval_seconds=time.perf_counter()-started,
            index_gpu_bytes=self.embeddings.numel()*4, state_gpu_bytes=self.agent_states.numel()*4,
            visual_top1_id=self.candidate_id(original), visual_top1_index=original,
            visual_top1_cosine=float(cosines[original].item()), visual_rank=rank,
            visual_cosine_gap=gap, visual_guard=self.visual_cosine_gap, eligible_count=eligible_count,
            selection_changed=selected != original, agent_state_weight=self.agent_state_weight,
            state_raw=raw.tolist(), state_normalized=normalized.tolist(), state_clipped=clipped.tolist(),
            candidate_state_raw=self.raw_agent_states[selected].tolist(),
            candidate_state_normalized=self.agent_states[selected].cpu().tolist(),
            velocity_scale=self.velocity_scale.tolist(), visual_distance=scalar["visual"],
            state_distance=scalar["state"], position_distance=scalar["position"],
            velocity_distance=scalar["velocity"], injection_distance=scalar["distance"],
            weighted_visual_distance=(1-self.agent_state_weight)*scalar["visual"],
            weighted_state_distance=self.agent_state_weight*scalar["state"],
            state_sidecar_sha256=self.state_manifest["states_sha256"], **self.encoder.last_metadata)
        return result

    def close(self):
        super().close()
        self.agent_states = None
        self.raw_agent_states = None
