"""Video-window retrieval and video/state late fusion."""
import json,time
from pathlib import Path
import numpy as np
from .video_pool import VideoRetrievalPool
from .config import load_config, RetrievalError
from .index import pool_manifest, sha256, validate_vectors
from .qwen_client import QwenClient
from .qwen_video_encoder import implementation_hash
class QwenVideoRetrieval(VideoRetrievalPool):
    def __init__(self,*args,retrieval_config,**kwargs):
        self.qwen_cfg=load_config(retrieval_config)
        if self.qwen_cfg.strategy not in ("qwen_video", "qwen_video_late_fusion", "qwen_video_agent_state"): raise RetrievalError("Requires video strategy")
        super().__init__(*args,**kwargs); root=Path(self.qwen_cfg.index_path); m=json.loads((root/"manifest.json").read_text())
        pm=pool_manifest(self)
        pm.update(self.window_metadata())
        for k in ("files", "candidate_ids", "window", "stride", "window_length", "window_type"):
            if m.get(k) != pm[k]: raise RetrievalError(f"Pool mismatch: {k}")
        if m.get("stride") != 1: raise RetrievalError("Video index must use stride=1")
        sig_cfg=self.qwen_cfg
        if self.qwen_cfg.strategy == "qwen_video_late_fusion":
            from dataclasses import replace
            sig_cfg=replace(self.qwen_cfg, strategy="qwen_video")
        if m.get("encoder")!=sig_cfg.encoder_signature() or m.get("implementation_hash")!=implementation_hash(): raise RetrievalError("Video index encoder mismatch")
        if m.get("embeddings_sha256")!=sha256(root/"embeddings.npy"): raise RetrievalError("Video index checksum mismatch")
        self.embeddings=np.load(root/"embeddings.npy",mmap_mode="r",allow_pickle=False); validate_vectors(self.embeddings,len(self._subframes),self.qwen_cfg.embedding_dim)
        self.client=QwenClient(self.qwen_cfg)
        if any(self.client.metadata[k] != m["worker"][k] for k in ("implementation_hash", "model_hashes", "versions")): raise RetrievalError("Worker/index mismatch")
    def get_retrieved_data(self,agent_pos=None,block_pos=None,block_angle=None,*,primary_image,primary_images=None,**kwargs):
        started=time.perf_counter(); frames=list(primary_images or [primary_image])[-8:]
        if not frames or not np.array_equal(frames[-1],primary_image): raise RetrievalError("Video query must end at current frame")
        vals,meta=self.client.encode_video([frames]); scores=self.embeddings@vals[0]; selected=int(np.argmax(scores)); result=self.get_candidate_data(selected)
        self.last_result={"strategy":"qwen_video","selected_id":self.candidate_id(selected),"selected_index":selected,"search_scope":"full_pool","candidate_count":len(self._subframes),"qwen_score":float(scores[selected]),"video_frames":len(frames),"window_type":"partial_window" if len(frames)<8 else "full_window","window_start_offset":-len(frames)+1,"window_end_offset":0,"retrieval_seconds":time.perf_counter()-started,**meta}; return result
    def close(self): self.client.close()

class QwenVideoAgentStateRetrieval(QwenVideoRetrieval):
    """Dense video-window + causal agent position/velocity joint embedding."""
    def __init__(self,*args,retrieval_config,**kwargs):
        super().__init__(*args,retrieval_config=retrieval_config,**kwargs)
        if self.qwen_cfg.strategy != "qwen_video_agent_state":
            raise RetrievalError("Requires strategy=qwen_video_agent_state")

    @staticmethod
    def _agent_state_text(state_history):
        states = np.asarray(state_history, dtype=np.float32)
        if states.ndim != 2 or states.shape[1] != 5 or not len(states) or not np.isfinite(states).all():
            raise RetrievalError("Expected finite causal state history (T,5)")
        pos = states[-1, :2] / 512.0
        vel = np.diff(states[-3:, :2] / 512.0, axis=0).mean(axis=0) if len(states) > 1 else np.zeros(2, np.float32)
        vals = np.concatenate([pos, vel])
        fmt=lambda a: "(" + ", ".join(f"{float(x):.6f}" for x in a) + ")"
        return "PushT agent state. Agent position xy=" + fmt(pos) + "; agent velocity xy=" + fmt(vel) + ". Positions and velocities are normalized by 512."

    def get_retrieved_data(self,agent_pos=None,block_pos=None,block_angle=None,block_pos_history=None,agent_pos_history=None,*,primary_image,primary_images=None,state_history=None,**kwargs):
        started=time.perf_counter(); frames=list(primary_images or [primary_image])[-8:]
        if not frames or not np.array_equal(frames[-1],primary_image): raise RetrievalError("Video query must end at current frame")
        if state_history is None: raise RetrievalError("Agent-state joint encoding requires state history")
        text=self._agent_state_text(state_history)
        vals,meta=self.client.encode_video([frames], [text]); scores=self.embeddings@vals[0]; selected=int(np.argmax(scores)); result=self.get_candidate_data(selected)
        self.last_result={"strategy":"qwen_video_agent_state","selected_id":self.candidate_id(selected),"selected_index":selected,"search_scope":"full_pool","candidate_count":len(self._subframes),"qwen_score":float(scores[selected]),"state_features":"agent_pos_xy+agent_vel_xy","state_text":text,"video_frames":len(frames),"window_type":"partial_window" if len(frames)<8 else "full_window","window_start_offset":-len(frames)+1,"window_end_offset":0,"retrieval_seconds":time.perf_counter()-started,**meta}; return result

class QwenVideoLateFusionRetrieval(QwenVideoRetrieval):
    def __init__(self,*args,retrieval_config,**kwargs):
        super().__init__(*args,retrieval_config=retrieval_config,**kwargs)
        if self.qwen_cfg.strategy != "qwen_video_late_fusion": raise RetrievalError("Requires strategy=qwen_video_late_fusion")
    def get_retrieved_data(self,agent_pos=None,block_pos=None,block_angle=None,block_pos_history=None,agent_pos_history=None,*,primary_image,primary_images=None,state_history=None,**kwargs):
        started=time.perf_counter(); frames=list(primary_images or [primary_image])[-8:]
        if not frames or not np.array_equal(frames[-1],primary_image): raise RetrievalError("Video query must end at current frame")
        vals,meta=self.client.encode_video([frames]); visual_raw=self.embeddings@vals[0]
        query_state=self._feat_state(state_history)
        distances=((self._feat-query_state)**2).sum(axis=1)
        visual = visual_raw
        # Fixed RBF similarity in the original normalized, weighted 10D space.
        # No candidate-pool-dependent rescaling; d^2=1 maps to exp(-1).
        state = np.exp(-distances.astype(np.float64))
        final = self.qwen_cfg.alpha * visual + self.qwen_cfg.beta * state
        selected = int(np.argmin(distances)) if self.qwen_cfg.alpha == 0 else int(np.argmax(final))
        result = self.get_candidate_data(selected)
        self.last_result={"strategy":"qwen_video_late_fusion","selected_id":self.candidate_id(selected),"selected_index":selected,"search_scope":"full_pool","candidate_count":len(self._subframes),"video_cosine":float(visual_raw[selected]),"video_score":float(visual[selected]),"score_version":"raw_cosine_rbf_state_v2","state_distance":float(distances[selected]),"state_score":float(state[selected]),"final_score":float(final[selected]),"alpha":self.qwen_cfg.alpha,"beta":self.qwen_cfg.beta,"video_frames":len(frames),"window_type":"partial_window" if len(frames)<8 else "full_window","retrieval_seconds":time.perf_counter()-started,**meta}
        if self.qwen_cfg.alpha == 0:
            original_ids, original_distances = self.get_state_candidates(
                agent_pos, block_pos, block_angle, block_pos_history, agent_pos_history)
            nearest = int(np.argmin(distances))
            original = int(original_ids[0])
            self.last_result.update(
                diagnostic_baseline_selected_id=self.candidate_id(original),
                diagnostic_baseline_selected_index=original,
                diagnostic_baseline_distance=float(original_distances[0]),
                diagnostic_full_pool_argmin=nearest,
                diagnostic_full_pool_distance=float(distances[nearest]),
                diagnostic_same_as_baseline=(selected == original),
                diagnostic_minmax_changed_argmin=(selected != nearest),
                diagnostic_selected_distance=float(distances[selected]))
        return result
    def _feat_state(self,state_history):
        from .state_features import state_feature, FEATURE_WEIGHTS
        return state_feature(state_history)*FEATURE_WEIGHTS

