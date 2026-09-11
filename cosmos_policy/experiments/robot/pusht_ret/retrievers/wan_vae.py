"""Full-pool Wan VAE cosine retrieval with the original policy payload."""
import json
from pathlib import Path
import time

import numpy as np

from ..retrieval import PushTRetrieval
from .config import RetrievalError, load_config
from .index import pool_manifest, sha256, validate_vectors
from .video_pool import VideoRetrievalPool
from .wan_vae_encoder import WanVAEEncoder, full_precision_search, implementation_hash


def index_identity(pool, cfg):
    identity = pool_manifest(pool)
    if cfg.strategy == "wan_vae_video":
        identity.update(pool.window_metadata(), early_padding="repeat_first_left", encoded_history_frames=8)
    return dict(identity, encoder=cfg.encoder_signature(), implementation_hash=implementation_hash())


def candidate_clip(pool, index, video):
    sf = pool._subframes[index]
    images = pool._base_data[(sf["split"], sf["demo"])]["images"]
    return list(images[sf["start"]:sf["t_last"] + 1]) if video else [images[sf["t_last"]]]


class WanVAERetrieval(PushTRetrieval):
    def __init__(self, *args, retrieval_config, **kwargs):
        import torch
        self.vae_cfg = load_config(retrieval_config)
        if self.vae_cfg.strategy not in ("wan_vae_image", "wan_vae_video"):
            raise RetrievalError("Requires a Wan VAE strategy")
        if kwargs.get("chunk_size", 8) != 8 or kwargs.get("ret_context_multiplier", 1) != 1 or kwargs.get("ret_image_subsample", 1) != 1:
            raise RetrievalError("Wan retrieval requires original 8-step payloads")
        self.video = self.vae_cfg.strategy == "wan_vae_video"
        if self.video:
            self.STRIDE = 1
        super().__init__(*args, **kwargs)
        root = Path(self.vae_cfg.index_path)
        manifest = json.loads((root / "manifest.json").read_text())
        for key, expected in index_identity(self, self.vae_cfg).items():
            if manifest.get(key) != expected:
                raise RetrievalError(f"Wan index identity mismatch: {key}")
        if sha256(root / "embeddings.npy") != manifest["embeddings_sha256"]:
            raise RetrievalError("Wan index checksum mismatch")
        vectors = np.load(root / "embeddings.npy", mmap_mode="r", allow_pickle=False)
        for start in range(0, len(vectors), 256):
            part = vectors[start:start+256]
            validate_vectors(part, len(part), self.vae_cfg.embedding_dim)
        if len(vectors) != len(self._subframes):
            raise RetrievalError("Wan index candidate count mismatch")
        self.encoder = WanVAEEncoder(self.vae_cfg)
        if self.encoder.metadata != manifest["worker"]:
            raise RetrievalError("Wan index and online encoder metadata differ")
        # Keep FP32 cosine on GPU to avoid streaming gigabytes from host per query.
        self.embeddings = torch.empty(vectors.shape, dtype=torch.float32, device="cuda")
        for start in range(0, len(vectors), 256):
            self.embeddings[start:start+256].copy_(torch.from_numpy(vectors[start:start+256].copy()))

    window_metadata = VideoRetrievalPool.window_metadata

    def get_retrieved_data(self, agent_pos=None, block_pos=None, block_angle=None,
                           block_pos_history=None, agent_pos_history=None, *, primary_image,
                           primary_images=None, **kwargs):
        import torch
        started = time.perf_counter()
        try:
            frames = (list(primary_images)[-8:] if primary_images is not None else [primary_image]) if self.video else [primary_image]
            if not frames or not np.array_equal(frames[-1], primary_image):
                raise RetrievalError("Video query must end at current frame")
            query = self.encoder.encode([frames])
            search_start = time.perf_counter()
            with torch.inference_mode(), full_precision_search():
                scores = self.embeddings @ query[0]
                if not torch.isfinite(scores).all():
                    raise RetrievalError("Nonfinite Wan cosine scores")
                selected = int(torch.argmax(scores).item())
                score = float(scores[selected].item())
            search_seconds = time.perf_counter() - search_start
            result = self.get_candidate_data(selected)
            self.last_result = dict(strategy=self.vae_cfg.strategy, selected_id=self.candidate_id(selected),
                                    selected_index=selected, selected_rank=1, cosine_score=score,
                                    search_scope="full_pool", candidate_count=len(self._subframes),
                                    history_frames=len(frames), history_padding_frames=8-len(frames) if self.video else 0,
                                    search_seconds=search_seconds, retrieval_seconds=time.perf_counter()-started,
                                    index_gpu_bytes=self.embeddings.numel()*4, **self.encoder.last_metadata)
            return result
        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError(f"Wan VAE retrieval failed: {exc}") from exc

    def close(self):
        self.encoder.close()
        self.embeddings = None
