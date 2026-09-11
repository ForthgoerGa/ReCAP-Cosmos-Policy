"""Cosmos Wan2.1 posterior means, flattened and normalized for cosine search."""
from contextlib import contextmanager
import hashlib
from pathlib import Path
import time

import numpy as np
from PIL import Image, __version__ as pillow_version

from .config import RetrievalError
from .index import sha256


def implementation_hash():
    root = Path(__file__).resolve().parents[5]
    paths = [Path(__file__), root / "cosmos_policy/_src/predict2/tokenizers/wan2pt1.py"]
    h = hashlib.sha256()
    for path in paths:
        h.update(path.read_bytes())
    return h.hexdigest()


def padded_history(frames):
    frames = list(frames)[-8:]
    if not frames:
        raise RetrievalError("Empty causal video history")
    return [frames[0]] * (8 - len(frames)) + frames


def prepare_clip(frames, video, image_size):
    frames = list(frames)
    if video:
        frames = padded_history(frames)
        # Wan requires 1+4k frames. One extra prefix retains all eight observations.
        frames = [frames[0]] + frames
    elif len(frames) != 1:
        raise RetrievalError("Image retrieval requires exactly one frame")
    resized = []
    for frame in frames:
        a = np.asarray(frame)
        if a.dtype != np.uint8 or a.ndim != 3 or a.shape[-1] != 3:
            raise RetrievalError("Expected uint8 RGB frames")
        resized.append(np.asarray(Image.fromarray(a).resize((image_size, image_size), Image.Resampling.BICUBIC)))
    return np.stack(resized).transpose(3, 0, 1, 2).astype(np.float32) / 127.5 - 1.0


@contextmanager
def full_precision_search():
    import torch
    old = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old


class WanVAEEncoder:
    def __init__(self, cfg, device="cuda"):
        import torch
        from cosmos_policy._src.predict2.tokenizers.wan2pt1 import WanVAE_

        self.cfg = cfg
        self.device = device
        self.video = cfg.strategy == "wan_vae_video"
        digest = sha256(cfg.model_path)
        if digest != cfg.model_revision:
            raise RetrievalError("Wan VAE checkpoint checksum mismatch")
        # Base encode returns the posterior mean and never resets policy RNG state.
        with torch.device("meta"):
            model = WanVAE_(dim=96, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=2,
                            attn_scales=[], temperal_downsample=[False, True, True])
        state = torch.load(cfg.model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True, assign=True)
        self.model = model.eval().requires_grad_(False).to(device)
        mean = [-.7571, -.7089, -.9113, .1075, -.1745, .9653, -.1517, 1.5508,
                .4134, -.0715, .5517, -.3632, -.1922, -.9497, .2503, -.2921]
        std = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
               3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160]
        self.scale = [torch.tensor(mean, device=device, dtype=torch.bfloat16),
                      1.0 / torch.tensor(std, device=device, dtype=torch.bfloat16)]
        self.metadata = dict(implementation_hash=implementation_hash(), model_sha256=digest,
                             versions=dict(torch=torch.__version__, numpy=np.__version__, pillow=pillow_version),
                             encoder=cfg.encoder_signature())
        self.last_metadata = {}

    def encode(self, clips):
        import torch
        if not clips:
            raise RetrievalError("Empty encoding batch")
        start = time.perf_counter()
        x = torch.from_numpy(np.stack([prepare_clip(c, self.video, self.cfg.image_size) for c in clips])).to(self.device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16), full_precision_search(), \
                torch.backends.cudnn.flags(benchmark=False, deterministic=True, allow_tf32=False):
            latent = self.model.encode(x, self.scale)
            expected = tuple(self.cfg.encoder_signature()["latent_shape"])
            if tuple(latent.shape[1:]) != expected:
                raise RetrievalError(f"Unexpected Wan latent {latent.shape}; expected {expected}")
            flat = latent.float().flatten(1)
            norms = torch.linalg.vector_norm(flat, dim=1, keepdim=True)
            if not torch.isfinite(flat).all() or torch.any(norms <= 0):
                raise RetrievalError("Invalid Wan latent")
            values = flat / norms
        torch.cuda.synchronize()
        self.last_metadata = dict(encode_seconds=time.perf_counter()-start,
                                  latent_shape=list(latent.shape[1:]), vae_input_frames=int(x.shape[2]))
        return values

    def close(self):
        self.model = None
        self.scale = None
