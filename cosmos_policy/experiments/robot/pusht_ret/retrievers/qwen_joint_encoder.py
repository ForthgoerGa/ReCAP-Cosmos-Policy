"""Image and canonical state text encoded together; legacy image encoder stays fixed."""
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from .qwen_encoder import QwenEncoder, implementation_hash as image_implementation_hash


def implementation_hash():
    h = hashlib.sha256(image_implementation_hash().encode())
    for name in ("qwen_joint_encoder.py", "state_features.py"):
        h.update((Path(__file__).parent / name).read_bytes())
    return h.hexdigest()


class QwenJointEncoder(QwenEncoder):
    def encode(self, frames, texts):
        import torch
        if not frames or texts is None or len(frames) != len(texts):
            raise ValueError("Joint encoding requires one state text for every image")
        inputs = []
        for frame, text in zip(frames, texts):
            frame = np.asarray(frame)
            if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3:
                raise ValueError("Expected uint8 RGB image")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Missing state text")
            pil = Image.fromarray(frame).resize(
                (self.cfg.image_size, self.cfg.image_size), Image.Resampling.BICUBIC)
            # State is user content alongside the image, not a replacement for the instruction.
            inputs.append({"image": pil, "text": text, "instruction": self.cfg.instruction})
        with torch.inference_mode():
            vectors = self.model.process(inputs, normalize=False).float()
            vectors = torch.nn.functional.normalize(vectors, p=2, dim=-1)
        result = vectors.cpu().numpy()
        if result.shape != (len(frames), self.cfg.embedding_dim) or not np.isfinite(result).all():
            raise ValueError("Invalid joint embedding")
        if not np.allclose(np.linalg.norm(result, axis=1), 1, atol=1e-5):
            raise ValueError("Non-unit joint embedding")
        return result
