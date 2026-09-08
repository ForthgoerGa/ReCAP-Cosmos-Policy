"""One strict image encoder shared by offline and online paths."""
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image


def implementation_hash():
    root = Path(__file__).parent
    h = hashlib.sha256()
    for name in ("qwen_encoder.py", "_vendor/qwen3_vl_embedding.py"):
        h.update((root / name).read_bytes())
    return h.hexdigest()


class QwenEncoder:
    def __init__(self, cfg):
        import torch
        from ._vendor.qwen3_vl_embedding import Qwen3VLEmbedder, process_vision_info

        # The upstream helper catches visual preprocessing errors and substitutes
        # NULL text. That is unacceptable for an auditable retrieval experiment.
        class StrictEmbedder(Qwen3VLEmbedder):
            def _preprocess_inputs(self, conversations):
                text = self.processor.apply_chat_template(
                    conversations, add_generation_prompt=True, tokenize=False)
                images, videos, kwargs = process_vision_info(
                    conversations, image_patch_size=16,
                    return_video_metadata=True, return_video_kwargs=True)
                if images is None or videos is not None:
                    raise ValueError("Expected image inputs; refusing missing visual content")
                inputs = self.processor(
                    text=text, images=images, padding=True, truncation=False,
                    do_resize=False, return_tensors="pt", **kwargs)
                if "pixel_values" not in inputs or inputs["input_ids"].shape[1] > self.max_length:
                    raise ValueError("Missing pixels or excessive token length")
                return inputs

        self.cfg = cfg
        self.model = StrictEmbedder(
            model_name_or_path=cfg.model_path,
            torch_dtype=torch.bfloat16, attn_implementation="sdpa",
            min_pixels=cfg.image_size ** 2, max_pixels=cfg.image_size ** 2,
            max_length=8192)

    def encode(self, frames):
        import torch
        if not frames:
            raise ValueError("No frames to encode")
        inputs = []
        for frame in frames:
            frame = np.asarray(frame)
            if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3:
                raise ValueError("Expected uint8 RGB image")
            pil = Image.fromarray(frame).resize(
                (self.cfg.image_size, self.cfg.image_size), Image.Resampling.BICUBIC)
            inputs.append({"image": pil, "instruction": self.cfg.instruction})
        with torch.inference_mode():
            values = self.model.process(inputs, normalize=False).float()
            values = torch.nn.functional.normalize(values, p=2, dim=-1)
        result = values.cpu().numpy()
        if result.shape != (len(frames), self.cfg.embedding_dim) or not np.isfinite(result).all():
            raise ValueError("Invalid embedding output")
        if not np.allclose(np.linalg.norm(result, axis=1), 1, atol=1e-5):
            raise ValueError("Non-unit embeddings")
        return result
