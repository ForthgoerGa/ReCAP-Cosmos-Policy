"""Strict Qwen3-VL video-window encoder."""
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image

def implementation_hash():
    h=hashlib.sha256()
    for n in ("qwen_video_encoder.py", "_vendor/qwen3_vl_embedding.py"):
        h.update((Path(__file__).parent/n).read_bytes())
    return h.hexdigest()

class QwenVideoEncoder:
    def __init__(self, cfg):
        import torch
        from ._vendor.qwen3_vl_embedding import Qwen3VLEmbedder, process_vision_info
        class Strict(Qwen3VLEmbedder):
            def _preprocess_inputs(self, conversations):
                text=self.processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
                images, video_inputs, kwargs=process_vision_info(conversations, image_patch_size=16, return_video_metadata=True, return_video_kwargs=True)
                if video_inputs is None or images is not None: raise ValueError("Expected nonempty video inputs")
                videos=[v[0] if isinstance(v, tuple) else v for v in video_inputs]
                metadata=[v[1] if isinstance(v, tuple) else None for v in video_inputs]
                inputs=self.processor(text=text, images=None, videos=videos, video_metadata=metadata, padding=True, truncation=False, do_resize=False, return_tensors="pt", **kwargs)
                if "pixel_values_videos" not in inputs or inputs["input_ids"].shape[1] > self.max_length: raise ValueError("Missing video pixels or excessive token length")
                return inputs
        self.cfg=cfg
        self.model=Strict(model_name_or_path=cfg.model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa", min_pixels=cfg.image_size**2, max_pixels=cfg.image_size**2, max_length=8192)
    def encode(self, videos):
        import torch
        if not videos: raise ValueError("No videos")
        inputs=[]
        for clip in videos:
            if not clip: raise ValueError("Empty video")
            frames=[]
            for frame in clip:
                a=np.asarray(frame)
                if a.dtype!=np.uint8 or a.ndim!=3 or a.shape[-1]!=3: raise ValueError("Expected uint8 RGB")
                frames.append(Image.fromarray(a).resize((self.cfg.image_size,self.cfg.image_size),Image.Resampling.BICUBIC))
            inputs.append({"video":frames,"instruction":self.cfg.instruction,"max_frames":len(frames)})
        with torch.inference_mode():
            vals=self.model.process(inputs, normalize=False).float(); vals=torch.nn.functional.normalize(vals,p=2,dim=-1)
        out=vals.cpu().numpy()
        if out.shape!=(len(videos),self.cfg.embedding_dim) or not np.isfinite(out).all(): raise ValueError("Invalid video embeddings")
        return out
