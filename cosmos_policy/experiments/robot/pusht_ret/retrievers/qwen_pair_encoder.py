"""Batched, strict image-pair adapter for the official Qwen3-VL reranker.

Uses the upstream prompt and yes/no head; performs the final one-dimensional
projection in FP32 and ranks logits to avoid sigmoid saturation ties.
"""
import hashlib
from pathlib import Path
import time

import numpy as np
from PIL import Image


def implementation_hash():
    root = Path(__file__).parent
    h = hashlib.sha256()
    for name in ('qwen_pair_encoder.py', '_vendor/qwen3_vl_reranker.py'):
        h.update((root / name).read_bytes())
    return h.hexdigest()


class QwenPairEncoder:
    def __init__(self, cfg):
        import torch
        from qwen_vl_utils import process_vision_info
        from ._vendor.qwen3_vl_reranker import Qwen3VLReranker

        class StrictReranker(Qwen3VLReranker):
            def tokenize(self, pairs, **kwargs):
                text = self.processor.apply_chat_template(pairs, tokenize=False, add_generation_prompt=True)
                images, videos, extra = process_vision_info(
                    pairs, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)
                if images is None or len(images) != 2 * len(pairs) or videos is not None:
                    raise ValueError('Every reranker pair must contain exactly two images')
                inputs = self.processor(text=text, images=images, padding=True,
                                        truncation=False, do_resize=False, return_tensors='pt', **extra)
                if ('pixel_values' not in inputs or len(inputs['image_grid_thw']) != 2 * len(pairs)
                        or inputs['input_ids'].shape[1] > self.max_length):
                    raise ValueError('Missing visual content or excessive reranker input length')
                return inputs

        self.cfg = cfg
        self.model = StrictReranker(
            model_name_or_path=cfg.reranker_model_path, torch_dtype=torch.bfloat16,
            attn_implementation='sdpa', min_pixels=cfg.image_size ** 2,
            max_pixels=cfg.image_size ** 2, max_length=8192)
        self.model.model.config.use_cache = False
        self.model.score_linear.float()

    def score(self, query, candidates):
        import torch
        if not candidates:
            raise ValueError('No reranker candidates')
        preprocess_seconds = 0.0
        forward_seconds = 0.0
        start = time.perf_counter()
        images = []
        for frame in [query, *candidates]:
            if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3:
                raise ValueError('Expected uint8 RGB')
            images.append(Image.fromarray(frame).resize(
                (self.cfg.image_size, self.cfg.image_size), Image.Resampling.BICUBIC))
        pairs = [self.model.format_mm_instruction(
            None, images[0], None, None, image, None,
            instruction=self.cfg.reranker_instruction) for image in images[1:]]
        preprocess_seconds += time.perf_counter() - start
        logits = []
        token_lengths = []
        batch_size = self.cfg.reranker_batch_size
        with torch.inference_mode():
            for offset in range(0, len(pairs), batch_size):
                start = time.perf_counter()
                inputs = self.model.tokenize(pairs[offset:offset + batch_size]).to(self.model.device)
                token_lengths.append(int(inputs['input_ids'].shape[1]))
                torch.cuda.synchronize()
                preprocess_seconds += time.perf_counter() - start
                start = time.perf_counter()
                hidden = self.model.model(**inputs, use_cache=False).last_hidden_state[:, -1]
                values = self.model.score_linear(hidden.float()).squeeze(-1)
                logits.extend(values.cpu().tolist())
                del hidden, values, inputs
                torch.cuda.synchronize()
                forward_seconds += time.perf_counter() - start
        logits = np.asarray(logits, dtype=np.float32)
        if logits.shape != (len(candidates),) or not np.isfinite(logits).all():
            raise ValueError('Invalid reranker logits')
        scores = torch.sigmoid(torch.from_numpy(logits)).numpy()
        return logits, scores, dict(preprocess_seconds=preprocess_seconds,
                                    forward_seconds=forward_seconds, batch_size=batch_size,
                                    batches=len(token_lengths), token_lengths=token_lengths)
