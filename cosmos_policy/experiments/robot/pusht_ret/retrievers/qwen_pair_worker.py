"""Local persistent image-pair worker; stdout is JSON-lines protocol only."""
import base64
import contextlib
import hashlib
import io
import json
from importlib.metadata import version
from pathlib import Path
import sys
import time
import traceback


def main():
    protocol = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        import numpy as np
        import torch
        from PIL import Image
        from .config import RetrievalConfig
        from .qwen_pair_encoder import QwenPairEncoder, implementation_hash
        from .index import sha256
        cfg = RetrievalConfig(**json.loads(sys.stdin.readline()))
        cfg.validate()
        if not torch.cuda.is_available():
            raise RuntimeError('Reranker formal evaluation requires CUDA')
        root = Path(cfg.reranker_model_path)
        download = json.loads((root / 'DOWNLOAD_MANIFEST.json').read_text())
        if download['revision'] != cfg.reranker_model_revision:
            raise ValueError('Reranker revision does not match downloaded model')
        model_hashes = {}
        for filename, expected in download['files'].items():
            actual = sha256(root / filename)
            if actual != expected:
                raise ValueError(f'Reranker model checksum mismatch: {filename}')
            model_hashes[filename] = actual
        encoder = QwenPairEncoder(cfg)

    def send(record):
        protocol.write(json.dumps(record) + '\n')
        protocol.flush()

    send(dict(ready=True, implementation_hash=implementation_hash(), model_hashes=model_hashes,
              model_revision=cfg.reranker_model_revision, device=torch.cuda.get_device_name(),
              versions={name: version(name) for name in ('torch', 'transformers', 'qwen-vl-utils', 'pillow')},
              allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved()))
    for line in sys.stdin:
        request = json.loads(line)
        if request.get('close'):
            break
        try:
            started = time.perf_counter()
            frames = [np.array(Image.open(io.BytesIO(base64.b64decode(x))).convert('RGB'))
                      for x in request['images']]
            if len(frames) < 2:
                raise ValueError('Need a query and candidate images')
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            with contextlib.redirect_stdout(sys.stderr):
                logits, scores, metrics = encoder.score(frames[0], frames[1:])
            send(dict(id=request['id'], logits=logits.tolist(), scores=scores.tolist(),
                      scoring_seconds=time.perf_counter() - started,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved(),
                      query_sha256=hashlib.sha256(frames[0].tobytes()).hexdigest(),
                      candidate_sha256=[hashlib.sha256(f.tobytes()).hexdigest() for f in frames[1:]],
                      **metrics))
        except Exception:
            send(dict(id=request.get('id'), error=traceback.format_exc()))
            return


if __name__ == '__main__':
    main()
