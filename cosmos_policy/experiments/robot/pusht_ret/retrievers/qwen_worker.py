"""Persistent local JSON-lines worker. stdout is exclusively protocol output."""
import base64
import contextlib
import io
import json
import sys
import time
import traceback
from importlib.metadata import version
from pathlib import Path
import hashlib


def main():
    protocol = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        import numpy as np
        import torch
        from PIL import Image
        from .config import RetrievalConfig
        from .qwen_encoder import QwenEncoder, implementation_hash
        cfg = RetrievalConfig(**json.loads(sys.stdin.readline()))
        cfg.validate()
        model_hashes = {}
        for path in sorted(Path(cfg.model_path).iterdir()):
            if path.suffix in (".json", ".safetensors", ".jinja", ".txt") and path.name != "DOWNLOAD_MANIFEST.json":
                h = hashlib.sha256()
                with open(path, "rb") as f:
                    for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
                        h.update(block)
                model_hashes[path.name] = h.hexdigest()
        encoder = QwenEncoder(cfg)
    def send(value):
        protocol.write(json.dumps(value) + "\n")
        protocol.flush()
    send({"ready": True, "implementation_hash": implementation_hash(),
          "model_hashes": model_hashes,
          "versions": {name: version(name) for name in
                       ("torch", "transformers", "qwen-vl-utils", "pillow")},
          "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"})
    for line in sys.stdin:
        request = json.loads(line)
        if request.get("close"):
            break
        try:
            started = time.perf_counter()
            frames = [np.array(Image.open(io.BytesIO(base64.b64decode(x))).convert("RGB"))
                      for x in request["images"]]
            with contextlib.redirect_stdout(sys.stderr):
                embeddings = encoder.encode(frames)
            send({"id": request["id"], "embeddings": embeddings.tolist(),
                  "encoding_seconds": time.perf_counter() - started,
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0})
        except Exception:
            send({"id": request.get("id"), "error": traceback.format_exc()})
            return


if __name__ == "__main__":
    main()
