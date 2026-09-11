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
        from .qwen_joint_encoder import QwenJointEncoder, implementation_hash as joint_implementation_hash
        from .qwen_video_encoder import QwenVideoEncoder, implementation_hash as video_implementation_hash
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
        joint = cfg.strategy in ("qwen_state_text", "qwen_history_state")
        video = cfg.strategy in ("qwen_video", "qwen_video_late_fusion", "qwen_video_agent_state")
        encoder = QwenJointEncoder(cfg) if joint else (QwenVideoEncoder(cfg) if video else QwenEncoder(cfg))
        active_hash = joint_implementation_hash() if joint else (video_implementation_hash() if video else implementation_hash())
    def send(value):
        protocol.write(json.dumps(value) + "\n")
        protocol.flush()
    send({"ready": True, "implementation_hash": active_hash,
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
            if video:
                videos = [[np.array(Image.open(io.BytesIO(base64.b64decode(x))).convert("RGB")) for x in clip] for clip in request["videos"]]
                with contextlib.redirect_stdout(sys.stderr): embeddings = encoder.encode(videos, request.get("texts"))
            else:
                frames = [np.array(Image.open(io.BytesIO(base64.b64decode(x))).convert("RGB")) for x in request["images"]]
                with contextlib.redirect_stdout(sys.stderr):
                    if joint: embeddings = encoder.encode(frames, request.get("texts"))
                    else:
                        if request.get("texts") is not None: raise ValueError("Image-only worker does not accept state text")
                        embeddings = encoder.encode(frames)
            send({"id": request["id"], "embeddings": embeddings.tolist(),
                  "encoding_seconds": time.perf_counter() - started,
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                  "allocated_bytes": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
                  "reserved_bytes": torch.cuda.memory_reserved() if torch.cuda.is_available() else 0,
                  "peak_reserved_bytes": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0})
        except Exception:
            send({"id": request.get("id"), "error": traceback.format_exc()})
            return


if __name__ == "__main__":
    main()
