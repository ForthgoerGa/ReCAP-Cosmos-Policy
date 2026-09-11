"""IPC client without any Qwen/Transformers dependency in the policy process."""
import atexit
import base64
import io
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time

import numpy as np
from PIL import Image

from .config import RetrievalError


class QwenClient:
    def __init__(self, cfg, worker_module=None):
        self.cfg = cfg
        self.counter = 0
        self.responses = queue.Queue()
        self.stderr_tail = []
        self.process = subprocess.Popen(
            [cfg.worker_python, "-u", "-m",
             worker_module or "cosmos_policy.experiments.robot.pusht_ret.retrievers.qwen_worker"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, env={**os.environ, "TOKENIZERS_PARALLELISM": "false",
                                     "HF_HUB_OFFLINE": "1"})
        atexit.register(self.close)
        def reader():
            for line in self.process.stdout:
                self.responses.put(line)
            self.responses.put(None)
        def stderr_reader():
            for line in self.process.stderr:
                self.stderr_tail.append(line.rstrip())
                del self.stderr_tail[:-30]
        threading.Thread(target=reader, daemon=True).start()
        threading.Thread(target=stderr_reader, daemon=True).start()
        try:
            self._resource_role = 'reranker' if worker_module and 'pair_worker' in worker_module else 'embedding'
            self._register_resources()
            self.process.stdin.write(json.dumps(cfg.to_dict()) + "\n")
            self.process.stdin.flush()
            self.metadata = self._receive()
            if not self.metadata.get("ready"):
                raise RetrievalError("Qwen worker failed readiness check")
            self._register_resources(self.metadata)
        except Exception as e:
            self.close()
            if isinstance(e, RetrievalError):
                raise
            raise RetrievalError(f"Worker initialization failed: {e}") from e

    def _register_resources(self, metadata=None):
        filename = os.environ.get('RECAP_RESOURCE_FILE')
        if not filename:
            return
        path = Path(filename)
        record = json.loads(path.read_text()) if path.exists() else {'policy_pid': os.getpid(), 'workers': {}}
        record['workers'][self._resource_role] = {'pid': self.process.pid, 'metadata': metadata}
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(record, indent=2))
        tmp.replace(path)

    def _receive(self):
        try:
            line = self.responses.get(timeout=self.cfg.timeout_seconds)
            if line is None:
                raise RetrievalError("Qwen worker exited: " + "\n".join(self.stderr_tail))
            result = json.loads(line)
            if "error" in result:
                raise RetrievalError(result["error"])
            return result
        except queue.Empty as e:
            raise RetrievalError("Qwen worker timeout: " + "\n".join(self.stderr_tail)) from e

    def encode(self, frames, texts=None):
        started = time.perf_counter()
        try:
            images = []
            for frame in frames:
                if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3:
                    raise ValueError("Expected uint8 RGB")
                buf = io.BytesIO()
                Image.fromarray(frame).save(buf, format="PNG")
                images.append(base64.b64encode(buf.getvalue()).decode())
            self.counter += 1
            request = {"id": self.counter, "images": images}
            if texts is not None:
                if len(texts) != len(frames):
                    raise ValueError("texts and frames must have equal length")
                request["texts"] = [str(x) for x in texts]
            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()
            result = self._receive()
            values = np.asarray(result["embeddings"], dtype=np.float32)
            if result["id"] != self.counter or values.shape != (len(frames), self.cfg.embedding_dim):
                raise ValueError("Worker response mismatch")
            if not np.isfinite(values).all() or not np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-5):
                raise ValueError("Invalid normalized embeddings")
            result.pop("embeddings")
            result["roundtrip_seconds"] = time.perf_counter() - started
            return values, result
        except RetrievalError:
            raise
        except Exception as e:
            raise RetrievalError(f"Embedding IPC failed: {e}") from e

    def encode_video(self, videos, texts=None):
        started = time.perf_counter()
        try:
            encoded = []
            for frames in videos:
                clip = []
                for frame in frames:
                    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3:
                        raise ValueError("Expected uint8 RGB")
                    buf = io.BytesIO(); Image.fromarray(frame).save(buf, format="PNG")
                    clip.append(base64.b64encode(buf.getvalue()).decode())
                encoded.append(clip)
            if texts is not None and len(texts) != len(videos):
                raise ValueError("texts and videos must have equal length")
            self.counter += 1
            request = {"id": self.counter, "videos": encoded}
            if texts is not None: request["texts"] = [str(x) for x in texts]
            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()
            result = self._receive()
            values = np.asarray(result["embeddings"], dtype=np.float32)
            if result["id"] != self.counter or values.shape != (len(videos), self.cfg.embedding_dim):
                raise ValueError("Worker response mismatch")
            if not np.isfinite(values).all() or not np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-5):
                raise ValueError("Invalid normalized embeddings")
            result.pop("embeddings"); result["roundtrip_seconds"] = time.perf_counter() - started
            return values, result
        except RetrievalError: raise
        except Exception as e: raise RetrievalError(f"Video embedding IPC failed: {e}") from e

    def close(self):
        if getattr(self, "process", None) is None:
            return
        if self.process.poll() is None:
            try:
                self.process.stdin.write('{"close":true}\n')
                self.process.stdin.flush()
                self.process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
