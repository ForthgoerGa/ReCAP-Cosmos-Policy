"""Portable index identity and validation, independent of model imports."""
import hashlib
from pathlib import Path
import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def pool_manifest(pool):
    paths = sorted(set(pool._source_files.values()))
    return {
        "files": {p: sha256(Path(pool.data_dir) / p) for p in paths},
        "candidate_ids": [pool.candidate_id(i) for i in range(len(pool._subframes))],
        "window": pool.WINDOW_SIZE, "stride": pool.STRIDE,
    }


def validate_vectors(values, n, dim):
    if values.shape != (n, dim) or not np.isfinite(values).all():
        raise ValueError("Index has invalid vector shape or nonfinite values")
    if not np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-5):
        raise ValueError("Index vectors are not unit normalized")
