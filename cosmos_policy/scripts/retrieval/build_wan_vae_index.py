"""Build disjoint image/video Wan VAE shards, then validate and merge them."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from cosmos_policy.experiments.robot.pusht_ret.retrieval import PushTRetrieval
from cosmos_policy.experiments.robot.pusht_ret.retrievers.config import load_config
from cosmos_policy.experiments.robot.pusht_ret.retrievers.index import sha256, validate_vectors
from cosmos_policy.experiments.robot.pusht_ret.retrievers.video_pool import VideoRetrievalPool
from cosmos_policy.experiments.robot.pusht_ret.retrievers.wan_vae import candidate_clip, index_identity
from cosmos_policy.experiments.robot.pusht_ret.retrievers.wan_vae_encoder import WanVAEEncoder


def make_pool(cfg, data_dir, split):
    cls = VideoRetrievalPool if cfg.strategy == "wan_vae_video" else PushTRetrieval
    return cls(data_dir, split=split)


def build(cfg, pool, start, end, out):
    total = len(pool._subframes)
    end = total if end is None else end
    if not 0 <= start < end <= total:
        raise ValueError("Invalid shard bounds")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() or out.with_suffix(".json").exists():
        raise FileExistsError(out)
    metadata = index_identity(pool, cfg)
    encoder = WanVAEEncoder(cfg)
    started = time.perf_counter()
    values = np.empty((end-start, cfg.embedding_dim), np.float32)
    try:
        for lo in range(start, end, cfg.batch_size):
            hi = min(end, lo + cfg.batch_size)
            clips = [candidate_clip(pool, i, cfg.strategy == "wan_vae_video") for i in range(lo, hi)]
            values[lo-start:hi-start] = encoder.encode(clips).cpu().numpy()
            if lo == start or (lo-start) % (cfg.batch_size * 20) == 0 or hi == end:
                print(json.dumps(dict(encoded=hi-start, total=end-start, seconds=time.perf_counter()-started)), flush=True)
        validate_vectors(values, end-start, cfg.embedding_dim)
        with out.open("xb") as f:
            np.save(f, values, allow_pickle=False)
        metadata.update(worker=encoder.metadata, embeddings_sha256=sha256(out), config=cfg.to_dict(),
                        shard_start=start, shard_end=end, elapsed_seconds=time.perf_counter()-started)
        with out.with_suffix(".json").open("x") as f:
            json.dump(metadata, f, indent=2)
    finally:
        encoder.close()


def merge(cfg, pool, shard_dir):
    identity = index_identity(pool, cfg)
    shards = []
    for path in Path(shard_dir).glob("*.npy"):
        m = json.loads(path.with_suffix(".json").read_text())
        for key, value in identity.items():
            if m.get(key) != value:
                raise ValueError(f"Shard identity mismatch: {path}: {key}")
        if sha256(path) != m["embeddings_sha256"]:
            raise ValueError(f"Shard checksum mismatch: {path}")
        shards.append((m["shard_start"], m["shard_end"], path, m))
    shards.sort()
    cursor, worker = 0, None
    for start, end, path, m in shards:
        if start != cursor or end <= start:
            raise ValueError("Overlapping or missing shard range")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        validate_vectors(array, end-start, cfg.embedding_dim)
        if worker is not None and worker != m["worker"]:
            raise ValueError("Worker metadata differs between shards")
        worker, cursor = m["worker"], end
    if cursor != len(pool._subframes):
        raise ValueError("Incomplete index shards")
    target = Path(cfg.index_path)
    target.mkdir(parents=True, exist_ok=False)
    output = target / "embeddings.npy"
    values = np.lib.format.open_memmap(output, mode="w+", dtype=np.float32, shape=(cursor, cfg.embedding_dim))
    for start, end, path, _ in shards:
        values[start:end] = np.load(path, mmap_mode="r", allow_pickle=False)
    values.flush()
    del values
    identity.update(worker=worker, embeddings_sha256=sha256(output), config=cfg.to_dict(),
                    shard_sources=[str(path) for _, _, path, _ in shards])
    with (target / "manifest.json").open("x") as f:
        json.dump(identity, f, indent=2)
    print(f"Index complete: {cursor} vectors at {target}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split", default="base_0,base_1,base_2,base_3,base_4")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--shard-out")
    parser.add_argument("--merge-shards")
    args = parser.parse_args()
    cfg = load_config(args.retrieval_config)
    pool = make_pool(cfg, args.data_dir, args.split.split(","))
    if args.merge_shards:
        merge(cfg, pool, args.merge_shards)
    elif args.shard_out:
        build(cfg, pool, args.start, args.end, args.shard_out)
    else:
        parser.error("--shard-out or --merge-shards is required")


if __name__ == "__main__":
    main()
