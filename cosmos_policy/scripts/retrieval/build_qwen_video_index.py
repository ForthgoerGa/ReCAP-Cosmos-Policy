"""Build and merge disjoint dense causal video embedding shards."""
import argparse
import json
from pathlib import Path
import numpy as np
from cosmos_policy.experiments.robot.pusht_ret.retrievers.video_pool import VideoRetrievalPool
from cosmos_policy.experiments.robot.pusht_ret.retrievers.config import load_config
from cosmos_policy.experiments.robot.pusht_ret.retrievers.index import pool_manifest, sha256, validate_vectors
from cosmos_policy.experiments.robot.pusht_ret.retrievers.qwen_client import QwenClient
from cosmos_policy.experiments.robot.pusht_ret.retrievers.qwen_video_encoder import implementation_hash

def identity(pool, cfg):
    return dict(pool_manifest(pool), **pool.window_metadata(),
                encoder=cfg.encoder_signature(), implementation_hash=implementation_hash())

def build(cfg, data_dir, split, shard_start=0, shard_end=None, shard_out=None):
    if cfg.strategy != "qwen_video":
        raise ValueError("Requires qwen_video")
    pool = VideoRetrievalPool(data_dir, split=split)
    metadata = identity(pool, cfg)
    total = len(pool._subframes)
    end = total if shard_end is None else shard_end
    if not 0 <= shard_start < end <= total:
        raise ValueError("Invalid shard bounds")
    out = Path(shard_out) if shard_out else Path(cfg.index_path) / "embeddings.npy"
    out.parent.mkdir(parents=True, exist_ok=bool(shard_out))
    if out.exists() or out.with_suffix(".json").exists():
        raise FileExistsError(out)
    client = QwenClient(cfg)
    try:
        vec = np.empty((end-shard_start, cfg.embedding_dim), np.float32)
        for start in range(shard_start, end, cfg.batch_size):
            stop = min(start+cfg.batch_size, end)
            clips = []
            for idx in range(start, stop):
                sf = pool._subframes[idx]
                images = pool._base_data[(sf["split"], sf["demo"])]["images"]
                clips.append(list(images[sf["start"]:sf["t_last"]+1]))
            values, _ = client.encode_video(clips)
            vec[start-shard_start:stop-shard_start] = values
            print(f"encoded {stop-shard_start}/{end-shard_start}", flush=True)
        validate_vectors(vec, end-shard_start, cfg.embedding_dim)
        with out.open("xb") as f:
            np.save(f, vec, allow_pickle=False)
        metadata.update(worker=client.metadata, embeddings_sha256=sha256(out),
                        config=cfg.to_dict(), shard_start=shard_start, shard_end=end)
        target = out.with_suffix(".json") if shard_out else out.parent/"manifest.json"
        with target.open("x") as f:
            json.dump(metadata, f, indent=2)
    finally:
        client.close()

def merge(cfg, data_dir, split, shard_dir):
    pool = VideoRetrievalPool(data_dir, split=split)
    expected = identity(pool, cfg)
    shards = []
    for p in Path(shard_dir).glob("*.npy"):
        m = json.loads(p.with_suffix(".json").read_text())
        for key, value in expected.items():
            if m.get(key) != value:
                raise ValueError(f"Shard identity mismatch: {p}: {key}")
        if sha256(p) != m["embeddings_sha256"]:
            raise ValueError(f"Shard checksum mismatch: {p}")
        shards.append((m["shard_start"], m["shard_end"], p, m))
    shards.sort()
    cursor, arrays, worker = 0, [], None
    for start, end, p, m in shards:
        if start != cursor or end <= start:
            raise ValueError("Overlapping or missing shard range")
        a = np.load(p, allow_pickle=False)
        validate_vectors(a, end-start, cfg.embedding_dim)
        if worker is not None and any(worker[k] != m["worker"][k] for k in ("implementation_hash", "model_hashes", "versions")):
            raise ValueError("Worker versions differ between shards")
        worker = m["worker"]
        arrays.append(a)
        cursor = end
    if cursor != len(pool._subframes):
        raise ValueError("Incomplete shards")
    out = Path(cfg.index_path)
    out.mkdir(parents=True, exist_ok=False)
    values = np.concatenate(arrays)
    validate_vectors(values, cursor, cfg.embedding_dim)
    with (out/"embeddings.npy").open("xb") as f:
        np.save(f, values, allow_pickle=False)
    expected.update(worker=worker, config=cfg.to_dict(),
                    embeddings_sha256=sha256(out/"embeddings.npy"),
                    shard_sources=[str(p) for _, _, p, _ in shards])
    with (out/"manifest.json").open("x") as f:
        json.dump(expected, f, indent=2)
    print(f"Index complete: {cursor} dense windows at {out}", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval-config", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--split", default="base_0,base_1,base_2,base_3,base_4")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int)
    ap.add_argument("--shard-out")
    ap.add_argument("--merge-shards")
    a = ap.parse_args()
    cfg = load_config(a.retrieval_config)
    if a.merge_shards:
        merge(cfg, a.data_dir, a.split.split(","), a.merge_shards)
    else:
        build(cfg, a.data_dir, a.split.split(","), a.start, a.end, a.shard_out)
