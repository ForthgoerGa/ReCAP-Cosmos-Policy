"""Encode each demo frame once; build single-frame or causal weighted-history indexes."""
import argparse
import json
from pathlib import Path

import numpy as np

from cosmos_policy.experiments.robot.pusht_ret.retrieval import PushTRetrieval
from cosmos_policy.experiments.robot.pusht_ret.retrievers.config import load_config
from cosmos_policy.experiments.robot.pusht_ret.retrievers.index import pool_manifest, sha256, validate_vectors
from cosmos_policy.experiments.robot.pusht_ret.retrievers.qwen_client import QwenClient
from cosmos_policy.experiments.robot.pusht_ret.retrievers.qwen_joint_encoder import implementation_hash
from cosmos_policy.experiments.robot.pusht_ret.retrievers.state_features import (
    history_embedding, representation_signature, state_feature, state_text,
)


def build(cfg, data_dir, split, frame_cache=None):
    if cfg.strategy not in ("qwen_state_text", "qwen_history_state"):
        raise ValueError("This builder only accepts joint image/state strategies")
    out = Path(cfg.index_path)
    out.mkdir(parents=True, exist_ok=False)
    pool = PushTRetrieval(data_dir, split=split)
    manifest = pool_manifest(pool)
    keys = sorted(pool._base_data)
    offsets, count = {}, 0
    for key in keys:
        offsets[key] = count
        count += len(pool._base_data[key]["images"])
    frame_layout = [{"source": pool._source_files[k], "demo": k[1], "offset": offsets[k],
                     "length": len(pool._base_data[k]["images"])} for k in keys]
    client = QwenClient(cfg)  # Also verify runtime/model when reusing a cache.
    try:
        if frame_cache:
            cache = Path(frame_cache)
            previous = json.loads((cache / "manifest.json").read_text())
            for key, value in manifest.items():
                if previous[key] != value:
                    raise ValueError(f"Frame-cache pool mismatch: {key}")
            if (previous["encoder"] != cfg.encoder_signature()
                    or previous["implementation_hash"] != implementation_hash()
                    or previous["frame_layout"] != frame_layout):
                raise ValueError("Frame-cache encoder/layout mismatch")
            for key in ("implementation_hash", "model_hashes", "versions"):
                if previous["worker"][key] != client.metadata[key]:
                    raise ValueError(f"Frame-cache worker mismatch: {key}")
            if sha256(cache / "frame_embeddings.npy") != previous["frame_embeddings_sha256"]:
                raise ValueError("Frame-cache checksum mismatch")
            frames = np.load(cache / "frame_embeddings.npy", mmap_mode="r", allow_pickle=False)
            validate_vectors(frames, count, cfg.embedding_dim)
            print(f"Reusing {count} joint frame embeddings from {cache}", flush=True)
        else:
            frames = np.empty((count, cfg.embedding_dim), dtype=np.float32)
            image_batch, text_batch, positions = [], [], []
            def flush():
                if positions:
                    frames[positions], _ = client.encode(image_batch, text_batch)
                    image_batch.clear()
                    text_batch.clear()
                    positions.clear()
            processed = 0
            for key in keys:
                item = pool._base_data[key]
                for t, img in enumerate(item["images"]):
                    image_batch.append(img)
                    text_batch.append(state_text(state_feature(item["states"][max(0, t - 2):t + 1])))
                    positions.append(offsets[key] + t)
                    if len(positions) == cfg.batch_size:
                        flush()
                    processed += 1
                    if processed % 800 == 0:
                        print(f"Encoded {processed}/{count} unique frames", flush=True)
            flush()
            validate_vectors(frames, count, cfg.embedding_dim)
            with (out / "frame_embeddings.npy").open("xb") as f:
                np.save(f, frames, allow_pickle=False)
        vectors = np.empty((len(pool._subframes), cfg.embedding_dim), dtype=np.float32)
        for i, sf in enumerate(pool._subframes):
            offset = offsets[(sf["split"], sf["demo"])]
            end = offset + sf["t_last"] + 1
            if cfg.strategy == "qwen_history_state":
                start = offset + max(0, sf["t_last"] - cfg.history_length + 1)
                vectors[i] = history_embedding(frames[start:end], cfg.history_weight_power)
            else:
                vectors[i] = frames[end - 1]
        validate_vectors(vectors, len(pool._subframes), cfg.embedding_dim)
        with (out / "embeddings.npy").open("xb") as f:
            np.save(f, vectors, allow_pickle=False)
        manifest.update(encoder=cfg.encoder_signature(), implementation_hash=implementation_hash(),
                        representation=representation_signature(cfg), worker=client.metadata,
                        embeddings_sha256=sha256(out / "embeddings.npy"), config=cfg.to_dict(),
                        frame_layout=frame_layout, frame_cache=str(frame_cache) if frame_cache else None)
        if not frame_cache:
            manifest["frame_embeddings_sha256"] = sha256(out / "frame_embeddings.npy")
        with (out / "manifest.json").open("x") as f:
            json.dump(manifest, f, indent=2)
        print(f"Index complete: {out}; {len(vectors)} candidates", flush=True)
    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split", default="base_0,base_1,base_2,base_3,base_4")
    parser.add_argument("--frame-cache", type=Path)
    args = parser.parse_args()
    build(load_config(args.retrieval_config), args.data_dir, args.split.split(","), args.frame_cache)


if __name__ == "__main__":
    main()
