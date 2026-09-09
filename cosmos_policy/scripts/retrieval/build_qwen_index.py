"""Run in ReCap; encode in the configured independent Qwen worker."""
import argparse
import json
from pathlib import Path

import numpy as np

from cosmos_policy.experiments.robot.pusht_ret.retrieval import PushTRetrieval
from cosmos_policy.experiments.robot.pusht_ret.retrievers.config import load_config
from cosmos_policy.experiments.robot.pusht_ret.retrievers.index import pool_manifest, sha256, validate_vectors
from cosmos_policy.experiments.robot.pusht_ret.retrievers.qwen_client import QwenClient
from cosmos_policy.experiments.robot.pusht_ret.retrievers.qwen_encoder import implementation_hash


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--retrieval-config", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--split", default="base_0,base_1,base_2,base_3,base_4")
    args = p.parse_args()
    cfg = load_config(args.retrieval_config)
    if cfg.strategy in ("qwen_state_text", "qwen_history_state"):
        from .build_qwen_multimodal_index import build
        return build(cfg, args.data_dir, args.split.split(","))
    out = Path(cfg.index_path)
    out.mkdir(parents=True, exist_ok=False)
    pool = PushTRetrieval(args.data_dir, split=args.split.split(","))
    manifest = pool_manifest(pool)
    vectors = np.empty((len(pool._subframes), cfg.embedding_dim), np.float32)
    client = QwenClient(cfg)
    try:
        for start in range(0, len(vectors), cfg.batch_size):
            stop = min(start + cfg.batch_size, len(vectors))
            frames = []
            for i in range(start, stop):
                sf = pool._subframes[i]
                frames.append(pool._base_data[(sf["split"], sf["demo"])]["images"][sf["t_last"]])
            vectors[start:stop], _ = client.encode(frames)
            if start % (cfg.batch_size * 20) == 0:
                print(f"Encoded {stop}/{len(vectors)}", flush=True)
    finally:
        client.close()
    validate_vectors(vectors, len(pool._subframes), cfg.embedding_dim)
    with open(out / "embeddings.npy", "xb") as f:
        np.save(f, vectors, allow_pickle=False)
    manifest.update(encoder=cfg.encoder_signature(), implementation_hash=implementation_hash(),
                    worker=client.metadata, embeddings_sha256=sha256(out / "embeddings.npy"),
                    config=cfg.to_dict())
    with open(out / "manifest.json", "x") as f:
        json.dump(manifest, f, indent=2)
    print(f"Index complete: {out}", flush=True)


if __name__ == "__main__":
    main()
