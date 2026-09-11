"""Real encoder invariants before an expensive index build."""
import json
import numpy as np
import torch
from cosmos_policy.experiments.robot.pusht_ret.retrievers.config import load_config
from cosmos_policy.experiments.robot.pusht_ret.retrievers.wan_vae_encoder import WanVAEEncoder

torch.manual_seed(42)
images = np.random.default_rng(42).integers(0, 256, (8, 128, 128, 3), dtype=np.uint8)
for mode in ("image", "video"):
    cfg = load_config(f"configs/retrieval/wan_vae_{mode}.yaml")
    encoder = WanVAEEncoder(cfg)
    clip = [images[0]] if mode == "image" else list(images)
    cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    query = encoder.encode([clip])
    assert torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(gpu_rng, torch.cuda.get_rng_state())
    values = encoder.encode([clip, clip])
    assert tuple(query.shape) == (1, cfg.embedding_dim)
    assert torch.allclose(torch.linalg.vector_norm(values, dim=1), torch.ones(2, device='cuda'), atol=1e-5)
    assert float((values @ query[0]).min()) > .999
    changed = [f.copy() for f in clip]
    changed[-1][:] = 0
    other = encoder.encode([changed])
    assert float((other-query).abs().max()) > 1e-5
    print(json.dumps(dict(mode=mode, dimension=cfg.embedding_dim, batch_cosine=(values @ query[0]).tolist(),
                          last_frame_changes_latent=True, rng_preserved=True, **encoder.last_metadata)), flush=True)
    encoder.close()
    del encoder, values, query, other
    torch.cuda.empty_cache()
