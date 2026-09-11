# Wan VAE flattened cosine experiment (2026-09-11)

Two independent formal groups: single image and causal sliding video window.
Both keep query-only blue-circle rendering, tri_default physics, policy inputs,
model_000007000.pt, seeds 42-91, and the existing formal inference settings.

- Backbone: Cosmos Predict2.5 tokenizer.pth, byte-identical to official Wan2.1_VAE.pth.
  SHA256: 38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981.
- Resize RGB to 224x224 with PIL bicubic; scale to [-1,1]. Use the deterministic
  posterior mean with Cosmos fixed channel mean/std, flatten C,T,H,W and FP32 L2.
  No training, spatial/temporal pooling, state gating, or reranking.
- Single image: [16,1,28,28], 12544 dimensions, original 12951-candidate pool.
- Video: logical history is exactly the latest 8 observations, stride 1 over all
  24397 demo endpoints. At episode/demo start replicate its first frame on the
  left until the logical history has 8 frames. Never cross episode boundaries.
- Compatibility detail: the current Cosmos Wan implementation raises a temporal
  convolution kernel-size error for 8 input frames. Add ONE more first frame at
  the left before encoding, giving legal 1+4k=9 frames. This preserves all eight
  observations, including the current endpoint. Output [16,3,28,28], 37632 dims.
  The same adaptation is used for index and online query and recorded in manifests.
- Cosine search: complete normalized FP32 index on GPU, TF32 disabled during
  search, stable first-row top-1 tie behavior. Return original 8-step policy payload.
- Use base WanVAE_.encode (no stochastic sample and no deterministic wrapper RNG
  reset) so retrieval does not change policy random state. GPU/backend flags are
  scoped and restored after each call.

Shared experiment base: /mnt4/cyh/ReCAP_qwen_visual_align.
Compute snapshot: repo_wan_20260911; authoritative Git worktree remains on ROG.
Environment and paths: cosmos_policy/scripts/retrieval/wan_shared_env.sh.
Build: python -m cosmos_policy.scripts.retrieval.build_wan_vae_index with
--retrieval-config, --data-dir, --start, --end, --shard-out; --merge-shards validates
coverage, checksums, identity, and worker versions before creating a fresh index.
Formal: bash cosmos_policy/scripts/retrieval/run_wan_formal_shared.sh image|video SEED GPU 5.
Each group has ten disjoint five-seed shards; all 50 seeds are required for completion.
