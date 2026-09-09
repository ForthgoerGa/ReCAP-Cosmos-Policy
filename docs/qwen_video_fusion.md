# Dense video retrieval and fixed-score fusion (v2)

VideoRetrievalPool (retrievers/video_pool.py) subclasses the original pool with STRIDE=1. The original STRIDE=2 pool and image/state retrievers are unchanged. Every demo frame t has one causal candidate [max(0,t-7), t]. Partial windows use real frames only; full windows have eight frames. Window length/type and complete candidate identities are validated by both builder and runtime. The existing sparse video_v1 index is intentionally rejected.

build_qwen_video_index.py supports disjoint --start/--end/--shard-out builds and --merge-shards. Merging checks exact coverage, source and encoder identity, checksums, normalized vectors and worker model/dependency versions. New output directories are required; old artifacts remain intact.

qwen_video.py uses raw cosine for pure-video top1. Video/state fusion v2 uses:
- d² = sum((weighted_demo_state10 - weighted_current_state10)²)
- state_score = exp(-d²), a fixed RBF similarity with unit scale in the original normalized/weighted feature space
- final_score = alpha * raw_cosine + beta * state_score

Neither score is min-max normalized against the candidate pool. The state mapping has no data-fitted scale. Alpha/beta are coefficient weights and do not guarantee dominance of one score: ranking depends on score differences. For alpha=0, raw argmin(d²) is used to avoid exponential underflow ties. Scores and score_version=raw_cosine_rbf_state_v2 are recorded in traces.

State remains the original 10 dimensions: block position(2), agent position(2), sin/cos yaw(2), block velocity(2), agent velocity(2), normalized by 512 with weights [2,2,2.5,2.5,1.5,1.5,1,1,1,1]. Full-pool search differs from the original top-100-demo gate.

Runtime entry: run_eval.py -> QwenVideoRetrieval / QwenVideoLateFusionRetrieval. Configs: qwen_video.yaml and qwen_video_late_08.yaml (alpha=.8,beta=.2). Set QWEN_VIDEO_INDEX_PATH to the new dense index. Formal launcher: scripts/retrieval/run_formal_ablation.py, seeds42–91, tri_default, model_000007000.pt.

Earlier sparse/min-max results are historical experiments, not results for this implementation. The previous history-forwarding bug run is also invalid as a video experiment.
