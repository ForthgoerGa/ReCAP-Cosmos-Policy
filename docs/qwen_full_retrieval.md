# Full-pool Qwen image retrieval

`qwen_full` is a separate backend alongside `standard`, `consistent`, `cumulative`,
and `qwen_rerank`. It searches every timestamp in the selected demo pool using
only image embeddings. It does not use initial block-position filtering, state
Top-K, pose/velocity distances, temporal penalties, or state fallback.

## Call chain and modules

1. `scripts/retrieval/build_qwen_index.py` enumerates the original
   `PushTRetrieval._subframes` and encodes the RGB image at each `t_last` through
   the shared `qwen_client.py` → `qwen_worker.py` → `qwen_encoder.py` path.
2. `run_eval.py` selects `retrievers/qwen_full.py::QwenFullRetrieval` when
   `retrieval_strategy=qwen_full`; the current rendered observation is passed as
   `primary_image` at every action-chunk boundary (8 steps).
3. The backend embeds that image and computes `index_embeddings @ query_embedding`
   over the complete index. FP32 L2-normalized vectors make this cosine similarity.
   `argmax` returns top-1; exact ties pick the earliest row in the index.
4. `retrieval.py::get_candidate_data` loads the selected original 8-frame/action
   clip with its starting proprio and unchanged tail padding. Cosmos Policy and
   residual action decoding consume the same payload as before.

The base_0..base_4 pool contains 12,951 candidate timestamps from 200 demos.
Full pool means all candidates in the configured split, not all visual-task splits.
The inherited loader still reads demo states for compatible payload/index handling;
query state arguments are accepted for eval compatibility but never used to rank
or filter. Policy proprioception and environment metrics remain unchanged.

## Configuration and use

`configs/retrieval/qwen_full.yaml` uses the same model, preprocessing, instruction,
and encoder signature as `qwen_rerank.yaml`. A compatible complete index can be
shared. `candidate_k: 1` describes the returned top-1; it does not restrict search.
Any other K is rejected to prevent a misleading ablation configuration.

Set `QWEN_MODEL_PATH`, `QWEN_MODEL_REVISION`, `QWEN_PYTHON`, `QWEN_INDEX_PATH`,
`DATA`, and `CKPT` as described in `qwen_retrieval.md`. Build a new index when the
source hash, data, model, preprocessing, or runtime metadata differs:

```bash
python -m cosmos_policy.scripts.retrieval.build_qwen_index \
  --retrieval-config configs/retrieval/qwen_full.yaml --data-dir "$DATA" \
  --split base_0,base_1,base_2,base_3,base_4

python -m cosmos_policy.scripts.retrieval.run_ablation \
  --retrieval-configs configs/retrieval/qwen_full.yaml \
  --seed 42 --num-trials 50 --gpu 1 --offline \
  --data-dir "$DATA" --ckpt "$CKPT" --output-root /path/to/new/run
```

To run all three modules in one ablation launcher, pass all three YAML files:

```bash
python -m cosmos_policy.scripts.retrieval.run_ablation \
  --retrieval-configs configs/retrieval/state.yaml \
    configs/retrieval/qwen_rerank.yaml configs/retrieval/qwen_full.yaml \
  --seed 42 --num-trials 50 --gpu 1 --offline \
  --data-dir "$DATA" --ckpt "$CKPT" --output-root /path/to/new/comparison
```

Do not set a global `--candidate-k` override for a mixed run; the YAML files retain
state/rerank K and full-search top-1 independently. These modes run sequentially
on the selected GPU with identical seeds and policy settings.

Each full-search trace records selected ID/index, complete candidate count,
selected cosine, payload hash, returned actions/proprio, final policy actions,
and retrieval latency. It deliberately has no `selected_state_rank`; the summary
reports state Top-1 change as null. Full candidate ordering is in the index
manifest; per-query traces avoid duplicating all 12,951 IDs and scores.

Run `python -m pytest tests/retrieval -q` in the policy environment. Tests verify
that state candidate generation is never called, full-pool selection reaches
candidates outside state shortlists, tie behavior, payload alignment, strict
configuration, worker failures, and summary compatibility.
