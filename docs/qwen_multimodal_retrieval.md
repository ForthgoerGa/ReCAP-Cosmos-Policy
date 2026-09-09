# PushT joint-state and late-fusion retrieval ablations

All three new strategies search the complete selected demo pool and return one
original 8-step RGB/action/proprio clip. They coexist with `standard`, `consistent`,
`cumulative`, `qwen_rerank`, and `qwen_full`. No initial-position filter, state Top-K,
continuity penalty, policy training, or residual-statistics change is introduced.

## Strategies

| Strategy / YAML | Query and demo representation | Selection |
|---|---|---|
| `qwen_state_text` / `qwen_state.yaml` | Current RGB + current canonical state text in the same Qwen input | Full-pool joint-embedding cosine top-1 |
| `qwen_history_state` / `qwen_history.yaml` | Each available frame in t-7..t jointly encoded with its own causal state text; weighted FP32 pooling + L2 normalization | Full-pool pooled-embedding cosine top-1 |
| `qwen_late_fusion` / `qwen_late.yaml` | Original image-only Qwen embedding; independent weighted 10D geometry | Full-pool weighted score top-1 |

State text has fixed field ordering and six decimal places: agent/block xy positions
in simulator units divided by 512, block orientation sin/cos, and agent/block
linear velocity computed from the last two available displacements divided by 512.
The same `state_feature` and `state_text` functions serve offline and online paths.
No angular velocity or future state is included. State comes from simulator GT,
so these modes do not remove the need for state observations.

The joint encoder passes state as the `text` content alongside the image, retaining
a fixed retrieval `instruction`. It does not substitute state text for the instruction.
Its vectors stay within retrieval and are not supplied as policy tokens.

History weights are `w_j = j ** history_weight_power`, oldest to newest. Default
length 8 and power 1 give weights 1..8 (normalized by their sum). At episode start,
only available frames participate, with weights 1..L. The result is L2-normalized.
The eval loop retains 8 images and 10 states so the oldest image still has two
preceding steps for its velocity. Buffers reset on every episode.
This is weighted pooling of per-frame joint embeddings, not a temporal transformer
or native video encoder. It emphasizes recency but does not directly encode motion
order beyond the weights and state velocities.

Late fusion uses the unchanged original image encoder/index and original weighted
10D feature distance, evaluated directly for all candidates:

```
c_i = dot(image_query_embedding, image_demo_embedding[i])
d_i = sum((weighted_query_state - weighted_demo_state[i]) ** 2)
v_i = (c_i - min(c)) / (max(c) - min(c))
s_i = 1 - (d_i - min(d)) / (max(d) - min(d))
score_i = (alpha * v_i + beta * s_i) / (alpha + beta)
```

A constant score channel contributes zero. All values must be finite; invalid
index/worker/visual/state data fails the evaluation instead of silently falling
back. Alpha=beta=0.5 is the fixed first experiment. Both score channels use
per-query full-pool min/max normalization so their observed scales are comparable.
These extrema are candidate-pool dependent; weights are configurable and are not
claimed to be optimal. Alpha=1,beta=0 recovers image top-1; alpha=0,beta=1 uses
full-pool geometry and still differs from the original initial-position-gated baseline.

## Modules and callers

Paths below are relative to the repo. `retrievers/` means
`cosmos_policy/experiments/robot/pusht_ret/retrievers/`.

| Module | Called by | Responsibility |
|---|---|---|
| `retrievers/state_features.py` | joint index builder, online retrievers | causal state features, canonical text, history pooling and representation identity |
| `retrievers/qwen_joint_encoder.py` | `qwen_worker.py` for the two joint modes | same-image/text Qwen input and separate implementation hash; old image encoder unchanged |
| `retrievers/qwen_multimodal.py` | `run_eval.py` | shared index verification; `QwenStateTextRetrieval` and `QwenHistoryStateRetrieval` |
| `retrievers/qwen_late_fusion.py` | `run_eval.py` | independent image/geometric scores and normalized weighted fusion |
| `scripts/retrieval/build_qwen_multimodal_index.py` under `cosmos_policy/` | CLI or existing `build_qwen_index.py` | joint frame cache and mode-specific indexes |
| `scripts/retrieval/run_formal_ablation.py` under `cosmos_policy/` | CLI | configurable multi-GPU seed partitioning, calling existing `run_ablation.py` |
| `tests/retrieval/test_qwen_multimodal.py` | pytest in policy env | causal alignment, payloads, image-only fusion path, fusion endpoints, signature separation |

`retrieval.py` additionally retains the raw five-field demo states already read
from HDF5; the legacy feature, candidate, and payload algorithms are unchanged.
`run_eval.py` selects the new classes and supplies causal histories.
`qwen_client.py` transports optional per-image state strings over existing IPC.
`qwen_worker.py` chooses the joint encoder only for the two joint modes.

## Index setup

Use the existing separate Qwen worker environment. Export the common model paths,
revision, `DATA`, and `CKPT` as described in `qwen_retrieval.md`, plus three index paths:

```bash
export QWEN_INDEX_PATH=/path/to/compatible/original_image_index
export QWEN_STATE_INDEX_PATH=/path/to/new/joint_state_index
export QWEN_HISTORY_INDEX_PATH=/path/to/new/joint_history_index
```

Build the joint frame cache once, then reuse it for history:

```bash
python -m cosmos_policy.scripts.retrieval.build_qwen_multimodal_index \
  --retrieval-config configs/retrieval/qwen_state.yaml --data-dir "$DATA"

python -m cosmos_policy.scripts.retrieval.build_qwen_multimodal_index \
  --retrieval-config configs/retrieval/qwen_history.yaml --data-dir "$DATA" \
  --frame-cache "$QWEN_STATE_INDEX_PATH"
```

Every destination directory must be new. The single-frame directory contains
`frame_embeddings.npy` for 24,397 unique frames plus the 12,951-candidate
`embeddings.npy`. The history directory only needs its pooled candidate vectors.
Manifest fields record data and vector checksums, model/runtime/encoder identity,
frame layout, text schema, history length/weights, and available-prefix handling.
The offline per-frame encoder signature is shared by the two joint modes; their
retrieval representation signatures are different and checked at load time.
Changing the history window or weights requires a new history index, but compatible
joint frame embeddings can be reused. Changing text, image preprocessing, model,
or encoding implementation requires a new frame cache.

## Formal runs

Single GPU, three modes sequentially, 50 identical seeds per mode:

```bash
python -m cosmos_policy.scripts.retrieval.run_ablation \
  --retrieval-configs configs/retrieval/qwen_state.yaml \
    configs/retrieval/qwen_history.yaml configs/retrieval/qwen_late.yaml \
  --seed 42 --num-trials 50 --gpu 1 --offline \
  --data-dir "$DATA" --ckpt "$CKPT" --output-root /path/to/new/runs
```

Multi-GPU, the same total 50 seeds per mode partitioned across the specified GPUs:

```bash
python -m cosmos_policy.scripts.retrieval.run_formal_ablation \
  --retrieval-configs configs/retrieval/qwen_state.yaml \
    configs/retrieval/qwen_history.yaml configs/retrieval/qwen_late.yaml \
  --seed 42 --num-trials 50 --gpus 1 2 3 4 5 --offline \
  --data-dir "$DATA" --ckpt "$CKPT" --output-root /path/to/new/formal_runs
```

`candidate_k: 1` means one returned clip; every new mode searches the full pool.
Each mode keeps its YAML settings when launched together. Do not apply a global K
sweep to a mixed full-pool/Top-K comparison.

All runs retain exact episodes, source and asset checksums, resolved config,
retrieval traces, timing, GPU samples, and raw/combined/future videos. Joint traces
include the actual state texts and history weights; fusion traces include raw
cosine/distance, normalized scores, min/max ranges, weights, and selected index.
Use `summarize_ablation.py` for per-run summaries; historical baselines can be read
without rerunning them.
