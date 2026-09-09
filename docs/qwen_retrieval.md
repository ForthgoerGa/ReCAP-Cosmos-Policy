# Qwen image reranking for PushT

Status: v1 uses frozen Qwen3-VL-Embedding-2B and the existing frozen ReCAP checkpoint.
No temporal continuity, video embedding, extra reranker model, or training is introduced.

## Data flow

`run_eval.py` selects `QwenRerankRetrieval` only for `retrieval_strategy=qwen_rerank`.

1. `retrieval.py::get_state_candidates` uses the unchanged initial-position top-100-demo filter and 10D state distance, returning the best K candidate timestamps.
2. `qwen_client.py` sends current uint8 RGB to a persistent local `qwen_worker.py`, launched with a separate Python interpreter. PNG IPC is lossless.
3. `qwen_encoder.py` encodes full-frame RGB resized to 224x224 with fixed instruction. Query and index use the same implementation. BF16 model output is normalized in FP32.
4. `qwen_rerank.py` ranks the K cached candidate vectors by cosine similarity. Ties retain state ranking.
5. `retrieval.py::get_candidate_data` returns original images/actions for s..s+7 and proprio at s. Tail padding repeats the last sample.
6. Existing `cosmos_utils.py::get_action` and residual unnormalization remain unchanged.

Qwen sees current/matched images, not future query observations. Its vectors do not become policy tokens.
K counts timestamps, not distinct demos. This strategy still uses object GT for candidate generation.
Similarity improvements do not automatically imply better task success.

## Files and responsibilities

| Module | Caller | Responsibility |
|---|---|---|
| `retrievers/config.py` | builder, backend, runner | shared validated YAML |
| `retrievers/index.py` | builder, backend | source identities, data checksums, vector validation |
| `retrievers/qwen_encoder.py` | worker | strict visual preprocessing and embedding |
| `retrievers/qwen_worker.py` | client | isolated Qwen runtime over JSON lines |
| `retrievers/qwen_client.py` | builder, backend | process lifecycle, timeouts, lossless RGB IPC |
| `retrievers/qwen_rerank.py` | existing eval | state candidates, cosine ranking, original payload |
| `scripts/retrieval/build_qwen_index.py` | CLI | offline candidate image index |
| `scripts/retrieval/run_ablation.py` | CLI | separate immutable run directories, subprocess evaluation |
| `scripts/retrieval/summarize_ablation.py` | CLI | results, latency, selection changes, video links |

`retrievers/` is under `cosmos_policy/experiments/robot/pusht_ret/`.
`scripts/retrieval/` is under `cosmos_policy/`.
Configs are `configs/retrieval/{state,qwen_rerank}.yaml`.

The standard backend needs no Qwen installation or index. Legacy consistent/cumulative strategies remain available.
Training still uses `pusht_dataset_ret.py` and its NPZ lookup; v1 does not change that path.

## Environment and setup

Run policy/index orchestration from the existing ReCap environment. Keep Qwen in a user-owned separate environment:

```bash
python -m pip install -r configs/retrieval/requirements-qwen.txt
```

The official encoder source is vendored with its license and pinned commit in
`retrievers/_vendor/SOURCE.md`. The wrapper overrides upstream visual-error-to-NULL
fallback: visual errors are fatal. SDPA is used so no extra FlashAttention build is needed.

Download Qwen/Qwen3-VL-Embedding-2B at revision
`9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda`.
Model safetensors SHA256:
`c73fa9caeddeb3ff831d46c085a7a5708343248ca777e90f2d486964464509c1`.

From the repo root, set paths for the local compute machine:

```bash
export QWEN_MODEL_PATH=/path/to/local/Qwen3-VL-Embedding-2B
export QWEN_MODEL_REVISION=9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda
export QWEN_PYTHON=/path/to/ReCap-Qwen/bin/python
export QWEN_INDEX_PATH=/path/to/new/index_directory
export DATA=/path/to/PushT-Cosmos-Policy/success_only
export CKPT=/path/to/model_000007000.pt
```

Build the index once in ReCap:

```bash
CUDA_VISIBLE_DEVICES=1 python -m cosmos_policy.scripts.retrieval.build_qwen_index \
  --retrieval-config configs/retrieval/qwen_rerank.yaml --data-dir "$DATA" \
  --split base_0,base_1,base_2,base_3,base_4
```

The output directory must not exist. A failed build retains partial outputs; choose a new
directory for the next build. An index is complete only when its manifest exists.
Metadata includes relative HDF5 file/demo/timestamp IDs, full data checksums, encoder configuration,
source checksum, model file checksums, runtime versions, and vector checksum.
Changes to data, preprocessing, model, or runtime versions require a new index.

## Experiments

Check K=1 equivalence at seed 45:

```bash
python -m cosmos_policy.scripts.retrieval.run_ablation \
  --retrieval-configs configs/retrieval/state.yaml configs/retrieval/qwen_rerank.yaml \
  --candidate-k 1 --seed 45 --num-trials 1 --gpu 1 \
  --data-dir "$DATA" --ckpt "$CKPT" --output-root /path/to/equivalence_runs
```

First paired smoke test, seeds 42..45:

```bash
python -m cosmos_policy.scripts.retrieval.run_ablation \
  --retrieval-configs configs/retrieval/state.yaml configs/retrieval/qwen_rerank.yaml \
  --seed 42 --num-trials 4 --gpu 1 \
  --data-dir "$DATA" --ckpt "$CKPT" --output-root /path/to/smoke_runs
```

Summarize once per run root:

```bash
python -m cosmos_policy.scripts.retrieval.summarize_ablation --root /path/to/run_root
python -m pytest tests/retrieval -q
```

Each run records resolved config, command, source checksums, checkpoint/stats checksums,
host/GPU, exit status, console, exact episode metrics, retrieval trace and videos.
Retrieval trace includes all candidate IDs, state distances, cosine scores, selected state rank,
retrieved payload hash, raw retrieved actions/proprio and final policy actions.
Reported sampled GPU memory covers all processes on the selected GPU; use an idle GPU.
Qwen peak allocated bytes cover the worker's PyTorch allocations.

Fatal index/worker/encoding errors invalidate the run, with a nonzero exit status; there is no
automatic state-only fallback. The smoke launcher also makes other episode exceptions fatal.

## New multimodal ablations

Three additional independent backends are available; their full design and usage are in [qwen_multimodal_retrieval.md](qwen_multimodal_retrieval.md):

- `qwen_state_text`: single RGB image plus canonical simulator state text in one Qwen input.
- `qwen_history_state`: causal t-7..t joint image/state embeddings, weighted 1..8 and pooled.
- `qwen_late_fusion`: image-only cosine and full-pool geometric state score fused with YAML `alpha`/`beta`.

The old backends remain available. Joint modes use `build_qwen_multimodal_index`; late fusion reuses the image-only index.

## Future ablations

Vary `candidate_k` through YAML or the launcher override, while reusing the same index.
Full-pool visual top-1 is available as `qwen_full`; see [its guide](qwen_full_retrieval.md). Future score fusion should use a separate backend/config name.
Video embedding requires a separately versioned historical-window index and causal query history.
Neither temporal penalties nor smoothing is present in v1.


## 实验边界与后续扩展

首轮只验证“状态 Top-K → Qwen 图像余弦重排 → 原 policy”能否跑通。
不加入连续性约束，不改变检索周期，不训练模型，也不修改原残差统计。

需要调整的首要参数是 YAML 的 candidate_k，默认 32。
模型路径、固定版本、worker Python、索引路径通过运行环境变量指定。
编码指令或预处理改变后必须重新建索引，不能复用旧向量。

纯图像全池检索已作为独立 qwen_full 策略接入，见 qwen_full_retrieval.md。
后续融合分数、历史视频检索分别建立新策略与配置；
它们继续复用候选来源映射、实验入口和日志汇总，不通过复制整个 eval 脚本实现。
