#!/usr/bin/env bash
set -euo pipefail
BASE=/mnt4/cyh/ReCAP_qwen_visual_align
REPO="$BASE/repo_full"
SHARDS="$BASE/indices/video_2b_matched_shards_v2"
mkdir -p "$SHARDS"
exec 9>"$SHARDS/build.lock"
flock -n 9 || exit 75
export PYTHONPATH="$REPO" PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export QWEN_MODEL_PATH="$BASE/models/Qwen3-VL-Embedding-2B-9f2f7e7"
export QWEN_MODEL_REVISION=9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda
export QWEN_PYTHON="$BASE/envs/ReCAP-Qwen/bin/python"
export QWEN_VIDEO_INDEX_PATH="$BASE/indices/video_2b_matched"
cd "$REPO"
pids=()
for gpu in 0 1 2 3 4 5 6 7; do
  start=$((24397*gpu/8)); end=$((24397*(gpu+1)/8))
  CUDA_VISIBLE_DEVICES="$gpu" "$BASE/envs/ReCAP/bin/python" -u \
    -m cosmos_policy.scripts.retrieval.build_qwen_video_index \
    --retrieval-config configs/retrieval/qwen_video.yaml \
    --data-dir "$BASE/data/PushT-Cosmos-Policy/success_only" \
    --start "$start" --end "$end" --shard-out "$SHARDS/shard_$gpu.npy" \
    > "$SHARDS/shard_$gpu.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if ((failed)); then echo 'One or more shards failed; merge skipped'; exit 1; fi
"$BASE/envs/ReCAP/bin/python" -u -m cosmos_policy.scripts.retrieval.build_qwen_video_index \
  --retrieval-config configs/retrieval/qwen_video.yaml \
  --data-dir "$BASE/data/PushT-Cosmos-Policy/success_only" --merge-shards "$SHARDS"
