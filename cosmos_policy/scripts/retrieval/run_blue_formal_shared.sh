#!/usr/bin/env bash
set -euo pipefail
# One invocation runs one formal 10-episode seed on one physical GPU.
GROUP=${1:?group required}
SEED=${2:?seed required}
GPU=${3:?physical GPU required}
BASE=/mnt4/cyh/ReCAP_qwen_visual_align
REPO="$BASE/repo_full"
QUERY_VISUAL=${RETRIEVAL_QUERY_VISUAL:-blue_circle}
SUITE=${FORMAL_SUITE:-blue_circle_formal_20260910_v3}
RUN="$BASE/results/$SUITE/$GROUP/seed_$SEED/${ATTEMPT:-attempt1}"
mkdir -p "$RUN"
exec 9>"$RUN/run.lock"
flock -n 9 || exit 75
if [[ -e "$RUN/started.json" ]]; then
  echo "Seed already submitted: $RUN" >&2
  exit 73
fi
export PYTHONPATH="$REPO" PYTHONDONTWRITEBYTECODE=1
export PATH="$BASE/envs/ReCAP/bin:$PATH"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy WANDB_MODE=disabled
export HF_HOME="$BASE/hf_cache" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export QWEN_PYTHON="$BASE/envs/ReCAP-Qwen/bin/python"
export QWEN_MODEL_PATH="$BASE/models/Qwen3-VL-Embedding-2B-9f2f7e7"
export QWEN_MODEL_REVISION=9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda
export QWEN_INDEX_PATH="$BASE/indices/full_2b_image_v1"
export QWEN_VIDEO_INDEX_PATH="$BASE/indices/video_2b_matched"
export QWEN8_MODEL_PATH="$BASE/models/Qwen3-VL-Embedding-8B-2c4565"
export QWEN8_INDEX_PATH="$BASE/indices/image_index" QWEN8_VIDEO_INDEX_PATH="$BASE/indices/video_index"
case "$GROUP" in
  image2b) CFG=qwen_full.yaml ;;
  image8b) CFG=qwen8_full.yaml ;;
  video8b) CFG=qwen8_video.yaml ;;
  video2b) CFG=qwen_video.yaml ;;
  *) echo "Unknown group $GROUP" >&2; exit 2 ;;
esac
cd "$REPO"
printf '{"host":"%s","group":"%s","seed":%s,"physical_gpu":%s,"pid":%s}\n' \
  "$(hostname)" "$GROUP" "$SEED" "$GPU" "$$" > "$RUN/started.json"
# run_ablation sets CUDA_VISIBLE_DEVICES itself: pass the physical GPU here.
set +e
"$BASE/envs/ReCAP/bin/python" -u -m cosmos_policy.scripts.retrieval.run_ablation \
  --retrieval-configs "$REPO/configs/retrieval/$CFG" \
  --data-dir "$BASE/data/PushT-Cosmos-Policy/success_only" \
  --ckpt "$BASE/checkpoints/model_000007000.pt" --output-root "$RUN" \
  --seed "$SEED" --num-trials 10 --gpu "$GPU" --offline \
  --visual-config tri_default --retrieval-query-visual "$QUERY_VISUAL"
RC=$?
printf '%s\n' "$RC" > "$RUN/exit_code.txt"
exit "$RC"
