#!/usr/bin/env bash
set -euo pipefail
GROUP=${1:?group required}
SEED=${2:?base seed required}
GPU=${3:?physical GPU required}
BASE=/mnt4/cyh/ReCAP_qwen_visual_align
REPO="$BASE/repo_full"
SUITE=${FORMAL_SUITE:-rerank_top30_formal_20260910}
RUN="$BASE/results/$SUITE/$GROUP/seed_$SEED/${ATTEMPT:-attempt1}"
mkdir -p "$RUN"
exec 9>"$RUN/run.lock"
flock -n 9 || exit 75
if [[ -e "$RUN/started.json" ]]; then
  echo "Already submitted: $RUN" >&2
  exit 73
fi
source "$REPO/cosmos_policy/scripts/retrieval/rerank_shared_env.sh"
case "$GROUP" in
  e2_r2|e2_r8|e8_r2|e8_r8) CFG="qwen_${GROUP}_top30.yaml" ;;
  image2b) CFG=qwen_full.yaml ;;
  image8b) CFG=qwen8_full.yaml ;;
  *) echo "Unknown group $GROUP" >&2; exit 2 ;;
esac
cd "$REPO"
printf '{"host":"%s","group":"%s","seed":%s,"physical_gpu":%s,"pid":%s}\n' \
  "$(hostname)" "$GROUP" "$SEED" "$GPU" "$$" > "$RUN/started.json"
set +e
"$BASE/envs/ReCAP/bin/python" -u -m cosmos_policy.scripts.retrieval.run_ablation \
  --retrieval-configs "$REPO/configs/retrieval/$CFG" \
  --data-dir "$BASE/data/PushT-Cosmos-Policy/success_only" \
  --ckpt "$BASE/checkpoints/model_000007000.pt" --output-root "$RUN" \
  --seed "$SEED" --num-trials "${NUM_TRIALS:-10}" --gpu "$GPU" --offline \
  --visual-config tri_default --retrieval-query-visual blue_circle
RC=$?
printf '%s\n' "$RC" > "$RUN/exit_code.txt"
exit "$RC"
