#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/wan_shared_env.sh"
GROUP=${1:?image or video required}
SEED=${2:?seed required}
GPU=${3:?physical GPU required}
TRIALS=${4:-5}
case "$GROUP" in image|video) ;; *) exit 2 ;; esac
RUN="$BASE/results/${WAN_SUITE:-wan_flat_formal_20260911}/$GROUP/seed_$SEED/${ATTEMPT:-attempt1}"
mkdir -p "$RUN"
exec 9>"$RUN/run.lock"
flock -n 9 || exit 75
if [[ -e "$RUN/started.json" ]]; then
  echo "Already submitted: $RUN" >&2
  exit 73
fi
cd "$REPO"
printf '{"host":"%s","group":"%s","seed":%s,"num_trials":%s,"physical_gpu":%s,"pid":%s}\n' \
  "$(hostname)" "$GROUP" "$SEED" "$TRIALS" "$GPU" "$$" > "$RUN/started.json"
set +e
"$BASE/envs/ReCAP/bin/python" -u -m cosmos_policy.scripts.retrieval.run_ablation \
  --retrieval-configs "$REPO/configs/retrieval/wan_vae_$GROUP.yaml" \
  --data-dir "$BASE/data/PushT-Cosmos-Policy/success_only" \
  --ckpt "$BASE/checkpoints/model_000007000.pt" --output-root "$RUN" \
  --seed "$SEED" --num-trials "$TRIALS" --gpu "$GPU" --offline \
  --visual-config tri_default --retrieval-query-visual blue_circle
RC=$?
printf '%s\n' "$RC" > "$RUN/exit_code.txt"
exit "$RC"
