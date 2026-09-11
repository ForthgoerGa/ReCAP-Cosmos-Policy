# Source this file before formal evaluation or the matched-query benchmark.
BASE=/mnt4/cyh/ReCAP_qwen_visual_align
REPO="$BASE/repo_full"
export PYTHONPATH="$REPO" PYTHONDONTWRITEBYTECODE=1
export PATH="$BASE/envs/ReCAP/bin:$PATH"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy WANDB_MODE=disabled
export HF_HOME="$BASE/hf_cache" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export QWEN_PYTHON="$BASE/envs/ReCAP-Qwen/bin/python"
export QWEN_MODEL_PATH="$BASE/models/Qwen3-VL-Embedding-2B-9f2f7e7"
export QWEN_MODEL_REVISION=9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda
export QWEN_INDEX_PATH="$BASE/indices/full_2b_image_v1"
export QWEN8_MODEL_PATH="$BASE/models/Qwen3-VL-Embedding-8B-2c4565"
export QWEN8_INDEX_PATH="$BASE/indices/image_index"
export RERANK2_MODEL_PATH="$BASE/models/Qwen3-VL-Reranker-2B-4bd860a"
export RERANK8_MODEL_PATH="$BASE/models/Qwen3-VL-Reranker-8B-b212dc8"
