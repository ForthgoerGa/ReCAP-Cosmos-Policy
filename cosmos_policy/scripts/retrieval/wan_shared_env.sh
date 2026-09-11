#!/usr/bin/env bash
BASE=/mnt4/cyh/ReCAP_qwen_visual_align
REPO="$BASE/repo_wan_20260911"
export PYTHONPATH="$REPO" PYTHONDONTWRITEBYTECODE=1
export PATH="$BASE/envs/ReCAP/bin:$PATH"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy WANDB_MODE=disabled
export HF_HOME="$BASE/hf_cache" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export WAN_VAE_PATH="$BASE/hf_cache/hub/models--nvidia--Cosmos-Predict2.5-2B/snapshots/85f8ae7bfe8f5525c8d103429524dcf12f98bf7b/tokenizer.pth"
export WAN_IMAGE_INDEX_PATH="$BASE/indices/wan_image_flat224_20260911"
export WAN_VIDEO_INDEX_PATH="$BASE/indices/wan_video_flat224_padfirst_20260911"
