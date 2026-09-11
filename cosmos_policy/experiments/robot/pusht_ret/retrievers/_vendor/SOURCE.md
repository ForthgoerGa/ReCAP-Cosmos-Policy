Vendored from https://github.com/QwenLM/Qwen3-VL-Embedding
Commit: 393e2978d27852b0d0230d6994f37f9c15bed73c
License: Apache-2.0 (see LICENSE). Encoder source unmodified.
Strict image preprocessing is implemented in ../qwen_encoder.py.

## Qwen3-VL reranker

Source: https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B/blob/4bd860ac4f15ad1897a214615cccc700f8f71818/scripts/qwen3_vl_reranker.py

License: Apache-2.0 (model repository). Vendored verbatim; the strict batched adapter lives in `qwen_pair_encoder.py`. It retains the upstream image-pair prompt and yes/no head, rejects visual preprocessing errors, disables KV caching, projects the final score in FP32, and ranks logits before sigmoid saturation.
