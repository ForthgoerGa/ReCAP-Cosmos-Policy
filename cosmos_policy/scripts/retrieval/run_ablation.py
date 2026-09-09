"""Run isolated paired evaluations; never overwrite an existing experiment."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import threading

import yaml

from cosmos_policy.experiments.robot.pusht_ret.retrievers.config import load_config
from cosmos_policy.experiments.robot.pusht_ret.retrievers.index import sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--retrieval-configs", nargs="+", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-trials", type=int, default=4)
    p.add_argument("--candidate-k", type=int)
    p.add_argument("--gpu", default="1")
    p.add_argument("--visual-config", default="tri_default")
    p.add_argument("--offline", action="store_true", help="Use already cached Cosmos assets without network checks")
    args = p.parse_args()
    if args.num_trials < 1:
        p.error("num-trials must be positive")
    root = Path(args.output_root) / datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    root.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[3]
    data = Path(args.data_dir).resolve()
    ckpt = Path(args.ckpt).resolve()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    diff = subprocess.check_output(["git", "diff"], cwd=repo, text=True)
    (root / "tracked_changes.diff").write_text(diff)
    # Capture untracked implementation as well as tracked modifications.
    files = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=repo, text=True).splitlines()
    code_hashes = {f: sha256(repo / f) for f in files
                   if (repo / f).is_file() and
                   (f.startswith("cosmos_policy/experiments/robot/pusht_ret/")
                    or f.startswith("cosmos_policy/scripts/retrieval/")
                    or f.startswith("configs/retrieval/"))
                   and "__pycache__" not in f}
    (root / "source_checksums.json").write_text(json.dumps(code_hashes, indent=2))
    for idx, config_path in enumerate(args.retrieval_configs):
        cfg = load_config(config_path)
        if args.candidate_k is not None:
            cfg.candidate_k = args.candidate_k
            cfg.validate()
        suffix = "all_top1" if cfg.strategy in ("qwen_full", "qwen_state_text", "qwen_history_state", "qwen_late_fusion") else f"k{cfg.candidate_k}"
        out = root / f"{idx:02d}_{cfg.strategy}_{suffix}"
        out.mkdir()
        resolved = out / "retrieval_config.yaml"
        resolved.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
        cmd = [
            sys.executable, "-u", "-m", "cosmos_policy.experiments.robot.pusht_ret.run_eval",
            "--config", "cosmos_predict2p5_2b_480p_pusht_ret_top100_residual_inference_only",
            "--ckpt_path", str(ckpt), "--config_file", "cosmos_policy/config/config.py",
            "--t5_text_embeddings_path", str(data / "t5_embeddings.pkl"),
            "--dataset_stats_path", str(data / "dataset_statistics.json"),
            "--retrieval_data_dir", str(data), "--use_residual_actions", "True",
            "--delta_stats_path", str(data / "delta_dataset_statistics.json"),
            "--visual_config", args.visual_config, "--seed", str(args.seed),
            "--num_trials", str(args.num_trials), "--chunk_size", "8",
            "--num_open_loop_steps", "8", "--num_denoising_steps_action", "5",
            "--predict_future_states", "True", "--local_log_dir", str(out),
            "--retrieval_strategy", cfg.strategy, "--retrieval_config", str(resolved),
            "--retrieval_trace", "True", "--fail_on_episode_error", "True",
        ]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": args.gpu,
               "SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy",
               "WANDB_MODE": "disabled", "BASE_DATASETS_DIR": str(data.parent.parent)}
        if args.offline:
            env["HF_HUB_OFFLINE"] = "1"
        manifest = {
            "command": cmd, "commit": commit, "config": cfg.to_dict(),
            "seed": args.seed, "num_trials": args.num_trials, "gpu": args.gpu,
            "host": os.uname().nodename, "checkpoint_sha256": sha256(ckpt),
            "stats_sha256": {f: sha256(data / f) for f in
                             ("dataset_statistics.json", "delta_dataset_statistics.json")},
            "status": "running",
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"Running {out}", flush=True)
        started = time.perf_counter()
        stop = threading.Event()
        gpu_samples = []
        def monitor_gpu():
            while not stop.is_set():
                try:
                    value = subprocess.check_output(
                        ["nvidia-smi", "-i", args.gpu, "--query-gpu=memory.used",
                         "--format=csv,noheader,nounits"], text=True, timeout=5)
                    gpu_samples.append(float(value.strip()))
                except (OSError, ValueError, subprocess.SubprocessError):
                    pass
                stop.wait(1)
        monitor = threading.Thread(target=monitor_gpu, daemon=True)
        monitor.start()
        try:
            with open(out / "console.log", "x") as log:
                result = subprocess.run(cmd, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
        finally:
            stop.set()
            monitor.join(timeout=6)
        manifest["sampled_gpu_peak_used_mib"] = max(gpu_samples, default=None)
        manifest.update(status="complete" if result.returncode == 0 else "error",
                        exit_code=result.returncode, elapsed_seconds=time.perf_counter() - started)
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        if result.returncode:
            raise SystemExit(f"Evaluation failed, inspect {out / 'console.log'}")
    print(root, flush=True)


if __name__ == "__main__":
    main()
