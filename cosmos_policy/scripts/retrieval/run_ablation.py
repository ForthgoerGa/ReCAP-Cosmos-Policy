"""Run isolated paired evaluations; never overwrite an existing experiment."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import yaml

from cosmos_policy.experiments.robot.pusht_ret.retrievers.config import load_config
from cosmos_policy.experiments.robot.pusht_ret.retrievers.index import sha256
from cosmos_policy.experiments.robot.pusht_ret.retrievers.query_visual import QUERY_VISUAL_MODES
from cosmos_policy.scripts.retrieval.resource_monitor import ResourceMonitor


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
    p.add_argument("--retrieval-query-visual", choices=QUERY_VISUAL_MODES, default="none")
    p.add_argument("--offline", action="store_true", help="Use already cached Cosmos assets without network checks")
    args = p.parse_args()
    if args.num_trials < 1:
        p.error("num-trials must be positive")
    root = Path(args.output_root) / datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    root.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[3]
    data = Path(args.data_dir).resolve()
    ckpt = Path(args.ckpt).resolve()
    # FileLock writes beside the cache. Remote accounts can have different UIDs,
    # so give each run its own byte-identical cache and lock file.
    t5_cache = root / "t5_embeddings.pkl"
    shutil.copyfile(data / "t5_embeddings.pkl", t5_cache)
    t5_cache_sha256 = sha256(t5_cache)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        diff = subprocess.check_output(["git", "diff"], cwd=repo, text=True)
        files = subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=repo, text=True).splitlines()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "no_git_metadata"
        diff = ""
        files = []
        for prefix in ("cosmos_policy/experiments/robot/pusht_ret",
                       "cosmos_policy/experiments/robot/pusht/gym_pusht",
                       "cosmos_policy/scripts/retrieval", "configs/retrieval"):
            base = repo / prefix
            if base.exists():
                files.extend(str(f.relative_to(repo)) for f in base.rglob("*") if f.is_file())
    (root / "tracked_changes.diff").write_text(diff)
    code_hashes = {f: sha256(repo / f) for f in files
                   if (repo / f).is_file() and
                   (f.startswith("cosmos_policy/experiments/robot/pusht_ret/")
                    or f.startswith("cosmos_policy/experiments/robot/pusht/gym_pusht/")
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
            "--t5_text_embeddings_path", str(t5_cache),
            "--dataset_stats_path", str(data / "dataset_statistics.json"),
            "--retrieval_data_dir", str(data), "--use_residual_actions", "True",
            "--delta_stats_path", str(data / "delta_dataset_statistics.json"),
            "--visual_config", args.visual_config, "--seed", str(args.seed),
            "--num_trials", str(args.num_trials), "--chunk_size", "8",
            "--num_open_loop_steps", "8", "--num_denoising_steps_action", "5",
            "--predict_future_states", "True", "--local_log_dir", str(out),
            "--retrieval_strategy", cfg.strategy, "--retrieval_config", str(resolved),
            "--retrieval_trace", "True", "--fail_on_episode_error", "True",
            "--retrieval_query_visual", args.retrieval_query_visual,
        ]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": args.gpu,
               "SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy",
               "WANDB_MODE": "disabled", "BASE_DATASETS_DIR": str(data.parent.parent)}
        env['RECAP_RESOURCE_FILE'] = str(out / 'retrieval_resources.json')
        if args.offline:
            env["HF_HUB_OFFLINE"] = "1"
        manifest = {
            "command": cmd, "commit": commit, "config": cfg.to_dict(),
            "seed": args.seed, "num_trials": args.num_trials, "gpu": args.gpu,
            "host": os.uname().nodename, "checkpoint_sha256": sha256(ckpt),
            "t5_embeddings_sha256": t5_cache_sha256,
            "stats_sha256": {f: sha256(data / f) for f in
                             ("dataset_statistics.json", "delta_dataset_statistics.json")},
            "status": "running",
            "retrieval_query_visual": args.retrieval_query_visual,
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"Running {out}", flush=True)
        started = time.perf_counter()
        monitor = ResourceMonitor(args.gpu, out)
        monitor.start()
        try:
            with open(out / "console.log", "x") as log:
                result = subprocess.run(cmd, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
        finally:
            resource_metrics = monitor.finish()
        manifest['resource_metrics'] = resource_metrics
        manifest["sampled_gpu_peak_used_mib"] = resource_metrics['gpu_peak_used_mib']
        manifest.update(status="complete" if result.returncode == 0 else "error",
                        exit_code=result.returncode, elapsed_seconds=time.perf_counter() - started)
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        if result.returncode:
            raise SystemExit(f"Evaluation failed, inspect {out / 'console.log'}")
    print(root, flush=True)


if __name__ == "__main__":
    main()
