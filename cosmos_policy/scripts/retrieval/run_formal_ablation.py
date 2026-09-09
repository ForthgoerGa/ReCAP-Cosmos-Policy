"""Partition paired seed ranges over explicit GPUs, retaining per-mode run manifests."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import json
from pathlib import Path
import subprocess
import sys


def partition_seeds(seed, trials, gpus):
    if trials < 1 or not gpus or len(set(gpus)) != len(gpus):
        raise ValueError("Need positive trials and distinct GPUs")
    base, extra = divmod(trials, len(gpus))
    result = []
    for i, gpu in enumerate(gpus):
        count = base + (i < extra)
        if count:
            result.append((gpu, seed, count))
            seed += count
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-configs", nargs="+", required=True)
    parser.add_argument("--gpus", nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-trials", type=int, default=50, help="Total trials per strategy")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--visual-config", default="tri_default")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    partitions = partition_seeds(args.seed, args.num_trials, args.gpus)
    root = args.output_root / datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    root.mkdir(parents=True, exist_ok=False)
    (root / "launcher_config.json").write_text(json.dumps({
        **vars(args), "output_root": str(args.output_root), "partitions": partitions,
    }, indent=2))
    print(f"Formal ablation root: {root}", flush=True)
    def run(partition):
        gpu, seed, count = partition
        command = [sys.executable, "-u", "-m", "cosmos_policy.scripts.retrieval.run_ablation",
                   "--retrieval-configs", *args.retrieval_configs, "--seed", str(seed),
                   "--num-trials", str(count), "--gpu", gpu, "--data-dir", args.data_dir,
                   "--ckpt", args.ckpt, "--output-root", str(root / f"shard_{seed}"),
                   "--visual-config", args.visual_config]
        if args.offline:
            command.append("--offline")
        with (root / f"shard_{seed}.log").open("x") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        return {"gpu": gpu, "seed": seed, "count": count, "command": command, "exit_code": result.returncode}
    with ThreadPoolExecutor(max_workers=len(partitions)) as pool:
        results = list(pool.map(run, partitions))
    (root / "shard_status.json").write_text(json.dumps(results, indent=2))
    failed = any(r["exit_code"] for r in results)
    (root / "exit_code.txt").write_text("1\n" if failed else "0\n")
    if failed:
        raise SystemExit(f"One or more shards failed: {root}")
    print(f"All strategies and seeds complete: {root}", flush=True)


if __name__ == "__main__":
    main()
