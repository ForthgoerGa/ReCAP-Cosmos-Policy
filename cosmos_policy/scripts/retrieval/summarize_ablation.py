"""Summarize evaluation and trace evidence; errors remain errors."""
import argparse
import json
from pathlib import Path
import re
import numpy as np


def summarize(root):
    results = []
    for path in sorted(root.rglob("manifest.json")):
        manifest = json.loads(path.read_text())
        if "command" not in manifest:
            continue
        out = path.parent
        log = (out / "console.log").read_text()
        trace_path = out / "retrieval_trace.jsonl"
        rows = [json.loads(line) for line in trace_path.read_text().splitlines()] if trace_path.exists() else []
        episodes = []
        current = []
        for line in log.splitlines():
            if line.startswith("INFO:"):
                continue
            match = re.search(r"t=\s*(\d+)\s+coverage=([\d.]+)", line)
            if match:
                current.append(float(match[2]))
            match = re.search(r"Episode (\d+): (SUCCESS|FAIL)", line)
            if match:
                episodes.append({"seed": manifest["seed"] + int(match[1]) - 1,
                                 "success": match[2] == "SUCCESS",
                                 "terminal_coverage": current[-1] if current else None,
                                 "steps": len(current)})
                current = []
        if (out / "episodes.jsonl").exists():
            episodes = [json.loads(line) for line in (out / "episodes.jsonl").read_text().splitlines()]
        timings = [r["total_retrieval_seconds"] for r in rows]
        ranks = [r["selected_state_rank"] for r in rows if r.get("selected_state_rank") is not None]
        results.append({
            "directory": str(out), "status": manifest["status"], "episodes": episodes,
            "success_rate": sum(e["success"] for e in episodes) / len(episodes) if episodes else None,
            "mean_terminal_coverage": float(np.mean([e["terminal_coverage"] for e in episodes])) if episodes else None,
            "retrieval_queries": len(rows),
            "changed_top1_fraction": float(np.mean(np.array(ranks) > 1)) if ranks else None,
            "retrieval_latency_p50_seconds": float(np.percentile(timings, 50)) if timings else None,
            "retrieval_latency_p95_seconds": float(np.percentile(timings, 95)) if timings else None,
            "qwen_peak_allocated_bytes": max([r.get("peak_allocated_bytes", 0) for r in rows], default=0),
            "sampled_gpu_peak_used_mib": manifest.get("sampled_gpu_peak_used_mib"),
            "episode_errors": log.count("Episode error:"),
            "combined_videos": [str(p) for p in sorted(out.rglob("*combined.mp4"))],
        })
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    results = summarize(args.root)
    with open(args.root / "summary.json", "x") as f:
        json.dump(results, f, indent=2)
    lines = ["# Retrieval smoke evaluation", "",
             "Small paired smoke runs validate integration, not statistical improvement.", ""]
    for r in results:
        lines.extend([f"## {Path(r['directory']).name}", "",
                      f"Status: {r['status']}; success: {r['success_rate']}; terminal coverage: {r['mean_terminal_coverage']}",
                      f"Queries: {r['retrieval_queries']}; changed state Top-1: {r['changed_top1_fraction']}",
                      f"Retrieval p50/p95: {r['retrieval_latency_p50_seconds']}/{r['retrieval_latency_p95_seconds']} seconds", ""])
        for video in r["combined_videos"]:
            lines.append(f"- [{Path(video).name}]({Path(video).relative_to(args.root)})")
        lines.append("")
    with open(args.root / "REPORT.md", "x") as f:
        f.write("\n".join(lines))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
