"""Read formal evaluation progress without modifying runs or requiring CUDA."""
import argparse
import datetime
import json
from pathlib import Path


def json_lines(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # The writer may still be appending the last record.
            continue
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(
        "/mnt4/cyh/ReCAP_qwen_visual_align/results/blue_circle_formal_20260910_v3"))
    args = parser.parse_args()
    report = {"checked_at": datetime.datetime.now().astimezone().isoformat(), "groups": {}}
    for group in ("image2b", "image8b", "video8b", "video2b"):
        if not (args.root / group).exists():
            continue
        runs, episodes = [], []
        for seed in (42, 52, 62, 72, 82):
            candidates = sorted((args.root / group / f"seed_{seed}").rglob("manifest.json"),
                                key=lambda path: path.parent.parent.name)
            if not candidates:
                runs.append({"seed": seed, "status": "not_started"})
                continue
            # Timestamp directory names remain ordered after later completion writes.
            manifest_path = candidates[-1]
            manifest = json.loads(manifest_path.read_text())
            output = manifest_path.parent
            done = json_lines(output / "episodes.jsonl")
            traces = json_lines(output / "retrieval_trace.jsonl")
            episodes.extend(done)
            row = {"seed": seed, "status": manifest["status"], "host": manifest["host"],
                   "gpu": manifest["gpu"], "episodes_complete": len(done),
                   "retrieval_calls": len(traces), "output": str(output)}
            if traces:
                last = traces[-1]
                row.update(current_episode=last["episode"] + 1, current_step=last["step"],
                           query_visual=last["retrieval_query_visual"],
                           candidate_count=last["candidate_count"],
                           query_differs_from_policy=(last["policy_frame_sha256"] != last["query_frame_sha256"]))
            if manifest["status"] == "error":
                row["error_tail"] = (output / "console.log").read_text()[-1200:]
            runs.append(row)
        report["groups"][group] = {
            "completed": len(episodes), "target": 50,
            "successes_so_far": sum(bool(row["success"]) for row in episodes),
            "mean_terminal_coverage_so_far": (
                sum(row["terminal_coverage"] for row in episodes) / len(episodes) if episodes else None),
            "runs": runs,
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
