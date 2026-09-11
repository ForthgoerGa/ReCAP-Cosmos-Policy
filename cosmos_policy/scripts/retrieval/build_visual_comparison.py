"""Package paired formal runs and a portable, offline HTML comparison gallery."""
import argparse
from collections import defaultdict
import datetime
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys

import h5py
import numpy as np
from PIL import Image


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_runs(root, strategy, mode):
    selected = {}
    for p in root.rglob("manifest.json"):
        m = json.loads(p.read_text())
        if m.get("config", {}).get("strategy") != strategy:
            continue
        if m.get("status") != "complete" or m.get("exit_code") != 0:
            continue
        if m.get("retrieval_query_visual", "none") != mode:
            continue
        seed = m["seed"]
        if seed not in selected or p.parent.parent.name > selected[seed][0].parent.parent.name:
            selected[seed] = (p, m)
    return selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=Path("/mnt4/cyh/ReCAP_qwen_visual_align"))
    ap.add_argument("--original", type=Path, default=Path("/home/chenyuhong/Projects/ReCAP"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--template", type=Path, required=True)
    ap.add_argument("--partial", action="store_true")
    args = ap.parse_args()
    repo = args.base / "repo_full"
    sys.path[:0] = [str(repo), str(repo / "cosmos_policy/experiments/robot/pusht")]
    from gym_pusht.envs.pusht import PushTEnv
    from cosmos_policy.experiments.robot.pusht_ret.retrievers.query_visual import retrieval_query_frame

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    (output / "assets").mkdir(exist_ok=True)
    env = PushTEnv(obs_type="pixels_agent_pos", render_mode="rgb_array",
                   observation_width=128, observation_height=128)
    data = args.base / "data/PushT-Cosmos-Policy/success_only"
    sources = {
        "image2b": ("2B · 图像", "qwen_full", args.original / "qwen_full_formal50_20260909_final"),
        "image8b": ("8B · 图像", "qwen_full", args.original / "qwen8_formal50_20260910/image"),
        "video2b": ("2B · Dense 视频", "qwen_video", args.base / "results/no_visual_formal_20260910_video2b/video2b"),
        "video8b": ("8B · Dense 视频", "qwen_video", args.original / "qwen8_formal50_20260910/video"),
    }
    corrected = args.base / "results/blue_circle_formal_20260910_v3"
    report = {"built_at": datetime.datetime.now().astimezone().isoformat(), "groups": [],
              "protocol": {"visual_config": "tri_default", "seeds": list(range(42, 92)),
                           "success_threshold": .85, "coverage": "terminal_coverage",
                           "max_steps": 300, "fps": 10}, "validation": {}}
    checksum_records = {}
    all_checkpoint_hashes = set()
    stats_hashes = set()
    query_verified = 0
    query_reconstructed = 0
    paired_initial_verified = 0
    frames_verified = 0

    def asset_path(name):
        return output / "assets" / name

    def decode(entry):
        return np.asarray(Image.open(io.BytesIO(entry.tobytes())).convert("RGB"))

    for group, (title, strategy, before_root) in sources.items():
        result = {"id": group, "title": title, "sides": {}, "pairs": []}
        side_rows = {}
        index_hashes = set()
        for side, root, mode in [("before", before_root, "none"),
                                 ("after", corrected / group, "blue_circle")]:
            runs = load_runs(root, strategy, mode)
            if not args.partial and set(runs) != {42, 52, 62, 72, 82}:
                raise ValueError(f"{group}/{side}: incomplete formal seeds: {sorted(runs)}")
            rows = {}
            for base_seed, (mp, m) in sorted(runs.items()):
                all_checkpoint_hashes.add(m["checkpoint_sha256"])
                stats_hashes.add(json.dumps(m["stats_sha256"], sort_keys=True))
                index = Path(m["config"]["index_path"]) / "manifest.json"
                im = json.loads(index.read_text())
                index_hashes.add(im["embeddings_sha256"])
                index_copy = output / "artifacts" / side / group / "index_manifest.json"
                index_copy.parent.mkdir(parents=True, exist_ok=True)
                if not index_copy.exists():
                    shutil.copy2(index, index_copy)
                assert m["config"]["candidate_k"] == 1
                command = m["command"]
                assert command[command.index("--visual_config") + 1] == "tri_default"
                assert command[command.index("--chunk_size") + 1] == "8"
                assert command[command.index("--num_open_loop_steps") + 1] == "8"
                destination = output / "artifacts" / side / group / f"seed_{base_seed}"
                if not destination.exists():
                    shutil.copytree(mp.parent, destination)
                for name in ("source_checksums.json", "tracked_changes.diff"):
                    provenance = mp.parent.parent / name
                    if provenance.is_file() and not (destination / name).exists():
                        shutil.copy2(provenance, destination / name)
                for f in destination.rglob("*"):
                    if f.is_file():
                        checksum_records[str(f.relative_to(output))] = sha(f)
                trace = defaultdict(list)
                for line in (mp.parent / "retrieval_trace.jsonl").read_text().splitlines():
                    row = json.loads(line)
                    trace[row["seed"]].append(row)
                episodes = [json.loads(line) for line in (mp.parent / "episodes.jsonl").read_text().splitlines()]
                assert len(episodes) == 10 and {e["seed"] for e in episodes} == set(range(base_seed, base_seed + 10))
                for episode in episodes:
                    seed = episode["seed"]
                    if seed in rows:
                        raise ValueError(f"Duplicate seed {group}/{side}/{seed}")
                    records = trace[seed]
                    first = records[0]
                    assert first["step"] == 0 and first.get("retrieval_query_visual", "none") == mode
                    episode_number = seed - base_seed + 1
                    videos = list(destination.rglob(f"ep{episode_number:03d}--*--combined.mp4"))
                    if len(videos) != 1:
                        raise ValueError(f"Missing/ambiguous combined video: {group}/{side}/{seed}")
                    raw = env.reset(seed=seed, options={"agent_shape": "triangle"})[0]["pixels"]
                    query = retrieval_query_frame(env, raw, mode)
                    raw_hash = hashlib.sha256(raw.tobytes()).hexdigest()
                    query_hash = hashlib.sha256(query.tobytes()).hexdigest()
                    if "policy_frame_sha256" in first:
                        assert raw_hash == first["policy_frame_sha256"], (group, side, seed, "policy render mismatch")
                        assert query_hash == first["query_frame_sha256"], (group, side, seed, "query render mismatch")
                        query_verified += 1
                    else:
                        query_reconstructed += 1
                    query_file = asset_path(f"query_{mode}_{seed}.png")
                    if not query_file.exists():
                        Image.fromarray(query).save(query_file)
                    relative_h5, demo, endpoint = first["selected_id"].split("::")
                    endpoint = int(endpoint)
                    with h5py.File(data / relative_h5, "r") as f:
                        images = f[f"data/{demo}/obs/images"]
                        clip = np.stack([decode(images[min(endpoint + i, len(images) - 1)]) for i in range(8)])
                    assert hashlib.sha256(clip.tobytes()).hexdigest() == first["frame_sha256"], (group, side, seed, "retrieval frame mismatch")
                    frames_verified += 1
                    selected_file = asset_path(f"retrieved_{side}_{group}_{seed}.png")
                    if not selected_file.exists():
                        Image.fromarray(clip[0]).save(selected_file)
                    condensed = [{k: t[k] for k in ("step", "selected_id", "qwen_score", "video_frames") if k in t} for t in records]
                    rows[seed] = dict(episode, video=str(videos[0].relative_to(output)),
                                      query=str(query_file.relative_to(output)),
                                      retrieved=str(selected_file.relative_to(output)),
                                      selected_id=first["selected_id"], score=first["qwen_score"],
                                      raw_hash=raw_hash, trace=condensed,
                                      manifest=str((destination / "manifest.json").relative_to(output)))
            side_rows[side] = rows
            result["sides"][side] = {
                "n": len(rows), "successes": sum(e["success"] for e in rows.values()),
                "coverage": sum(e["terminal_coverage"] for e in rows.values()) / len(rows) if rows else None,
            }
        if len(index_hashes) > 1:
            raise ValueError(f"Paired index vectors differ: {group}")
        for seed in sorted(set(side_rows["before"]) | set(side_rows["after"])):
            before, after = side_rows["before"].get(seed), side_rows["after"].get(seed)
            if before and after:
                assert before["raw_hash"] == after["raw_hash"]
                paired_initial_verified += 1
            result["pairs"].append({"seed": seed, "before": before, "after": after})
        complete_pairs = [p for p in result["pairs"] if p["before"] and p["after"]]
        result["changed_first_top1"] = sum(p["before"]["selected_id"] != p["after"]["selected_id"] for p in complete_pairs)
        result["paired_n"] = len(complete_pairs)
        report["groups"].append(result)
        print(title, result["sides"], "paired", result["paired_n"], flush=True)
    env.close()
    assert len(all_checkpoint_hashes) == 1 and len(stats_hashes) == 1
    report["protocol"]["checkpoint_sha256"] = next(iter(all_checkpoint_hashes))
    report["protocol"]["stats_sha256"] = json.loads(next(iter(stats_hashes)))
    report["validation"] = {"same_checkpoint_and_stats": True, "same_index_per_pair": True,
                            "query_hashes_verified": query_verified, "retrieval_clips_verified": frames_verified,
                            "queries_reconstructed_without_trace_hash": query_reconstructed,
                            "paired_seeds_verified": paired_initial_verified}
    payload = json.dumps(report, ensure_ascii=False).replace("</", "<\\/")
    html = args.template.read_text().replace("__REPORT_DATA__", payload)
    (output / "index.html").write_text(html)
    (output / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    lines = ["# 检索视觉修正对照", "", "同一 checkpoint / tri_default / 每组 50 个相同 seed / top1。", "",
             "| 组别 | 未修正成功率 | 修正成功率 | 未修正终局 coverage | 修正终局 coverage |", "|---|---:|---:|---:|---:|"]
    for g in report["groups"]:
        a, b = g["sides"]["before"], g["sides"]["after"]
        rate = lambda x: f'{100*x["successes"]/x["n"]:.1f}%' if x["n"] else "待完成"
        cov = lambda x: f'{100*x["coverage"]:.2f}%' if x["coverage"] is not None else "待完成"
        lines.append(f'| {g["title"]} | {rate(a)} | {rate(b)} | {cov(a)} | {cov(b)} |')
    lines += ["", "双击 index.html 即可离线打开，或在本目录执行 python3 -m http.server 8765 --bind 127.0.0.1。",
              "同 seed 只保证起点一致；后续动作与场景会分叉。首页的首个查询用于比较相同初始场景的 top1 变化。",
              "assets 中查询截图由真实环境重放，已对照可用 trace 的像素哈希；retrieved 图片来自原始数据，8 帧 payload 哈希全部核验。",
              "旧基线没有保存查询像素哈希，其查询图按相同 seed 重建，不计入查询哈希核验数量。",
              "artifacts 保留全部有效评测日志、trace、结果和三类视频；失败启动不计入统计。",
              "2B dense 未修正对照于本次补跑，与修正组使用同一新索引。"]
    (output / "README.md").write_text("\n".join(lines) + "\n")
    for f in output.rglob("*"):
        if f.is_file() and f.name != "checksums.json":
            checksum_records[str(f.relative_to(output))] = sha(f)
    (output / "checksums.json").write_text(json.dumps(checksum_records, indent=2))
    print("GALLERY_READY", output, report["validation"], flush=True)


if __name__ == "__main__":
    main()
