"""Compare one new 50-seed evaluation to the preserved original run."""
import argparse
import csv
import hashlib
import html
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    root, baseline = args.root, args.baseline
    out = root / "comparison"
    out.mkdir(exist_ok=False)
    old = list(csv.DictReader((baseline / "episodes.csv").open()))
    assert len(old) == 50 and [int(e["seed"]) for e in old] == list(range(42, 92))
    old = {int(e["seed"]): e for e in old}
    new, traces, manifests = {}, [], []
    expected_ckpt = (baseline / "checkpoint.sha256").read_text().split()[0]
    source_fingerprints = []
    for mp in sorted(root.glob("shard_*/*/*/manifest.json")):
        manifest = json.loads(mp.read_text())
        assert manifest["status"] == "complete", mp
        assert manifest["num_trials"] == 10
        assert manifest["checkpoint_sha256"] == expected_ckpt
        assert manifest["config"]["candidate_k"] == 32
        manifests.append(manifest)
        source_fingerprints.append(json.loads((mp.parent.parent / "source_checksums.json").read_text()))
        episodes = [json.loads(s) for s in (mp.parent / "episodes.jsonl").read_text().splitlines()]
        assert len(episodes) == 10
        for i, episode in enumerate(episodes, 1):
            seed = episode["seed"]
            assert seed not in new
            video = next(mp.parent.rglob(f"ep{i:03d}--*--combined.mp4"))
            new[seed] = dict(episode, video=video)
        traces += [json.loads(s) for s in (mp.parent / "retrieval_trace.jsonl").read_text().splitlines()]
    assert len(manifests) == 5 and sorted(new) == list(range(42, 92))
    assert all(x == source_fingerprints[0] for x in source_fingerprints)
    assert all(m["stats_sha256"] == manifests[0]["stats_sha256"] for m in manifests)
    for name, checksum in manifests[0]["stats_sha256"].items():
        cmd = manifests[0]["command"]
        data = Path(cmd[cmd.index("--retrieval_data_dir") + 1])
        assert hashlib.sha256((data / name).read_bytes()).hexdigest() == checksum
    videos = out / "videos"
    videos.mkdir()
    rows = []
    for seed, episode in sorted(new.items()):
        prev = old[seed]
        prev_success = prev["success"] == "True"
        category = ("both_success" if prev_success and episode["success"] else
                    "regression" if prev_success else
                    "improvement" if episode["success"] else "both_fail")
        prev_video = next(baseline.rglob(f"ep{int(prev['episode']):03d}--*--combined.mp4"))
        for label, source in [("state", prev_video), ("qwen", episode["video"])]:
            shutil.copy2(source, videos / f"seed{seed}_{label}.mp4")
        rows.append(dict(seed=seed, state_success=prev_success, qwen_success=episode["success"],
                         state_coverage=float(prev["terminal_coverage_rounded"]),
                         qwen_coverage=episode["terminal_coverage"],
                         state_steps=int(prev["steps"]), qwen_steps=episode["steps"],
                         category=category))
    with (out / "episodes.csv").open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts = {name: sum(r["category"] == name for r in rows)
              for name in ("both_success", "improvement", "regression", "both_fail")}
    latency = np.array([r["total_retrieval_seconds"] for r in traces])
    summary = dict(trials=50, seeds=[42, 91], state_successes=sum(r["state_success"] for r in rows),
                   qwen_successes=sum(r["qwen_success"] for r in rows),
                   state_mean_coverage_rounded=float(np.mean([r["state_coverage"] for r in rows])),
                   qwen_mean_coverage=float(np.mean([r["qwen_coverage"] for r in rows])),
                   paired_outcomes=counts,
                   queries=len(traces), changed_top1=sum(r["selected_state_rank"] > 1 for r in traces),
                   retrieval_p50_seconds=float(np.percentile(latency, 50)),
                   retrieval_p95_seconds=float(np.percentile(latency, 95)),
                   gpu_peak_used_mib=max(m["sampled_gpu_peak_used_mib"] for m in manifests),
                   baseline=str(baseline), host="a2 / lyg2150", gpus=[1, 2, 3, 4, 5],
                   baseline_rerun=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "manifests.json").write_text(json.dumps(manifests, indent=2))
    # Full sequential decoding of every newly generated raw/combined/future video.
    paths = sorted(root.glob("shard_*/*/*/**/videos/*.mp4"))
    assert len(paths) == 150, len(paths)
    def validate(p):
        reader = imageio.get_reader(p)
        count = sum(1 for _ in reader)
        reader.close()
        assert count > 0
        return {"path": str(p.relative_to(root)), "frames": count,
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    with ThreadPoolExecutor(max_workers=4) as pool:
        validation = list(pool.map(validate, paths))
    (out / "video_validation.json").write_text(json.dumps(validation, indent=2))
    lines = ["# Qwen Top-32 正式 50 次消融", "",
             "原版直接复用已有评估；本轮只运行新版。相同 seeds 42–91、tri_default、checkpoint、示教池、8 步动作、5 步去噪。", "",
             "| 方案 | 成功次数 | 成功率 | 平均终态 coverage |",
             "|---|---:|---:|---:|",
             f"| 原状态 | {summary['state_successes']}/50 | {summary['state_successes']*2:.1f}% | {summary['state_mean_coverage_rounded']:.1%} |",
             f"| Qwen Top-32 | {summary['qwen_successes']}/50 | {summary['qwen_successes']*2:.1f}% | {summary['qwen_mean_coverage']:.1%} |", "",
             f"成功率差：{(summary['qwen_successes']-summary['state_successes'])*2:+.1f} 个百分点。",
             f"成对结果：共同成功 {counts['both_success']}；原失败→新成功 {counts['improvement']}；原成功→新失败 {counts['regression']}；共同失败 {counts['both_fail']}。", "",
             f"检索 {len(traces)} 次，改变原 Top-1 {summary['changed_top1']} 次；检索 p50/p95 {summary['retrieval_p50_seconds']*1000:.1f}/{summary['retrieval_p95_seconds']*1000:.1f} ms。",
             f"a2（lyg2150）GPU 1–5 各跑 10 个种子；采样显存峰值 {summary['gpu_peak_used_mib']:.0f} MiB。",
             "新生成 150 个视频全部完整解码通过。原版 coverage 来自三位小数日志，新版来自精确 episode 记录。",
             "性能计时受五卡同时运行时的共享 CPU/I/O 影响；成功率比较使用固定种子的成对结果。",
             "未增加连续性约束、未训练模型、未调整原 checkpoint 的残差统计。", "",
             "[查看逐种子视频对比](index.html)", ""]
    (out / "REPORT.md").write_text("\n".join(lines))
    page = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>Qwen 50 次消融</title>',
            '<style>body{font-family:sans-serif;max-width:1150px;margin:30px auto;background:#f6f7fa;color:#172235}section{background:white;padding:15px;margin:15px 0;border-radius:10px}.row{display:flex;gap:18px;flex-wrap:wrap}.card{flex:1;min-width:390px}video{width:100%}button{margin:8px;padding:8px}</style>',
            '<h1>Qwen Top-32：正式 50 次消融</h1>',
            f'<p>原状态 {summary["state_successes"]}/50；Qwen {summary["qwen_successes"]}/50。原版复用已有结果。</p>',
            '<p><a href="REPORT.md">报告</a> · <a href="episodes.csv">逐种子 CSV</a></p>',
            '<div>' + ''.join(f'<button onclick="filterRows(\'{key}\')">{label}</button>' for key,label in
                             [('all','全部'),('improvement','改善'),('regression','退化'),('both_success','共同成功'),('both_fail','共同失败')]) + '</div>']
    for r in rows:
        page.append(f'<section data-category="{r["category"]}"><h2>Seed {r["seed"]} · {r["category"]}</h2><div class="row">')
        for key,label in [('state','原状态'),('qwen','Qwen')]:
            page.append(f'<div class="card"><h3>{label}：{"成功" if r[key+"_success"] else "失败"}，coverage {r[key+"_coverage"]:.1%}</h3>'
                        f'<video controls preload="none" src="videos/seed{r["seed"]}_{key}.mp4"></video></div>')
        page.append('</div></section>')
    page.append("<script>function filterRows(k){document.querySelectorAll('section').forEach(s=>s.hidden=k!=='all'&&s.dataset.category!==k)}</script></html>")
    (out / "index.html").write_text("\n".join(page))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
