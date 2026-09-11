"""Archive four reranker ablations, their blue-circle baselines, and resource costs."""
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
from PIL import Image, ImageDraw
from cosmos_policy.scripts.retrieval.build_visual_comparison import load_runs, sha


def distribution(values):
    a = np.asarray(values, dtype=float)
    return dict(n=len(a), mean=float(a.mean()), p50=float(np.percentile(a, 50)), p95=float(np.percentile(a, 95))) if len(a) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', type=Path, default=Path('/mnt4/cyh/ReCAP_qwen_visual_align'))
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--template', type=Path, required=True)
    ap.add_argument('--partial', action='store_true')
    args = ap.parse_args()
    sys.path.insert(0, str(args.base / 'repo_full/cosmos_policy/experiments/robot/pusht'))
    from gym_pusht.envs.pusht import PushTEnv
    from cosmos_policy.experiments.robot.pusht_ret.retrievers.query_visual import retrieval_query_frame
    output = args.output
    (output / 'assets').mkdir(parents=True, exist_ok=True)
    data_root = args.base / 'data/PushT-Cosmos-Policy/success_only'
    report = dict(built_at=datetime.datetime.now().astimezone().isoformat(), groups=[], performance=[],
                  protocol=dict(seeds=list(range(42, 92)), visual_config='tri_default', query_visual='blue_circle',
                                coarse_k=30, selected_k=1, image_size=224, success_threshold=.85, coverage='terminal_coverage', fps=10))
    env = PushTEnv(obs_type='pixels_agent_pos', render_mode='rgb_array', observation_width=128, observation_height=128)
    checkpoints, stats, cache = set(), set(), {}
    query_count = clip_count = rerank_queries = 0

    def decode(entry):
        return np.asarray(Image.open(io.BytesIO(entry.tobytes())).convert('RGB'))

    def read_clip(candidate_id, length=1):
        file, demo, endpoint = candidate_id.split('::')
        with h5py.File(data_root / file, 'r') as f:
            images = f[f'data/{demo}/obs/images']
            return np.stack([decode(images[min(int(endpoint) + i, len(images) - 1)]) for i in range(length)])

    def setting(name):
        nonlocal query_count, clip_count, rerank_queries
        if name in cache:
            return cache[name]
        staged = name.startswith('e')
        suite = 'rerank_top30_formal_20260910' if staged else 'blue_circle_formal_20260910_v3'
        runs = load_runs(args.base / 'results' / suite / name, 'qwen_two_stage' if staged else 'qwen_full', 'blue_circle')
        if not args.partial:
            assert set(runs) == {42, 52, 62, 72, 82}, (name, 'missing batches')
        rows, timings, manifests, index_hashes = {}, [], [], set()
        for base_seed, (mp, m) in sorted(runs.items()):
            manifests.append(m)
            checkpoints.add(m['checkpoint_sha256'])
            stats.add(json.dumps(m['stats_sha256'], sort_keys=True))
            assert m['config']['candidate_k'] == (30 if staged else 1)
            for flag, expected in [('--visual_config', 'tri_default'), ('--chunk_size', '8'), ('--num_open_loop_steps', '8')]:
                assert m['command'][m['command'].index(flag) + 1] == expected
            index_path = Path(m['config']['index_path']) / 'manifest.json'
            index_hashes.add(json.loads(index_path.read_text())['embeddings_sha256'])
            dst = output / 'artifacts' / name / f'seed_{base_seed}'
            if not dst.exists():
                shutil.copytree(mp.parent, dst)
            for filename in ('source_checksums.json', 'tracked_changes.diff'):
                src = mp.parent.parent / filename
                if src.is_file() and not (dst / filename).exists():
                    shutil.copy2(src, dst / filename)
            if not (dst.parent / 'index_manifest.json').exists():
                shutil.copy2(index_path, dst.parent / 'index_manifest.json')
            by_seed = defaultdict(list)
            for line in (mp.parent / 'retrieval_trace.jsonl').read_text().splitlines():
                t = json.loads(line)
                assert t['retrieval_query_visual'] == 'blue_circle'
                if staged:
                    assert t['rerank_count'] == 30 and len(set(t['candidate_ids'])) == 30
                    assert len(t['coarse_scores']) == len(t['reranker_logits']) == len(t['reranker_scores']) == 30
                    rank = int(np.argmax(t['reranker_logits']))
                    assert t['selected_id'] == t['candidate_ids'][rank] and t['selected_coarse_rank'] == rank + 1
                    assert t['coarse_top1_id'] == t['candidate_ids'][0]
                    assert t['query_frame_sha256'] == t['reranker_metrics']['query_sha256']
                    rerank_queries += 1
                by_seed[t['seed']].append(t)
                timings.append(t)
            episodes = [json.loads(line) for line in (mp.parent / 'episodes.jsonl').read_text().splitlines()]
            assert len(episodes) == 10 and {e['seed'] for e in episodes} == set(range(base_seed, base_seed + 10))
            for episode in episodes:
                seed = episode['seed']
                assert seed not in rows
                trace, first = by_seed[seed], by_seed[seed][0]
                assert first['step'] == 0
                raw = env.reset(seed=seed, options={'agent_shape': 'triangle'})[0]['pixels']
                query = retrieval_query_frame(env, raw, 'blue_circle')
                assert hashlib.sha256(raw.tobytes()).hexdigest() == first['policy_frame_sha256']
                assert hashlib.sha256(query.tobytes()).hexdigest() == first['query_frame_sha256']
                query_count += 1
                query_file = output / 'assets' / f'query_{seed}.png'
                if not query_file.exists():
                    Image.fromarray(query).save(query_file)
                clip = read_clip(first['selected_id'], 8)
                assert hashlib.sha256(clip.tobytes()).hexdigest() == first['frame_sha256']
                clip_count += 1
                selected_file = output / 'assets' / f'retrieved_{name}_{seed}.png'
                if not selected_file.exists():
                    Image.fromarray(clip[0]).save(selected_file)
                videos = list(dst.rglob(f'ep{seed - base_seed + 1:03d}--*--combined.mp4'))
                assert len(videos) == 1
                keys = ('step', 'selected_id', 'qwen_score', 'selected_coarse_rank', 'reranker_score', 'reranker_logit',
                        'coarse_retrieval_seconds', 'rerank_seconds', 'retrieval_module_seconds')
                row = dict(episode, video=str(videos[0].relative_to(output)), query=str(query_file.relative_to(output)),
                           retrieved=str(selected_file.relative_to(output)), selected_id=first['selected_id'], score=first['qwen_score'],
                           raw_hash=first['policy_frame_sha256'], query_hash=first['query_frame_sha256'],
                           trace=[{k:t[k] for k in keys if k in t} for t in trace], manifest=str((dst / 'manifest.json').relative_to(output)))
                if staged:
                    row.update({k:first[k] for k in ('reranker_score', 'selected_coarse_rank', 'coarse_top1_id', 'candidate_ids', 'coarse_scores', 'reranker_logits', 'reranker_scores')})
                    sheet_file = output / 'assets' / f'top30_{name}_{seed}.png'
                    if not sheet_file.exists():
                        sheet = Image.new('RGB', (6 * 128, 5 * 154), 'white')
                        draw = ImageDraw.Draw(sheet)
                        for rank, candidate_id in enumerate(first['candidate_ids']):
                            frame = read_clip(candidate_id)[0]
                            assert hashlib.sha256(frame.tobytes()).hexdigest() == first['reranker_metrics']['candidate_sha256'][rank]
                            x, y = rank % 6 * 128, rank // 6 * 154
                            sheet.paste(Image.fromarray(frame), (x, y + 24))
                            draw.rectangle((x, y, x + 127, y + 23), fill='#2569d4' if candidate_id == first['selected_id'] else '#142a3a')
                            draw.text((x + 4, y + 6), f"#{rank+1} R={first['reranker_scores'][rank]:.4f}", fill='white')
                        sheet.save(sheet_file)
                    row['top30_sheet'] = str(sheet_file.relative_to(output))
                rows[seed] = row
        assert len(index_hashes) <= 1
        summary = dict(n=len(rows), successes=sum(e['success'] for e in rows.values()),
                       coverage=float(np.mean([e['terminal_coverage'] for e in rows.values()])) if rows else None,
                       retrieval_calls=len(timings), latency=distribution([t.get('retrieval_module_seconds', t['total_retrieval_seconds']) for t in timings]),
                       gpu_peak_mib=max((m.get('sampled_gpu_peak_used_mib', 0) for m in manifests), default=0))
        cache[name] = (rows, summary, next(iter(index_hashes), None))
        return cache[name]

    for emb in (2, 8):
        for rerank in (2, 8):
            name = f'e{emb}_r{rerank}'
            before, bs, bi = setting(f'image{emb}b')
            after, rs, ri = setting(name)
            if ri:
                assert bi == ri
            pairs = [dict(seed=seed, before=before.get(seed), after=after.get(seed)) for seed in sorted(set(before) | set(after))]
            complete = [p for p in pairs if p['before'] and p['after']]
            assert all(p['before']['query_hash'] == p['after']['query_hash'] for p in complete)
            report['groups'].append(dict(id=name, title=f'{emb}B Emb → {rerank}B Reranker', embedding=emb, reranker=rerank,
                                         baseline=f'image{emb}b', sides=dict(before=bs, after=rs), pairs=pairs, paired_n=len(complete),
                                         changed_first_top1=sum(p['before']['selected_id'] != p['after']['selected_id'] for p in complete),
                                         coarse_top1_matches_baseline=sum(p['before']['selected_id'] == p['after']['coarse_top1_id'] for p in complete)))
            print(name, bs, rs, flush=True)
    env.close()
    reference_queries, devices = None, set()
    benchmark_root = args.base / 'results/rerank_matched_benchmark_20260910'
    for name in ['image2b', 'image8b', 'e2_r2', 'e2_r8', 'e8_r2', 'e8_r8']:
        valid = [p for p in sorted((benchmark_root / name).rglob('summary.json')) if json.loads(p.read_text()).get('status') == 'complete']
        if not valid:
            assert args.partial, ('Missing performance benchmark', name)
            continue
        path = valid[-1]
        summary = json.loads(path.read_text())
        assert summary['query_count'] == 50 and summary['warmup_count'] == 5 and not summary['resource_metrics']['errors']
        devices.add(summary['resource_metrics']['hardware']['uuid'])
        measured = [r for r in map(json.loads, (path.parent / 'queries.jsonl').read_text().splitlines()) if not r['warmup']]
        identities = [(r['seed'], r['query_sha256']) for r in measured]
        if reference_queries is None:
            reference_queries = identities
        assert identities == reference_queries
        dst = output / 'performance' / name
        if not dst.exists():
            shutil.copytree(path.parent, dst)
        summary.update(id=name, mean_coarse_seconds=float(np.mean([r.get('coarse_retrieval_seconds', r['retrieval_call_seconds']) for r in measured])),
                       mean_rerank_seconds=float(np.mean([r.get('rerank_seconds', 0) for r in measured])), archive=str((dst / 'summary.json').relative_to(output)))
        report['performance'].append(summary)
    assert len(devices) <= 1 and len(checkpoints) == len(stats) == 1
    report['protocol'].update(checkpoint_sha256=next(iter(checkpoints)), stats_sha256=json.loads(next(iter(stats))))
    report['validation'] = dict(query_hashes_verified=query_count, retrieval_clips_verified=clip_count, unique_episodes=query_count,
                                paired_seeds_verified=sum(g['paired_n'] for g in report['groups']), rerank_queries_verified=rerank_queries,
                                same_index_per_pair=True, benchmark_identical_queries=True, benchmark_same_physical_gpu=True)
    payload = json.dumps(report, ensure_ascii=False).replace('</', '<\\/')
    (output / 'index.html').write_text(args.template.read_text().replace('__REPORT_DATA__', payload))
    (output / 'comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    lines = ['# 图像 top30 → Qwen3-VL-Reranker top1', '', '所有设置保持 blue_circle；Cosmos 输入保持原始三角形。每组 50 个相同 seed，tri_default，同 checkpoint。', '',
             '| 设置 | 直接 top1 成功率 | rerank 成功率 | 直接 top1 coverage | rerank coverage |', '|---|---:|---:|---:|---:|']
    for g in report['groups']:
        b, a = g['sides']['before'], g['sides']['after']
        pct = lambda s: f"{100*s['successes']/s['n']:.1f}%" if s['n'] else '待完成'
        cov = lambda s: f"{100*s['coverage']:.2f}%" if s['coverage'] is not None else '待完成'
        lines.append(f"| {g['title']} | {pct(b)} | {pct(a)} | {cov(b)} | {cov(a)} |")
    lines += ['', '| 同查询性能对照 | 均值 ms | P50 ms | P95 ms | 检索进程峰值 GiB |', '|---|---:|---:|---:|---:|']
    for p in report['performance']:
        lines.append(f"| {p['id']} | {p['latency_mean_seconds']*1000:.1f} | {p['latency_p50_seconds']*1000:.1f} | {p['latency_p95_seconds']*1000:.1f} | {p['resource_metrics']['retrieval_process_peak_mib']/1024:.2f} |")
    lines += ['', '性能对照：同一张 A100 80GB、同一组 50 张蓝色圆形查询，先预热 5 次；模型初始化单独记录，耗时不包含 Cosmos 推理。',
              '检索显存：NVML 每 200ms 采样 embedding/reranker 进程之和；formal 整卡峰值另包含 Cosmos。采样可能漏掉更短的瞬时峰值，trace 同时记录 PyTorch 分配峰值。',
              'Reranker：224px、BF16 backbone、SDPA、batch=30；官方 yes/no 头，最终投影 FP32，按 sigmoid 前 logit 排序。',
              '同 seed 保证相同起点；后续策略动作可能导致场景分叉。Top30 图集按 embedding 排名排列，蓝色标题为最终选中候选。',
              '原始日志、完整 trace、模型身份、源代码校验、显存采样与三类视频均保留。',
              '双击 index.html 离线打开，或执行 python3 -m http.server 8767 --bind 127.0.0.1。',
              '模型来源：https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B 和 https://huggingface.co/Qwen/Qwen3-VL-Reranker-8B。']
    (output / 'README.md').write_text('\n'.join(lines) + '\n')
    checksums = {str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file() and p.name != 'checksums.json'}
    (output / 'checksums.json').write_text(json.dumps(checksums, indent=2))
    print('RERANK_GALLERY_READY', report['validation'], flush=True)


if __name__ == '__main__':
    main()
