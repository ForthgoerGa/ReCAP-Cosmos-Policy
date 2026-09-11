"""Create auditable summaries for both Wan VAE formal groups."""
import json
from pathlib import Path
import numpy as np


def summarize(root, group):
    rows, traces = [], []
    for path in sorted((root/group).rglob('manifest.json')):
        manifest = json.loads(path.read_text())
        if manifest.get('status') not in ('complete', 'error'):
            continue
        ep = path.parent/'episodes.jsonl'
        episodes = [json.loads(line) for line in ep.read_text().splitlines()] if ep.exists() else []
        trace = path.parent/'retrieval_trace.jsonl'
        records = [json.loads(line) for line in trace.read_text().splitlines()] if trace.exists() else []
        rows.extend(episodes)
        traces.extend(records)
    timings = [r.get('retrieval_module_seconds', r.get('retrieval_seconds')) for r in traces]
    timings = [float(x) for x in timings if x is not None]
    enc = [float(r['encode_seconds']) for r in traces if r.get('encode_seconds') is not None]
    search = [float(r['search_seconds']) for r in traces if r.get('search_seconds') is not None]
    return dict(group=group, episodes=len(rows), successes=sum(bool(r['success']) for r in rows),
                success_rate=sum(bool(r['success']) for r in rows)/len(rows) if rows else None,
                mean_terminal_coverage=float(np.mean([r['terminal_coverage'] for r in rows])) if rows else None,
                retrieval_queries=len(traces), retrieval_p50_seconds=float(np.percentile(timings,50)) if timings else None,
                retrieval_p95_seconds=float(np.percentile(timings,95)) if timings else None,
                encoder_p50_seconds=float(np.percentile(enc,50)) if enc else None,
                search_p50_seconds=float(np.percentile(search,50)) if search else None,
                index_gpu_bytes=max([r.get('index_gpu_bytes',0) for r in traces], default=0),
                videos=[str(p) for p in sorted((root/group).rglob('*combined.mp4'))])


def main():
    root = Path('/mnt4/cyh/ReCAP_qwen_visual_align/results/wan_flat_formal_20260911')
    output = dict(protocol=dict(seeds=[42,52,62,72,82], episodes_per_seed=10, visual='blue_circle',
                               visual_config='tri_default', checkpoint='model_000007000.pt'),
                  groups=[summarize(root, 'image'), summarize(root, 'video')])
    (root/'summary.json').write_text(json.dumps(output, indent=2))
    lines=['# Wan VAE formal summary','',json.dumps(output, indent=2)]
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
