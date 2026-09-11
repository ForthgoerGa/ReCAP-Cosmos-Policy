"""Summarize completed shards and formal seeds without importing GPU libraries."""
import json
from pathlib import Path

base = Path('/mnt4/cyh/ReCAP_qwen_visual_align')
for mode, name, total in [('image', 'wan_image_flat224_20260911', 12951),
                          ('video', 'wan_video_flat224_padfirst_20260911', 24397)]:
    shards = base/'indices'/f'{name}_shards'
    encoded = 0
    errors = []
    details = []
    for path in sorted(shards.glob('*.attempt2.log')):
        progress = None
        lines = path.read_text().splitlines()
        for line in lines:
            if line.startswith('{"encoded"'):
                progress = json.loads(line)
        if progress:
            encoded += progress['encoded']
        if any('Traceback (most recent call last)' in line for line in lines):
            errors.append(dict(shard=path.name, error=lines[-1]))
        details.append(dict(log=path.name, progress=progress, last_line=lines[-1] if lines else 'starting'))
    manifests = list(shards.glob('shard_*.json'))
    runs = []
    for path in (base/'results'/'wan_flat_formal_20260911'/mode).rglob('manifest.json'):
        m = json.loads(path.read_text())
        if 'command' not in m:
            continue
        ep = path.parent/'episodes.jsonl'
        rows = [json.loads(line) for line in ep.read_text().splitlines()] if ep.exists() else []
        runs.append(dict(seed=m['seed'], trials=m['num_trials'], status=m['status'],
                         done=len(rows), successes=sum(r['success'] for r in rows), host=m['host']))
    print(json.dumps(dict(group=mode, encoded=encoded, total=total, completed_shards=len(manifests),
                          merged=(base/'indices'/name/'manifest.json').exists(), errors=errors,
                          details=details, formal=runs)), flush=True)
