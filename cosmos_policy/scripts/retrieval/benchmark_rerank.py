"""Compare retrieval on identical blue-circle queries with no policy inference."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['RECAP_RESOURCE_FILE'] = str(args.output / 'retrieval_resources.json')
    repo = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo / 'cosmos_policy/experiments/robot/pusht'))
    from gym_pusht.envs.pusht import PushTEnv
    from cosmos_policy.experiments.robot.pusht_ret.retrievers.config import load_config
    from cosmos_policy.experiments.robot.pusht_ret.retrievers.query_visual import retrieval_query_frame
    from cosmos_policy.experiments.robot.pusht_ret.retrievers.qwen_full import QwenFullRetrieval
    from cosmos_policy.experiments.robot.pusht_ret.retrievers.qwen_two_stage import QwenTwoStageRetrieval
    from cosmos_policy.scripts.retrieval.resource_monitor import ResourceMonitor

    cfg = load_config(args.config)
    monitor = ResourceMonitor(args.gpu, args.output)
    monitor.start()
    started = time.perf_counter()
    cls = QwenTwoStageRetrieval if cfg.strategy == 'qwen_two_stage' else QwenFullRetrieval
    index_manifest = json.loads((Path(cfg.index_path) / 'manifest.json').read_text())
    split = sorted({str(Path(name).parent) for name in index_manifest['files']})
    retrieval = cls(data_dir='/mnt4/cyh/ReCAP_qwen_visual_align/data/PushT-Cosmos-Policy/success_only',
                    split=split, retrieval_config=args.config)
    initialization_seconds = time.perf_counter() - started
    env = PushTEnv(obs_type='pixels_agent_pos', render_mode='rgb_array',
                   observation_width=128, observation_height=128)
    rows = []
    try:
        with (args.output / 'queries.jsonl').open('x') as f:
            for i, seed in enumerate([42, 43, 44, 45, 46] + list(range(42, 92))):
                raw = env.reset(seed=seed, options={'agent_shape': 'triangle'})[0]['pixels']
                start = time.perf_counter()
                query = retrieval_query_frame(env, raw, 'blue_circle')
                visual_seconds = time.perf_counter() - start
                start = time.perf_counter()
                retrieval.get_retrieved_data(primary_image=query)
                seconds = time.perf_counter() - start
                row = dict(retrieval.last_result, seed=seed, warmup=i < 5,
                           query_sha256=hashlib.sha256(query.tobytes()).hexdigest(),
                           query_visual_seconds=visual_seconds, retrieval_call_seconds=seconds,
                           retrieval_module_seconds=visual_seconds + seconds)
                f.write(json.dumps(row) + '\n')
                f.flush()
                rows.append(row)
                print('query', seed, 'warmup' if i < 5 else 'measured', seconds, flush=True)
        resources = monitor.finish()
        measured = [r for r in rows if not r['warmup']]
        timings = np.array([r['retrieval_module_seconds'] for r in measured])
        summary = dict(status='complete', config=cfg.to_dict(), host=os.uname().nodename,
                       gpu=args.gpu, initialization_seconds=initialization_seconds,
                       query_count=len(measured), warmup_count=5,
                       latency_mean_seconds=float(timings.mean()),
                       latency_p50_seconds=float(np.percentile(timings, 50)),
                       latency_p95_seconds=float(np.percentile(timings, 95)),
                       resource_metrics=resources)
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        if monitor.is_alive():
            monitor.finish()
        retrieval.close()
        env.close()


if __name__ == '__main__':
    main()
