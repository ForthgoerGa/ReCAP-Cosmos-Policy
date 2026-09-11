"""Check real online self-retrieval against saved index rows and policy payloads."""
import argparse
import json
import numpy as np
import torch
from cosmos_policy.experiments.robot.pusht_ret.retrievers.wan_vae import WanVAERetrieval, candidate_clip

parser = argparse.ArgumentParser()
parser.add_argument('--retrieval-config', required=True)
parser.add_argument('--data-dir', required=True)
args = parser.parse_args()
obj = WanVAERetrieval(args.data_dir, split=[f'base_{i}' for i in range(5)], retrieval_config=args.retrieval_config)
try:
    for index in (0, 7, len(obj._subframes)//2, len(obj._subframes)-1):
        frames = candidate_clip(obj, index, obj.video)
        expected = obj.get_candidate_data(index)
        result = obj.get_retrieved_data(primary_image=frames[-1], primary_images=frames)
        record = obj.last_result
        assert record['cosine_score'] > .999, record
        selected = obj.get_candidate_data(record['selected_index'])
        for a, b in zip(result, selected):
            np.testing.assert_array_equal(a, b)
        assert result[0].shape == expected[0].shape and result[1].shape == expected[1].shape
        print(json.dumps(dict(source_index=index, **record)), flush=True)
    print('ONLINE_INDEX_VALIDATION_PASSED', flush=True)
finally:
    obj.close()
