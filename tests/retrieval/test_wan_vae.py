"""Causal window and search invariants for the flattened Wan baseline."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from test_qwen_rerank import pool
from cosmos_policy.experiments.robot.pusht_ret.retrievers.config import RetrievalError
from cosmos_policy.experiments.robot.pusht_ret.retrievers.wan_vae import WanVAERetrieval, candidate_clip
from cosmos_policy.experiments.robot.pusht_ret.retrievers.wan_vae_encoder import prepare_clip, padded_history


@pytest.mark.parametrize("length", range(1, 11))
def test_history_retains_last_eight_and_current_frame(length):
    frames = [np.full((8, 8, 3), i, np.uint8) for i in range(length)]
    expected = [max(0, length-8)] * max(0, 8-length) + list(range(max(0, length-8), length))
    assert [int(f[0, 0, 0]) for f in padded_history(frames)] == expected
    tensor = prepare_clip(frames, True, 8)
    assert tensor.shape == (3, 9, 8, 8)
    recovered = np.rint((tensor[0, :, 0, 0] + 1) * 127.5).astype(int)
    np.testing.assert_array_equal(recovered, [expected[0]] + expected)
    assert int(recovered[-1]) == length-1


def test_invalid_pixels_fail_instead_of_silent_fallback():
    with pytest.raises(RetrievalError):
        padded_history([])
    with pytest.raises(RetrievalError):
        prepare_clip([np.zeros((8, 8, 3), np.float32)], False, 8)
    with pytest.raises(RetrievalError):
        prepare_clip([np.zeros((8, 8, 3), np.uint8)]*2, False, 8)


def test_demo_query_use_identical_causal_frames():
    obj = pool()
    for sf in obj._subframes:
        sf['start'] = 0
    for i, sf in enumerate(obj._subframes):
        frames = obj._base_data[(sf['split'], sf['demo'])]['images']
        clip = candidate_clip(obj, i, True)
        np.testing.assert_array_equal(prepare_clip(clip, True, 8),
                                      prepare_clip(frames[:sf['t_last']+1], True, 8))
        np.testing.assert_array_equal(candidate_clip(obj, i, False)[0], frames[sf['t_last']])


def test_visual_cosine_selects_payload_and_stable_first_tie():
    obj = pool(WanVAERetrieval)
    obj.vae_cfg = SimpleNamespace(strategy='wan_vae_video')
    obj.video = True
    obj.embeddings = torch.tensor([[0., 1.], [1., 0.], [1., 0.], [-1., 0.]])
    current = np.zeros((8, 8, 3), np.uint8)
    seen = []
    def encode(clips):
        seen.extend(clips)
        return torch.tensor([[1., 0.]])
    def forbidden(*args, **kwargs):
        raise AssertionError('State must not affect visual selection')
    obj.get_state_candidates = forbidden
    obj.encoder = SimpleNamespace(encode=encode, last_metadata={})
    expected = obj.get_candidate_data(1)
    result = obj.get_retrieved_data(agent_pos=object(), block_pos=object(), primary_image=current,
                                  primary_images=[current])
    for a, b in zip(result, expected):
        np.testing.assert_array_equal(a, b)
    assert obj.last_result['selected_index'] == 1
    assert obj.last_result['history_padding_frames'] == 7
    assert len(seen[0]) == 1
    with pytest.raises(RetrievalError, match='end at current'):
        obj.get_retrieved_data(primary_image=current, primary_images=[current + 1])
