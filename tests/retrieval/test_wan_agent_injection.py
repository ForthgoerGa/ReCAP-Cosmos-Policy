"""Scientific invariants for state injection, causal alignment and visual dominance."""
import copy
import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from cosmos_policy.experiments.robot.pusht_ret.retrievers.config import RetrievalError, RetrievalConfig
from cosmos_policy.experiments.robot.pusht_ret.retrievers.wan_agent_state import (
    STATE_SPEC, implementation_hash, causal_states, query_state, normalize_states,
    validate_sidecar, injection_distances)
from cosmos_policy.experiments.robot.pusht_ret.retrievers.wan_agent_injection import WanAgentInjectionRetrieval
from cosmos_policy.experiments.robot.pusht_ret.retrievers.wan_vae import WanVAERetrieval
from test_qwen_rerank import pool

@pytest.mark.parametrize("weight", [0., .01, .03, .05, .1, .15, .2, .3, .4, .5, .6, .7, .8, .9, 1.])
def test_compact_score_matches_literal_injection(weight):
    rng = np.random.default_rng(142)
    z = torch.tensor(rng.normal(size=(5, 37632)), dtype=torch.float64)
    z /= z.norm(dim=1, keepdim=True)
    u = torch.tensor(rng.uniform(-1, 1, size=(5, 4)), dtype=torch.float64)
    cos = z[1:] @ z[0]
    _, d = injection_distances(cos, u[1:], u[0], weight, .02)
    tiled = u.repeat(1, 3136).reshape(5, 16, 1, 28, 28)
    zv = z.reshape(5, 16, 3, 28, 28) * np.sqrt(1-weight)/2
    zs = tiled * np.sqrt(weight)/(2*np.sqrt(12544))
    literal = torch.cat([zv, zs], dim=2).flatten(1)
    torch.testing.assert_close(d["distance"], ((literal[1:] - literal[0])**2).sum(1), rtol=1e-10, atol=1e-10)

def test_zero_weight_preserves_legacy_argmax_even_at_rounding_ties():
    cos = torch.tensor([.8, .9, .9, .7], dtype=torch.float32)
    states = torch.zeros(4,4)
    selected, _ = injection_distances(cos, states, torch.zeros(4), 0., .02)
    assert selected == int(torch.argmax(cos)) == 1

def test_state_changes_only_visually_eligible_choices():
    cos = torch.tensor([.9, .89, .86])
    candidates = torch.tensor([[-1.]*4, [0.]*4, [1.]*4])
    chosen, d = injection_distances(cos, candidates, torch.ones(4), .1, .02)
    assert chosen == 1
    assert d["eligible"].tolist() == [True, True, False]
    assert float(cos.max()-cos[chosen]) <= .020001

def test_guard_singleton_cannot_change_visual_choice():
    c = torch.tensor([.9, .5])
    chosen, _ = injection_distances(c, torch.tensor([[-1.]*4,[1.]*4]), torch.ones(4), .1, .02)
    assert chosen == 0

def test_causal_velocity_is_identical_for_full_demo_and_query_prefix():
    p=np.array([[10,20],[14,18],[20,18],[50,50]],np.float32)
    all_states=causal_states(p)
    np.testing.assert_array_equal(all_states[:3,2:], [[0,0],[4,-2],[5,-1]])
    for t in range(len(p)):
        history=p[max(0,t-7):t+1] if t else None
        np.testing.assert_array_equal(query_state(p[t], history),all_states[t])
    np.testing.assert_array_equal(causal_states(p[:3]),all_states[:3])
    np.testing.assert_array_equal(causal_states(p[-1:])[0,2:],[0,0])
    with pytest.raises(RetrievalError):
        query_state(p[-1],p[:3])

def test_scaling_preserves_speed_magnitude_and_clips():
    states=np.array([[256,512,0,0],[256,512,1,2],[256,512,10,20]],np.float32)
    u, clipped=normalize_states(states,[2,4])
    np.testing.assert_array_equal(u,[[0,1,0,0],[0,1,.5,.5],[0,1,1,1]])
    assert clipped[-1].tolist()==[False,False,True,True]
    assert not clipped[:2].any()

@pytest.mark.parametrize("bad", [float("nan"), -.01, 1.01])
def test_invalid_weight_fails(bad):
    with pytest.raises(RetrievalError):
        injection_distances(torch.ones(1),torch.zeros(1,4),torch.zeros(4),bad,.02)

def test_sidecar_reordered_or_missing_candidates_are_rejected():
    visual={"candidate_ids":["demo::0","demo::1"],"files":{"base_0/data.hdf5":"x"},"embeddings_sha256":"z"}
    manifest={"spec":STATE_SPEC,"implementation_hash":implementation_hash(),"visual_index":copy.deepcopy(visual)}
    validate_sidecar(manifest,visual)
    manifest["visual_index"]["candidate_ids"].reverse()
    with pytest.raises(RetrievalError,match="candidate_ids"):
        validate_sidecar(manifest,visual)
    manifest["visual_index"]["candidate_ids"]=[]
    with pytest.raises(RetrievalError):
        validate_sidecar(manifest,visual)

def test_zero_weight_actual_retriever_returns_legacy_payload():
    legacy=pool(WanVAERetrieval)
    injected=pool(WanAgentInjectionRetrieval)
    frame=np.zeros((8,8,3),np.uint8)
    for obj in (legacy,injected):
        obj.video=True
        obj.vae_cfg=SimpleNamespace(strategy="wan_vae_video")
        obj.embeddings=torch.tensor([[0.,1.],[1.,0.],[1.,0.],[-1.,0.]])
        obj.encoder=SimpleNamespace(encode=lambda clips:torch.tensor([[1.,0.]]), last_metadata={})
    injected.velocity_scale=np.array([1.,1.],np.float32)
    injected.agent_states=torch.zeros(4,4)
    injected.raw_agent_states=np.zeros((4,4),np.float32)
    injected.agent_state_weight=0.
    injected.visual_cosine_gap=.02
    injected.state_manifest={"states_sha256":"test"}
    kwargs=dict(primary_image=frame,primary_images=[frame],agent_pos=np.array([256,256],np.float32))
    original=legacy.get_retrieved_data(**kwargs)
    current=injected.get_retrieved_data(**kwargs)
    assert legacy.last_result["selected_id"]==injected.last_result["selected_id"]
    for a,b in zip(original,current):np.testing.assert_array_equal(a,b)

def test_invalid_state_or_guard_fails():
    with pytest.raises(RetrievalError):causal_states(np.empty((0,2)))
    with pytest.raises(RetrievalError):normalize_states([0,0,float("nan"),0],[1,1])
    with pytest.raises(RetrievalError):
        injection_distances(torch.ones(1),torch.zeros(1,4),torch.zeros(4),.05,float("nan"))

def test_pure_state_is_independent_of_visual_scores_and_visual_ties():
    candidates = torch.tensor([[1.,1.,1.,1.], [0.,0.,0.,0.], [0.,0.,0.,0.]])
    query = torch.zeros(4)
    for cos in (torch.tensor([1.,-1.,.9]), torch.tensor([-.5,.7,1.]), torch.ones(3)):
        selected, details = injection_distances(cos, candidates, query, 1., None)
        assert selected == 1  # First index resolves a state-distance tie, regardless of visual score.
        assert details['eligible'].all()
        torch.testing.assert_close(details['distance'], ((candidates-query)**2).mean(1)/4)

def test_weight_one_with_guard_differs_from_genuine_pure_state():
    cos = torch.tensor([.9,.5])
    candidates = torch.tensor([[1.]*4,[0.]*4])
    assert injection_distances(cos,candidates,torch.zeros(4),1.,.02)[0] == 0
    assert injection_distances(cos,candidates,torch.zeros(4),1.,None)[0] == 1

@pytest.mark.parametrize('gap', [None,.02])
def test_config_accepts_weight_one_and_optional_guard(tmp_path,gap):
    checkpoint=tmp_path/'tokenizer.pth';checkpoint.touch()
    cfg=RetrievalConfig(strategy='wan_vae_video',model_path=str(checkpoint),
        model_revision='a'*64,index_path='visual',candidate_k=1,image_size=224,
        embedding_dim=37632,agent_state_weight=1.,agent_state_index_path='state',visual_cosine_gap=gap)
    cfg.validate()
