"""Regression tests for video history forwarding and state/video weighting."""
import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from test_qwen_rerank import pool
from cosmos_policy.experiments.robot.pusht_ret.retrievers.config import RetrievalConfig
from cosmos_policy.experiments.robot.pusht_ret.retrievers.qwen_video import QwenVideoLateFusionRetrieval

@pytest.mark.parametrize("length", [1, 2, 7, 8])
@pytest.mark.parametrize("alpha,beta,expected", [(0.2,0.8,0),(0.8,0.2,3)])
def test_video_history_and_fusion_selection(length, alpha, beta, expected):
    obj=pool(QwenVideoLateFusionRetrieval)
    obj.qwen_cfg=RetrievalConfig(strategy="qwen_video_late_fusion", candidate_k=1, alpha=alpha,beta=beta)
    states=np.zeros((length,5),np.float32)
    feature=obj._feat_state(states)
    obj._feat=np.tile(feature,(4,1))
    obj._feat[:,0]+=np.array([0,np.sqrt(1/3),np.sqrt(2/3),1])
    obj.embeddings=np.array([[0,1],[1/3,np.sqrt(8/9)],[2/3,np.sqrt(5/9)],[1,0]],np.float32)
    images=[np.full((8,8,3),i,np.uint8) for i in range(length)]
    received=[]
    def encode_video(videos):
        received.extend(videos)
        return np.array([[1,0]],np.float32),{}
    obj.client=SimpleNamespace(encode_video=encode_video)
    result=obj.get_retrieved_data(primary_image=images[-1],primary_images=images,state_history=states)
    assert len(received)==1 and len(received[0])==length
    for actual,expected_frame in zip(received[0],images):
        np.testing.assert_array_equal(actual,expected_frame)
    assert obj.last_result["selected_index"]==expected
    assert obj.last_result["video_frames"]==length
    for actual,payload in zip(result,obj.get_candidate_data(expected)):
        np.testing.assert_array_equal(actual,payload)

def test_eval_dispatch_forwards_history_to_video_fusion():
    path=Path("cosmos_policy/experiments/robot/pusht_ret/run_eval.py")
    tree=ast.parse(path.read_text())
    found=False
    for node in ast.walk(tree):
        if not isinstance(node,ast.If): continue
        if not any(isinstance(child,ast.Constant) and child.value=="qwen_video_late_fusion" for child in ast.walk(node.test)): continue
        if any(isinstance(child,ast.Assign) and any(isinstance(target,ast.Subscript) and isinstance(target.slice,ast.Constant) and target.slice.value=="primary_images" for target in child.targets) for statement in node.body for child in ast.walk(statement)):
            found=True
    assert found, "Video fusion must receive the causal image deque"

@pytest.mark.parametrize("length", [1, 6, 7, 8, 9, 17])
def test_dense_pool_covers_every_endpoint(tmp_path, monkeypatch, length):
    import h5py
    from cosmos_policy.experiments.robot.pusht_ret import retrieval
    from cosmos_policy.experiments.robot.pusht_ret.retrievers.video_pool import VideoRetrievalPool
    root=tmp_path/"base_0"
    root.mkdir()
    path=root/"data.hdf5"
    with h5py.File(path,"w") as f:
        g=f.create_group("data/demo_0")
        g["obs/images"]=np.zeros((length,2,2,3),np.uint8)
        g["obs/states"]=np.zeros((length,5),np.float32)
        g["actions"]=np.zeros((length,2),np.float32)
    monkeypatch.setattr(retrieval,"get_hdf5_files",lambda *a,**kw:[str(path)])
    monkeypatch.setattr(retrieval,"decode_jpeg_bytes_dataset",lambda d:d[:])
    obj=VideoRetrievalPool(str(tmp_path),split=["base_0"])
    assert [sf["t_last"] for sf in obj._subframes]==list(range(length))
    m=obj.window_metadata()
    assert m["window_length"]==[min(t+1,8) for t in range(length)]
    assert m["window_type"]==["partial_window" if t<7 else "full_window" for t in range(length)]
    assert retrieval.PushTRetrieval.STRIDE==2

def test_raw_cosine_and_fixed_state_scores():
    obj=pool(QwenVideoLateFusionRetrieval)
    obj.qwen_cfg=RetrievalConfig(strategy="qwen_video_late_fusion",candidate_k=1,alpha=.8,beta=.2)
    states=np.zeros((1,5),np.float32)
    obj._feat=np.tile(obj._feat_state(states),(4,1))
    obj._feat[:,0]+=[0,.4,.8,2]
    cos=np.array([.81,.82,.83,-.9],np.float32)
    obj.embeddings=np.column_stack([cos,np.sqrt(1-cos*cos)])
    obj.client=SimpleNamespace(encode_video=lambda videos:(np.array([[1,0]],np.float32),{}))
    im=np.zeros((8,8,3),np.uint8)
    obj.get_retrieved_data(primary_image=im,state_history=states)
    scores=.8*cos+.2*np.exp(-np.array([0,.16,.64,4]))
    chosen=int(scores.argmax())
    assert obj.last_result["selected_index"]==chosen
    assert obj.last_result["video_score"]==pytest.approx(cos[chosen])
    assert obj.last_result["final_score"]==pytest.approx(scores[chosen])
    # Changing an irrelevant outlier must not rescale the surviving candidates.
    obj.embeddings[-1]=[-.1,np.sqrt(.99)]
    obj._feat[-1,0]+=100
    obj.get_retrieved_data(primary_image=im,state_history=states)
    assert obj.last_result["selected_index"]==chosen
    assert obj.last_result["final_score"]==pytest.approx(scores[chosen])
