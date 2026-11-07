import json
import numpy as np
import pytest
from genie_diep.data import ClipDataset, source_split, center_tank_present


def test_split_is_source_stable_and_missing_validation_is_error(tmp_path):
    assert source_split("video_a")==source_split("video_a")
    row=dict(path="a.npy",source_id="a",split="train",shape=[5,32,32,3])
    np.save(tmp_path/"a.npy",np.zeros(row["shape"],dtype=np.uint8))
    (tmp_path/"a.jsonl").write_text(json.dumps(row))
    assert ClipDataset(tmp_path,"train")[0].shape==(5,3,32,32)
    with pytest.raises(ValueError,match="No val clips"):
        ClipDataset(tmp_path,"val")
    row["split"]="val"
    (tmp_path/"b.jsonl").write_text(json.dumps(row))
    with pytest.raises(ValueError,match="leakage"):
        ClipDataset(tmp_path)


def test_center_detector_rejects_empty_and_offcenter_cyan():
    image=np.full((64,64,3),180,dtype=np.uint8)
    assert not center_tank_present(image)
    image[3:8,3:8]=[20,170,220]
    assert not center_tank_present(image)
    image[30:35,30:35]=[20,170,220]
    assert center_tank_present(image)
