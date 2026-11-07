import json
from types import SimpleNamespace
import numpy as np
from genie_diep.audit_duplicates import run


def test_duplicate_pixels_detected_despite_npy_storage_order(tmp_path):
    clip=np.full((16,64,64,3),180,dtype=np.uint8)
    clip[:,:,10:24,0]=25
    np.save(tmp_path/'a.npy',clip)
    np.save(tmp_path/'b.npy',np.asfortranarray(clip))
    rows=[dict(path='a.npy',split='train'),dict(path='b.npy',split='val')]
    (tmp_path/'clips.jsonl').write_text('\n'.join(map(json.dumps,rows)))
    out=tmp_path/'report';run(SimpleNamespace(data=str(tmp_path),out=str(out)))
    report=json.loads((out/'duplicates.json').read_text())
    assert report['exact_cross_split']==1
    assert report['matches'][0]['exact_pixels']
