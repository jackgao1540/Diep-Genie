import copy
import hashlib
import io
import json

import numpy as np
from PIL import Image
import pytest

from scripts.export_reconstruction_viewer import (
    KIND, PUBLIC_IDS, atlas_image, decode_public_frames, lossless_webp,
    private_bundle, selected_public_clips, validate_destinations,
)
from scripts.validate_reconstruction_viewer import UI_FILES, validate


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def public_source(cid):
    return dict(id=cid, title=cid, description='Approved example', split='val', status='accepted',
        source_id='video', source_title='Video', creator='Creator', source_url='https://www.youtube.com/watch?v=video',
        fps=10, frames=40, processed_kind='retained_proxy',
        media={'processed':{'path':f'assets/{cid}-processed.webp','sha256':'x','bytes':1}})


def test_lossless_atlas_round_trip_preserves_every_frame_and_order(tmp_path):
    frames=np.zeros((40,128,128,3),dtype=np.uint8)
    for i in range(40):
        frames[i]=[i*5,255-i*4,i*3]
        frames[i,3:7,17:31]=[i,93,201]
    blob=lossless_webp(frames); path=tmp_path/'atlas.webp'; path.write_bytes(blob)
    media=dict(bytes=len(blob),sha256=digest(path))
    assert np.array_equal(decode_public_frames(path,media),frames)
    media['sha256']='0'*64
    with pytest.raises(ValueError,match='changed'):
        decode_public_frames(path,media)
    with Image.open(io.BytesIO(lossless_webp(frames[:16]))) as image:
        assert image.size==(1024,256)


@pytest.mark.parametrize('frames',[
    np.zeros((16,128,128,3),dtype=np.float32),
    np.zeros((41,128,128,3),dtype=np.uint8),
    np.zeros((16,64,64,3),dtype=np.uint8),
])
def test_atlas_rejects_wrong_tensor_or_unbounded_frames(frames):
    with pytest.raises(ValueError):
        atlas_image(frames)


@pytest.mark.parametrize('field,value',[
    ('status','held'),('split','test'),('processed_kind','illustrative_resize_not_training_data'),('frames',41),
])
def test_public_selection_rejects_unapproved_or_wrong_contract(field,value):
    showcase=dict(clips=[public_source(cid) for cid in PUBLIC_IDS]); recipe=dict(sample_ids=list(PUBLIC_IDS))
    assert len(selected_public_clips(showcase,recipe))==4
    showcase['clips'][0][field]=value
    with pytest.raises(ValueError):
        selected_public_clips(showcase,recipe)


def test_public_selection_rejects_new_source_or_path(tmp_path):
    showcase=dict(clips=[public_source(cid) for cid in PUBLIC_IDS]); recipe=dict(sample_ids=list(PUBLIC_IDS))
    recipe['sample_ids'][-1]='sample-03'
    with pytest.raises(ValueError,match='selection'):
        selected_public_clips(showcase,recipe)
    recipe['sample_ids']=list(PUBLIC_IDS)
    showcase['clips'][0]['media']['processed']['path']='../private.webp'
    with pytest.raises(ValueError,match='path'):
        selected_public_clips(showcase,recipe)


def test_private_output_cannot_enter_published_tree_or_replace_export(tmp_path):
    root=tmp_path/'project'; (root/'docs/assets').mkdir(parents=True)
    destination=root/'exports/viewer/heldout.json'
    assert validate_destinations(root,root/'docs',destination)[2]==destination.resolve()
    for wrong in [root/'docs/local.json',root.parent/'outside.json']:
        with pytest.raises(ValueError,match='Private bundle'):
            validate_destinations(root,root/'docs',wrong)
    destination.parent.mkdir(parents=True); destination.write_text('preserved')
    with pytest.raises(FileExistsError):
        validate_destinations(root,root/'docs',destination)
    assert destination.read_text()=='preserved'


def test_private_bundle_rejects_checkpoint_or_artifact_escape_before_decoding(tmp_path):
    expected={'before':dict(step=219,preflight=False,sha256='a'),'after':dict(step=518,preflight=True,sha256='b')}
    report=dict(windows=64,sources=16,checkpoints=copy.deepcopy(expected),selection_sha256='selection',
        records=[dict(index=i,selection=dict(length=16,source_id=str(i//4)),artifact=dict(path='../private.npz',sha256='x')) for i in range(64)])
    path=tmp_path/'quality-report.json'; write_json(path,report)
    with pytest.raises(ValueError,match='artifact path'):
        private_bundle(tmp_path,expected)
    report['checkpoints']['after']['sha256']='changed'; write_json(path,report)
    with pytest.raises(ValueError,match='checkpoint mismatch'):
        private_bundle(tmp_path,expected)


def public_fixture(root):
    docs=root/'docs'; (docs/'assets').mkdir(parents=True)
    for name in UI_FILES:
        (docs/name).write_text('viewer',encoding='utf-8')
    checkpoints={'before':dict(step=219,preflight=False,sha256='a'),'after':dict(step=518,preflight=True,sha256='b')}
    recipe=dict(sample_ids=list(PUBLIC_IDS),checkpoints=checkpoints,max_reconstruction_media_bytes=15000000)
    recipe_path=root/'configs/public-reconstructions-v1.json'; write_json(recipe_path,recipe)
    blob=lossless_webp(np.full((40,128,128,3),125,dtype=np.uint8)); size=len(blob); checksum=hashlib.sha256(blob).hexdigest()
    showcase=dict(clips=[]); clips=[]
    for cid in PUBLIC_IDS:
        source=public_source(cid); media={}
        for label in ['source','before','after']:
            relative=f'assets/{cid}-processed.webp' if label=='source' else f'assets/recon-{cid}-{label}.webp'
            (docs/relative).write_bytes(blob)
            media[label]=dict(path=relative,sha256=checksum,bytes=size)
        source['media']['processed']=media['source']; showcase['clips'].append(source)
        clips.append({**{k:source[k] for k in ['id','title','description','split','source_id','source_title','creator','source_url','fps','frames']},
            'width':128,'height':128,'columns':8,'media':media})
    write_json(docs/'showcase.json',showcase)
    data=dict(schema_version=1,kind=KIND,mode='public',checkpoints=checkpoints,optimizer_updates=0,
        recipe_sha256=digest(recipe_path),showcase_sha256=digest(docs/'showcase.json'),clips=clips,reconstruction_media_bytes=size*8)
    write_json(docs/'reconstructions.json',data)
    return data


def test_validator_binds_exact_existing_public_inputs_and_derived_assets(tmp_path):
    data=public_fixture(tmp_path)
    expected,total=validate(tmp_path)
    assert len(expected)==12 and total==data['reconstruction_media_bytes']
    (tmp_path/'docs/assets/recon-sample-01-after.webp').write_bytes(b'changed')
    with pytest.raises(ValueError,match='media changed'):
        validate(tmp_path)


@pytest.mark.parametrize('mutation', ['local_mode','embedded_source','wrong_source'])
def test_validator_refuses_private_bundle_or_new_source_pixels(tmp_path,mutation):
    data=public_fixture(tmp_path)
    if mutation=='local_mode': data['mode']='local'
    elif mutation=='embedded_source': data['clips'][0]['media']['source']={'data_url':'data:image/webp;base64,AAAA'}
    else: data['clips'][0]['media']['source']['path']='assets/unapproved.webp'
    write_json(tmp_path/'docs/reconstructions.json',data)
    with pytest.raises(ValueError):
        validate(tmp_path)
