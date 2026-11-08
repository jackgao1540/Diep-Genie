import json
from pathlib import Path
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest
from PIL import Image
import io

from scripts.serve_footage_review import Library, make_handler, ThreadingHTTPServer
from genie_diep.data import file_hash


@pytest.fixture
def library(tmp_path):
    library=Library.__new__(Library)
    library.root=tmp_path;library.feedback_path=tmp_path/'feedback.jsonl'
    library.lock=threading.RLock();library.render_lock=threading.Lock()
    from collections import OrderedDict
    library.cache=OrderedDict();library.token='test-token';library.manifest_hash='manifest'
    proxy=tmp_path/'clip.npz'
    frames=np.zeros((40,128,128,3),np.uint8)
    for i in range(40):frames[i,:,:]=[i,2*i,3*i]
    np.savez_compressed(proxy,frames=frames)
    row=dict(source_id='source',block_id='000_000001',video_sha256='source-hash',proxy_sha256=file_hash(proxy),source_frame_indices=list(range(40)),crop=[0,0,128,128],_proxy=proxy)
    library.records={'known':row};library.feedback={};library.public={'known':{'status':'held'}}
    return library


def test_review_is_durable_content_bound_and_not_approval(library):
    result=library.save({'id':'known','decision':'good','note':'Clearly visible.'})
    saved=json.loads(library.feedback_path.read_text())
    assert saved==result and saved['content_sha256']==library.content_hash(library.records['known'])
    assert saved['training_approved'] is False
    assert library.public['known']['status']=='held'
    library.save({'id':'known','decision':'clear'})
    assert len(library.feedback_path.read_text().splitlines())==2
    assert library.feedback['known']['decision']=='clear'


def test_unknown_clip_and_invalid_feedback_are_rejected(library):
    for payload in [{'id':'test-clip','decision':'good'}, {'id':'known','decision':'approve_training'}, {'id':'known','decision':'flag','note':'x'*4001}]:
        with pytest.raises(ValueError):library.save(payload)
    with pytest.raises(KeyError):library.picture('../secret','proxy')
    assert not library.feedback_path.exists()


def test_proxy_atlas_preserves_exact_frame_order(library):
    atlas=np.asarray(Image.open(io.BytesIO(library.picture('known','proxy'))))
    assert atlas.shape==(640,1024,3)
    for i in range(40):assert atlas[i//8*128,i%8*128].tolist()==[i,2*i,3*i]
    library.cache.clear()
    library.records['known']['_proxy'].write_bytes(b'changed')
    with pytest.raises(ValueError,match='Proxy changed'):library.picture('known','proxy')


def test_http_prevents_cross_origin_writes_and_arbitrary_files(library,tmp_path):
    (tmp_path/'index.html').write_text('review')
    server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(library,tmp_path))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    base=f'http://127.0.0.1:{server.server_port}'
    try:
        body=json.dumps({'id':'known','decision':'flag','note':'test'}).encode()
        def post(origin,token):
            return urllib.request.urlopen(urllib.request.Request(base+'/api/review',data=body,headers={'Origin':origin,'X-Review-Token':token,'Content-Type':'application/json'}))
        for origin,token in [('https://unrelated.example',library.token),(base,'bad')]:
            with pytest.raises(urllib.error.HTTPError) as error:post(origin,token)
            assert error.value.code==403
        with post(base,library.token) as response:assert json.load(response)['decision']=='flag'
        for path in ['/HISTORY.md','/../secret','/api/image/reserved-test/proxy']:
            with pytest.raises(urllib.error.HTTPError) as error:urllib.request.urlopen(base+path)
            assert error.value.code==404
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(urllib.request.Request(base+'/',headers={'Host':'unrelated.example'}))
        assert error.value.code==403
    finally:server.shutdown();server.server_close();thread.join()
