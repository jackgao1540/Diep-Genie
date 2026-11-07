import json
from types import SimpleNamespace
import numpy as np
import torch
from genie_diep.action_pool import initialize_book,update_books,collect_features
from genie_diep.model import VQ,Config,LatentActionModel
from genie_diep.data import ClipDataset


def test_spherical_initialization_separates_unlabeled_clusters():
    book=VQ(4,2)
    features=torch.tensor([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]]).repeat(8,1)
    initialize_book(book,features)
    ids=book(features)[1].reshape(8,4)
    assert ids[0].unique().numel()==4
    assert torch.equal(ids,ids[0].expand_as(ids))


def test_revival_clears_only_replaced_optimizer_rows():
    book=VQ(4,2)
    with torch.no_grad():book.embedding.weight.copy_(torch.tensor([[1.,0.]]).repeat(4,1))
    optimizer=torch.optim.AdamW(book.parameters())
    optimizer.state[book.embedding.weight]={'exp_avg':torch.ones(4,2),'exp_avg_sq':torch.ones(4,2)}
    features=torch.tensor([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])[:,None]
    result=update_books(SimpleNamespace(books=[book]),features,optimizer)
    assert result[0]['changed']==[1,2,3]
    assert all(count>0 for count in result[0]['counts'])
    state=optimizer.state[book.embedding.weight]
    assert state['exp_avg'][0].sum()==2 and state['exp_avg'][1:].sum()==0


def test_pool_covers_sources_and_does_not_read_validation(tmp_path):
    rows=[]
    for source in ('a','b'):
        for i in range(3):
            name=f'{source}{i}.npy'
            np.save(tmp_path/name,np.zeros((4,16,16,3),dtype=np.uint8))
            rows.append(dict(path=name,source_id=source,split='train',start=i,shape=[4,16,16,3]))
    # Missing validation file would fail if it entered the feature pool.
    rows.append(dict(path='missing.npy',source_id='val',split='val',shape=[4,16,16,3]))
    (tmp_path/'clips.jsonl').write_text('\n'.join(map(json.dumps,rows)))
    dataset=ClipDataset(tmp_path,'train',2)
    indices=dataset.representative_indices(4)
    assert {dataset.rows[i]['source_id'] for i in indices}=={'a','b'}
    assert len(indices)==len(set(indices))==4
    lam=LatentActionModel(Config(size=16,width=8,action_dim=4,action_codes=(4,),lam_variant='delta'))
    features,selected=collect_features(lam,dataset,'cpu',limit=4,batch_size=2)
    assert selected==indices and features.shape==(4,1,4)
