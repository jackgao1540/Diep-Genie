"""Create bound triage evidence for the remaining preserved low-motion queue."""
from pathlib import Path
import json
from PIL import Image, ImageDraw
from genie_diep.data import file_hash
from scripts.preserve_screened_candidates import content_id


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def run():
    bank=Path('data/preserved-candidates-v1');out=Path('runs/remaining-low-motion-v1')
    out.mkdir(exist_ok=False)
    queue=read(bank/'low-motion-review.json')['rows']
    assert file_hash(bank/'low-motion-review.json')==read(bank/'summary.json')['hashes']['low-motion-review.json']
    active={content_id(json.loads(s)) for s in Path('data/curated-longform-v7/manifest.jsonl').read_text().splitlines()}
    remaining=[dict(r,queue_index=i) for i,r in enumerate(queue) if r['id'] not in active]
    assert len(remaining)==84
    visual=[r for r in remaining if not r['reserved_test']]
    sheets=[]
    for start in range(0,len(visual),5):
        page=Image.new('RGB',(1240,5*232),'#f5f5ef');draw=ImageDraw.Draw(page)
        for slot,r in enumerate(visual[start:start+5]):
            art=r['review_artifacts']
            for name in ['retained-frames.png','context.png','frames.npz']:
                assert file_hash(bank/art[name]['path'])==art[name]['sha256']
            w,h=r['crop'][2:];full=Image.open(bank/art['retained-frames.png']['path'])
            y=slot*232;draw.text((3,y+2),f"Queue {r['queue_index']} | {r['source_id']} | {r['block_id']} | {r['start']:.2f}s | {r['split']}",fill='black')
            for j,f in enumerate([0,20,39]):
                crop=full.crop((f%8*w,f//8*(h+18)+18,(f%8+1)*w,f//8*(h+18)+18+h))
                page.paste(crop.resize((200,200)),(j*200,y+25))
            context=Image.open(bank/art['context.png']['path']);context.thumbnail((640,200));page.paste(context,(600,y+25))
        path=out/f'triage-{start//5:02}.png';page.save(path);sheets.append(dict(path=path.as_posix(),sha256=file_hash(path),queue_indices=[r['queue_index'] for r in visual[start:start+5]]))
    result=dict(parent_manifest_sha256=file_hash('data/curated-longform-v7/manifest.jsonl'),queue_sha256=file_hash(bank/'low-motion-review.json'),remaining=remaining,sheets=sheets,reserved_test_visual_review=False)
    (out/'index.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(remaining=len(remaining),visual=len(visual),reserved_test=len(remaining)-len(visual),sheets=len(sheets))))


if __name__=='__main__':run()
