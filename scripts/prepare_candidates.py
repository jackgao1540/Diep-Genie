"""Extract a reviewed source shortlist into a NEW, unapproved candidate dataset."""
import argparse
import json
from pathlib import Path
from genie_diep.data import extract


def main():
    p=argparse.ArgumentParser();p.add_argument('recipe');p.add_argument('out');p.add_argument('--raw',default='data/raw')
    args=p.parse_args();recipe=json.loads(Path(args.recipe).read_text());out=Path(args.out)
    if out.exists():raise FileExistsError('Preserve existing candidate datasets; choose a new output')
    out.mkdir(parents=True);records=[]
    for interval in recipe['intervals']:
        source=interval['source_id'];folder=out/source/f"{interval['start']:07.2f}"
        video=Path(args.raw)/f'{source}.mp4'
        rows=extract(video,folder,source,start=interval['start'],seconds=interval['seconds'],
                     crop=recipe['crop'],size=recipe['size'],require_tank=True)
        for row in rows:
            row['path']=(folder/row['path']).relative_to(out).as_posix()
            row['source_review_note']=interval['note'];row['training_approved']=False
            records.append(row)
    (out/'clips.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records),encoding='utf-8')
    (out/'recipe.json').write_text(json.dumps(recipe,indent=2),encoding='utf-8')
    print(json.dumps(dict(candidates=len(records),approved=0)))


if __name__=='__main__':main()
