"""Resumable local archive from a previously fetched public channel inventory.

Title matching selects candidates, not approved training footage. Ambiguous
titles stay marked for visual review. No cookies, remote code or paid services.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from genie_diep.data import source_split,file_hash


def catalog(inventory):
    rows=[]
    for entry in inventory.get('entries',[]):
        title=entry.get('title','');video_id=entry['id']
        if not re.fullmatch(r'[A-Za-z0-9_-]{11}',video_id):raise ValueError('Invalid public video identifier')
        # A rival game's title can also contain "Diep.io"; exclusion comes first.
        if re.search(r'(?:obstar|tanksmith)[.\s-]*io',title,re.I):status='excluded_other_game'
        elif re.search(r'diep[.\s-]*io|diepio',title,re.I):status='diep_title_candidate'
        else:status='ambiguous_review_required'
        rows.append(dict(id=video_id,title=title,duration=entry.get('duration'),
                         channel=inventory.get('channel'),url=f'https://www.youtube.com/watch?v={video_id}',
                         split=source_split(video_id),selection=status,training_approved=False))
    return rows


def run(args):
    root=Path(args.out);root.mkdir(parents=True,exist_ok=True)
    rows=catalog(json.loads(Path(args.inventory).read_text(encoding='utf-8-sig')))
    (root/'channel-catalog.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    selected=[r for r in rows if r['selection']!='excluded_other_game']
    if args.limit:selected=selected[:args.limit]
    logs=root/'acquisition-logs';logs.mkdir(exist_ok=True)
    print(json.dumps(dict(total=len(rows),selected=len(selected),hours=sum(r['duration'] or 0 for r in selected)/3600)),flush=True)
    for row in selected:
        video_id=row['id'];existing=[p for ext in ('mp4','webm','mkv') if (p:=root/f'{video_id}.{ext}').exists()]
        if existing:
            print(json.dumps(dict(id=video_id,status='already_present')),flush=True);continue
        used=sum(p.stat().st_size for p in root.iterdir() if p.is_file())
        if used>=args.max_gib*1024**3 or shutil.disk_usage(root).free<args.min_free_gib*1024**3:
            raise RuntimeError('Local storage guard reached; preserve existing files and review capacity')
        command=[sys.executable,'-m','yt_dlp','--ignore-config','--no-playlist','--no-remote-components',
                 '--socket-timeout','20','--retries','2','--fragment-retries','2','--no-progress',
                 '--max-filesize','500M','--write-info-json','--no-write-playlist-metafiles',
                 '--download-archive',str(root/'archive.txt'),'-f','best[height<=480]/bestvideo[height<=480]',
                 '-o',str(root/'%(id)s.%(ext)s'),'--js-runtimes','node:'+args.node,row['url']]
        started=time.monotonic()
        with (logs/f'{video_id}.log').open('a',encoding='utf-8') as output:
            try:
                completed=subprocess.run(command,stdout=output,stderr=subprocess.STDOUT,timeout=240)
                status='downloaded' if completed.returncode==0 else 'failed'
            except subprocess.TimeoutExpired:status='timeout'
        files=[p for ext in ('mp4','webm','mkv') if (p:=root/f'{video_id}.{ext}').exists()]
        if status=='downloaded' and not files:status='skipped_without_media'
        record=dict(id=video_id,status=status,seconds=round(time.monotonic()-started,2),
                    files=[dict(path=p.name,bytes=p.stat().st_size,sha256=file_hash(p)) for p in files])
        with (root/'acquisition.jsonl').open('a',encoding='utf-8') as output:output.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
        # Respect service limits: do not keep retrying or switch identities.
        if re.search(r'HTTP Error 429|Too Many Requests', (logs/f'{video_id}.log').read_text(encoding='utf-8'),re.I):
            raise RuntimeError('Service rate limit encountered; stop and retry later')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('inventory');p.add_argument('--out',default='data/raw')
    p.add_argument('--node',required=True);p.add_argument('--limit',type=int,default=0)
    p.add_argument('--max-gib',type=float,default=20);p.add_argument('--min-free-gib',type=float,default=20)
    run(p.parse_args())
