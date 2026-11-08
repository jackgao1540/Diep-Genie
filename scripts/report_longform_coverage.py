"""Describe a frozen cohort and prepare a content-bound semantic review sample.

Test metadata is summarized, but test pixels are never selected for this review.
Eight original-crop frames establish observable coverage, not action ground truth
or a new all-frame quality audit. Sampling is uniform within train/val separately.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def key(row):
    return row['source_id'], row['block_id']


def select_sample(rows, seed, counts):
    if set(counts) - {'train', 'val'}:
        raise ValueError('Reserved test must not enter semantic review')
    if len({key(r) for r in rows}) != len(rows):
        raise ValueError('Repeated source block')
    durations = {r['frames'] / r['fps'] for r in rows}
    if len(durations) != 1:
        raise ValueError('Uniform block sampling requires equal durations')
    chosen = []
    for split, count in counts.items():
        pool = [r for r in rows if r['split'] == split]
        if count < 1 or count > len(pool):
            raise ValueError('Invalid sample size')
        rank = lambda r: hashlib.sha256((seed + '|' + '|'.join(key(r))).encode()).hexdigest()
        chosen.extend(sorted(pool, key=rank)[:count])
    return chosen


def describe(rows, sequences, sources):
    result = {}
    for split in ['all', 'train', 'val', 'test']:
        rr = [r for r in rows if split == 'all' or r['split'] == split]
        runs = [r for r in sequences if split == 'all' or r['split'] == split]
        by_source, by_channel, by_pool = Counter(), Counter(), Counter()
        for r in rr:
            seconds = r['frames'] / r['fps']
            s = sources[r['source_id']]
            by_source[r['source_id']] += seconds
            by_channel[s['channel']] += seconds
            by_pool[s['pool']] += seconds
        total = sum(by_source.values())
        if total != sum(r['seconds'] for r in runs):
            raise ValueError('Manifest/sequence duration mismatch')
        ranked = by_source.most_common()
        result[split] = dict(seconds=total, blocks=len(rr), videos=len(by_source),
            seconds_by_channel=dict(by_channel), seconds_by_pool=dict(by_pool),
            sources=[dict(source_id=sid, seconds=sec, channel=sources[sid]['channel']) for sid, sec in ranked],
            top1_share=ranked[0][1]/total, top5_share=sum(v for _, v in ranked[:5])/total,
            effective_source_count=total**2/sum(v*v for v in by_source.values()),
            runs=len(runs), longest_seconds=max(r['seconds'] for r in runs),
            seconds_in_runs_at_least={str(n):sum(r['seconds'] for r in runs if r['seconds'] >= n) for n in [10,20,30,60]})
    return result


def summarize_annotations(index, review):
    rows = index['rows']
    if any(r['split'] not in {'train', 'val'} for r in rows):
        raise ValueError('Reserved test in semantic sample')
    if [r['review_index'] for r in rows] != list(range(len(rows))):
        raise ValueError('Noncanonical review indices')
    annotations = review['annotations']
    if sorted(a['review_index'] for a in annotations) != list(range(len(rows))):
        raise ValueError('Missing, repeated, or extra semantic annotations')
    fields = list(review['definitions'])
    for a in annotations:
        if set(a) != {'review_index', *fields}:
            raise ValueError('Annotation schema mismatch')
        for f in fields:
            allowed = {'seen', 'not_seen', 'uncertain'}
            if f == 'player_palette':
                allowed = {'cyan', 'red', 'green', 'purple', 'uncertain'}
            elif f == 'barrel_layout':
                allowed = {'single', 'multi', 'uncertain'}
            if a[f] not in allowed:
                raise ValueError('Invalid semantic value')
    result = {}
    for split in ['train', 'val']:
        selected = [a for a in annotations if rows[a['review_index']]['split'] == split]
        source_counts = Counter(rows[a['review_index']]['source_id'] for a in selected)
        result[split] = dict(blocks=len(selected), seconds=sum(rows[a['review_index']]['seconds'] for a in selected),
            videos=len(source_counts), sampled_blocks_by_source=dict(source_counts),
            counts={f:dict(Counter(a[f] for a in selected)) for f in fields},
            sources_with_observation={f:len({rows[a['review_index']]['source_id'] for a in selected if a[f]=='seen'})
                                      for f in fields if f not in ['player_palette', 'barrel_layout']})
    return result


def finalize(args):
    review = read(args.review)
    review_root = Path(review['review'])
    index_path = review_root/'index.json'
    if digest(index_path) != review['index_sha256']:
        raise ValueError('Semantic review evidence changed')
    index = read(index_path)
    root = Path(args.cohort)
    for name in ['manifest', 'sequences']:
        path = root/('manifest.jsonl' if name == 'manifest' else 'sequences.json')
        if digest(path) != index[name+'_sha256']:
            raise ValueError('Frozen cohort changed')
    if digest(args.recipe) != index['recipe_sha256']:
        raise ValueError('Recipe changed')
    for sheet in index['sheets']:
        if digest(review_root/sheet['path']) != sheet['sha256']:
            raise ValueError('Rendered evidence changed')
    rows = [json.loads(l) for l in (root/'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    counts = Counter(r['split'] for r in index['rows'])
    selected = select_sample(rows, index['seed'], counts)
    if [key(r) for r in selected] != [key(r) for r in index['rows']]:
        raise ValueError('Sample is not reproducible')
    for r, sampled in zip(selected, index['rows']):
        expected = dict(review_index=sampled['review_index'], source_id=r['source_id'], block_id=r['block_id'],
            split=r['split'], start=r['start'], seconds=r['frames']/r['fps'], proxy_sha256=r['proxy_sha256'],
            video_sha256=r['video_sha256'], crop=r['crop'],
            original_frame_indices=[r['source_frame_indices'][f] for f in index['frame_offsets']], page=sampled['page'])
        if expected != sampled:
            raise ValueError('Sample provenance mismatch')
    recipe = read(args.recipe)
    sources = {s['id']:dict(s) for p in recipe['source_recipes'] for s in read(p)['sources']}
    channels = {}
    for sid, meta in index['source_metadata'].items():
        source = sources[sid]
        meta_path = Path(source['media_path']).with_suffix('.info.json')
        if digest(meta_path) != meta['metadata_sha256']:
            raise ValueError('Channel metadata changed')
        raw = read(meta_path)
        channel_id = raw.get('channel_id')
        if not channel_id:
            raise ValueError('Missing stable channel ID')
        channels[sid] = channel_id
        source.update(channel=meta['channel'], pool=Path(source['media_path']).parts[1])
    metadata = describe(rows, read(root/'sequences.json'), sources)
    for split, stats in metadata.items():
        rr = [r for r in rows if split == 'all' or r['split'] == split]
        seconds = Counter()
        for r in rr:
            seconds[channels[r['source_id']]] += r['frames']/r['fps']
        stats['channel_count'] = len(seconds)
        stats['seconds_by_channel_id'] = dict(seconds)
    result = dict(manifest_sha256=index['manifest_sha256'], sequences_sha256=index['sequences_sha256'],
        review_sha256=digest(args.review), index_sha256=review['index_sha256'],
        metadata=metadata, semantic_sample=summarize_annotations(index, review),
        limitations=[__doc__, review['method'], review['value_policy'],
            'Source/channel counts describe uploads, not independent creators or gameplay sessions; transformed reuse may remain.',
            'Sample is uniform over equal-duration blocks within each split. Split sample sizes need not be corpus proportional; do not pool percentages.',
            'A single reviewer supplied exploratory observations. No inter-rater validation, calibrated action labels, or exhaustive class taxonomy.'],
        corpus_complete=False, training_approved=False)
    out = Path(args.out)
    if out.exists():
        raise FileExistsError('Choose a new immutable summary path')
    out.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result['semantic_sample'], indent=2))


def run(args):
    import cv2
    from PIL import Image, ImageDraw
    cv2.setNumThreads(1)
    root, out = Path(args.cohort), Path(args.out)
    if out.exists():
        raise FileExistsError('Choose a new immutable review directory')
    recipe = read(args.recipe)
    sources = {s['id']:dict(s) for p in recipe['source_recipes'] for s in read(p)['sources']}
    rows = [json.loads(l) for l in (root/'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    source_splits, hash_splits = defaultdict(set), defaultdict(set)
    for r in rows:
        s = sources[r['source_id']]
        if (r['video_sha256'], r['crop'], r['split']) != (s['video_sha256'], s['crop'], s['split']):
            raise ValueError('Source provenance mismatch')
        if not r['curation_accepted'] or r['training_approved']:
            raise ValueError('Unexpected frozen cohort status')
        source_splits[r['source_id']].add(r['split'])
        hash_splits[r['video_sha256']].add(r['split'])
    if any(len(v) != 1 for v in [*source_splits.values(), *hash_splits.values()]):
        raise ValueError('Source split leakage')
    for sid in source_splits:
        s = sources[sid]
        meta_path = Path(s['media_path']).with_suffix('.info.json')
        meta = read(meta_path)
        if meta['id'] != sid:
            raise ValueError('Metadata identity mismatch')
        s.update(channel=meta.get('channel') or meta.get('uploader') or 'unknown',
                 pool=Path(s['media_path']).parts[1], metadata_sha256=digest(meta_path))
    selected = select_sample(rows, args.seed, {'train':args.train_count, 'val':args.val_count})
    selected_sources = {r['source_id'] for r in selected}
    for sid in selected_sources:
        if digest(sources[sid]['media_path']) != sources[sid]['video_sha256']:
            raise ValueError('Original media changed')
    for r in selected:
        if digest(root/r['proxy_path']) != r['proxy_sha256']:
            raise ValueError('Sample proxy changed')
    out.mkdir(parents=True)
    stats = describe(rows, read(root/'sequences.json'), sources)
    frame_offsets = [0,5,10,15,20,25,30,39]
    sample_rows, sheets = [], []
    for page in range((len(selected)+3)//4):
        sheet = Image.new('RGB', (1600, 900), 'white')
        draw = ImageDraw.Draw(sheet)
        for slot, r in enumerate(selected[page*4:(page+1)*4]):
            i = page*4+slot
            y = slot*225
            draw.text((4,y+3), f"#{i} {r['split']} {r['source_id']} {r['start']:.2f}s | original crop; offsets 0 .5 1 1.5 2 2.5 3 3.9s", fill='black')
            cap = cv2.VideoCapture(sources[r['source_id']]['media_path'])
            x0,y0,w,h = r['crop']
            for col, offset in enumerate(frame_offsets):
                cap.set(cv2.CAP_PROP_POS_FRAMES, r['source_frame_indices'][offset])
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError('Original frame decode failed')
                crop = frame[y0:y0+h,x0:x0+w]
                if crop.shape[:2] != (h,w):
                    raise ValueError('Invalid crop')
                im = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                sheet.paste(im.resize((200,200)), (col*200,y+23))
            cap.release()
            sample_rows.append(dict(review_index=i, source_id=r['source_id'], block_id=r['block_id'],
                split=r['split'], start=r['start'], seconds=r['frames']/r['fps'],
                proxy_sha256=r['proxy_sha256'], video_sha256=r['video_sha256'], crop=r['crop'],
                original_frame_indices=[r['source_frame_indices'][f] for f in frame_offsets], page=page+1))
        path = out/f'coverage-{page+1:03}.png'
        sheet.save(path)
        sheets.append(dict(path=path.name, sha256=digest(path)))
        print(f'Rendered {page+1}/{(len(selected)+3)//4}', flush=True)
    index = dict(cohort=str(root), manifest_sha256=digest(root/'manifest.jsonl'),
        sequences_sha256=digest(root/'sequences.json'), recipe_sha256=digest(args.recipe),
        seed=args.seed, selection='Smallest SHA256(seed|source_id|block_id), within each split; equal-duration blocks, no replacement',
        frame_offsets=frame_offsets, rows=sample_rows, sheets=sheets,
        test_pixels_reviewed=False, training_approved=False,
        source_metadata={sid:dict(channel=sources[sid]['channel'], metadata_sha256=sources[sid]['metadata_sha256']) for sid in sorted(source_splits)})
    (out/'index.json').write_text(json.dumps(index, indent=2)+'\n', encoding='utf-8')
    (out/'statistics.json').write_text(json.dumps(stats, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({split:{k:v for k,v in s.items() if k not in ['sources']} for split,s in stats.items()}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort', default='data/curated-longform-v5')
    p.add_argument('--recipe', default='configs/longform-cohort-v5.json')
    p.add_argument('--out', required=True)
    p.add_argument('--seed', default='genie-semantic-coverage-v1-2026-11-08')
    p.add_argument('--train-count', type=int, default=64)
    p.add_argument('--val-count', type=int, default=32)
    p.add_argument('--review', help='Validate a completed review and write summary JSON to --out instead of rendering')
    args = p.parse_args()
    finalize(args) if args.review else run(args)
