"""Validate a design/coverage checkpoint locally; does not authorize training."""
import argparse
import json
from pathlib import Path

from scripts.report_longform_coverage import read, digest


def validate_annotations(rows, annotations):
    expected = {r['review_index'] for r in rows}
    actual = [a['review_index'] for a in annotations]
    if len(expected) != len(rows) or len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError('Incomplete or duplicate annotations')
    if any(r['split'] not in {'train', 'val'} for r in rows):
        raise ValueError('Reserved test forbidden')
    for a in annotations:
        if a['firing'] not in {'seen', 'none_visible', 'uncertain'} or not a['notes']:
            raise ValueError('Invalid visual observation')


def validate(review_path, baseline_path):
    review, spec = read(review_path), read(baseline_path)
    assert review['training_approved'] is False and review['test_pixels_reviewed'] is False
    assert spec['training_approved'] is False and spec['execution_ready'] is False
    root = Path(spec['cohort'])
    assert digest(root/'manifest.jsonl') == spec['manifest_sha256']
    manifest = [json.loads(line) for line in (root/'manifest.jsonl').read_text().splitlines()]
    members = {(r['source_id'], r['block_id']): r for r in manifest}
    counts = {'blocks': 0, 'retained_frames': 0, 'accepted_blocks': 0, 'rejected_blocks': 0}
    holds = {h['source_id'] for h in review['training_holds']}
    for batch in review['batches']:
        index_path = Path(batch['index'])
        assert digest(index_path) == batch['index_sha256']
        index = read(index_path)
        assert index['request_sha256'] == digest(batch['request'])
        assert index['manifest_sha256'] == spec['manifest_sha256']
        validate_annotations(index['rows'], batch['annotations'])
        for row in index['rows']:
            assert len(row['source_frame_indices']) == 40
            for field in ('sheet', 'context'):
                assert digest(index_path.parent/row[field]) == row[field+'_sha256']
            if row['accepted']:
                original = members[(row['source_id'], row['block_id'])]
                for field in ('split', 'start', 'crop', 'proxy_sha256', 'video_sha256', 'source_frame_indices'):
                    assert row[field] == original[field]
                assert digest(root/original['proxy_path']) == row['proxy_sha256']
            counts['blocks'] += 1
            counts['retained_frames'] += 40
            counts['accepted_blocks' if row['accepted'] else 'rejected_blocks'] += 1
        lookup = {r['review_index']: r for r in index['rows']}
        for annotation in batch['annotations']:
            if annotation.get('quality_hold'):
                assert lookup[annotation['review_index']]['source_id'] in holds
    assert spec['coverage_review'] == str(review_path).replace('\\', '/')
    assert spec['training_hold_sources'] == sorted(holds)
    assert sum(s['max_compute_units'] for s in spec['stages']) == spec['budget']['total_max_compute_units']
    assert sum(s['max_allocated_hours'] for s in spec['stages']) == spec['budget']['total_max_allocated_hours']
    assert spec['data']['window_frames'] == spec['model']['context']
    assert spec['data']['window_span_seconds'] == (spec['model']['context']-1)/spec['data']['fps']
    assert spec['data']['spatial_tokens_per_frame'] == (spec['model']['size']//spec['model']['downsample'])**2
    from genie_diep.model import Config, VideoTokenizer, LatentActionModel, Dynamics
    cfg = Config(**spec['model'])
    models = {'tokenizer': VideoTokenizer(cfg), 'lam': LatentActionModel(cfg), 'dynamics': Dynamics(cfg)}
    parameters = {name: sum(p.numel() for p in model.parameters()) for name, model in models.items()}
    parameters['total'] = sum(parameters.values())
    assert parameters == spec['parameter_counts']
    return dict(counts=counts, parameter_counts=parameters, training_approved=False,
                execution_ready=False, test_pixels_reviewed=False,
                manifest_sha256=spec['manifest_sha256'], training_hold_sources=sorted(holds),
                scope='Hash/provenance, annotation completeness and static model sizing only; no training or performance validation.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review', default='configs/longform-focused-review-v1.json')
    parser.add_argument('--baseline', default='configs/baseline-v1.json')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    result = validate(args.review, args.baseline)
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result))
