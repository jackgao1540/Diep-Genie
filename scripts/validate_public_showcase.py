"""Validate the bounded static publication without accessing the private dataset."""
import hashlib
import json
from pathlib import Path
from PIL import Image


def validate(root):
    root = Path(root)
    data = json.loads((root / 'docs/showcase.json').read_text(encoding='utf-8'))
    recipe = json.loads((root / 'configs/public-showcase-v1.json').read_text(encoding='utf-8'))
    assert hashlib.sha256((root / 'configs/public-showcase-v1.json').read_bytes()).hexdigest() == data['selection_sha256']
    assert data['training_approved'] is False
    assert len(data['clips']) == data['sample_count'] == 13
    assert data['sample_seconds'] == 52
    assert {c['id'] for c in data['clips']} == {c['id'] for c in recipe['selection']}
    assert [sum(c['status'] == s for c in data['clips']) for s in ['accepted', 'held', 'rejected']] == [8, 1, 4]
    total = 0
    expected = {'index.html', 'style.css', 'app.js', 'showcase.json', '.nojekyll'}
    for clip in data['clips']:
        assert clip['split'] in {'train', 'val'}
        assert clip['frames'] == 40 and clip['fps'] == 10 and clip['duration'] == 4
        assert len(clip['provenance']['source_frame_indices']) == 40
        assert clip['source_url'].startswith('https://www.youtube.com/watch?v=' + clip['source_id'] + '&t=')
        kinds = {'accepted': 'retained_proxy', 'held': 'withheld_proxy', 'rejected': 'illustrative_resize_not_training_data'}
        assert clip['processed_kind'] == kinds[clip['status']]
        for name, media in clip['media'].items():
            relative = f"assets/{clip['id']}-{name}.webp"
            assert media['path'] == relative
            expected.add(relative)
            path = root / 'docs' / relative
            blob = path.read_bytes()
            assert hashlib.sha256(blob).hexdigest() == media['sha256'], relative
            assert len(blob) == media['bytes']
            total += len(blob)
            w, h = clip['source_crop_size'] if name == 'original' else [128, 128]
            with Image.open(path) as image:
                assert image.size == ((128, 128) if name == 'thumbnail' else (w * 8, h * 5)), relative
    assert {p.relative_to(root / 'docs').as_posix() for p in (root / 'docs').rglob('*') if p.is_file()} == expected
    assert total == data['media_bytes'] <= recipe['max_media_bytes']
    public_text = '\n'.join((root / 'docs' / name).read_text(encoding='utf-8') for name in ['index.html', 'app.js', 'style.css', 'showcase.json'])
    for private in ['127.0.0.1', 'localhost', 'C:\\Users', 'feedback.jsonl', '/api/', 'feedback_token']:
        assert private not in public_text, private
    print(json.dumps({'samples': len(data['clips']), 'media_bytes': total, 'public_files': len(expected), 'reserved_test_samples': 0, 'validation': 'passed'}))


if __name__ == '__main__':
    validate(Path(__file__).resolve().parents[1])
