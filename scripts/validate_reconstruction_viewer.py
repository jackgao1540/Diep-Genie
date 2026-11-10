"""Verify public reconstruction scope, hashes and exact published-file allowlist."""
import hashlib
import json
from pathlib import Path

from PIL import Image

from scripts.export_reconstruction_viewer import KIND, PUBLIC_IDS, selected_public_clips


UI_FILES = {'reconstructions.html', 'reconstruction-viewer.js', 'reconstruction-viewer.css'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(root):
    root = Path(root); docs = root/'docs'
    data = json.loads((docs/'reconstructions.json').read_text(encoding='utf-8'))
    recipe_path = root/'configs/public-reconstructions-v1.json'
    recipe = json.loads(recipe_path.read_text(encoding='utf-8'))
    showcase = json.loads((docs/'showcase.json').read_text(encoding='utf-8'))
    selected = selected_public_clips(showcase, recipe)
    allowed = {'schema_version','kind','mode','title','description','checkpoints','optimizer_updates','clips',
        'recipe_sha256','showcase_sha256','policy','device','precision','torch_version','exporter_sha256',
        'model_implementation_sha256','reconstruction_media_bytes'}
    if not set(data) <= allowed:
        raise ValueError('Unexpected public manifest fields')
    if data['schema_version'] != 1 or data['kind'] != KIND or data['mode'] != 'public':
        raise ValueError('Unexpected viewer manifest')
    if data['recipe_sha256'] != sha(recipe_path) or data['showcase_sha256'] != sha(docs/'showcase.json'):
        raise ValueError('Public provenance binding changed')
    if data['checkpoints'] != recipe['checkpoints'] or data['optimizer_updates'] != 0:
        raise ValueError('Public checkpoint metadata changed')
    if [c['id'] for c in data['clips']] != list(PUBLIC_IDS):
        raise ValueError('Public sample allowlist differs')
    expected = {'reconstructions.json'} | UI_FILES; total = 0
    for clip, source in zip(data['clips'], selected):
        if set(clip) != {'id','title','description','split','source_id','source_title','creator','source_url','fps','frames',
                'width','height','columns','media'}:
            raise ValueError('Unexpected public clip fields')
        for key in ['id','title','description','split','source_id','source_title','creator','source_url','fps','frames']:
            if clip[key] != source[key]:
                raise ValueError('Published sample provenance changed: '+key)
        if (clip['width'],clip['height'],clip['columns']) != (128,128,8) or set(clip['media']) != {'source','before','after'}:
            raise ValueError('Unexpected frame atlas contract')
        if clip['media']['source'] != source['media']['processed']:
            raise ValueError('Viewer input is not the already-public processed atlas')
        for label, media in clip['media'].items():
            relative = f'assets/{clip["id"]}-processed.webp' if label=='source' else f'assets/recon-{clip["id"]}-{label}.webp'
            if set(media) != {'path','bytes','sha256'} or media['path'] != relative:
                raise ValueError('Unexpected public media path or embedded/private media')
            path = docs/relative
            if sha(path) != media['sha256'] or path.stat().st_size != media['bytes']:
                raise ValueError('Published media changed')
            with Image.open(path) as image:
                if image.size != (1024,640) or image.format != 'WEBP':
                    raise ValueError('Invalid frame atlas')
            if label != 'source':
                expected.add(relative); total += media['bytes']
    if total != data['reconstruction_media_bytes'] or total > recipe['max_reconstruction_media_bytes']:
        raise ValueError('Public reconstruction size cap changed')
    for name in UI_FILES:
        if not (docs/name).is_file():
            raise ValueError('Viewer UI file missing: '+name)
    text = (docs/'reconstructions.json').read_text(encoding='utf-8')
    for private in ['data:image','C:\\Users','localhost','127.0.0.1','quality-report.json','heldout-','feedback.jsonl']:
        if private in text:
            raise ValueError('Private material found in public manifest: '+private)
    return expected, total


if __name__ == '__main__':
    files, size = validate(Path(__file__).resolve().parents[1])
    print(json.dumps(dict(validation='passed', additional_public_files=len(files), reconstruction_media_bytes=size)))
