"""Export bounded public derivatives and a separate, private local viewer bundle.

Only existing approved public processed atlases feed public reconstruction.
The complete held-out comparison is never written into the published directory.
"""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from genie_diep.cohort import read
from genie_diep.data import file_hash
from genie_diep.model import Config, WorldModel
from scripts.evaluate_tokenizer_quality import reconstruct, uint8_frames, validate_state


LABELS = ('source', 'before', 'after')
PUBLIC_IDS = ('sample-01', 'sample-02', 'sample-06', 'sample-07')
KIND = 'diep-genie-reconstructions'


def sha_bytes(blob):
    return hashlib.sha256(blob).hexdigest()


def atlas_image(frames, columns=8):
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[1:] != (128, 128, 3):
        raise ValueError('Expected uint8 THWC 128px frames')
    if not 1 <= len(frames) <= 40 or columns != 8:
        raise ValueError('Invalid bounded frame layout')
    image = Image.new('RGB', (128*columns, 128*((len(frames)+columns-1)//columns)))
    for index, frame in enumerate(frames):
        image.paste(Image.fromarray(frame), (index % columns*128, index//columns*128))
    return image


def lossless_webp(frames):
    image = atlas_image(frames)
    buffer = io.BytesIO()
    image.save(buffer, format='WEBP', lossless=True, method=6)
    blob = buffer.getvalue()
    with Image.open(io.BytesIO(blob)) as decoded:
        if not np.array_equal(np.asarray(image), np.asarray(decoded.convert('RGB'))):
            raise ValueError('Lossless export changed pixels')
    return blob


def decode_public_frames(path, media):
    blob = Path(path).read_bytes()
    if len(blob) != media['bytes'] or sha_bytes(blob) != media['sha256']:
        raise ValueError('Approved public input changed')
    with Image.open(io.BytesIO(blob)) as image:
        if image.size != (1024, 640):
            raise ValueError('Unexpected public atlas dimensions')
        pixels = np.asarray(image.convert('RGB'))
        return np.stack([pixels[i//8*128:(i//8+1)*128, i%8*128:(i%8+1)*128] for i in range(40)])


def selected_public_clips(showcase, recipe):
    if tuple(recipe['sample_ids']) != PUBLIC_IDS:
        raise ValueError('Public reconstruction selection changed')
    lookup = {clip['id']: clip for clip in showcase['clips']}
    if len(lookup) != len(showcase['clips']):
        raise ValueError('Duplicate showcase identity')
    selected = []
    for cid in PUBLIC_IDS:
        clip = lookup[cid]
        if clip['status'] != 'accepted' or clip['split'] not in {'train', 'val'}:
            raise ValueError('Only already-public retained train/validation samples are eligible')
        if clip['processed_kind'] != 'retained_proxy' or clip['frames'] != 40 or clip['fps'] != 10:
            raise ValueError('Unexpected public input contract')
        if clip['media']['processed']['path'] != f'assets/{cid}-processed.webp':
            raise ValueError('Public input path differs from exact allowlist')
        selected.append(clip)
    return selected


def validate_destinations(root, public_dir, private_out):
    root, public_dir, private_out = map(lambda p: Path(p).resolve(), (root, public_dir, private_out))
    if public_dir != root/'docs':
        raise ValueError('Public output must be the canonical docs directory')
    if not private_out.is_relative_to(root/'exports') or private_out.is_relative_to(public_dir):
        raise ValueError('Private bundle must stay under canonical exports, outside docs')
    if private_out.exists() or private_out.with_suffix('.receipt.json').exists() or (public_dir/'reconstructions.json').exists():
        raise FileExistsError('Choose a fresh export; never overwrite completed evidence')
    for cid in PUBLIC_IDS:
        for label in ('before', 'after'):
            if (public_dir/f'assets/recon-{cid}-{label}.webp').exists():
                raise FileExistsError('Reconstruction asset already exists')
    return root, public_dir, private_out


def manifest_base(mode, title, checkpoints):
    return dict(schema_version=1, kind=KIND, mode=mode, title=title,
        description='Same observed frames reconstructed by the tokenizer. These are not future predictions or playable controls.',
        checkpoints=checkpoints, optimizer_updates=0, clips=[])


def private_bundle(quality_dir, expected_checkpoints):
    quality_dir = Path(quality_dir)
    report = read(quality_dir/'quality-report.json')
    if report['windows'] != 64 or report['sources'] != 16 or len(report['records']) != 64:
        raise ValueError('Expected complete fixed held-out comparison')
    for label, expected in expected_checkpoints.items():
        if any(report['checkpoints'][label][key] != value for key, value in expected.items()):
            raise ValueError('Private evaluation checkpoint mismatch')
    bundle = manifest_base('local', 'Full held-out comparison: 64 fixed windows', expected_checkpoints)
    bundle.update(quality_report_sha256=file_hash(quality_dir/'quality-report.json'),
        selection_sha256=report['selection_sha256'], publication='Local import only; never upload this bundle or its frames.')
    for index, row in enumerate(report['records']):
        if row['index'] != index or row['selection']['length'] != 16:
            raise ValueError('Held-out window ordering changed')
        if row['artifact']['path'] != f'reconstruction-{index:03}.npz':
            raise ValueError('Unexpected private artifact path')
        path = quality_dir/row['artifact']['path']
        if file_hash(path) != row['artifact']['sha256']:
            raise ValueError('Held-out reconstruction changed')
        with np.load(path, allow_pickle=False) as arrays:
            if set(arrays.files) != set(LABELS):
                raise ValueError('Unexpected reconstruction arrays')
            media = {}
            for label in LABELS:
                frames = arrays[label]
                if frames.shape != (16, 128, 128, 3):
                    raise ValueError('Unexpected held-out frame shape')
                blob = lossless_webp(frames)
                media[label] = dict(data_url='data:image/webp;base64,'+base64.b64encode(blob).decode('ascii'),
                    bytes=len(blob), sha256=sha_bytes(blob))
        sid = row['selection']['source_id']
        bundle['clips'].append(dict(id=f'heldout-{index:03}', title=f'Fixed window {index:02d} · {sid}',
            description='Frozen held-out validation window. Frame 16 is the previously selected visual anchor.',
            split='val', source_id=sid, source_title=sid, creator='Recorded gameplay',
            source_url='https://www.youtube.com/watch?v='+sid, fps=10, frames=16, width=128, height=128,
            columns=8, diagnostic_frame=15, selection=row['selection'], media=media))
    if len({clip['source_id'] for clip in bundle['clips']}) != 16:
        raise ValueError('Held-out source coverage changed')
    return bundle


def run(args):
    root, public_dir, private_out = validate_destinations(args.root, args.public_dir, args.private_out)
    torch.set_num_threads(2)
    recipe_path = root/'configs/public-reconstructions-v1.json'
    recipe = read(recipe_path)
    showcase_path = public_dir/'showcase.json'; showcase = read(showcase_path)
    if file_hash(root/'configs/public-showcase-v1.json') != showcase['selection_sha256']:
        raise ValueError('Approved showcase recipe changed')
    selected = selected_public_clips(showcase, recipe)
    spec = read(root/'configs/training-baseline-v1.json')
    binding_hash = file_hash(root/spec['data_binding']); selection_hash = file_hash(root/spec['evaluation_selection'])
    quality_dir = Path(args.quality_dir).resolve()
    quality_summary = read(root/'configs/tokenizer-quality-v1-results.json')
    if file_hash(quality_dir/'quality-report.json') != quality_summary['artifacts']['quality-report.json']:
        raise ValueError('Private evaluation report differs from the reviewed result')
    quality_report = read(quality_dir/'quality-report.json')
    selection = read(root/spec['evaluation_selection'])
    if (quality_report['selection_sha256'] != selection_hash or quality_report['binding_sha256'] != binding_hash
            or [row['selection'] for row in quality_report['records']] != selection['validation']):
        raise ValueError('Private evaluation no longer matches the frozen validation selection')
    paths = {'before':Path(args.checkpoint_before).resolve(), 'after':Path(args.checkpoint_after).resolve()}
    models = {}
    for label, path in paths.items():
        expected = recipe['checkpoints'][label]
        if file_hash(path) != expected['sha256']:
            raise ValueError('Checkpoint file differs from the approved comparison')
        state = torch.load(path, map_location='cpu', weights_only=True)
        validate_state(state, spec, binding_hash, selection_hash)
        if state['step'] != expected['step'] or state['contract']['preflight'] != expected['preflight']:
            raise ValueError('Checkpoint stage metadata differs')
        model = WorldModel(Config(**state['config'])).cpu().float().eval()
        model.load_state_dict(state['model']); model.requires_grad_(False); models[label] = model
    # Prepare all private material in memory before any output. It cannot be
    # reached by a public path, even if caller supplies an unexpected location.
    bundle = private_bundle(quality_dir, recipe['checkpoints'])
    public = manifest_base('public', 'Reconstruction comparison', recipe['checkpoints'])
    public.update(recipe_sha256=file_hash(recipe_path), showcase_sha256=file_hash(showcase_path),
        policy='Four already-public accepted samples only. Illustrative examples, not the fixed held-out quality score.',
        device='cpu', precision='fp32', torch_version=str(torch.__version__),
        exporter_sha256=file_hash(__file__), model_implementation_sha256=hashlib.sha256((root/'genie_diep/model.py').read_bytes().replace(b'\r\n',b'\n')).hexdigest())
    generated = {}; original_inputs = {}
    for clip in selected:
        source_media = clip['media']['processed']; path = public_dir/source_media['path']
        original_inputs[path] = file_hash(path)
        frames = decode_public_frames(path, source_media)
        x = torch.from_numpy(frames).permute(0, 3, 1, 2).float()/255
        media = {'source':dict(source_media)}
        for label, model in models.items():
            prediction, _ = reconstruct(model.tokenizer, x, 4)
            blob = lossless_webp(uint8_frames(prediction))
            relative = f'assets/recon-{clip["id"]}-{label}.webp'
            generated[relative] = blob
            media[label] = dict(path=relative, bytes=len(blob), sha256=sha_bytes(blob))
        public['clips'].append({**{key:clip[key] for key in ['id','title','description','split','source_id','source_title','creator','source_url','fps','frames']},
            'width':128, 'height':128, 'columns':8, 'media':media})
        print(json.dumps(dict(reconstructed=clip['id'])), flush=True)
    public['reconstruction_media_bytes'] = sum(len(blob) for blob in generated.values())
    if public['reconstruction_media_bytes'] > recipe['max_reconstruction_media_bytes']:
        raise ValueError('Public reconstruction media exceeds bounded export')
    for path, expected in original_inputs.items():
        if file_hash(path) != expected:
            raise ValueError('Public source changed during reconstruction')
    for label, path in paths.items():
        if file_hash(path) != recipe['checkpoints'][label]['sha256']:
            raise ValueError('Checkpoint changed during export')
    encoded = (json.dumps(bundle, ensure_ascii=False, separators=(',',':'))+'\n').encode('utf-8')
    if len(encoded) > 200*1024*1024:
        raise ValueError('Private bundle exceeds viewer import cap')
    # Exclusive writes prevent accidental replacement of an earlier export.
    private_out.parent.mkdir(parents=True, exist_ok=True)
    with private_out.open('xb') as handle:
        handle.write(encoded)
    for relative, blob in generated.items():
        with (public_dir/relative).open('xb') as handle:
            handle.write(blob)
    with (public_dir/'reconstructions.json').open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(public, handle, indent=2, ensure_ascii=False, allow_nan=False); handle.write('\n')
    receipt = dict(public_manifest_sha256=file_hash(public_dir/'reconstructions.json'),
        private_bundle=dict(path=private_out.name, bytes=len(encoded), sha256=sha_bytes(encoded)),
        checkpoint_files_unchanged=True, public_inputs_unchanged=True, optimizer_updates=0,
        public_samples=len(selected), private_windows=64, private_sources=16)
    with private_out.with_suffix('.receipt.json').open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(receipt, handle, indent=2); handle.write('\n')
    print(json.dumps(receipt), flush=True)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--public-dir', required=True)
    parser.add_argument('--private-out', required=True)
    parser.add_argument('--checkpoint-before', required=True)
    parser.add_argument('--checkpoint-after', required=True)
    parser.add_argument('--quality-dir', required=True)
    run(parser.parse_args())
