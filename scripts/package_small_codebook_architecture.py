"""Build a fresh disabled launcher/source overlay; never allocate a runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from genie_diep.baseline import validate_readiness
from genie_diep.data import file_hash
from scripts.build_drive_trial_notebook import notebook as base_notebook
from scripts.run_small_codebook_architecture import validate_plan


def notebook(manifest):
    document = base_notebook()
    document['cells'][0]['source'] = [
        '# Two fresh 32-code tokenizer tests\n\n'
        'CNN32 first, then near-size-matched causal transformer32. Both receive '
        '5,000 unique updates, identical frozen data order, 4-sequence microbatches '
        'and four accumulation steps. Both CUDA pilots run before either full run. '
        'Shared hard limits: 4 compute units / four allocated hours, preserving '
        '50 units. Setup, idle, precision/replay checks, validation and exports '
        'all count. No action/dynamics training or quality promotion.\n\n'
        'The overlay contains a verified untrained starting checkpoint and must '
        'remain private. This exported launcher is disabled.\n']
    config = ''.join(document['cells'][1]['source'])
    config += '\nRESUME_RUN_DIRECTORY = ""  # Existing absolute result directory on mounted Drive, only for reviewed recovery\n'
    document['cells'][1]['source'] = config.splitlines(keepends=True)
    source = ''.join(document['cells'][2]['source'])
    source = source.replace('min(3600, 2 / DISPLAYED_UNITS_PER_HOUR * 3600,', 'min(14400, 4 / DISPLAYED_UNITS_PER_HOUR * 3600,')
    source = source.replace("'maximum_compute_units': 2, 'maximum_allocated_seconds': 3600,", "'maximum_compute_units': 4, 'maximum_allocated_seconds': 14400,")
    source = source.replace("'purpose': 'GPU execution measurement only'", "'purpose': 'Sequential fresh CNN32 and ST32; shared allocation'")
    source = source.replace("result = pathlib.Path('/content/drive/MyDrive/Diep-Genie/baseline-v1-v9') / time.strftime('pilot-%Y%m%d-%H%M%S', time.gmtime())\n    result.mkdir(parents=True, exist_ok=False)",
        "result = pathlib.Path(RESUME_RUN_DIRECTORY) if RESUME_RUN_DIRECTORY else pathlib.Path('/content/drive/MyDrive/Diep-Genie/small-codebook-architecture-v1') / time.strftime('run-%Y%m%d-%H%M%S', time.gmtime())\n    assert result.resolve().is_relative_to(pathlib.Path('/content/drive/MyDrive/Diep-Genie').resolve()), 'Unsafe recovery result path'\n    result.mkdir(parents=True, exist_ok=bool(RESUME_RUN_DIRECTORY))")
    source = source.replace("pathlib.Path('/content/genie-baseline-v1-v9')", "pathlib.Path('/content/genie-small-codebook-architecture-v1')")
    status_write = next(line for line in source.splitlines() if line.startswith("    (result/'launch-status.json').write_text"))
    source = source.replace(status_write,
        "    if RESUME_RUN_DIRECTORY and (result/'launch-status.json').exists():\n"
        "        shutil.copyfile(result/'launch-status.json', result/('launch-status-prior-'+str(time.time_ns())+'.json'))\n"+status_write, 1)
    marker = "    assert time.time() + 180 < stop_at, 'Setup exhausted the measurement budget'"
    overlay = '''    overlay_contract = OVERLAY_MANIFEST
    overlay_path = pathlib.Path('/content/drive/MyDrive') / overlay_contract['overlay_name']
    assert overlay_path.stat().st_size == overlay_contract['overlay_bytes'], 'Overlay size mismatch'
    with overlay_path.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == overlay_contract['overlay_sha256'], 'Overlay digest mismatch'
    with zipfile.ZipFile(overlay_path) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names)), 'Duplicate overlay member'
        assert set(names) == set(overlay_contract['files']) | {'overlay-sha256.json'}, 'Unexpected overlay member'
        assert json.loads(archive.read('overlay-sha256.json')) == overlay_contract['files'], 'Overlay member table differs'
        assert sum(info.file_size for info in archive.infolist()) <= 64*1024**2, 'Expanded overlay too large'
        payloads = {}
        for name, digest in overlay_contract['files'].items():
            target = (root/name).resolve(); info = archive.getinfo(name)
            assert target.is_relative_to(root.resolve()), 'Unsafe overlay path'
            assert not info.is_dir() and (info.external_attr >> 16) & 0o170000 != 0o120000, 'Overlay symlink denied'
            assert info.file_size <= (32 if name.endswith('.pt') else 2)*1024**2, 'Oversized overlay member'
            data = archive.read(name)
            assert hashlib.sha256(data).hexdigest() == digest, 'Overlay member changed: '+name
            payloads[name] = data
        for name, data in payloads.items():
            target = root/name; target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    status['overlay_sha256'] = overlay_contract['overlay_sha256']
    status['plan_sha256'] = overlay_contract['plan_sha256']
'''.replace('OVERLAY_MANIFEST', repr(manifest))
    if source.count(marker) != 1: raise ValueError('Base launcher setup marker changed')
    source = source.replace(marker, overlay+marker)
    begin = source.index('    # The pilot requires a new directory')
    end = source.index('\nexcept BaseException as error:', begin)
    command = '''    command = [sys.executable, '-u', '-m', 'scripts.run_small_codebook_architecture',
        '--plan', 'configs/small-codebook-architecture-v1.json', '--out', str(result/'experiment'),
        '--device', 'cuda', '--authorized', '--rate', str(DISPLAYED_UNITS_PER_HOUR),
        '--balance', str(LIVE_BALANCE), '--allocation-start', str(accounting_start)]
    if RESUME_RUN_DIRECTORY: command.append('--resume')
    status['status'] = 'training'
    (result/'launch-status.json').write_text(json.dumps(status, indent=2)+'\\n')
    with (result/'experiment.log').open('a' if RESUME_RUN_DIRECTORY else 'w', buffering=1) as log:
        subprocess.run(command, cwd=root, check=True, stdout=log, stderr=subprocess.STDOUT,
            timeout=max(1, stop_at-time.time()-180))
    report = json.loads((result/'experiment/experiment-report.json').read_text())
    status['status'] = report['status']
    print('Private experiment saved:', str(result), json.dumps(report, indent=2), flush=True)
'''
    source = source[:begin]+command+source[end:]
    # Configuration validation also releases an already connected runtime.
    start = source.index('assert math.isfinite(LIVE_BALANCE)')
    finish = source.index('finished = threading.Event()', start)
    validation = source[start:finish]
    source = source[:start]+'try:\n'+''.join('    '+line if line.strip() else line for line in validation.splitlines(keepends=True))+'except BaseException:\n    runtime.unassign()\n    raise\n'+source[finish:]
    document['cells'][2]['source'] = source.splitlines(keepends=True)
    for index, cell in enumerate(document['cells']):
        if cell['cell_type'] == 'code': compile(''.join(cell['source']), f'cell-{index}', 'exec')
    return document


def package(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    plan = validate_plan(root, 'configs/small-codebook-architecture-v1.json')
    for entry in plan['arms'].values(): validate_readiness(root, entry['spec'])
    if output.exists(): raise FileExistsError('Choose a fresh immutable export')
    files = list((root/'genie_diep').glob('*.py'))
    files += [root/name for name in ['scripts/run_small_codebook_architecture.py', 'scripts/run_tokenizer_comparison.py',
        'scripts/diagnose_tokenizer_precision.py', 'configs/small-codebook-architecture-v1.json', plan['initial_checkpoint']['path']]]
    for entry in plan['arms'].values():
        files.append(root/entry['spec']); files.append(root/read_spec(root/entry['spec'])['readiness'])
    hashes = {path.relative_to(root).as_posix(): file_hash(path) for path in sorted(set(files))}
    output.mkdir(parents=True); overlay = output/'genie-small-codebook-architecture-v1.zip'
    with zipfile.ZipFile(overlay, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for name in hashes: archive.write(root/name, name)
        archive.writestr('overlay-sha256.json', json.dumps(hashes, indent=2)+'\n')
    manifest = dict(overlay_name=overlay.name, overlay_sha256=file_hash(overlay), overlay_bytes=overlay.stat().st_size,
        files=hashes, plan_sha256=file_hash(root/'configs/small-codebook-architecture-v1.json'),
        contains_private_untrained_checkpoint=True, maximum_compute_units=4, maximum_allocated_seconds=14400,
        minimum_remaining_units=50, execution_authorized_in_file=False)
    with zipfile.ZipFile(overlay) as archive:
        for name, digest in hashes.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest: raise ValueError('Overlay readback mismatch')
    document = notebook(manifest)
    target = output/'small-codebook-architecture.ipynb'
    target.write_text(json.dumps(document, indent=2)+'\n')
    manifest['notebook_sha256'] = file_hash(target)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def read_spec(path):
    return json.loads(Path(path).read_text())


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='.'); p.add_argument('--out', required=True)
    args = p.parse_args(); print(json.dumps(package(args.root, args.out), indent=2))
