"""Build fresh private staged-tokenizer launchers; never allocate compute."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import zipfile

from genie_diep.data import file_hash
from scripts.build_drive_trial_notebook import ARCHIVE_BYTES, ARCHIVE_SHA256
from scripts.package_small_codebook_architecture import notebook as overlay_notebook


def notebook(manifest):
    # Keep the independently tested archive/overlay validation and pre-mount watchdog.
    contract = dict(manifest, maximum_compute_units=1, maximum_allocated_seconds=2700,
                    minimum_remaining_units=50)
    document = overlay_notebook(contract)
    document['cells'][0]['source'] = [
        '# Private staged 16-code transformer: bounded CUDA pilot\n\n'
        'Execution and resource measurement only. Maximum 1 compute unit and 45 allocated minutes, '
        'including every setup, idle and retry debit; retain 50 units. No purchases. '
        'Both stages must pass the runner checks. This pilot never launches full training.\n\n'
        'Before connecting, upload and verify the private overlay and immutable base archive. '
        'Record the actual allocation start immediately on connection; enter current balance/rate '
        'and cumulative prior debits below. Run configuration then guarded execution immediately. '
        'If execution cannot start, disconnect and delete runtime in the UI immediately. '
        'The watchdog arms before Drive mount. Success and failure both release the runtime. '
        'Do not share this notebook or its outputs.\n']
    code = ''.join(document['cells'][2]['source'])
    code = code.replace("pathlib.Path('/content/drive/MyDrive/Diep-Genie').resolve()",
                        "pathlib.Path('/content/drive/MyDrive/Diep-Genie/staged-tokenizer-run-v1').resolve()")
    # This runner accepts separate prior time/CU debits. Use the same exact
    # remaining limits so the outer shell never preempts its checkpoint reserve.
    begin = code.index('    # The immutable pilot accepts')
    end = code.index('    # Begin shutdown', begin)
    code = code[:begin]+'''    deadline = allocation_start + min(2700 - PRIOR_ALLOCATED_SECONDS,
        (1 - PRIOR_COMPUTE_UNITS) / DISPLAYED_UNITS_PER_HOUR * 3600,
        (LIVE_BALANCE - 50) / DISPLAYED_UNITS_PER_HOUR * 3600)
'''+code[end:]
    code = code.replace("'conservative_accounting_offset_seconds': accounting_offset",
                        "'accounting_policy': 'separate_cumulative_time_and_units'")
    code = code.replace('assert AUTHORIZED_TO_RUN, "Authorize only this bounded measurement first"',
        'try:\n    assert AUTHORIZED_TO_RUN, "Authorize only this bounded pilot first"\n'
        'except BaseException:\n    runtime.unassign()\n    raise')
    start = code.index('    command = [sys.executable')
    end = code.index('\nexcept BaseException as error:', start)
    command = """    command = [sys.executable, '-u', '-m', 'scripts.run_staged_tokenizer',
        '--root', str(root), '--plan', PLAN_PATH, '--out', str(result/'experiment'),
        '--private-output-root', '/content/drive/MyDrive/Diep-Genie/staged-tokenizer-run-v1',
        '--mode', 'pilot', '--device', 'cuda', '--authorized',
        '--rate', str(DISPLAYED_UNITS_PER_HOUR), '--balance', str(LIVE_BALANCE),
        '--allocation-start', str(allocation_start),
        '--prior-seconds', str(PRIOR_ALLOCATED_SECONDS), '--prior-units', str(PRIOR_COMPUTE_UNITS)]
    if RESUME_RUN_DIRECTORY: command.append('--resume')
    status['status'] = 'pilot'
    (result/'launch-status.json').write_text(json.dumps(status, indent=2)+'\\n')
    with (result/'experiment.log').open('a' if RESUME_RUN_DIRECTORY else 'w', buffering=1) as log:
        subprocess.run(command, cwd=root, check=True, stdout=log, stderr=subprocess.STDOUT,
            timeout=max(1, stop_at-time.time()-60))
    status['status'] = 'pilot_runner_completed'
    print('Private pilot saved:', str(result), flush=True)
""".replace('PLAN_PATH', repr(manifest['plan_path']))
    code = code[:start]+command+code[end:]
    # Always leave a local receipt, even when mounting Drive fails. Drive receipt remains authoritative.
    marker = "        if result is not None and result.exists():"
    code = code.replace(marker,
        "        pathlib.Path('/content/staged-pilot-launch-status.json').write_text(json.dumps(status, indent=2)+'\\n')\n"+marker)
    code = code.replace("'purpose': 'Sequential fresh CNN32 and ST32; shared allocation'",
                        "'purpose': 'Staged transformer pilot only; full training forbidden'")
    document['cells'][2]['source'] = code.splitlines(keepends=True)
    for index, cell in enumerate(document['cells']):
        if cell['cell_type'] == 'code': compile(''.join(cell['source']), f'pilot-cell-{index}', 'exec')
    return document


def full_notebook(manifest):
    """A distinct bound full launcher; never mutate or promote the pilot launcher."""
    if manifest.get('mode') != 'full' or not manifest.get('full_budget_path'):
        raise ValueError('Full launch requires a bound measured-pilot budget receipt')
    document = notebook(manifest)
    document['cells'][0]['source'] = [
        '# Private staged 16-code transformer: full training\n\n'
        'Two fresh 5,000-update stages, using the successful exact bound CUDA pilot. '
        'Shared TOTAL maximum: 30 compute units and 26 allocated hours including pilot, '
        'setup, idle and retries; preserve 50 units. No purchases. The measured forecast '
        'must fit remaining budget. The second stage imports the actual full first-stage '
        'encoder/codebook; pilot weights are never promoted. This remains a tokenizer '
        'experiment, with held-out quality and controllability unmeasured.\n\n'
        'Before connecting, verify private uploads and enter fresh live balance/rate, actual '
        'allocation start and ALL earlier allocated time/CU debits (including pilot). '
        'Run the guarded cell immediately after configuring. If it cannot start, delete '
        'runtime through the UI immediately. Resume only the exact bound full result '
        'directory with cumulative accounting. Both success and failure release runtime.\n']
    code = ''.join(document['cells'][2]['source'])
    code = code.replace('min(2700 - PRIOR_ALLOCATED_SECONDS,',
                        'min(93600 - PRIOR_ALLOCATED_SECONDS,')
    code = code.replace('(1 - PRIOR_COMPUTE_UNITS) / DISPLAYED_UNITS_PER_HOUR',
                        '(30 - PRIOR_COMPUTE_UNITS) / DISPLAYED_UNITS_PER_HOUR')
    code = code.replace("'maximum_compute_units': 1, 'maximum_allocated_seconds': 2700",
                        "'maximum_compute_units': 30, 'maximum_allocated_seconds': 93600")
    code = code.replace("'--mode', 'pilot'", "'--mode', 'full', '--full-budget', str(root/"+repr(manifest['full_budget_path'])+")")
    code = code.replace('/Diep-Genie/staged-tokenizer-run-v1', '/Diep-Genie/staged-tokenizer-full-v1')
    code = code.replace("status['status'] = 'pilot'", "status['status'] = 'full_training'")
    code = code.replace("status['status'] = 'pilot_runner_completed'", "status['status'] = 'full_runner_completed'")
    code = code.replace('Staged transformer pilot only; full training forbidden', 'Bound measured-pilot full staged training')
    code = code.replace('Authorize only this bounded pilot first', 'Authorize only this bound full experiment')
    code = code.replace('Private pilot saved:', 'Private full experiment saved:')
    code = code.replace('staged-pilot-launch-status.json', 'staged-full-launch-status.json')
    document['cells'][2]['source'] = code.splitlines(keepends=True)
    for index, cell in enumerate(document['cells']):
        if cell['cell_type'] == 'code': compile(''.join(cell['source']), f'full-cell-{index}', 'exec')
    return document


def package(root, output, plan_path='configs/staged-tokenizer-run-v1.json',
            pilot_report=None, prior_seconds=None, prior_units=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists(): raise FileExistsError('Choose a fresh immutable private export')
    from scripts.run_staged_tokenizer import validate_plan
    plan = validate_plan(root, plan_path)
    # All current source modules plus explicit runnable scripts; never package datasets or checkpoints.
    files = list((root/'genie_diep').glob('*.py'))
    files += [root/name for name in [
        'pyproject.toml', 'scripts/run_staged_tokenizer.py', 'scripts/run_small_codebook_architecture.py',
        'scripts/run_tokenizer_comparison.py', 'scripts/diagnose_tokenizer_precision.py', plan_path]]
    # Include only runtime-read configurations. The data binding's historical audit
    # references are provenance, not input dependencies; do not leak those private notes.
    for entry in plan['stages'].values():
        spec_path = root/entry['spec']; files.append(spec_path)
        spec = json.loads(spec_path.read_text())
        files += [root/spec[key] for key in ('data_binding', 'evaluation_selection')]
    hashes = {path.relative_to(root).as_posix(): file_hash(path) for path in sorted(set(files))}
    for name in hashes:
        if not (name.startswith(('genie_diep/', 'scripts/', 'configs/')) or name == 'pyproject.toml'):
            raise ValueError('Non-source payload denied: '+name)
    extra = {}
    full_budget_path = 'runs/staged-tokenizer-full-gate-v1/full-budget.json'
    if pilot_report is not None:
        pilot_bytes = Path(pilot_report).read_bytes()
        pilot = json.loads(pilot_bytes)
        if (pilot['identity']['plan_sha256'] != file_hash(root/plan_path)
                or pilot['status'] != 'pilot_complete'
                or pilot['identity'].get('mode') != 'pilot'
                or pilot['identity'].get('local_check') is not False):
            raise ValueError('Successful pilot must bind this exact plan')
        if prior_seconds is None or prior_units is None or not all(
            math.isfinite(v) and v >= 0 for v in (prior_seconds, prior_units)) or prior_seconds < pilot['spent_seconds'] or prior_units < pilot['spent_units']:
            raise ValueError('Full launch must retain complete pilot/prior allocation debits')
        pilot_path = 'runs/staged-tokenizer-full-gate-v1/pilot-report.json'
        receipt = dict(authorized=True, purpose='staged16-full-5000-each',
            max_compute_units=30, max_allocated_hours=26, minimum_remaining_units=50,
            pilot=dict(report_path=pilot_path, sha256=hashlib.sha256(pilot_bytes).hexdigest()),
            minimum_prior_seconds=prior_seconds, minimum_prior_units=prior_units)
        extra[pilot_path] = pilot_bytes
        extra[full_budget_path] = (json.dumps(receipt, indent=2)+'\n').encode()
        hashes.update({name: hashlib.sha256(data).hexdigest() for name, data in extra.items()})
    output.mkdir(parents=True)
    overlay = output/('genie-'+output.name+'.zip')
    with zipfile.ZipFile(overlay, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for name in hashes:
            if name in extra: archive.writestr(name, extra[name])
            else: archive.write(root/name, name)
        archive.writestr('overlay-sha256.json', json.dumps(hashes, indent=2)+'\n')
    with zipfile.ZipFile(overlay) as archive:
        for name, digest in hashes.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest: raise ValueError('Overlay readback mismatch')
    manifest = dict(overlay_name=overlay.name, overlay_bytes=overlay.stat().st_size,
        overlay_sha256=file_hash(overlay), files=hashes, plan_path=plan_path,
        plan_sha256=file_hash(root/plan_path), maximum_compute_units=1,
        maximum_allocated_seconds=2700, minimum_remaining_units=50,
        base_archive_name='genie-baseline-v1-v9.zip', base_archive_sha256=ARCHIVE_SHA256,
        base_archive_bytes=ARCHIVE_BYTES, execution_authorized_in_file=False,
        mode='pilot', full_training_enabled=False, contains_checkpoints=False)
    if pilot_report is not None:
        manifest.update(mode='full', full_training_enabled=True, full_budget_path=full_budget_path,
            maximum_compute_units=30, maximum_allocated_seconds=93600,
            full_budget_sha256=hashes[full_budget_path], pilot_report_sha256=hashes[pilot_path])
    target = output/('staged-tokenizer-'+manifest['mode']+'.ipynb')
    target.write_text(json.dumps(full_notebook(manifest) if pilot_report is not None else notebook(manifest), indent=2)+'\n', encoding='utf-8')
    manifest['notebook_sha256'] = file_hash(target)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='.')
    parser.add_argument('--out', required=True)
    parser.add_argument('--plan', default='configs/staged-tokenizer-run-v1.json')
    parser.add_argument('--pilot-report', help='Build separate full launcher from this exact successful private pilot')
    parser.add_argument('--prior-seconds', type=float)
    parser.add_argument('--prior-units', type=float)
    args = parser.parse_args()
    print(json.dumps(package(args.root, args.out, args.plan, args.pilot_report,
        args.prior_seconds, args.prior_units), indent=2))
