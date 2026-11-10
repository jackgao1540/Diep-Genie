"""Summarize an already audited paired run; no training, images or gate scoring.

Only complete paired fixed assessments enter comparisons. Source win fractions
are descriptive signs, not significance tests. Training calibration has one
window from each of 84 sources; validation has four from each of 16 other sources.
Their difference is therefore a probe difference, not an isolated generalization
gap. The raw baseline/detail optimization losses are intentionally not compared:
the arms optimize different objectives.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


METRICS = ('mse', 'foreground_mse', 'temporal_mse')
ARMS = ('baseline', 'detail')
TIE_ABSOLUTE_TOLERANCE = 1e-12


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def change(reference, candidate):
    if reference is None or candidate is None:
        return dict(reference=reference, candidate=candidate, decrease=None, percent_lower=None)
    require(all(type(value) in (int, float) and math.isfinite(value) and value >= 0
                for value in (reference, candidate)), 'Invalid reconstruction metric')
    return dict(reference=reference, candidate=candidate, decrease=reference-candidate,
                percent_lower=100*(reference-candidate)/reference if reference else None)


def compare_metrics(baseline, detail):
    require(baseline['complete'] and detail['complete'], 'Cannot compare partial assessments')
    require(baseline['samples'] == detail['samples'] and baseline['sources'] == detail['sources']
            and baseline['by_source'].keys() == detail['by_source'].keys(), 'Arm assessment populations differ')
    result = {}
    for metric in METRICS:
        per_source = {}
        wins = losses = ties = unavailable = 0
        for source in sorted(baseline['by_source']):
            a, b = baseline['by_source'][source], detail['by_source'][source]
            require(a['samples'] == b['samples'], 'Arm source sample counts differ')
            pair = change(a[metric], b[metric])
            per_source[source] = pair
            delta = pair['decrease']
            if delta is None:
                unavailable += 1
            elif abs(delta) <= TIE_ABSOLUTE_TOLERANCE:
                ties += 1
            elif delta > 0:
                wins += 1
            else:
                losses += 1
        paired = [pair for pair in per_source.values() if pair['decrease'] is not None]
        equal = change(sum(pair['reference'] for pair in paired)/len(paired),
                       sum(pair['candidate'] for pair in paired)/len(paired)) if paired else change(None, None)
        result[metric] = dict(pooled=change(baseline[metric], detail[metric]), equal_source=equal,
            source_signs=dict(detail_lower=wins, detail_higher=losses, tied=ties, unavailable=unavailable,
                              assessable=wins+losses+ties, total_sources=len(per_source),
                              detail_lower_fraction=wins/(wins+losses+ties) if wins+losses+ties else None),
            per_source=per_source)
    return result


def summarize(report_path, audit_path):
    report_path, audit_path = Path(report_path).resolve(), Path(audit_path).resolve()
    report_sha, audit_sha = sha(report_path), sha(audit_path)
    report, audit = read(report_path), read(audit_path)
    require(audit['status'] == 'validated' and audit['promotion_approved'] is False
            and audit['quality_gates'] == 'unmeasured', 'Require an unpromoted artifact audit')
    require(report['experiment_sha256'] == audit['experiment_sha256']
            and report['latest_common_step'] == audit['latest_common_step']
            and report['status'] == audit['run_status'], 'Report differs from artifact audit')
    bound = [row for row in audit['files'] if row['path'] == 'continuation-report.json']
    require(len(bound) == 1 and bound[0]['sha256'] == report_sha, 'Report bytes differ from artifact audit')
    require(audit['sampling_independently_replayed'] is True and audit['parent_checkpoints_unchanged'] is True
            and audit['frozen_lam_dynamics_exact'] is True, 'Incomplete coverage or lineage evidence')
    require(all(report['coverage'][arm] == audit['coverage'] for arm in ARMS), 'Coverage differs from audit')
    entries = {}
    for entry in report['evaluations']:
        key = (entry['common_step'], entry['arm'], entry['report']['split'])
        require(key not in entries, 'Duplicate fixed assessment')
        entries[key] = entry['report']
    steps = sorted({key[0] for key in entries})
    paired = []
    excluded = []
    gaps = []
    for step in steps:
        for split in ('train', 'val'):
            records = {arm: entries.get((step, arm, split)) for arm in ARMS}
            if not all(record is not None and record['complete'] for record in records.values()):
                excluded.append(dict(step=step, split=split, reason='Missing or incomplete paired assessment'))
                continue
            paired.append(dict(step=step, split=split, samples=records['baseline']['samples'],
                               sources=records['baseline']['sources'], **compare_metrics(**records)))
        for arm in ARMS:
            train, val = entries.get((step, arm, 'train')), entries.get((step, arm, 'val'))
            if train and val and train['complete'] and val['complete']:
                gaps.append(dict(step=step, arm=arm, metrics={metric: dict(
                    train=train[metric], validation=val[metric],
                    validation_minus_train=val[metric]-train[metric] if train[metric] is not None and val[metric] is not None else None,
                    validation_over_train=val[metric]/train[metric] if train[metric] and val[metric] is not None else None)
                    for metric in METRICS}))
    changes = []
    for step in steps:
        if step == 999:
            continue
        for arm in ARMS:
            for split in ('train', 'val'):
                start, finish = entries.get((999, arm, split)), entries.get((step, arm, split))
                if start and finish and start['complete'] and finish['complete']:
                    changes.append(dict(step=step, arm=arm, split=split,
                                        metrics={metric: change(start[metric], finish[metric]) for metric in METRICS}))
    require(sha(report_path) == report_sha and sha(audit_path) == audit_sha, 'Input changed during summary')
    return dict(schema_version=1, status='descriptive_summary', experiment_sha256=report['experiment_sha256'],
        report_sha256=report_sha, artifact_audit_sha256=audit_sha, latest_common_step=report['latest_common_step'],
        run_status=report['status'], coverage=audit['coverage'], paired_assessments=paired,
        excluded_assessments=excluded, arm_changes_from_999=changes, train_validation_probe_differences=gaps,
        policy=dict(lower_is_better=True, paired_positive_percent_lower_favors_detail=True,
                    arm_change_positive_percent_lower='That arm improved relative to its own step999 probe',
                    source_tie_absolute_tolerance=TIE_ABSOLUTE_TOLERANCE,
                    equal_source='Mean over the same sources with available metrics in both arms',
                    counts='Fixed windows and sources are correlated; source wins are descriptive, not statistical significance',
                    train_validation='Different source populations and coverage; probe differences do not isolate overfitting or data sufficiency',
                    temporal='Successive-frame reconstruction differences; not prediction or control',
                    coverage='Prepared TRAIN frames, not validation/test or all original native frames; full coverage does not imply convergence'),
        inference_limits=['One matched initialization seed and one explicitly changed shared sampling schedule',
                          'Repeated fixed validation informs diagnosis; it is not an untouched final test',
                          'No semantic geometry, food/bullet count, angle or control conclusions follow from these metrics'],
        quality_gates='unmeasured', promotion_approved=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True)
    parser.add_argument('--audit', required=True)
    parser.add_argument('--out')
    args = parser.parse_args()
    result = summarize(args.report, args.audit)
    encoded = json.dumps(result, indent=2, allow_nan=False)+'\n'
    if args.out:
        destination = Path(args.out).resolve()
        require(not destination.is_relative_to(Path(args.report).resolve().parent), 'Summary must be outside the audited run')
        with destination.open('x', encoding='utf-8') as stream:
            stream.write(encoded)
    else:
        print(encoded, end='')


if __name__ == '__main__':
    main()
