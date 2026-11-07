"""Check selected channel files and decode endpoints; never approve training data."""
import argparse
import json
from pathlib import Path
import cv2
from genie_diep.data import file_hash


def run(root):
    root = Path(root)
    rows = json.loads((root / 'channel-catalog.json').read_text(encoding='utf-8'))
    selected = [r for r in rows if r['selection'] != 'excluded_other_game']
    files, missing, failures = [], [], []
    for row in selected:
        matches = [p for ext in ('mp4', 'webm', 'mkv') if (p := root / f"{row['id']}.{ext}").exists()]
        if not matches:
            missing.append(row['id'])
            continue
        path = matches[0]
        cap = cv2.VideoCapture(str(path))
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        first, _ = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, count - 2))
        last, _ = cap.read()
        cap.release()
        if not first or not last:
            failures.append(row['id'])
        files.append(dict(id=row['id'], path=path.name, bytes=path.stat().st_size,
                          sha256=file_hash(path)))
    report = dict(selected=len(selected), present=len(files), missing=missing,
                  endpoint_decode_failures=failures,
                  gib=sum(f['bytes'] for f in files) / 1024**3,
                  selected_hours=sum(r['duration'] or 0 for r in selected) / 3600,
                  present_hours=sum(r['duration'] or 0 for r in selected if r['id'] not in missing) / 3600,
                  warning='Endpoint decode checks only, not exhaustive integrity or gameplay curation.', files=files)
    (root / 'archive-verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'files'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', default='data/raw')
    run(parser.parse_args().raw)
