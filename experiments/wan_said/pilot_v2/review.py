"""Convert a completed human review sheet into a dataset-bound approval record."""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from ..data import load_split
from ..io_utils import sha256, write_json
from .config import load_config, signature


def approve(cfg, sheet, audit_path, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Do not overwrite an existing human review record')
    audit = json.loads(Path(audit_path).read_text())
    if (audit['preprocessing_hash'] != signature(cfg) or
            audit['split_lock_sha256'] != sha256(Path(cfg['manifests'])/'split_lock.json')):
        raise ValueError('Review audit belongs to another preprocessing protocol or split')
    allowed = {r['sample_id']: r for s in ('train', 'val') for r in load_split(cfg['manifests'], s)}
    samples = {r['sample_id']: r for r in audit['samples']}
    expected = set(samples)
    with Path(sheet).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows or len(rows) != len(expected) or {r['sample_id'] for r in rows} != expected:
        raise ValueError('Review must cover every audited sample exactly once')
    rejected = []
    for row in rows:
        if (row['sample_id'] not in allowed or row['caption'] != allowed[row['sample_id']]['caption'] or
                row['split'] != allowed[row['sample_id']]['split'] or
                row['preview'] != samples[row['sample_id']]['preview']):
            raise ValueError('Changed caption, unknown sample, or sealed test sample in review')
        if not row['reviewer'].strip() or row['approve'] not in ('yes', 'no'):
            raise ValueError('Each row needs a named human reviewer and approve=yes/no')
        flags = [row[k] for k in ('event_coverage', 'caption_match', 'crop_preserves_subject')]
        if any(flag not in ('yes', 'no') for flag in flags):
            raise ValueError('Complete all event/caption/crop decisions with yes/no')
        if row['approve'] == 'yes' and 'no' in flags:
            raise ValueError('A failed event/caption/crop check cannot be approved')
        if row['approve'] == 'no':
            if not row['rejection_reason'].strip():
                raise ValueError('Rejected samples require a recorded reason')
            rejected.append(row['sample_id'])
    result = dict(approved=not rejected, reviewer=sorted({r['reviewer'] for r in rows}),
                  recorded_at_utc=datetime.now(timezone.utc).isoformat(),
                  reviewed_sample_ids=sorted(expected), rejected_sample_ids=rejected,
                  decisions=rows, review_sheet_sha256=sha256(sheet), audit_sha256=sha256(audit_path),
                  preprocessing_hash=signature(cfg), split_lock_sha256=audit['split_lock_sha256'],
                  scope='Representative pilot audit only, not a claim of corpus-wide semantic validity',
                  policy='Any rejection blocks formal training; revise/filter and freeze a new dataset before approval')
    write_json(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--sheet', required=True)
    parser.add_argument('--audit', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    approve(load_config(args.config), args.sheet, args.audit, args.output)
