"""Download selected ZIP members through HF-mirror, never entire multi-GB archives."""
import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import random
import re
import shutil
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
from remotezip import RemoteZip

from .data import audit_splits, caption_key, perceptual_signature
from .io_utils import read_jsonl, sha256, write_json, write_jsonl
from .mirror_download import StrictRangeSession, resolve_url
from .runtime import ROOT

REPO = 'nkp37/OpenVid-1M'
REVISION = 'd8a63bd22989c80b5734ec2bb989f4e1b61a5807'
DEFAULT_ROOT = ROOT / 'datasets/openvid'
MAX_DATA_BYTES = 100_000_000_000


def source_identity(filename):
    stem = Path(filename).stem
    if stem.startswith('celebv_'):
        match = re.fullmatch(r'celebv_([A-Za-z0-9_-]{11})_\d+', stem)
        if match:
            return 'youtube:' + match.group(1), 'celebv'
    match = re.fullmatch(r'([A-Za-z0-9_-]{11})_\d+_\d+to\d+', stem)
    if match:
        return 'youtube:' + match.group(1), 'panda'
    if stem.startswith('pixabay_'):
        match = re.fullmatch(r'(pixabay_.+_\d+)_\d+', stem)
        if match:
            return match.group(1), 'pixabay'
    raise ValueError(f'Unknown parent-video naming scheme; refusing unsafe split: {filename}')


def index_archive(part, destination):
    destination = Path(destination)
    if destination.exists():
        return read_jsonl(destination)
    url = resolve_url(REPO, REVISION, f'OpenVid_part{part}.zip')
    with StrictRangeSession() as session:
        with RemoteZip(url, session=session) as archive:
            rows = []
            for info in archive.infolist():
                if not info.filename.endswith('.mp4') or not 50_000 <= info.file_size <= 4_000_000:
                    continue
                try:
                    source, family = source_identity(info.filename)
                except ValueError:
                    continue
                rows.append(dict(archive=f'OpenVid_part{part}.zip', member=info.filename,
                                 name=Path(info.filename).name, bytes=info.file_size,
                                 compressed_bytes=info.compress_size, crc32=info.CRC,
                                 source_id=source, family=family))
    write_jsonl(destination, rows)
    print(f'Indexed part {part}: {len(rows)} candidates <= 4 MB', flush=True)
    return rows


def build_plan(root, parts, samples, seed):
    root = Path(root)
    plan_path = root / 'download_plan.jsonl'
    if plan_path.exists():
        plan = read_jsonl(plan_path)
        lock = json.loads((root / 'selection.json').read_text())
        if lock['samples_requested'] != samples or lock['seed'] != seed or lock['parts'] != parts:
            raise ValueError('Selection arguments differ from existing immutable plan')
        return plan
    candidates = []
    for part in parts:
        candidates += index_archive(part, root / 'metadata' / f'part{part}_index.jsonl')
    by_name = {r['name']: r for r in candidates}
    csv_path = root / 'metadata/OpenVid-1M.csv'
    if not csv_path.exists():
        raise FileNotFoundError('Caption CSV download must finish and pass SHA256 first')
    with csv_path.open(newline='', encoding='utf-8') as stream:
        for meta in csv.DictReader(stream):
            record = by_name.get(meta['video'])
            if record is None:
                continue
            seconds = float(meta['seconds'])
            text = meta['caption'].strip()
            if not 2.1 <= seconds <= 30 or not 8 <= len(text.split()) <= 200:
                continue
            record.update(caption=text, caption_group=caption_key(text), duration=seconds,
                          original_fps=float(meta['fps']), aesthetic_score=float(meta['aesthetic score']),
                          motion_score=float(meta['motion score']),
                          window_start=round(max(0., (seconds - 2.0) / 2), 6))
    candidates = sorted([r for r in candidates if 'caption' in r], key=lambda r: r['name'])
    random.Random(seed).shuffle(candidates)
    # Limit stock-only clips and talking-head material; one clip per original video globally.
    quotas = dict(pixabay=round(samples * .2), celebv=round(samples * .1), panda=samples)
    chosen, sources, captions, families = [], set(), set(), Counter()
    for record in candidates:
        if len(chosen) >= samples:
            break
        if (record['source_id'] in sources or record['caption_group'] in captions or
                families[record['family']] >= quotas[record['family']]):
            continue
        chosen.append(record)
        sources.add(record['source_id'])
        captions.add(record['caption_group'])
        families[record['family']] += 1
    if len(chosen) < samples:
        raise ValueError(f'Only {len(chosen)} unique eligible sources; expand parts or lower --samples')
    bytes_total = sum(r['bytes'] for r in chosen)
    # 40 GB raw videos + <= 30 GB anticipated feature cache leaves ample headroom below 100 GB.
    if bytes_total > 40_000_000_000:
        raise ValueError('Selection exceeds 40 GB raw-video limit')
    if bytes_total + samples * 6_500_000 + 1_000_000_000 > 95_000_000_000:
        raise ValueError('Estimated raw videos + three feature caches exceed 95 GB safety budget')
    chosen.sort(key=lambda r: r['name'])
    write_jsonl(plan_path, chosen)
    write_json(root / 'selection.json', dict(repo=REPO, revision=REVISION, mirror='https://hf-mirror.com',
               samples_requested=samples, seed=seed, parts=parts, raw_bytes=bytes_total,
               family_counts=dict(families), plan_sha256=sha256(plan_path),
               hard_budget_bytes=MAX_DATA_BYTES, video_budget_bytes=40_000_000_000,
               policy='Seeded source-unique subset; maximum 4 MB/clip; no official heldout split exists',
               license='Dataset card: CC-BY-4.0, research/non-commercial intent; source licenses also apply',
               limitations=['Size cap may bias scene length, bitrate and resolution.',
                            'Selected archive IDs are a subset, not a representative million-video sample.',
                            'Captions are provided by OpenVid and require visual spot checks.']))
    return chosen


def current_bytes(root):
    return sum(p.stat().st_size for p in Path(root).rglob('*') if p.is_file())


def download_plan(root, workers=8):
    root = Path(root)
    plan = read_jsonl(root / 'download_plan.jsonl')
    expected = json.loads((root / 'selection.json').read_text())
    if sha256(root / 'download_plan.jsonl') != expected['plan_sha256']:
        raise ValueError('Download plan was modified')
    if current_bytes(root) + expected['raw_bytes'] > MAX_DATA_BYTES:
        raise ValueError('Insufficient headroom under 100 GB data budget')
    if shutil.disk_usage(root).free < expected['raw_bytes'] + 5_000_000_000:
        raise OSError('Insufficient disk space')
    (root / 'videos').mkdir(parents=True, exist_ok=True)
    local = threading.local()
    handles = []
    handle_lock = threading.Lock()

    def fetch(record):
        destination = root / 'videos' / record['name']
        report = destination.with_suffix('.json')
        if destination.exists() and report.exists():
            result = json.loads(report.read_text())
            if destination.stat().st_size == record['bytes'] and sha256(destination) == result['sha256']:
                return result
            raise ValueError(f'Previously downloaded video is corrupt: {destination}')
        for attempt in range(3):
            try:
                if not hasattr(local, 'archives'):
                    local.archives = {}
                name = record['archive']
                if name not in local.archives:
                    session = StrictRangeSession()
                    archive = RemoteZip(resolve_url(REPO, REVISION, name), session=session)
                    local.archives[name] = (archive, session)
                    with handle_lock:
                        handles.append((archive, session))
                archive, _ = local.archives[name]
                info = archive.getinfo(record['member'])
                if info.file_size != record['bytes'] or info.CRC != record['crc32']:
                    raise ValueError('Remote member differs from pinned download plan')
                temporary = destination.with_suffix('.part')
                with archive.open(info) as source, temporary.open('wb') as target:
                    received = 0
                    while True:
                        chunk = source.read(1 << 20)
                        if not chunk:
                            break
                        received += len(chunk)
                        if received > record['bytes']:
                            raise ValueError('Unbounded ZIP member rejected')
                        target.write(chunk)
                if received != record['bytes']:
                    raise ValueError('Truncated ZIP member')
                # zipfile verifies CRC on EOF; also record SHA256 for all downstream accesses.
                os.replace(temporary, destination)
                result = dict(record, video=str(destination.resolve()), sha256=sha256(destination))
                write_json(report, result)
                return result
            except Exception:
                if hasattr(local, 'archives') and record['archive'] in local.archives:
                    archive, session = local.archives.pop(record['archive'])
                    archive.close()
                    session.close()
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)

    started = time.monotonic()
    completed, failed = [], []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(fetch, row): row for row in plan}
            for future in concurrent.futures.as_completed(futures):
                row = futures[future]
                try:
                    result = future.result()
                    completed.append(result)
                except Exception as error:
                    failed.append(dict(name=row['name'], error=repr(error)))
                if (len(completed) + len(failed)) % 25 == 0:
                    write_json(root / 'download_progress.json', dict(completed=len(completed),
                               failed=failed, total=len(plan), seconds=time.monotonic() - started))
                    print(f'OpenVid {len(completed)}/{len(plan)}, failures={len(failed)}, '
                          f'elapsed={time.monotonic() - started:.1f}s', flush=True)
    finally:
        for archive, session in handles:
            archive.close()
            session.close()
    write_jsonl(root / 'downloaded.jsonl', sorted(completed, key=lambda r: r['name']))
    write_json(root / 'download_result.json', dict(completed=len(completed), total=len(plan), failed=failed,
               raw_bytes=sum(r['bytes'] for r in completed), data_directory_bytes=current_bytes(root),
               seconds=time.monotonic() - started))
    if failed:
        raise RuntimeError(f'{len(failed)} downloads failed; rerun to retry, not to change the selected data')


def prepare_manifests(root, output, workers=12, seed=42):
    root, output = Path(root), Path(output)
    if (output / 'split_lock.json').exists():
        raise FileExistsError('Manifests already frozen')
    plan = read_jsonl(root / 'download_plan.jsonl')
    records = read_jsonl(root / 'downloaded.jsonl')
    if len(records) != len(plan):
        raise ValueError('Downloads are incomplete; no partial or silently biased training split')

    def inspect(row):
        if sha256(row['video']) != row['sha256']:
            raise ValueError(f'Content changed: {row["video"]}')
        cap = cv2.VideoCapture(row['video'], cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
        try:
            fps, count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if fps <= 0 or count / fps < row['window_start'] + 2.0 or min(width, height) < 224:
                raise ValueError(f'Invalid video geometry/duration: {row["name"]}')
        finally:
            cap.release()
        return dict(row, actual_fps=fps, width=width, height=height,
                    perceptual_hashes=perceptual_signature(row['video']), visual_review='pending')

    rows, rejected = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(inspect, row): row for row in records}
        for future in concurrent.futures.as_completed(futures):
            try:
                rows.append(future.result())
            except Exception as error:
                rejected.append(dict(name=futures[future]['name'], error=repr(error)))
    write_json(root / 'quality_rejections.json', rejected)
    if len(rows) < .95 * len(plan):
        raise ValueError('More than 5 percent of downloaded videos failed QC; inspect quality_rejections.json')
    # Source identity and exact-caption uniqueness are enforced before downloading.
    # Deduplicate near-identical videos before assigning any train/val/test labels.
    buckets, retained, removed = defaultdict(list), [], []
    rng = random.Random(seed)
    rows.sort(key=lambda r: r['name'])
    rng.shuffle(rows)
    exact = set()
    for row in rows:
        first = int(row['perceptual_hashes'][0], 16)
        candidates = set()
        for band in range(7):
            candidates.update(buckets[(band, (first >> (band * 9)) & 511)])
        duplicate = row['sha256'] in exact
        for index in candidates:
            other = retained[index]
            if all((int(a, 16) ^ int(b, 16)).bit_count() <= 6
                   for a, b in zip(row['perceptual_hashes'], other['perceptual_hashes'])):
                duplicate = True
                break
        if duplicate:
            removed.append(row['name'])
            continue
        for band in range(7):
            buckets[(band, (first >> (band * 9)) & 511)].append(len(retained))
        retained.append(row)
        exact.add(row['sha256'])
    splits = dict(train=[], val=[], test=[])
    families = defaultdict(list)
    for row in retained:
        families[row['family']].append(row)
    for family, group in sorted(families.items()):
        rng.shuffle(group)
        train_end, val_end = round(len(group) * .8), round(len(group) * .9)
        for i, row in enumerate(group):
            split = 'train' if i < train_end else 'val' if i < val_end else 'test'
            row.update(split=split, group_id=row['source_id'], previous_split='unassigned',
                       sample_id=hashlib.sha256((row['sha256'] + '|' + row['caption_group'] + '|' +
                                                str(row['window_start'])).encode()).hexdigest()[:24])
            splits[split].append(row)
    counts = audit_splits(splits)
    for name, rows in splits.items():
        write_jsonl(output / f'{name}.jsonl', sorted(rows, key=lambda r: r['source_id']))
    write_json(output / 'split_lock.json', dict(seed=seed, counts=counts, repo=REPO, revision=REVISION,
               selection_sha256=sha256(root / 'selection.json'), near_duplicates_removed=removed,
               quality_rejections=rejected,
               manifest_sha256={n: sha256(output / f'{n}.jsonl') for n in splits},
               family_counts={n: dict(Counter(r['family'] for r in rows)) for n, rows in splits.items()},
               policy='One clip/original source, dedup before split, family-stratified 80/10/10',
               test_usage='Sealed; no training, hyperparameter selection, mask pretraining or negative pools',
               limitations=['pHash does not prove semantic deduplication.',
                            'YouTube source IDs do not prove episode/channel independence.',
                            'Base Wan pretraining overlap is unknown.',
                            'Captions and generated-video attributes require manual review.']))
    print(json.dumps(counts), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['index', 'plan', 'download', 'split'])
    parser.add_argument('--root', default=str(DEFAULT_ROOT))
    parser.add_argument('--parts', nargs='+', type=int, default=[20, 60, 100, 150])
    parser.add_argument('--samples', type=int, default=5000)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default=str(ROOT / 'manifests_openvid'))
    args = parser.parse_args()
    if args.stage == 'index':
        for part in args.parts:
            index_archive(part, Path(args.root) / 'metadata' / f'part{part}_index.jsonl')
    elif args.stage == 'plan':
        build_plan(args.root, args.parts, args.samples, args.seed)
    elif args.stage == 'download':
        download_plan(args.root, args.workers)
    else:
        prepare_manifests(args.root, args.output, args.workers, args.seed)
