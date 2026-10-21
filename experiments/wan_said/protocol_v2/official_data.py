"""Exact VideoREPA CSV acquisition, resumable per-file downloads and decode verification."""
import argparse
import concurrent.futures
import csv
import gzip
import json
import os
import shutil
import threading
import time
import zlib
from collections import Counter
from pathlib import Path

import av
from remotezip import RemoteZip

from ..io_utils import read_jsonl, sha256, write_json, write_jsonl
from ..mirror_download import MIRROR, resolve_url
from ..openvid_subset import REPO, REVISION, source_identity
from ..runtime import ROOT
from .mirror_transport import MirrorRangeSession

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = ROOT / 'datasets/openvid64k'
RESERVE = 50_000_000_000


def official_rows(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['video', 'caption', 'seconds']:
            raise ValueError(f'Unexpected official CSV schema: {reader.fieldnames}')
        rows = list(reader)
    names = set()
    for row in rows:
        name = row['video']
        if Path(name).name != name or not name.endswith('.mp4') or not row['caption'].strip():
            raise ValueError(f'Unsafe name or empty caption: {name}')
        if name in names or float(row['seconds']) <= 0:
            raise ValueError(f'Duplicate name or invalid duration: {name}')
        names.add(name)
    return rows


def audit_csvs(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    vendor = ROOT / 'vendor/VideoREPA/finetune/openvid'
    sets = {}
    hashes = {}
    for scale, filename in [('32k', 'openvid_3w2.csv'), ('64k', 'openvid_6w4.csv')]:
        source = vendor / filename
        rows = official_rows(source)
        sets[scale] = {r['video']: r for r in rows}
        hashes[scale] = sha256(source)
        destination = HERE / 'official' / (filename + '.gz')
        destination.parent.mkdir(exist_ok=True)
        with source.open('rb') as inp, destination.open('wb') as out:
            with gzip.GzipFile(filename=filename, mode='wb', fileobj=out, mtime=0) as compressed:
                shutil.copyfileobj(inp, compressed)
    a, b = sets['32k'], sets['64k']
    result = dict(official_csv_rows={k: len(v) for k, v in sets.items()},
                  unique_video_count=len(b), csv_sha256=hashes,
                  strict_32k_subset=set(a) < set(b), missing_32k_from_64k=sorted(set(a)-set(b)),
                  caption_mismatch_32k_64k=[n for n in a.keys() & b.keys()
                                          if a[n]['caption'] != b[n]['caption']],
                  parts=list(range(30, 50)), full_dataset_ready=False)
    write_json(root / 'csv_audit.json', result)
    return result


def index_part(part, root):
    destination = root / 'indexes' / f'part{part}.jsonl'
    if destination.exists():
        return read_jsonl(destination)
    rows = []
    for attempt in range(4):
        try:
            with MirrorRangeSession() as session:
                url = resolve_url(REPO, REVISION, f'OpenVid_part{part}.zip')
                with RemoteZip(url, session=session) as archive:
                    for item in archive.infolist():
                        if item.filename.endswith('.mp4'):
                            rows.append(dict(name=Path(item.filename).name, member=item.filename,
                                             archive=f'OpenVid_part{part}.zip', bytes=item.file_size,
                                             compressed_bytes=item.compress_size, crc32=item.CRC))
            break
        except Exception:
            rows = []
            if attempt == 3:
                raise
            time.sleep(2**attempt)
    write_jsonl(destination, rows)
    print(f'Indexed full part {part}: {len(rows)} MP4s (no size filter)', flush=True)
    return rows


def plan(root, workers=6):
    root = Path(root)
    audit = audit_csvs(root)
    rows = official_rows(HERE / 'official/openvid_6w4.csv.gz')
    if (root / 'plan_lock.json').exists():
        lock = json.loads((root / 'plan_lock.json').read_text())
        if sha256(root / 'download_plan.jsonl') != lock['plan_sha256']:
            raise ValueError('Immutable official download plan changed')
        if lock['csv_sha256'] != audit['csv_sha256']:
            raise ValueError('Official CSV revision changed')
        return lock
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        indexes = list(pool.map(lambda p: index_part(p, root), range(30, 50)))
    names = {r['video'] for r in rows}
    mapping = {}
    for index in indexes:
        for record in index:
            if record['name'] in names:
                if record['name'] in mapping:
                    raise ValueError(f'Ambiguous official ZIP mapping: {record["name"]}')
                mapping[record['name']] = record
    missing = names - mapping.keys()
    write_json(root / 'mapping_errors.json', dict(missing=sorted(missing)))
    if missing:
        raise ValueError(f'{len(missing)} official names not found in parts 30-49')
    records = []
    for row in rows:
        name = row['video']
        try:
            source_id, family = source_identity(name)
        except ValueError:
            source_id, family = 'unresolved:' + name, 'unresolved'
        records.append(dict(mapping[name], caption=row['caption'], duration=float(row['seconds']),
                            source_id=source_id, family=family, caption_scope='full_source_video',
                            official_32k=False))
    subset = {r['video'] for r in official_rows(HERE / 'official/openvid_3w2.csv.gz')}
    for row in records:
        row['official_32k'] = row['name'] in subset
    write_jsonl(root / 'download_plan.jsonl', records)
    raw = sum(r['bytes'] for r in records)
    count = len(records)
    # Exact uncompressed payload sizes at 49x480x720; serialization overhead is additional.
    cache = dict(text_uncompressed=count*512*4096*2,
                 vae_latents=count*16*13*60*90*2,
                 teacher_tokens_bf16=count*3600*768*2,
                 teacher_pooled_fp32=count*768*4)
    lock = dict(audit, plan_sha256=sha256(root / 'download_plan.jsonl'),
                repo=REPO, revision=REVISION, raw_bytes=raw, cache_estimate_bytes=cache,
                storage_available_bytes=shutil.disk_usage(root).free,
                safety_reserve_bytes=RESERVE, part_counts=dict(Counter(r['archive'] for r in records)),
                full_dataset_ready=False,
                storage_note='No 100 GB limit. Raw acquisition and optional token caches have separate capacity gates.')
    write_json(root / 'plan_lock.json', lock)
    if shutil.disk_usage(root).free < raw + RESERVE:
        raise OSError(f'Raw acquisition needs {raw + RESERVE} bytes free, including reserve')
    return lock


def inspect_video(path):
    count, first, last, width, height = 0, None, None, None, None
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = 'SLICE'
        stream.codec_context.thread_count = 1
        fps = float(stream.average_rate)
        for frame in container.decode(video=0):
            if frame.time is None:
                raise ValueError('Missing frame PTS')
            timestamp = float(frame.time)
            if last is not None and timestamp < last:
                raise ValueError('Nonmonotonic decoded PTS')
            if first is None:
                first = timestamp
            last = timestamp
            width, height = frame.width, frame.height
            count += 1
    if count < 2 or fps <= 0 or not width or not height:
        raise ValueError('Empty or invalid video')
    return dict(decoded_frames=count, source_fps=fps, source_duration=last-first+1/fps,
                first_pts=first, last_pts=last, width=width, height=height)


def verify_file(row, path):
    if path.stat().st_size != row['bytes']:
        raise ValueError('Byte size mismatch')
    crc = 0
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            crc = zlib.crc32(chunk, crc)
    if crc != row['crc32']:
        raise ValueError('ZIP CRC32 mismatch')
    return dict(row, video=str(path.resolve()), sha256=sha256(path),
                download_status='verified_and_decoded', **inspect_video(path))


def download(root, workers=6, attempts=4):
    if workers < 1 or attempts < 1:
        raise ValueError('Workers and attempts must be positive')
    root = Path(root)
    lock = plan(root, workers=workers)
    records = read_jsonl(root / 'download_plan.jsonl')
    (root / 'videos').mkdir(exist_ok=True)
    (root / 'records').mkdir(exist_ok=True)
    existing_bytes = sum((root/'videos'/r['name']).stat().st_size for r in records
                         if (root/'videos'/r['name']).exists())
    remaining_bytes = max(0,lock['raw_bytes']-existing_bytes)
    if shutil.disk_usage(root).free < remaining_bytes + RESERVE:
        write_json(root/'storage_blocked.json',dict(required_bytes=remaining_bytes+RESERVE,
                   available_bytes=shutil.disk_usage(root).free,full_dataset_ready=False))
        raise OSError('Raw download capacity gate failed; no old files were removed')
    local, handles, mutex = threading.local(), [], threading.Lock()
    started = time.monotonic()
    completed, failures = [], []

    def fetch(row):
        destination = root / 'videos' / row['name']
        report = root / 'records' / (row['name'] + '.json')
        if destination.exists() and report.exists():
            try:
                saved = json.loads(report.read_text())
                if (saved.get('download_status') == 'verified_and_decoded' and
                        all(saved.get(k) == v for k,v in row.items()) and
                        destination.stat().st_size == row['bytes'] and sha256(destination) == saved['sha256']):
                    return saved
            except (ValueError,OSError,KeyError):
                pass
        # Reuse only identity+CRC+full-decode verified existing clips, never filename alone.
        old = ROOT / 'datasets/openvid/videos' / row['name']
        if not destination.exists() and old.exists():
            try:
                result = verify_file(row, old)
                result['reused_from'] = str(old)
                write_json(report, result)
                return result
            except Exception:
                pass
        for attempt in range(attempts):
            try:
                if shutil.disk_usage(root).free < RESERVE + row['bytes']:
                    raise OSError('Safety disk reserve reached; no files deleted')
                if destination.exists():
                    try:
                        result = verify_file(row, destination)
                        write_json(report, result)
                        return result
                    except Exception:
                        os.replace(destination, destination.with_suffix(f'.corrupt.{time.time_ns()}'))
                if not hasattr(local, 'archives'):
                    local.archives = {}
                name = row['archive']
                if name not in local.archives:
                    session = MirrorRangeSession()
                    try:
                        archive = RemoteZip(resolve_url(REPO, REVISION, name), session=session)
                    except Exception:
                        session.close()
                        raise
                    local.archives[name] = (archive, session, MIRROR)
                    with mutex:
                        handles.append((archive, session))
                archive, _, host = local.archives[name]
                info = archive.getinfo(row['member'])
                if info.file_size != row['bytes'] or info.CRC != row['crc32']:
                    raise ValueError('Remote member differs from immutable plan')
                temporary = destination.with_suffix('.part')
                with archive.open(info) as source, temporary.open('wb') as target:
                    shutil.copyfileobj(source, target, length=1 << 20)
                result = verify_file(row, temporary)
                os.replace(temporary, destination)
                result.update(video=str(destination.resolve()), download_host=host)
                write_json(report, result)
                return result
            except Exception:
                if hasattr(local, 'archives') and row['archive'] in local.archives:
                    archive, session, _ = local.archives.pop(row['archive'])
                    archive.close()
                    session.close()
                if attempt + 1 == attempts:
                    raise
                time.sleep(min(30, 2**attempt))

    def snapshot():
        count = len(completed)
        result = dict(official_csv_rows=lock['official_csv_rows'], unique_video_count=len(records),
                      downloaded_count=count, verified_count=count, failed_count=len(failures),
                      missing_count=len(records)-count, corrupted_count=sum('CRC' in x['error'] or
                          'corrupt' in x['error'] for x in failures),
                      total_raw_bytes=sum(r['bytes'] for r in completed), total_cache_bytes=0,
                      estimated_training_storage=lock['cache_estimate_bytes'],
                      full_dataset_ready=count == len(records) and not failures,
                      elapsed_seconds=time.monotonic()-started, stage='raw_download_and_full_decode',
                      feature_cache_enabled=False, training_enabled=False,
                      download_source=MIRROR, download_workers=workers,
                      status='complete' if count == len(records) else 'in_progress',
                      note='Only completed files count; restart verifies SHA256 before reuse. No automatic training.')
        write_json(root / 'progress.json', result)
        write_json(root / 'errors.json', failures)
        print(json.dumps(result), flush=True)
        return result

    snapshot()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(fetch, row): row for row in records}
            for future in concurrent.futures.as_completed(futures):
                try:
                    completed.append(future.result())
                except Exception as error:
                    failures.append(dict(name=futures[future]['name'], error=repr(error)))
                if (len(completed)+len(failures)) % 50 == 0:
                    snapshot()
    finally:
        for archive, session in handles:
            archive.close()
            session.close()
        write_jsonl(root / 'downloaded.jsonl', sorted(completed, key=lambda r: r['name']))
        result = snapshot()
    if not result['full_dataset_ready']:
        raise RuntimeError(f'Incomplete: {result["missing_count"]} official targets remain')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['audit', 'plan', 'download'])
    parser.add_argument('--root', default=str(DEFAULT_ROOT))
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    if args.stage == 'audit':
        print(json.dumps(audit_csvs(args.root), indent=2))
    elif args.stage == 'plan':
        print(json.dumps(plan(args.root, args.workers), indent=2))
    else:
        download(args.root, args.workers)
