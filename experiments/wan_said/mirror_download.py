"""Bounded HF-mirror downloads, pinned revisions and verified content hashes."""
import argparse
import concurrent.futures
import os
import shutil
import time
from pathlib import Path

import requests

from .io_utils import sha256, write_json

MIRROR = 'https://hf-mirror.com'


class StrictRangeSession(requests.Session):
    def request(self, method, url, **kwargs):
        kwargs.setdefault('timeout', (30, 90))
        response = super().request(method, url, **kwargs)
        response.raise_for_status()
        requested = kwargs.get('headers', {}).get('Range')
        if requested and (response.status_code != 206 or 'Content-Range' not in response.headers):
            response.close()
            raise RuntimeError(f'Server ignored Range {requested}; refusing full-archive transfer')
        return response


def tree(repo, revision, dataset=True):
    kind = 'datasets' if dataset else 'models'
    response = requests.get(f'{MIRROR}/api/{kind}/{repo}/tree/{revision}',
                            params={'recursive': 'true', 'limit': 1000}, timeout=45)
    response.raise_for_status()
    if response.links.get('next'):
        raise RuntimeError('Tree is paginated; explicit pagination is required')
    return response.json()


def resolve_url(repo, revision, name, dataset=True):
    prefix = 'datasets/' if dataset else ''
    return f'{MIRROR}/{prefix}{repo}/resolve/{revision}/{name}'


def download_file(url, destination, size, expected_hash, workers=8, max_bytes=100_000_000_000):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if size > max_bytes:
        raise ValueError('Download exceeds storage budget')
    if destination.exists():
        if destination.stat().st_size == size and sha256(destination) == expected_hash:
            return destination
        raise ValueError(f'Existing destination failed verification: {destination}')
    if shutil.disk_usage(destination.parent).free < size + 5_000_000_000:
        raise OSError('Insufficient free space with 5 GB safety reserve')
    temporary = destination.with_suffix(destination.suffix + '.part')
    # Disjoint writes to a fixed-size file avoid duplicate archive copies or unbounded buffering.
    fd = os.open(temporary, os.O_CREAT | os.O_RDWR, 0o644)
    os.ftruncate(fd, size)
    chunk_size = 16 << 20
    ranges = [(start, min(start + chunk_size, size)) for start in range(0, size, chunk_size)]
    start_time = time.monotonic()

    def fetch(bounds):
        start, stop = bounds
        for attempt in range(4):
            try:
                with StrictRangeSession() as session:
                    with session.get(url, headers={'Range': f'bytes={start}-{stop - 1}'}, stream=True) as r:
                        content_range = r.headers['Content-Range']
                        if content_range != f'bytes {start}-{stop - 1}/{size}':
                            raise ValueError(f'Unexpected Content-Range: {content_range}')
                        position = start
                        for chunk in r.iter_content(1 << 20):
                            if position + len(chunk) > stop:
                                raise ValueError('Range response exceeds expected size')
                            written = os.pwrite(fd, chunk, position)
                            if written != len(chunk):
                                raise OSError('Short disk write')
                            position += written
                        if position != stop:
                            raise ValueError('Truncated response')
                return stop - start
            except (requests.RequestException, ValueError):
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
    try:
        done = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for count in pool.map(fetch, ranges):
                done += count
                print(f'{destination.name}: {done / 1e6:.1f}/{size / 1e6:.1f} MB '
                      f'({done / 1e6 / max(1, time.monotonic() - start_time):.2f} MB/s)', flush=True)
        os.fsync(fd)
    finally:
        os.close(fd)
    actual = sha256(temporary)
    if actual != expected_hash:
        raise ValueError(f'Download SHA256 mismatch: {actual} != {expected_hash}')
    os.replace(temporary, destination)
    write_json(destination.with_suffix(destination.suffix + '.provenance.json'),
               dict(url=url, size=size, sha256=actual, seconds=time.monotonic() - start_time))
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--file', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    entry = next(x for x in tree(args.repo, args.revision) if x['path'] == args.file)
    download_file(resolve_url(args.repo, args.revision, args.file), args.output,
                  entry['size'], entry['lfs']['oid'], args.workers)
