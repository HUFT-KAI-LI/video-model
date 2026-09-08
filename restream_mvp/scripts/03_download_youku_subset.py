"""Bounded parallel Youku downloads with streaming metadata and exact byte quota."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import signal
import threading


class ByteBudget:
    """Committed files AND in-flight reserved bytes share one limit."""
    def __init__(self, limit, used=0):
        self.limit, self.allocated = limit, used
        self.lock = threading.Lock()

    def reserve(self, size):
        with self.lock:
            if size < 0 or self.allocated + size > self.limit:
                return False
            self.allocated += size
            return True

    def release(self, size):
        with self.lock:
            self.allocated -= size


def download_one(item, oss, videos, quota, min_duration):
    import av
    remote, caption = item["video_id:FILE"], item["golden_caption"]
    source = Path(remote).stem
    dest = videos / (hashlib.sha256(source.encode()).hexdigest()[:24] + ".mp4")
    partial = dest.with_suffix(".partial")
    reserved, committed = 0, False
    try:
        if dest.exists() or partial.exists():
            raise RuntimeError("Unmanifested destination requires review")
        key = oss.oss_dir.rstrip("/") + "/" + remote
        if not oss.bucket.object_exists(key):
            key = oss.oss_backup_dir.rstrip("/") + "/" + remote
        size = oss.bucket.head_object(key).content_length
        if size <= 0 or size > 256 * 1024**2 or not quota.reserve(size):
            return None
        reserved = size
        response = oss.bucket.get_object(key)
        received, digest = 0, hashlib.sha256()
        try:
            with partial.open("xb") as f:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    received += len(chunk)
                    if received > size:
                        raise ValueError("Object exceeded reserved byte budget")
                    f.write(chunk)
                    digest.update(chunk)
        finally:
            response.close()
        if received != size:
            raise ValueError("Truncated video")
        with av.open(str(partial)) as video:
            stream = video.streams.video[0]
            stream.thread_count = 1
            duration = float(stream.duration * stream.time_base) if stream.duration else float(video.duration / av.time_base)
            if duration < min_duration:
                return None
            frames = sum(1 for _ in video.decode(stream))
            if frames < 2:
                raise ValueError("Undecodable video")
        row = {"video": str(dest), "caption": caption, "source_id": source, "duration": duration,
               "bytes": received, "sha256": digest.hexdigest(), "source_path": remote, "visual_review": "pending"}
        partial.replace(dest)
        committed = True
        return row
    except Exception as error:
        # Do not expose temporary signed OSS URLs embedded in SDK exceptions.
        return {"source_id": source, "error_type": type(error).__name__}
    finally:
        if partial.exists():
            partial.unlink()
        if reserved and not committed:
            quota.release(reserved)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-gb", type=float, default=8)
    p.add_argument("--max-items", type=int, default=1200)
    p.add_argument("--min-duration", type=float, default=4)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    if not 0 < a.max_gb <= 10 or a.max_items < 1 or not 1 <= a.workers <= 8:
        p.error("Require max-gb in (0,10], positive max-items, workers in [1,8]")
    a.output = a.output.resolve()
    videos = a.output / "videos"
    videos.mkdir(parents=True, exist_ok=True)
    manifest = a.output / "raw_manifest.jsonl"
    existing = [json.loads(x) for x in manifest.read_text().splitlines()] if manifest.exists() else []
    seen = {r["source_id"] for r in existing}
    if len(seen) != len(existing):
        raise RuntimeError("Duplicate source ID in existing manifest")
    paths = {Path(r["video"]).resolve() for r in existing}
    actual = {f.resolve() for f in videos.iterdir() if f.is_file()}
    if paths != actual:
        raise RuntimeError("Missing/orphaned files: resolve against manifest before resuming")
    for row in existing:
        if Path(row["video"]).stat().st_size != row["bytes"]:
            raise RuntimeError("Existing video size mismatch")
    used = sum(r["bytes"] for r in existing)
    quota = ByteBudget(int(a.max_gb * 1e9), used)
    if used > quota.limit:
        raise ValueError("Existing files already exceed requested byte limit")
    os.environ["MODELSCOPE_CACHE"] = str(a.output / "sdk_cache")
    from modelscope import MsDataset
    from modelscope.hub.api import HubApi
    from modelscope.msdatasets.utils.oss_utils import OssUtilities
    if os.environ.get("MODELSCOPE_TOKEN"):
        HubApi().login(os.environ["MODELSCOPE_TOKEN"])
    ds = MsDataset.load("Youku-AliceMind", namespace="modelscope", subset_name="caption",
                        split="train", use_streaming=True, cache_dir=str(a.output / "sdk_cache"))
    oss = OssUtilities("Youku-AliceMind", "modelscope", "master")
    # Native __iter__ downloads before yielding; iterate ONLY metadata here.
    metadata = iter(ds.ds_instance.iter(batch_size=1))
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    count, errors, exhausted = len(existing), 0, False
    with ThreadPoolExecutor(max_workers=a.workers) as pool, manifest.open("a", buffering=1) as out:
        while not stop.is_set() and count < a.max_items and not exhausted and quota.allocated < quota.limit:
            batch_items = []
            while len(batch_items) < min(a.workers, a.max_items - count):
                try:
                    batch = next(metadata)
                except StopIteration:
                    exhausted = True
                    break
                item = {k: v[0] for k, v in batch.items()}
                source = Path(item["video_id:FILE"]).stem
                caption = item["golden_caption"]
                if source in seen:
                    continue
                seen.add(source)
                if any(w in caption for w in ["动画", "卡通", "动漫", "小羊", "灰狼", "游戏画面"]):
                    continue
                if not any(w in caption for w in ["人", "男子", "女子", "男人", "女人", "男孩", "女孩"]):
                    continue
                batch_items.append(item)
            # Bound the queue to one small batch; never submit the full dataset.
            futures = [pool.submit(download_one, item, oss, videos, quota, a.min_duration) for item in batch_items]
            for future in futures:
                row = future.result()
                if row is None:
                    continue
                if "error_type" in row:
                    errors += 1
                    print(json.dumps(row), flush=True)
                else:
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    count, used, errors = count + 1, used + row["bytes"], 0
                    print(json.dumps({"items": count, "bytes": used, "duration": row["duration"]}), flush=True)
            # Only stop AFTER all in-flight successes have been committed to manifest.
            if errors >= 10:
                raise RuntimeError("Ten consecutive download/decode failures")
    print(json.dumps({"complete": not stop.is_set(), "items": count, "bytes": used}), flush=True)


if __name__ == "__main__":
    main()
