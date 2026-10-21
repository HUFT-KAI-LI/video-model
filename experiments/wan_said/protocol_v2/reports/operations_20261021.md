# Mirror Download Operations: 2026-10-21 (Asia/Shanghai)

## Scope

The user requested domestic HF mirror acquisition with fewer workers and
proper 429 waits. Only the independent `protocol_v2` acquisition code changed.
No base FM, SAID, LongLive, or Reality Memory code was edited. No training or
feature caching was started.

## Observations Before The Change

- The old service ran with `--workers 24`.
- A bounded 8 MiB Range probe against `OpenVid_part30.zip` returned HTTP 429
  from the domestic mirror; no full archive was transferred.
- The equivalent official-HF probe returned HTTP 206, 8,388,608 bytes in
  17.13 seconds (0.49 MB/s). These single probes are not sustained throughput
  measurements and cannot establish the mirror's exact rate-limit policy.
- The old progress snapshot showed 2,400 verified videos, 13,807,730,317 bytes,
  zero recorded failures, and zero generated cache bytes before shutdown.
  Per-file reports may be ahead of the 50-file progress snapshot.

## Applied Change

- Stopped `wan-said-openvid64k-20261021.service` before starting its replacement.
- Removed official-HF fallback from both ZIP indexing and member acquisition.
- Added one shared request gate, 0.5-second request-start spacing, and global
  429 cooldowns honoring Retry-After seconds and HTTP dates (minimum 60s).
- Closed HTTP error responses before retrying; retained strict Range checks.
- Started `wan-said-openvid64k-mirror-20261021.service` at 14:47:34 CST,
  `--workers 6`, MemoryMax=8G, Nice=10, Restart=on-failure, RestartSec=300.
- Existing completed videos are rechecked and reused. No old files were
  removed. Original dataset plan/revision and the 50 GB reserve are unchanged.

## Verification

The first post-change test run passed all 11 download-only and transport tests
in 1.221 seconds. Coverage includes SHA/CRC/full decode, corruption quarantine,
completed-file reuse, insufficient-space refusal, failed-file status, mirror
restriction, 429 retry exhaustion, HTTP-date parsing, shared cooldown, request
pacing, and rejection of servers that ignore Range.

The new service was observed active/running with the six-worker command. The
initial startup recounted existing files rather than downloading them again.
Production dataset completion and sustained speed improvements are not claimed.
Subsequent generated test logs and live-status snapshots are stored alongside
this note for review.

The recorded rerun passed 11 acquisition/transport tests in 1.235 seconds and
all 18 existing compatibility tests in 5.879 seconds. The latter includes the
native FM contract and existing real-video source-split test, not a new 64K
training split. A later production snapshot reached 2,450 verified files
(14,104,193,631 bytes), with zero recorded failures and no generated cache.
An independent per-file report check found nine files newly acquired from the
mirror after the restart, confirming actual transfers beyond startup reuse.
This is an intermediate checkpoint, not the current live total or a claim
that all 64,000 files are ready.
