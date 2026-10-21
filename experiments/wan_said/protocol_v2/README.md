# Official OpenVid Acquisition

This isolated project currently downloads and verifies raw videos only. It does
not change the existing Wan FM or SAID algorithms, start training, generate
benchmarks, or extract text, VAE, or teacher features. Teacher migration and the
remaining benchmark protocol work are deferred.

## Dataset And Storage

- The target is the exact 64,000 unique filenames in the pinned VideoREPA
  `openvid_6w4.csv`, with full original clips and captions, not cropped segments.
- `official/*.csv.gz` contains deterministic lossless copies of the official
  32K and 64K CSV files. Pins and hashes are in `sources_lock.json` and the
  dataset-local `plan_lock.json`.
- The official 32K list is **not** a strict subset of the 64K list. The audit
  records the differences; downloading 64K does not imply acquiring all 32K.
- The selected MP4 payload totals 369,319,691,213 bytes (369.3 GB decimal).
  Allow roughly 375 GB including operational metadata and partial transfers.
  This full target exceeds the earlier 100 GB pilot budget.
- Acquisition keeps a separate 50,000,000,000-byte free-space reserve. Existing
  experiment data, model weights, and old feature caches are not deleted.
- `estimated_training_storage` in the plan/progress is a hypothetical budget,
  not allocated storage. All new feature caches remain disabled.

## Mirror And Retry Policy

The default is six workers, using `https://hf-mirror.com` as the only resolver
entry point, with no fallback to the Hugging Face official resolver or local
proxy. Mirror-controlled storage redirects remain permitted; this is not a
guarantee that every storage CDN is geographically in China.

Workers share a request gate with at least 0.5 seconds between request starts.
A 429 blocks new requests across all workers. `Retry-After` accepts seconds or
an HTTP date, with a minimum 60-second wait. Without a valid header, retries
wait 60, 120, 240, 480, then up to 900 seconds. Already in-flight requests may
finish. A persistent failure is recorded; the background service retries the
job after five minutes. A speed increase is not guaranteed.

Range requests must return HTTP 206 with a Content-Range header; full archive
fallback is rejected. Only selected MP4 members are saved, not complete ZIPs.
Each completed file passes expected byte size, ZIP CRC32, SHA256, and full CPU
video decode. Restarts validate existing reports and SHA256 before reuse.
Incomplete individual members restart from their beginning; this is not
byte-level resume within a compressed member. Corrupt files are quarantined.

## Commands

Run from the isolated worktree root with the existing environment:

```bash
experiments/wan_said/.venv/bin/python -m experiments.wan_said.protocol_v2.official_data download --workers 6
experiments/wan_said/.venv/bin/python -m unittest discover -s experiments/wan_said/protocol_v2/tests -v
systemctl --user status wan-said-openvid64k-mirror-20261021.service
```

The earlier 24-worker unit `wan-said-openvid64k-20261021.service` was stopped.
The replacement mirror-only unit has an 8 GiB memory ceiling and uses no GPU.
Do not start a second downloader against the same output directory.

Live status is in `../datasets/openvid64k/progress.json`; per-file reports,
`errors.json`, and `download.log` are in the same ignored dataset directory.
Progress is refreshed every 50 processed files and recounts verified existing
files at startup. A dataset is ready only when every target is verified and
`full_dataset_ready` is true. Train/validation/test manifests for this new 64K
target have not yet been created; no new training is authorized here.

## Review Artifacts

`reports/` contains operation notes and test output snapshots, not an automatic
live GitHub mirror. Test stdout includes synthetic one-video fixtures whose
`full_dataset_ready=true` does not refer to the production 64K dataset.
Raw videos, weights, vendor checkouts, and feature caches stay outside Git.
