# Fixed-history paired intervention protocol

This supersedes the history sweep runner at `60a8d62`. No full GPU sweep has
been run or approved by this change.

For each `(edit, seed)`, generate B0/P0 once with g=1 and save one checkpoint
per requested target. Generate the full-regeneration P1 reference once with g=1.
For each requested `(target, gate)`, reload the same B0 checkpoint independently
for P0/replay and P1/text-rebind. Only `replay_chunk` executes within the temporary
gate context. Restore RNG and cached initial noise for both policies. All module
gates are restored on context exit, including exceptions and nested contexts.
The old reverse, recache and cached-text controls are not executed by this
mechanism runner: the smoke is exactly 18 chunk replays.

Manifest schema 2 has one entry per `(edit, seed, target_chunk, history_gate)`.
P0/P1 are executed internally, not encoded as a policy dimension. Entries are
consumed directly; sparse selections are preserved, seeds are never replaced by
prompt indices, and duplicate entries, schema 1, policy fields, unknown prompts,
and invalid gates are rejected before loading the model. CLI selection overrides
cannot be combined with a manifest. The default sweep is 4 edits × seeds 42/43 ×
chunks 0/1/4 × 5 gates = 120 pairs, in 8 groups, with 16 full-video generations.

Each pair records `history_gate`, `reference_history_gate=1`, checkpoint path/hash,
chunk hashes, noise checks, and:

- `S_P0`, `S_P1`, `E = S_P1 - S_P0`, `R_k = E / S_full`.
- `R_k=null` when the fixed full reference is nonpositive; negative E is retained.
- `D_drift`: target-chunk latent MSE of P0/g versus B0 (with pixel distances also recorded).
- Boundary deltas and outside-latent exactness for both policies; VAE leakage separately.
- Optional `--dino` appearance distances. These are identity-preservation proxies,
  not validated identity scores; omission is explicitly `not_measured`.

`gates.gate_b_editability.frontier` contains one row per pair, retaining edit,
seed, chunk and gate, scores, drift, boundary and preservation. Chunk 0 is only
calibration; qualitative probes do not count as automatic semantic successes.
Gate D is `invalid_diagnostic`, `passed=null`, `counted_cases=0` for **all** gates;
no timing ratio is computed. Intermediate gates invoke native attention twice.
The legacy MVP summarizer rejects these results rather than applying its old
ratio/cost protocol. Historical MVP files and their summary behavior are retained.

Every invocation creates a unique default run directory. Per-case directories
include g. Existing result files or nonempty output/cache directories are rejected;
explicit paths cannot silently overwrite earlier evidence. Provenance includes the
entire manifest, selected gates, shard, reference gate and sealed source hashes.

## Required GPU invariant smoke

From `restream_mvp`, on the original CUDA/model environment:

```bash
python3 scripts/run_chunk_edit.py \
  --manifest validation/history_release_smoke_manifest.json \
  --sealed-reference validation/edit_ready_mvp/local_edit_main_shard0.json \
  --invariant-smoke --reviewed --gpu 0 --dino
```

This is exactly dress-red-to-blue × seed 42 × k={0,1,4} × g={1,.5,0}:
9 P0/P1 pairs, 18 chunk replays and 2 full-video generations. The runner checks
sealed config/model identity and prompt/seed/target coverage before generation.
It aborts if g=1/P0 differs from B0; g=1/P1 differs from the prior sealed chunk
hash; B0 or the fixed full reference differs from sealed hashes; chunk 0/P0
or P1 differs from its gate-independent reference; recorded replay noise differs;
any outside latent differs for either policy; or all P0 and P1 hashes remain
unchanged across multiple gates at a committed-history target. Therefore
`D_drift(1)=0` is also enforced and an inert gate cannot pass. An exception marks
the run failed, never smoke-passed. Individual
completed pair metrics remain in the run video directory if a later pair fails.

A successful CPU fake-pipeline run is not GPU evidence. Review the actual sealed
GPU smoke before authorizing the 120-pair sweep. No automatic sweep is launched.

## Local validation

```bash
python3 -m unittest discover -s tests -p 'test_history*.py' -v
python3 -m unittest discover -s tests -p 'test_edit_cache.py' -v
```

CPU tests cover full-history g=1 exact native attention, no-history gate
invariance, nested/exception gate restoration, manifest seed and sparse case
selection, paired drift subtraction, invalid Gate D, 2 full generations / 18
replays, shared checkpoint provenance, sealed mismatch failure and reference
validation. The additional CUDA kernel test skips if CUDA is unavailable.

Local result: 44 CPU tests passed. The original 32 edit-cache regressions remain
green. Fault injection also verifies that P0
exactness, chunk-0 invariance and outside-preservation failures abort the runner.

The first GPU attempt after `21a669d` was rejected during result review: the
active baseline uses `causal_model.py`, while the gate had only been connected to
`causal_model_infinity.py`. Identical committed-history hashes across all gates
exposed the inactive intervention. The active model path and fail-closed check
were fixed before rerunning the smoke; those rejected artifacts are not evidence.

## GPU smoke result

The corrected smoke ran on one NVIDIA A800-SXM4-80GB at commit `9f30f13`.
Its provenance records a clean tree and the fixed gate set `[1,.5,0]`. The
native CUDA attention endpoint test passed separately. All 9 pairs / 18 replays
passed sealed hashes, replay-noise equality, chunk-0 gate invariance and outside
latent exactness. Boundary and DINO appearance diagnostics were recorded for both
policies in all 9 pairs. Gate D remains invalid and uncounted.

| chunk | gate | E | R | D_drift (latent MSE) |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1 / .5 / 0 | 0.093521 | 1.000000 | 0 |
| 1 | 1 | -0.000136 | -0.001410 | 0 |
| 1 | .5 | 0.045815 | 0.473808 | 0.287206 |
| 1 | 0 | 0.106611 | 1.102553 | 1.076333 |
| 4 | 1 | 0.001328 | 0.013151 | 0 |
| 4 | .5 | 0.029110 | 0.288180 | 0.517063 |
| 4 | 0 | 0.083023 | 0.821901 | 0.942856 |

The full JSON is `validation/history_release_smoke_gpu_9f30f13.json`, SHA-256
`a67959627c9b36330beeb5dcea10f199b9ba69b5cc5486666495d14d3bee59c4`.
This single edit/seed is invariant evidence, not a scientific conclusion. The
120-pair sweep was not started and still requires explicit review approval.
