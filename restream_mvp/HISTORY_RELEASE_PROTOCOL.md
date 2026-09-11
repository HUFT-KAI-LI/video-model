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
cannot be combined with a manifest. The confirmatory sweep is 4 edits × held-out
seeds 101/202 ×
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

## Frozen full-sweep analysis

`configs/history_release_analysis_plan.json` is the machine-readable decision
rule. It must be committed before the full sweep. Every result shard records its
SHA-256, and the analyzer refuses a different plan, dirty provenance, incomplete
grid, duplicate case, failed invariant, or enabled Gate D.

Frozen plan SHA-256:
`0f1469c45727e21f2e80988d9e47c526b31e5cf374cce85174ff97f691cded60`.

The confirmatory contrast is g=.5 versus g=1. For each of the 8 independent
`(edit, seed)` units, first average `delta_E = E(.5)-E(1)` over repeated chunks
1 and 4. The causal criterion requires at least 7/8 positive clusters, an exact
one-sided sign-test p <= .05, positive median and majority-positive replication
at each chunk, and positive edit-level effects for at least 3/4 edits. The 7/8
sign pattern has exact p=.03515625; 6/8 does not pass. This is named an exact
consistency test on the preregistered edit-seed units, not population-level
significance for all video editing tasks. There is no raw magnitude
cutoff because color and luma proxy scales are not commensurate. The deterministic
10,000-resample cluster bootstrap interval is descriptive and cannot change the
decision. Gates .75, .25 and 0 are secondary, with no confirmatory p-values.

Pareto coordinates are dimensionless median edit-seed cluster
`delta_R = R(g)-R(1) = delta_E/S_full` (maximize) and median cluster `D_drift`
(minimize). If any full-reference effect is nonpositive, the cross-edit frontier
is marked not estimable without dropping or imputing that unit. Raw `delta_E`
remains stratified by edit. Both point-estimate and 7/8 paired dominance are
reported. Every non-dominated gate remains on the frontier; the analyzer never
selects one operating point, applies a post-result drift budget, or counts timing.

The g=.5 primary choice was pilot-informed by seed 42. The confirmatory manifest
therefore uses held-out seeds 101/202; a pre-sweep recursive audit found neither
value in repository JSON/JSONL `seed`, `noise_seed`, or `generation_seed` fields.
Old generation experiments used seeds 42-57, so no confirmatory unit has a
historical seal by design. Every new group generates B0 and FullReg once at g=1,
fixes those references for all gates, and requires g=1/P0 exact base.

Gate activity is checked independently inside every `(edit, seed, target_chunk)`
group by both runner and analyzer. For committed-history chunks, at least one of
P0/P1 must change across gates; differences from another prompt or seed cannot
satisfy this invariant.

After explicit full-sweep approval, the 8 whole `(edit, seed)` groups can be
distributed evenly over four A800s. Each shard runs 30 pairs / 60 chunk replays
and two groups; across all shards this is 120 pairs / 240 replays and 16 full
generations. `--full-sweep-approved` is a separate fail-closed launch flag:

```bash
cd /workspace/video-model/restream_mvp
for gpu_index in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$gpu_index OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false \
    .venv/bin/python scripts/run_chunk_edit.py \
      --manifest validation/history_release_sweep_manifest.json \
      --reviewed --full-sweep-approved --shard $gpu_index --shards 4 --gpu 0 \
      --dino > /tmp/history_release_shard${gpu_index}.log 2>&1 &
done
wait
```

Do not add `--full-sweep-approved` until review explicitly releases the sweep.
After all four result JSONs are complete, run the frozen analyzer before reading
or interpreting per-case results:

```bash
.venv/bin/python scripts/analyze_history_release_sweep.py \
  --results validation/history_release/paired_shard*/results.json \
  --output validation/history_release_analysis.json
```

## Held-out full-sweep result

The four-A800 sweep completed at clean commit `4609a4b`: 120/120 unique pairs,
240 chunk replays, held-out seeds 101/202 and all 8 preregistered `(edit, seed)`
clusters. Every replay RNG and outside-latent invariant passed; all g=1 P0 chunks
were exact base with `D_drift=0`; Gate D remained invalid. Historical sealed
coverage was 0/24 as preregistered for held-out trajectories.

The frozen g=.5 consistency test **passed**: 8 positive, 0 negative and 0 tied
cluster effects, exact one-sided sign-test p=.00390625. Median cluster `delta_E`
was .0510653; the deterministic 95% cluster-bootstrap interval was
[.0384488, .0696843] and remained descriptive. Both chunks replicated at 8/8
positive (`k=1` median .0573422; `k=4` median .0449485), and all 4/4 edits had
positive edit-level effects. This supports the preregistered claim on these
held-out edit-seed units; it is not population-level significance for all video
editing tasks.

| gate | median cluster delta_E | median cluster delta_R | median D_drift |
| ---: | ---: | ---: | ---: |
| 1 | 0 | 0 | 0 |
| .75 | .012073 | .035353 | .028016 |
| .5 | .051065 | .252592 | .260369 |
| .25 | .138601 | .604286 | .877645 |
| 0 | .168525 | .750262 | .837763 |

All full-reference effects were positive (range .0680670-.5532694), so the
normalized frontier was estimable. Its point-estimate non-dominated gates are
`{1,.75,.5,0}`; g=.25 is dominated by g=0 on the aggregate coordinates. No gate
met the separate 7/8 paired-dominance reporting threshold. The result selects no
operating point and makes no practical drift-acceptability claim.

The frozen report is
`validation/history_release_analysis_heldout_4609a4b.json`, SHA-256
`3e11b9bcee30ae027b9e37b968c454b3e30fea463702f529620605fa824cab66`.
Four source shard JSONs are stored alongside it with their hashes embedded in the
report. Multi-GB checkpoints and videos remain local and ignored by Git.
