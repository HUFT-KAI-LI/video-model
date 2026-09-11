# History Component Decomposition: D1 Screen

Stage C is sealed as a held-out GPU PASS. D1 does not revisit whether history
suppresses editability. It asks which part of committed visual history supplies
semantic inertia and whether selective release improves the editability versus
preservation tradeoff over the unchanged Stage C global `g=.5` intervention.

## Partition and interventions

At each replay attention call, the cache window is partitioned in its actual
storage order:

```
sink | older local history | recent history | current query chunk
```

`sink` contains the first `sink_size=3` latent frames. `recent` contains the
last `num_frame_per_block=3` historical latent frames before the query. `old`
contains the remaining attended non-sink history. The runner refuses
configurations that do not use a 3-frame sink, 3-frame AR chunks and a 12-frame
local window.

The upstream 12-frame budget includes the current query chunk. At chunk 4, the
checkpoint contains C0-C3, but inserting C4 makes the native cache roll C1 out.
The actual attended partition is therefore `C0 sink | C2 old | C3 recent | C4
current`. D1 follows this native path so `full_history` remains exact base. Each
component replay records the observed per-layer token partition and the runner
requires it to equal 3/3/3 attended history frames at chunk 4. At chunk 1 the
partition is 3/0/0 frames.

The frozen screen has six conditions:

| condition | sink | old | recent | implementation |
| --- | ---: | ---: | ---: | --- |
| `full_history` | 1 | 1 | 1 | component router, exact native attention |
| `global_release` | .5 | .5 | .5 | unchanged Stage C global interpolation |
| `sink_release` | .5 | 1 | 1 | component interpolation |
| `old_release` | 1 | .5 | 1 | component interpolation |
| `recent_release` | 1 | 1 | .5 | component interpolation |
| `non_sink_release` | 1 | .5 | .5 | independent multilinear interpolation |

For one fractional component, output is
`g*A(full) + (1-g)*A(without_component)`, where both terms use the native
attention kernel. Multiple fractional components use the multilinear
expectation over native-attention inclusion subsets. Timing is invalid for this
mechanism screen because the probes intentionally invoke attention more than
once.

## Fixed-history paired protocol

For each `(edit, seed=303)` group, B0 and FullReg P1 are generated once with
unmodified full history. For each target `k in {1,4}`, every condition reloads
the same B0 checkpoint independently and runs both P0 and P1 with checkpoint
noise and RNG restoration. The runner records

```
E(m)       = S(P1,m) - S(P0,m)
delta_E(m) = E(m) - E(full_history)
delta_R(m) = delta_E(m) / S_full
D_drift(m) = MSE(Ck(P0,m), Ck(base))
```

along with outside exactness, boundary diagnostics and frozen DINOv2 appearance
distance. The 4 edits by 1 new exploratory seed by 2 chunks by 6 conditions form
48 P0/P1 pairs and 96 chunk replays. Seeds 42, 101 and 202 are excluded.

Chunk 1 is a routing invariant. Its only history is C0, which is entirely sink:

- `global_release` must equal `sink_release` bit-for-bit for both P0 and P1.
- `old_release`, `recent_release`, and `non_sink_release` must equal
  `full_history` bit-for-bit for both P0 and P1.

At chunk 4, every non-full intervention must change at least one of P0/P1. Any
missing case, duplicate, RNG mismatch, outside change, checkpoint mismatch,
dirty provenance, absent DINO/boundary diagnostic, or failed routing invariant
aborts generation or analysis.

## Frozen assets

- Manifest: `validation/history_component_screen_manifest.json`
- Manifest SHA-256: `0fb24666304e916fac477b19ba55fe810eff9e266cf04663d85bc432af7cab23`
- Plan: `configs/history_component_screen_plan.json`

The plan is exploratory. It assigns no p-value and makes no confirmatory claim.
It reports per-edit results and the median across four edits after averaging
chunks 1 and 4 within edit. A selective condition dominates global `.5` only
when its aggregate `delta_R` is no worse and drift is lower, or its `delta_R` is
higher and drift is no worse. No operating point is selected automatically.

## Four-A800 launch

Run only from the committed clean tree whose code should own the evidence:

```bash
cd /workspace/video-model/restream_mvp
scripts/run_history_component_screen_4gpu.sh
```

The wrapper launches one edit group per GPU, waits for all four shards and runs
the fail-closed analyzer only after every shard succeeds. Outputs are written to
`validation/history_components/screen_<commit>/` and ignored by Git because the
checkpoints and videos are large.
