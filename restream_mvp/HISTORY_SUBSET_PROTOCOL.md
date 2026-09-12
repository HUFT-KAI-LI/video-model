# History Subset Interaction D2

D1 showed that soft release of a single coarse temporal component did not recover the
Stage C global-release editability gain. D2 tests whether coarse components have
non-additive subset structure before any score/value or layer/head decomposition.

## Frozen screen

The main screen uses chunk 4 and exploratory seed 404. Its native attention window is:

`C0 sink | C2 old | C3 recent | C4 current`

For each of four edits, B0 and full-regeneration are generated once at unmodified full
history. P0 and P1 then replay the same checkpoint under all eight binary subsets of
`sink`, `old`, and `recent`. Each binary subset invokes native attention once over the
included history plus current tokens. `global_release` at Stage C `g=.5` is retained as
a ninth reference and is excluded from factorial decomposition. The grid is 36 pairs,
or 72 chunk replays.

The factorial response is `F(B)=R(B)`. The analyzer reports all pairwise inclusion-
exclusion contrasts, the third-order contrast, `L(B)=F(empty)-F(B)`, and the exact
three-component Shapley allocation of `L(SOR)`. It also reports `delta_R` against SOR
and P0 latent drift for every condition.

No equivalence threshold or significance test is defined. The analyzer therefore does
not automatically assign the qualitative A/B/C/D patterns. Nonzero factorial contrasts
may suggest interactions, but do not prove a mechanistic interaction. Chunk 1 remains
a routing and kernel control and is not part of the main GPU grid.

## Fail-closed rules

The runner and analyzer reject incomplete or duplicate grids, changed plan or manifest
hashes, dirty or mismatched code/model/config provenance, checkpoint differences across
conditions, replay RNG mismatch, outside-chunk changes, missing DINO or boundary
diagnostics, an inexact SOR P0 baseline, an incorrect 3/3/3/3 partition, or an inert
current-only endpoint. Timing remains an invalid diagnostic because the global reference
and binary subset conditions use different operators.

After review and explicit screen approval, run:

```bash
cd /workspace/video-model/restream_mvp
scripts/run_history_subset_interaction_4gpu.sh
```

## Recorded result

The approved 4xA800 screen completed on commit `daaf075311b14db837d468410cc00d00d373ef8a`.
All 36 pairs and 72 chunk replays passed the frozen invariants. The machine-readable
analysis is archived at
`validation/history_subset_analysis_daaf075.json`; its SHA-256 is
`5058ce48a7ccb65fab89768c4a08c96eb67d4fef6ccd4b31a6806b192767884c`.

The result suggests distributed, redundant locking with edit-dependent subset structure.
Most two-component subsets recover nearly all `SOR` locking. Singleton behavior varies:
for example, recent history supplies only 12.3% of full locking for dress red-to-blue,
but 99.6% for jacket green-to-yellow. Empty-baseline Mobius coefficients are descriptive
exploratory contrasts; this screen defines no significance test or equivalence threshold.
