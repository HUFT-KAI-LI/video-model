# M0/M1 Oracle Layer-wise Release Mask

This experiment asks whether a non-uniform, case-specific 30-layer release mask can move
the editability-preservation frontier beyond the fixed Stage-C global `.5` gate. LongLive
and its LoRA weights remain frozen. Layer `l` uses

`O_l = O_l_full + m_l * (O_l_current - O_l_full)`, with `m_l in [0,1]`.

`m=0` branches to native full-history attention, `m=1` to native current-only attention,
and an all-half mask is bit-exact with Stage-C global `.5`. All references are generated
with native full history, and every candidate replays the same chunk-4 checkpoint and
stored denoising noise for P0 and P1.

## Frozen optimization

The exploratory grid contains four directional edits and new seeds 606/707: eight
independent edit-seed units. Each unit optimizes its own 30 coefficients for the three
pre-registered drift weights `lambda={.05,.2,.8}`. The loss is

`L = -E + lambda * D_drift + .001 * sum(m)`.

The existing color/luma probes are discrete CPU metrics, so the experiment uses 16-step
deterministic SPSA rather than substituting a differentiable surrogate. The initialization
is the all-half Stage-C baseline. Candidate selection is frozen to the lowest exact loss
among initialization and every plus/minus SPSA probe. Final reporting contains 48 pairs:
full, global `.5`, current-only, and three optimized masks for each unit. There is no
significance test or post-run magnitude threshold.

The frozen PASS rule requires at least one optimized mask to weakly improve both
`delta_R` and `D_drift` over global `.5`, with a strict improvement on at least one axis.
The analyzer also reports the number of successful units so the weak existential rule is
not mistaken for uniform generalization.

## Recorded result

The 4xA800 run completed on clean commit
`312580ad8c0d15d899056ce64bd44243ae953650`. All 48 final pairs passed checkpoint,
30-layer routing, native endpoint, all-half Stage-C identity, RNG, outside-preservation,
DINO, boundary, model/config, plan, manifest, and clean-provenance checks.

The frozen decision is **PASS**: 6 of 8 edit-seed units contained an Oracle mask that
strictly Pareto-dominated global `.5`. Both seeds passed for dress red-to-blue, jacket
green-to-yellow, and lighting darker. Neither warm-to-cool seed passed, so this is evidence
for oracle feasibility with clear edit dependence, not a universal controller result.

Across eight units, global `.5` had median `(delta_R, D_drift)=(0.13171, 0.24872)`.
The middle weight `lambda=.2` had median `(0.22920, 0.21773)` and supplied a dominating
point in all six successful units. `lambda=.05` traded more drift for response, with median
`(0.27287, 0.33680)`; `lambda=.8` favored preservation, with median
`(0.07448, 0.10627)`. Optimized masks were non-uniform across layers, with observed
coefficients spanning approximately 0.31 to 0.67.

The machine-readable analysis is
`validation/oracle_layer_mask_analysis_312580a.json` (SHA-256
`3b178af7714de4fbd9268780845f53164a433d190c23e12725b96addb7bf4318`). Four complete
clean-provenance shard JSON files are archived beside it. Generated videos and caches stay
under the ignored `validation/oracle_layer_masks/screen_312580ad8c0d/` directory.
