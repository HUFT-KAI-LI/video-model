# M1-A Oracle Mask Distillation

M1-A tests whether per-case SPSA layer masks can be replaced by a single controller
forward pass. LongLive and the frozen T5 encoder remain unchanged. Two small controllers
are trained only with mask MSE against 30-layer lambda `.2` SPSA teachers.

## Frozen data and models

The teacher set contains 40 units: four directional edits, existing seeds 606/707, and
new seeds 801 through 808, all at chunk 4. New teachers use exactly the M0/M1 16-step
SPSA protocol and exact loss `-E + .2 D_drift + .001 sum(m)`. Weak or unsuccessful
teacher searches are retained without filtering.

The Prompt-only controller receives mean-pooled frozen-T5 `P1-P0`, projected from 4096
to 64 dimensions with a fixed seed-1701 Rademacher matrix. The Prompt+State controller
also receives a `30x24` summary: per layer and per head mean and RMS of the history V
cache that remains visible after current-chunk insertion (three sink and six local latent
frames). The current chunk is excluded.

Prompt-only has 6,110 parameters. Prompt+State has 6,513. Both use a sigmoid output and
were trained for 3,000 full-batch Adam epochs at learning rate `.003`. No video loss,
mask regularizer, validation selection, LongLive gradient, or post-training tuning was
used. Training-set mask MSE was 0.012764 for Prompt-only and 0.000216 for Prompt+State;
these numbers are diagnostics only.

Before held-out generation, the 40-unit dataset, controller checkpoint, and training
report were hash-locked in `configs/mask_distillation_plan.json` and committed as
`209160b09dcc4c3d4ae357a4fa74c99678f7f6a3`. Held-out seeds 1001/1002 were disjoint
from every teacher seed. Each of eight units evaluated full, global `.5`, current-only,
Prompt-only, Prompt+State, and a fresh lambda `.2` SPSA search reference: 48 final pairs.

The frozen PASS rule requires one controller to Pareto-dominate global `.5` in at least
6 of 8 units. Video replay `delta_R` and P0 latent drift decide the result; mask MSE does
not enter the decision.

## Recorded result

M1-A is **NO PASS**. Prompt-only won 5/8 held-out units and Prompt+State won 4/8, below
the frozen 6/8 threshold. All 48 pairs passed checkpoint, RNG, layer-routing,
outside-preservation, DINO, boundary, model/config, controller, artifact-hash, and clean
provenance checks.

The aggregate median does show useful signal. Global `.5` produced
`(delta_R,D)=(0.19218,0.30963)`, Prompt-only `(0.17788,0.18065)`, and Prompt+State
`(0.42090,0.22375)`. Median exact-loss recovery relative to the SPSA reference was 0.467
for Prompt-only and 0.846 for Prompt+State. This aggregate improvement cannot override
the failed unit-level rule.

The state-conditioned model won dress/1002, jacket/1002, darker/1001, and
warm-to-cool/1001. Prompt-only won dress/1002, both jacket units, and both darker units.
No controller was uniformly better. The result supports learning signal and some value
from state, while showing that 40 teachers plus this summary/controller are insufficient
for the frozen consistency target.

The finite 16-step SPSA reference was exceeded by a predictor on several units. It is an
equal-budget search reference, not a mathematical upper bound; reported recovery above
1 has that precise meaning. The machine-readable analysis is
`validation/mask_distillation_analysis_209160b.json` with SHA-256
`9d21cea092d4736c50812258a91e6144b0a06c5c56ff9bd8b0053ccfafeb26e1`.
