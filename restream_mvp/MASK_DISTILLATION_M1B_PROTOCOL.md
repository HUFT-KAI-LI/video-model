# M1-B Expanded Oracle Teacher Protocol

## Question

Does the unchanged Prompt+State controller become a stable predictor of per-case
30-layer history-release masks when the Oracle teacher set is expanded?

M1-B changes the amount of teacher data only. LongLive and T5 remain frozen. The
prompt/state features, controller architectures, mask-MSE objective, Adam settings,
and deterministic 16-step SPSA teacher procedure are identical to M1-A.

## Frozen data split

- Locked M1-A teachers: 40 units from seeds 606, 707, and 801-808.
- New M1-B teachers: 160 units from four edits and seeds 1101-1140.
- Combined training set: 200 Oracle masks.
- Held-out evaluation: four edits and seeds 2001-2004, for 16 units.
- All units use chunk 4 and the Oracle objective with lambda 0.2.

The new teacher and held-out manifests are disjoint and hash-locked in
`configs/mask_distillation_m1b_plan.json` before GPU generation.

## Controllers

Prompt-only remains an ablation. Prompt+State is the sole decision-bearing model.
Both architectures and inputs are unchanged from M1-A. Training uses full-batch
mask MSE for 3000 Adam epochs with seed 314159. Mask MSE is a training diagnostic;
it is not an experimental endpoint.

## Held-out evaluation

Every held-out unit starts from one fixed base checkpoint and replays these five
conditions with matched RNG:

1. full history
2. global layer release 0.5
3. Prompt-only prediction
4. Prompt+State prediction
5. same-budget 16-step SPSA reference

The SPSA result is a finite search reference, not a mathematical upper bound.
Evaluation uses real paired P0/P1 replay, `delta_R`, `D_drift`, outside exactness,
RNG exactness, boundary diagnostics, DINO identity preservation, and 30-layer
routing checks.

## Frozen decision

A unit is a strict Pareto win over global 0.5 when its prediction has no lower
`delta_R` and no higher `D_drift`, with at least one strict inequality. M1-B passes
only if Prompt+State wins on at least 12 of the 16 held-out units. Prompt-only is
reported as an ablation and cannot trigger PASS.

If M1-B does not pass, the next experiment changes controller representation or
loss rather than adding another teacher-only scale-up.
