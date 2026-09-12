# M1-B Result: Expanded Oracle Teacher Distillation

M1-B tests whether teacher scale alone makes the unchanged Prompt+State controller
generalize reliably. The experiment added 160 new Oracle masks to the locked 40
M1-A teachers, trained on all 200 units, and evaluated 16 held-out edit-seed units.

## Decision

**NO PASS.** Prompt+State achieved 8/16 strict Pareto wins over global 0.5. The
frozen requirement was at least 12/16. Prompt-only, retained as a non-decision
ablation, achieved 7/16.

| Controller | Pareto wins | Required for M1-B PASS |
| --- | ---: | ---: |
| Prompt+State | 8/16 | 12/16 |
| Prompt-only ablation | 7/16 | not decision-bearing |

Prompt+State wins by edit were dress 2/4, jacket 2/4, darker 3/4, and warm-to-cool
1/4. This is not stable held-out generalization.

## Descriptive replay results

| Condition | Median delta_R | Median D_drift |
| --- | ---: | ---: |
| global 0.5 | 0.15803 | 0.22657 |
| Prompt-only | 0.17710 | 0.17937 |
| Prompt+State | 0.19039 | 0.19508 |
| 16-step SPSA reference | 0.29727 | 0.17581 |

Prompt+State improves the aggregate median, but the frozen decision is unit-level
Pareto consistency, so the aggregate improvement does not override the NO PASS.
The median recovery of finite-SPSA objective advantage is 0.613 for Prompt+State.

## Evidence validity

All 80 final pairs passed checkpoint identity, full-history P0 exactness, matched
RNG, outside exactness, 30-layer routing, DINO, boundary, plan/manifest/controller
hash, and clean cross-shard provenance checks. All held-out results came from clean
commit `0fded5fa6a9597e4ea5cd410d463b6d7fefc895d`.

The result rejects the hypothesis that teacher scale alone is sufficient for this
unchanged controller. The SPSA reference retains a better median frontier, so the
next experiment should change controller representation or the supervised loss,
as frozen in the protocol, rather than add another teacher-only scale-up.
