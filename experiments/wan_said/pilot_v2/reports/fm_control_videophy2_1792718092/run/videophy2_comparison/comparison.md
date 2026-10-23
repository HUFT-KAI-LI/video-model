# Base / FM-Only A / FM+SAID C

VideoPhy-2: identical 100 prompts/seeds. VBench: identical 100 prompts x 5 seeds.
Scores/rates retain official scales. Rates are fractions, not percentages.

| Suite | Family | Metric | Base | A (FM) | C (FM+SAID) | A - Base | C - A |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| videophy2 | scores | joint_at_4 | 0.250000 | 0.240000 | 0.320000 | -0.010000 | +0.080000 |
| videophy2 | scores | pc_at_4 | 0.600000 | 0.680000 | 0.670000 | +0.080000 | -0.010000 |
| videophy2 | scores | pc_mean | 3.670000 | 3.760000 | 3.780000 | +0.090000 | +0.020000 |
| videophy2 | scores | sa_at_4 | 0.320000 | 0.290000 | 0.350000 | -0.030000 | +0.060000 |
| videophy2 | scores | sa_mean | 3.130000 | 3.080000 | 3.160000 | -0.050000 | +0.080000 |

## Limits

- 100-prompt pilot subsets, not full official benchmark results.
- A is FM-only LoRA, not full-parameter fine-tuning; C is FM+SAID with the same LoRA protocol.
- C minus A estimates the alignment effect under this single matched training seed, not universal significance.
- Official color exclusions may change effective denominators; consult per-model coverage.
- Caption/crop human review remains pending; the sealed 500-video dataset test split is not used.
