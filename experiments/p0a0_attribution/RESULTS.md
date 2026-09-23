# P0-A0 results

Validated videos: 160/160.

**Primary Initial Attribute Accuracy requires human review. CLIP below is only a whole-frame color proxy.**

| Model | Human correct / judgeable | Pending | CLIP color proxy |
|---|---:|---:|---:|
| wan | 0 / 0 | 80 | 38 / 80 |
| longlive | 0 / 0 | 80 | 37 / 80 |

The same integer seeds pair prompt conditions, not identical noise tensors. Wan uses 50-step UniPC, CFG 6, shift 8 and its official negative prompt; LongLive uses its released causal 4-step model plus LoRA. A difference is a pipeline-level observation, not isolated evidence against causalization.

All initial mismatches are retained. Recoverability analysis must separately require initially correct → later failure.

Exploratory uncertainty resamples the eight shared motion/template groups. The 20 prompts are not 20 independent semantic templates. No automatic threshold was fitted to these outcomes.
