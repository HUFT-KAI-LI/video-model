# History Attention-Path Decomposition D3

D1 and D2 found that coarse temporal history locking is distributed, redundant, and
exploratorily edit-dependent. D3 stops decomposing history by temporal position and asks
whether semantic inertia is carried by attention access/competition or by history value
content written into the current representation.

## Frozen operators

For history logits `l_H` and value vectors `V_H`, the score/access intervention is:

`l_H' = l_H + log(alpha)`

Thus `alpha=.5` halves history attention odds relative to current keys without changing
key geometry. `alpha=0` branches to native current-only attention and is checked against
that kernel bit-exactly. The value/content intervention is:

`V_H' = beta * V_H`

It leaves queries, keys, logits, and joint history/current softmax normalization unchanged.
In particular, `beta=0` leaves history competing for attention mass while injecting zero
history value content.

For `0<alpha<1`, the A800 path evaluates history-only and current-only native
FlashAttention, requests their native log-sum-exp normalizers, adds `log(alpha)` to the
history normalizer, and combines the two outputs with the resulting joint-softmax mass.
This is algebraically the same attention operator without changing backend or constructing
a dense query-by-key mask. `alpha=1,beta=1` branches directly to native full-history
attention; `alpha=0` branches directly to native current-only attention. Runs produced by
a fallback backend are rejected by the frozen analyzer.

## Frozen screen

The exploratory screen uses seed 505, chunk 4, and the four directional edits. Its seven
conditions are `full`, `global_.5`, `current_only`, `score_.5`, `value_.5`, `value_0`, and
`score_.5_value_.5`: 28 pairs and 56 chunk replays. B0 and full regeneration always use
unmodified full history. Every P0/P1 condition replays the same checkpoint and cached noise.

The response is `delta_R(c)=R(c)-R(full)` with P0 latent drift. A descriptive 2x2
score/value inclusion-exclusion contrast is frozen at the `.5` levels. The analyzer also
reports `R(current_only)-R(value_0)` to measure the editability still withheld when history
keeps normalization mass but injects no value content. Stage C global `.5` and current-only
are references and do not enter the score/value contrast.

There is no significance test, magnitude threshold, or automatic score/value classification.
Timing is invalid because the mechanisms use different attention implementations.

After review and explicit D3 approval, run:

```bash
cd /workspace/video-model/restream_mvp
scripts/run_history_attention_path_4gpu.sh
```
