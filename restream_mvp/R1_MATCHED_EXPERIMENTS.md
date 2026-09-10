# R1-A matched Top-1 experiments

本轮固定 Test 数据资格、K=1 oracle，并按零更新 smoke → 三组 2-update sanity → 三组 10-update 的顺序运行。所有效果分析只用 Train/Dev；不启动 30/100 updates，不以 train loss 代替内容利用证据。

## Test eligibility 已封存：0 / 100

在效果训练前，`scripts/seal_r1_test.py` 先写入不可变的 `data/r1_test_eligibility_policy.json`，绑定 source lock、shot metadata、selection identity 和执行代码哈希，再对原先固定的 100 个 source 使用与 Train/Dev 完全相同的候选构建规则。脚本不加载 DINO/LongLive，不计算生成或检索指标。

结果：97 个 source 没有足够长的连续 shot；3 个无法凑齐满足时间间隔/相似度条件的 async pool。因此 `data/r1_test_targets.jsonl` 为空，`data/r1_test_eligibility.json` 记录全部 100 个 rejection、manifest SHA 和 policy SHA。未替换 source、未降低阈值。100 个 source 过去大多已被质量筛选排除，这次结果确认了原有风险。

**当前没有可用的 untouched Test，不能进行最终泛化结论。** 本轮继续的是获准的 Train/Dev 机制实验。若将来要建立新的评测集，需单独预注册新的采集/资格规则并保持与模型指标隔离，不能暗中替换此次被拒绝的 source。

## 核心分支与控制

| 分支 | 生成实际读取的 memory |
|---|---|
| Global-Async | 1 份去重 strict-online Train async mean，8 tokens |
| Correct-Top1 | 在 correct async ×2 中用同一 frozen prefix router 选 1 张，8 tokens |
| Routed | correct async ×2 + hard-wrong ×2 中选 1 张，8 tokens |
| Correct-All2（仅 Dev/smoke） | 全部 2 张 correct，16 tokens |
| Wrong-Top1（仅 Dev） | 同一 hard donor 的 2 张参考中选 1 张，8 tokens |
| No Memory（仅 Dev/dropout） | false mask，context 精确恒等 |

`correct_only` 保留为 `correct_all2` 的兼容别名；新增训练核心 oracle 使用 `correct_top1`。hard donor 仍只由 correct reference 的 DINO 均值选择，不使用 prefix query，且不能跨 split/source/content SHA。Routed 与 Correct-Top1 的唯一区别是有没有 distractors；两者如果选同一张参考，其实际生成输入相同。

## 零更新 smoke 与数值精度

`check_r1_matched_backward.py` 同一个真实 Train target、adapter seed=42、noise seed=123，加载一次 frozen LongLive，逐分支 fresh zero-init RealityMemory；对四组做 backward，并额外重复一次 Routed 来测量 BF16/FlexAttention backward 的归约噪声。optimizer steps=0。

四组及 base loss 均为 0.3283647298812866，初始化哈希一致。Global 与 Correct-Top1 的 output-weight 梯度差远大于相同输入重复噪声。此次 Routed 和 Correct-Top1 都选中候选 1，它们的梯度应在数值噪声内一致，不能要求它们必然不同。

初始逐位梯度比较未通过；加入相同输入重复后测得梯度差范数约 0.00446（梯度范数约 0.326），Correct-Top1 与 Routed 差约 0.00411；Global 与 Routed 差约 0.32646。检查要求相同内容差异 ≤3 倍相同输入重复差、且 <5% 梯度范数；Global/Correct 差异须 >5 倍重复差。CPU 的同输入 exact-equivalence 测试仍保留。不能把微小 GPU 梯度差解读为科学差异。

## 固定训练设计

专用入口 `train_reality_r1.py`，配置 `configs/reality_memory_r1_matched.yaml`。仅训练 fresh RealityMemory；冻结 router、DINO、LongLive（含其官方 LoRA）。目标为 `L_video + 1e-5 * L_delta`，没有 InfoNCE、contrast 或 wrong-reference GT loss。

- 三组初始化 seed=42、LR=1e-4、AdamW weight_decay=0.01、warmup=10、grad_clip=1。
- 每张 GPU batch=1，clean GT prefix teacher forcing，57×256×432 pixels，6 prefix latents、1 future block。
- 25% 整组 No-Memory dropout。空 memory batch 仍验证零梯度，但不执行 AdamW、不推进 scheduler、不计为有效 update，避免 decay/momentum 造成变化。
- sample order、dropout、noise 分别用 `(seed, epoch/batch_step)` 的独立确定性 RNG，三个分支与 resume 共享完全相同序列。
- fresh invocation 最多 2 updates；通过 sanity 后从同分支 R1 checkpoint resume 到 10。拒绝 R0 checkpoint、跨分支/配置/数据/代码签名不一致的 resume。
- 每个 2-update checkpoint 必须有非零 projector 梯度；保存 optimizer、scheduler、batch position、RNG 和 manifest/encoder/global mean/Test seal/实现代码签名。
- Dev 在训练前固定：从 83 个 target 中以 seed=20260911 选 8 个，noise seed=50000；对每个训练出的 adapter 跑上表 6 个控制，计算 target-level bootstrap CI。此小样本诊断不是 full-Dev 或 Test 结论。

三组短跑在三张 A800 上同时运行，单分支有效预算相同。checkpoint 位于忽略目录 `checkpoints/r1_matched_v1/`，只上传配置、代码、数据资格和 JSON 指标，不上传模型权重。

## 复现

在 `restream_mvp` 目录，先确认原始视频、模型和严格在线 feature cache 就绪：

```bash
.venv/bin/python scripts/seal_r1_test.py
.venv/bin/python scripts/check_r1_matched_backward.py
OMP_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -v
```

每个 branch 取 `global_async`、`correct_top1`、`routed` 之一，使用不同 GPU/输出目录：

```bash
.venv/bin/python train_reality_r1.py --branch routed --max-updates 2 --output checkpoints/r1_matched_v1/routed --report validation/reality_memory/r1_matched/routed
.venv/bin/python train_reality_r1.py --branch routed --max-updates 10 --resume checkpoints/r1_matched_v1/routed/update_0002.pt --output checkpoints/r1_matched_v1/routed --report validation/reality_memory/r1_matched/routed --evaluate-dev
.venv/bin/python scripts/summarize_r1_matched.py --updates 2
.venv/bin/python scripts/summarize_r1_matched.py --updates 10
```

## 2-update sanity 验收

71 项 CPU tests 通过。三组各运行 3 个 batch，首个 No-Memory batch 梯度为 0 且无参数变化，随后完成 2 个有效更新。初始化哈希、样本/噪声/dropout/学习率序列全部一致。第二次更新的 projector 梯度范数分别为 Global 1.403e-4、Correct-Top1 1.407e-4、Routed 1.349e-4，证明首步 output 学习后梯度已进入上游 adapter。详见 [summary_0002.json](validation/reality_memory/r1_matched/summary_0002.json)。

## 判断依据

对每个训练 checkpoint 用相同 Dev target/noise 做反事实比较：

- `G_content_matched = L_global_async − L_correct_top1`：视觉注入是否利用 target-specific content。
- `S_route = L_wrong_top1 − L_routed`：相同 K 下，路由参考是否优于 wrong-only。
- `L_routed − L_correct_top1`：distractors 造成多少损失。

若 Correct-Top1 约等于 Global，不应继续磨 router；若 Correct-Top1 优于 Global 而 Routed 落后，才检查 routing/candidate bank。Correct-All2 作为额外上界，不能拿 K=2 对 K=1 的差异单独归因于 router。same-source/different-shot fingerprint 诊断仍留在论文前完成。


## 10-update 结果：尚无 content utility 证据

三组各完成 13 batch / 10 有效更新 / 3 No-Memory batch。`summary_0010.json` 核对初始化、代码/数据签名、每个 sample/noise/dropout/LR 完全一致；8 个 Dev target 的 No-Memory loss 跨 checkpoint 差为 **0**。每个 checkpoint 均完成 8×6=48 次固定噪声控制前向，合计 144 次。

下面每行是一个**训练出的 adapter checkpoint**，各列为该 checkpoint 上的反事实输入，loss 越小越好：

| 训练分支 | No Memory | Global | Correct-Top1 | Correct-All2 | Routed | Wrong-Top1 |
|---|---:|---:|---:|---:|---:|---:|
| Global-Async | 0.135039 | 0.133933 | 0.134442 | 0.134424 | 0.134417 | 0.134321 |
| Correct-Top1 | 0.135039 | 0.134528 | 0.134612 | 0.134696 | 0.134608 | 0.134625 |
| Routed | 0.135039 | 0.133975 | 0.134464 | 0.134404 | 0.134463 | 0.134416 |

关键 paired differences（单位 `1e-4`；95% CI 为 8 个 target 的 percentile bootstrap）：

| 训练分支 | G_content：Global − Correct-Top1 | S_route：Wrong-Top1 − Routed |
|---|---:|---:|
| Global-Async | −5.093 [−7.422, −2.772] | −0.961 [−2.642, +0.381] |
| Correct-Top1 | −0.837 [−1.861, +0.194] | +0.170 [−0.807, +1.182] |
| Routed | −4.884 [−7.931, −2.260] | −0.468 [−2.480, +1.334] |

Correct-Top1 训练后，真实参考仍未显示优于 Global 的证据；其 G_content CI 跨零。Routed checkpoint 的 Global loss 更低，G_content 为负，Wrong/Routed 区分也不稳定。Routed 与 Correct-Top1 在 8 个 Dev target 中有 7 个选中完全相同参考，因此推理时两者接近本身并不能说明 content 被利用。

结论：在这次 **10 updates / 8 Dev targets / 1 noise seed** 的小型机制实验中，核心 content gate 未通过。相对 No Memory 的总体下降不能替代 target-specific content 优势；单看 train loss 也不足以支持 routing 有效。

按预定诊断树，**不扩到 30/100 updates，不继续优化 router**；下一步优先检查 `RealityMemory → text context` 的视觉注入位置及其表示能力。本实验不能证明该 path 在所有训练预算下都无效，也不能支持 Test/泛化结论；当前封存 Test 的有效 target 为 0。BF16 backward 的重复噪声及仅单 seed、小 Dev 样本均限制了结果精度。

报告：[zero_update_smoke.json](validation/reality_memory/r1_matched/zero_update_smoke.json)、[summary_0002.json](validation/reality_memory/r1_matched/summary_0002.json)、[summary_0010.json](validation/reality_memory/r1_matched/summary_0010.json)。各分支目录保留完整训练记录、初始化/输入签名、参数梯度/变化范数和逐 Dev target 的六控制结果。
