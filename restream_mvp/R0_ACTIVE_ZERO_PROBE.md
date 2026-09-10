# R0 Active-Zero / Global-Mean / Correct 无更新反事实 probe

本轮按审阅批准，只做**不更新参数**的 teacher-forcing 反事实 probe：加载现有 30-update paired checkpoint 的 memory `state_dict`，不恢复 optimizer/scheduler/RNG/训练偏移，不执行 backward。它回答的是控制变量问题，不是 Reality Memory 有效性结论。

## 目的

在同一 target、同一 history、同一 noise 下比较：

```text
Base = No Memory
Active Zero   : K 个有效 mask + 全零 DINO 特征（分支启用，无图像内容）
Global Mean   : 去重训练正参考池的固定均值（通用真实图像统计，无当前场景内容）
Pair Mean     : 当前 target correct+wrong 的混合参考（消融）
Correct       : 当前 target 同 source 过去参考
Wrong Source  : donor source 参考
```

核心差值（loss 越低越好，正值表示后者更好）：

$$
U_{\text{correct}}=L_{\text{none}}-L_{\text{correct}},\quad
G_{\text{branch}}=L_{\text{none}}-L_{\text{active-zero}},
$$
$$
G_{\text{generic}}=L_{\text{active-zero}}-L_{\text{global}},\quad
G_{\text{content}}=L_{\text{global}}-L_{\text{correct}},\quad
S_{\text{reference}}=L_{\text{wrong}}-L_{\text{correct}}.
$$

统计单位是**唯一 target**：先在同一个 target 内对 noise seed 求平均，再对 target 做 bootstrap 95% CI。noise seed 不是独立样本。

## 复现命令

```bash
cd /workspace/video-model/restream_mvp
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/cache_reality_features.py \
  --config configs/reality_memory_paired.yaml --overfit-samples 16
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/probe_existing_memory_checkpoint.py \
  --config configs/reality_memory_paired.yaml \
  --checkpoint checkpoints/reality_memory_paired_review/step_0030 \
  --split val --cases 16 --noise-seeds 50000 60000 70000 \
  --output outputs/reality_memory/ee73b8e_val_controls
.venv/bin/python scripts/summarize_existing_probe.py \
  --probe outputs/reality_memory/ee73b8e_val_controls/paired_existing_val.json \
  --output validation/reality_memory/active_zero_probe/summary.json
```

## 结果

2026-09-10 执行：单卡 A800，`checkpoints/reality_memory_paired_review/step_0030`（30 次有效更新，sha256 `2137f50c…`），16 个独立 val target、3 个 noise seed、clean/mild 两种 prefix，共 96 个 target×noise 组合。全程只读 memory state_dict，未恢复 optimizer/scheduler/RNG，未执行 backward。校验记录见 [verification.json](validation/reality_memory/active_zero_probe/verification.json)，完整报告见 [paired_existing_val.json](validation/reality_memory/active_zero_probe/paired_existing_val.json)。

### 变体均值（video loss，越低越好）

| prefix | No Memory | Active Zero | Global Mean | Pair Mean | Correct | Wrong Source |
|---|---:|---:|---:|---:|---:|---:|
| clean | 0.201685 | 0.200301 | **0.196547** | 0.197922 | 0.198113 | 0.198258 |
| mild | 0.201432 | 0.199606 | **0.196676** | 0.197087 | 0.197481 | 0.197920 |

### 唯一 target 配对差值（正值 = 后者更好；bootstrap 95% CI，20k 重采样）

| 差值 | clean mean [95% CI]（正/16） | mild mean [95% CI]（正/16） |
|---|---|---|
| `U_correct = L_none − L_correct` | +0.003572 [0.00204, 0.00575]（16/16） | +0.003951 [0.00213, 0.00638]（16/16） |
| `G_branch = L_none − L_active_zero` | +0.001384 [−0.00006, 0.00265]（14/16） | +0.001825 [0.00068, 0.00319]（14/16） |
| `G_generic = L_active_zero − L_global` | +0.003754 [0.00171, 0.00633]（14/16） | +0.002930 [0.00166, 0.00447]（16/16） |
| `G_content = L_global − L_correct` | **−0.001566 [−0.00261, −0.00062]（4/16）** | −0.000805 [−0.00198, 0.00055]（3/16） |
| `S_reference = L_wrong − L_correct` | +0.000146 [−0.00067, 0.00099]（9/16） | +0.000440 [−0.00051, 0.00156]（10/16） |
| `H_wrong = L_wrong − L_none` | −0.003426 [−0.00513, −0.00210]（0/16） | −0.003511 [−0.00537, −0.00206]（1/16） |

### 结论：落在情况 B（单 checkpoint 诊断，不是普遍否证）

- `U_correct > 0` 且在 16/16 个 target 上一致（两种 prefix 都是），说明**打开 memory 分支并给它任何图像特征**确实比 No Memory 好，幅度约 1.7–2.0%。
- `G_generic > 0` 且 CI 不含 0，说明收益的主要来源是**通用真实图像统计**：固定训练均值显著优于全零特征。
- `G_content ≤ 0`：clean 下固定训练均值**显著优于当前 target 的正确照片**（−0.0016，CI 不含 0，只有 4/16 个 target 为正），mild 下方向相同但不显著。这是在本 probe 条件下对“正确照片内容提供额外价值”的**反方向证据**；它只覆盖一个 checkpoint、16 个 target、first-block teacher forcing 与离线协议，不能普遍否证场景内容机制，也不替代 prefix 条件检索的独立检验（见下文“后续”）。
- `S_reference ≈ 0`（9/16、10/16），correct 与 wrong-source 没有可区分的收益；`H_wrong < 0` 说明 wrong-source 也整体优于 No Memory，进一步支持“内容无关”的解释。
- `G_branch` 很小，clean 下 CI 跨 0，说明单靠 adapter/bias 的收益有限，主要增益来自输入了图像特征这件事本身。

因此，在这个 30-update checkpoint（`offline_target_filtered`、text-only query）上：**没有证据支持场景特定记忆；证据指向一个通用 video-domain context prior**。不能把本结果写成 Reality Memory 有效。

**局限**：单个 checkpoint、16 个 val target、first-future-block teacher forcing、离线协议；绝对幅度小（相对 No Memory 约 2.5%）；bootstrap CI 在 16 个 target 下较宽；checkpoint 训练时并没有 global-mean/active-zero 控制，因此该排序是事后诊断，不能反推“用这些控制重新训练”的结果。本轮的完整 val（83 target）matched-control 复测与唯一 target 统计见 [full_val_controls/summary.json](validation/reality_memory/full_val_controls/summary.json)（`G_content_matched` 使用与 `correct_kind` 同角色的 `global_async` 均值）。

**后续（已完成）**：本仓库下一提交把本 probe 扩展到完整 83 个 val target 并加入 role-matched 的 `global_async` 控制（[full_val_controls/summary.json](validation/reality_memory/full_val_controls/summary.json)）：`G_content_matched = −0.00117`（CI [−0.0017, −0.0007]，19/83），结论仍落在情况 B；同时新增 prefix-state 检索 gate（[R1_PREFIX_AWARE_PLAN.md](R1_PREFIX_AWARE_PLAN.md)、[prefix_retrieval/summary.json](validation/reality_memory/prefix_retrieval/summary.json)）证明**可见 prefix 本身能识别正确参考**（accuracy 1.000 / 0.940）。该 gate 的首版曾误用 24 个可见帧（含 3 帧未来），已修复为 21 帧 `[0,10,20]` 并重跑，结论不变。两者合起来说明：瓶颈在生成路径的 query，而不是参考数据是否可识别。

**下一步（按审阅 §九/§十）**：不要继续加对比损失，也不要在没有新证据前扩训练预算。若继续 R0，应加入最小的 prefix scene feature（可见历史）作为 memory query，让模型有条件区分“哪张照片与当前画面相关”；否则应重新审视 reference 语义与训练目标。

<!-- RESULTS-END -->


## 解释框架（事先约定）

- **A**：`G_branch > 0` 且 `G_generic ≈ 0, G_content ≈ 0` → 收益主要来自可训练 context adapter/bias，照片内容没有证据。
- **B**：`G_generic > 0, G_content ≈ 0` → 通用真实图像统计有帮助，但不是场景特定 memory。
- **C**：`G_content > 0` 且 `S_reference > 0`，并在 held-out target 上稳定 → 正确照片内容有额外价值的第一批机制证据。
- **D**：所有差值 ≈ 0 → text-only R0 不足，下一步应加入最小的 prefix scene feature 作为 query，而不是继续加对比损失。

无论结果落在哪一档，本轮都是 first-future-block teacher-forcing 诊断，不是 AR 或长时抗漂移证据；也不改变 paired objective、不重启 50/200-update 训练。
