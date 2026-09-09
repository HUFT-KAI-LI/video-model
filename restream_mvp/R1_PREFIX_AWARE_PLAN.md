# R1 计划：Prefix-State-Aware Reality Memory

R0 作为基线**冻结**：不再加 loss、不扩训练步数。R0 的无更新反事实 probe 已定位问题——text-only query 主要学到通用视觉条件 prior，而不是“按当前世界选择现实照片”的能力（[R0_ACTIVE_ZERO_PROBE.md](R0_ACTIVE_ZERO_PROBE.md)）。

R1 分三阶段推进，每阶段都有独立、可证伪的 gate；本仓库当前提交只完成**第一阶段的诊断与必要控制变量**，不训练任何视频模型。

## 第一阶段（本提交）：prefix 能否选对照片

脚本：`scripts/check_prefix_reference_retrieval.py`。冻结 DINO 编码可见 prefix，得到

$$
q_v = E_{\text{DINO}}(V_{\le t}),
$$

再与三组参考比较：correct（同 source 过去）、easy wrong（manifest 随机 donor）、hard wrong（**按与 correct 参考的 DINO 相似度**从其它 source 中选出的最难负例；不使用 $q_v$ 选择，避免选择泄漏）。

指标（唯一 target 为统计单位）：pair accuracy、margin 的 target-level bootstrap 95% CI、AUROC、Recall@1/K、以及 shuffled-query 对照。

**通过标准**：pair accuracy > 70%、margin > 0、CI 不跨 0。

**实测（83 个完整 val target，2026-09-10）**：

| 指标 | 值 |
|---|---:|
| accuracy vs easy wrong | 1.000 |
| accuracy vs hard wrong | 0.940 |
| margin (correct − easy) | +0.5469，CI [0.5049, 0.5890]，83/83 为正 |
| margin (correct − hard) | +0.2776，CI [0.2415, 0.3145]，78/83 为正 |
| AUROC | 0.947 |
| Recall@1 / Recall@2 / top-1 correct | 0.488 / 0.922 / 0.976 |
| shuffled-query 对照 | 0.566 |

**结论**：gate **通过**。可见 prefix 携带足够的 source/scene 信息来区分同源过去参考与随机 donor，甚至在 DINO-hard 跨 source 负例上仍有 94% 正确率；shuffled 对照说明是 query 内容而非度量本身在起作用。

注：每个 target 的 gallery 为 2 个 correct + 2 个 easy-wrong + 2 个 hard-wrong，Recall@1 的理论上限受“2 个正确项只能占一个第 1 位”限制（随机约 1/3）；因此 gate 的主指标是 pair accuracy 与 top-1 correct（0.976），Recall@1=0.488 与 Recall@2=0.922 作为补充。

**局限**：这是同 encoder、同 video、同 shot 的检索，测的是 source/scene 身份可分性，不等于“世界语义理解”；hard negative 只由一个正确参考的相似度选出；仍是离线 manifest、teacher-forcing 数据，不涉及生成质量。

## 第一阶段的配套：R0 matched-control 复测（完整 val）

同一 checkpoint 的**无更新视频 probe** 也扩展到完整 83 个 val target × 2 seeds，并加入 role-matched 控制（`global_async` 与 `correct_kind=async` 同角色）。统计单位仍是唯一 target，target-level bootstrap 95% CI。

| 差值（正=后者更好） | clean（83 target） | mild（83 target） |
|---|---:|---:|
| `U_correct = L_none − L_correct` | +0.00595 [0.0037, 0.0088]，77/83 | +0.00549 [0.0033, 0.0082]，79/83 |
| `G_branch = L_none − L_active_zero` | +0.00228 [0.0016, 0.0031]，75/83 | +0.00222 [0.0015, 0.0030]，75/83 |
| `G_generic = L_active_zero − L_global` | +0.00469 [0.0028, 0.0071]，74/83 | +0.00469 [0.0028, 0.0072]，78/83 |
| `G_content = L_global_positive − L_correct` | −0.00102 [−0.0016, −0.0004]，16/83 | −0.00143 [−0.0021, −0.0008]，14/83 |
| **`G_content_matched = L_global_async − L_correct`** | **−0.00117 [−0.0017, −0.0007]，19/83** | **−0.00149 [−0.0022, −0.0009]，14/83** |
| `S_reference = L_wrong − L_correct` | +0.00059 [−0.00004, 0.0015]，45/83 | +0.00024 [−0.0004, 0.0012]，40/83 |
| `H_wrong = L_wrong − L_none` | −0.00535 [−0.0077, −0.0035]，6/83 | −0.00524 [−0.0076, −0.0034]，4/83 |

变体均值（clean）：none 0.198904；active_zero 0.196625（−1.15%）；correct 0.192959（−2.99%）；wrong_source 0.193550（−2.69%）；global_async 0.191791（−3.58%）；global_aligned 0.191777（−3.58%）；global_positive 0.191936（−3.50%）；pair_mean 0.192732（−3.10%）。

**联合解读**：

- 检索层（本提交实测）：可见 prefix **能**识别同源参考（vs easy 1.000，vs DINO-hard 0.940）。
- 生成层（R0 matched-control 复测）：即使使用与 `correct_kind` 同角色的固定训练均值，它仍**显著优于**当前 target 的正确照片（`G_content_matched` 的 CI 完全在 0 以下）；correct 与 wrong-source 仍不可区分（`S_reference` CI 跨 0）。

因此 R0 的瓶颈**不在“参考数据是否可识别”，而在“生成路径没有使用当前世界状态”**：text-only query 无法把已经可检索的对应关系转成对正确照片的利用。这正好支持下一阶段的最小修改 R1-A（`text + visible-prefix DINO` 作为 query），而不是继续在 R0 上加 loss。

完整 val probe 的逐 target 统计见 [full_val_controls/summary.json](validation/reality_memory/full_val_controls/summary.json) 与 [verification.json](validation/reality_memory/full_val_controls/verification.json)；完整逐 case 报告位于 `outputs/reality_memory/4e_full_val_controls/`（未入库）。

## 第二阶段：R1-A 最小修改（下一步，本提交未实现）

保持冻结的 LongLive 与 soft context residual 不变，只把 query 从

$$
q = Q(E_{\text{text}})
$$

改为

$$
q = Q(E_{\text{text}}, E_{\text{prefix}}),
$$

其中 $E_{\text{prefix}}$ 是可见 prefix 的 1–3 帧经同一冻结 DINO 编码后的 pooled 特征。Soft Context Guidance、Global Mean / Active Zero 控制保持不变。

**Retrieval 层成功标准**：held-out target 上 $s_{\text{correct}} > s_{\text{wrong}}$，且 hard negative 也能区分。
**Generation 层成功标准**：$G_{\text{content}}^{matched} = L_{\text{global\_async}} - L_{\text{correct}} > 0$ 且 $S_{\text{reference}} = L_{\text{wrong}} - L_{\text{correct}} > 0$。两者同时成立，才能第一次说“当前视频状态确实选择并利用了匹配的现实照片”。

## 第三阶段：训练目标修正（下一步，本提交未实现）

不再让 wrong reference 帮助预测同一 GT future。新的目标：

$$
L = L_{\text{video}}^{\text{correct}} + \lambda_r L_{\text{retrieval}} + \lambda_w L_{\text{wrong-inert}} + \lambda_\Delta L_\Delta,
$$

其中

$$
L_{\text{retrieval}} = -\log \frac{e^{s_c/\tau}}{e^{s_c/\tau} + e^{s_w/\tau}},
\qquad
L_{\text{wrong-inert}} = \|\Delta C_{\text{wrong}}\|^2,
$$

并保留 20–30% No-Memory dropout，保证 No Memory ≈ Base。retrieval loss 只训练 prefix projector / query / key，不要求 gate 开启。

## 之后：AR 与长时

等 R1-A 在 teacher forcing 下成立后，先做 3 / 6 个 AR block（10–15 秒）对比 No Memory / Global Mean / Correct / Wrong，再考虑 30s / 60s / 100s。人工 mild corruption 降级为 stress test，不再作为主要科学任务；真正的 drift 来自 self-rollout 自然积累。

## 本提交的边界

- 不训练 R1 视频模型、不改 paired objective、不扩训练预算、不跑 50/200 updates、不开始 R1。
- 数据不重新下载；hard negative 在检索诊断中即时构造，尚未写入 manifest。
- strict-online 仍未构建数据、仍无 GPU 结果。
