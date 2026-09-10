# R1 计划：Prefix-State-Aware Reality Memory

R0 作为基线**冻结**：不再加 loss、不扩训练步数。R0 的无更新反事实 probe 已定位问题——text-only query 主要学到通用视觉条件 prior，而不是“按当前世界选择现实照片”的能力（[R0_ACTIVE_ZERO_PROBE.md](R0_ACTIVE_ZERO_PROBE.md)）。

R1 分三阶段推进，每阶段都有独立、可证伪的 gate；本仓库当前提交**不训练任何视频模型**，只完成第一阶段的诊断、必要的控制变量、以及 R1-A 路由模块的结构实现（仅 CPU forward 测试）。

## 0. 已修复的协议 bug（prefix 检索的可见帧数）

早期检索脚本用 `4 * prefix_latents = 24` 作为可见 pixel frame 数，于是 3 帧 query 取了 `[0, 12, 23]`。Wan 的因果时间映射是

```text
frame 0 -> latent 0,  frames 1..4 -> latent 1,  frames 5..8 -> latent 2, ...
```

因此 $L$ 个 latent 只暴露

$$
F = 4(L-1)+1
$$

个 pixel frame（索引 $0..4(L-1)$）。6 个 latent → **21 帧，索引 0..20**，frame 23 已经越过可见边界（最多约 0.19s 的未来）。

修复：

- 新增 `restream/reality_temporal.py` 作为唯一真值：`pixel_count_for_latent_prefix`、`latent_prefix_boundary_index`、`prefix_visible_seconds`、`assert_visible_frame_indices`；
- `RealityDataset`（运行时断言与 `visible_until` 上界）、manifest builder（`arrival`）与检索脚本全部改用它；
- 检索脚本对 query 帧硬断言 `max(frame) <= 4*(prefix_latents-1)`，6 latent 的三帧 query 现在是 **`[0, 10, 20]`**；
- 回归测试覆盖 latent=3/6/9 → 9/21/33 帧、边界 8/20/32，并断言旧的 `[0, 12, 23]` 会被拒绝。

旧报告保留为 `validation/reality_memory/prefix_retrieval/summary_superseded_pixel_count_bug.json`，**不得**再作为 R1 依据；下列数字均为修复后重跑。

## 第一阶段：prefix 能否选对照片

脚本：`scripts/check_prefix_reference_retrieval.py`。冻结 DINO 编码可见 prefix，得到

$$
q_v = E_{\text{DINO}}(V_{\le t}),
$$

再与三组参考比较：correct（同 source 过去）、easy wrong（manifest 随机 donor）、hard wrong（**按与 correct 参考的 DINO 相似度**从其它 source 中选出的最难负例；不使用 $q_v$ 选择，避免选择泄漏）。

指标（唯一 target 为统计单位）：pair accuracy、margin 的 target-level bootstrap 95% CI、AUROC、Recall@1/K、top-1，以及 **1000 个确定性 derangement 的 null test**（query target ≠ reference target 且 source 互斥）。

**通过标准**：pair accuracy > 70%、margin > 0、CI 不跨 0。

### 结果（全部 83 个 val target，2026-09-10，query 帧 `[0, 10, 20]`）

| 指标 | offline_target_filtered | strict_online + causal_previous |
|---|---:|---:|
| accuracy vs easy wrong | 1.000 | 1.000 |
| accuracy vs hard wrong | 0.940 | 0.940 |
| margin (correct − easy) | +0.5457，CI [0.5040, 0.5877]，83/83 | +0.5454，CI [0.5034, 0.5877]，83/83 |
| margin (correct − hard) | +0.2740，CI [0.2378, 0.3106] | +0.2750，CI [0.2386, 0.3122] |
| AUROC | 0.944 | 0.944 |
| Recall@1 / @2 / top-1 correct | 0.488 / 0.922 / 0.976 | 0.488 / 0.922 / 0.976 |
| derangement null（easy / hard） | 0.492 / 0.436（1000 perms） | 0.492 / 0.437 |
| signal over null（easy / hard） | +0.508 / +0.503 | +0.508 / +0.502 |

**结论**：gate **通过**。修复可见帧计数后结果几乎不变（说明此前的结果并非主要靠那 3 帧未来信息），且 null test 显示真实准确率比同类 derangement 高约 0.50。strict-online 版本用**完全不同的参考池**（83 个 target 的 async pools 与 offline 全部不同，且所有正参考只由因果可见信息筛出）得到同样结论，说明这不是 offline 筛选造成的假象。

**局限**：correct 仍是 same-source / same-shot 过去参考，hard negative 由一个正确参考的相似度选出；这证明的是 **source/scene 身份可检索**，还不是“同一 world 内该查哪一块记忆”（within-world / different-view retrieval），后者需要多视角数据（例如 Ego-Exo4D）。检索也不涉及生成质量。

## 第二阶段：R1-A = Frozen DINO State Router（本提交只实现模块 + CPU 测试）

修完泄漏后 frozen DINO prefix query 已经是 90%+ 的可靠状态信号，因此**不再**先训练一个新的 `Q(text, prefix)` MLP——那会重新引入一个没必要的不确定变量。

结构（`restream/reality_router.py::FrozenStateRouter`，**无可训练参数**）：

```text
Visible Prefix → frozen DINO → q_t = Pool(tokens)
Candidate Memory Bank {M_k}  → m_k = Pool(M_k)
s_k = cos(q_t, m_k);  w = softmax(s_k / τ)
soft 模式:  M~ = Σ_k w_k M_k         （一个有效 memory slot）
topk 模式:  取权重最高的 k 个候选并重新归一化
        ↓
M~ 直接作为 RealityMemory 的输入 → Memory Projector → Gated Context Residual → frozen LongLive
```

关键点：**权重直接决定哪些 reference token 进入 residual**，retrieval 与 generation 在结构上耦合，而不是只记录一个“好看的 accuracy”。这与 R0 的教训（会分类 ≠ 会利用）直接对应。

本提交包含：模块实现、mask 处理（无效候选权重为 0、全 mask 时输出恒等）、top-k 选择、以及“不同 prefix → 不同 routed memory → 不同 fused context”的 CPU 前向测试。**没有** optimizer 训练、没有接入训练循环。

### R1-A 第一轮训练设计（尚未执行）

- 冻结 LongLive 与 DINO；只训练 memory-to-context adapter（现有 `RealityMemory` 参数）。
- 单次前向面对一个 **candidate bank**（例如 correct async ×2 + hard wrong ×2），由 router 加权，而不是“correct forward / wrong forward 两套输入”。
- 目标函数只保留
  $$
  L = L_{\text{video}}^{\text{routed}} + \lambda_\Delta L_\Delta
  $$
  加 20–30% No-Memory dropout；**暂时不加** InfoNCE / retrieval loss（上一轮已验证“单独把 score 学好”不等于 generation 有用），**也不给 wrong reference 单独的 GT video loss**。先看 frozen router 自己能否让 wrong 权重低。
- 短程 10 / 30 / 100 updates，之后才考虑更长。

### 成功标准

**Retrieval 层**：held-out target 上 $s_{\text{correct}} > s_{\text{wrong}}$，hard negative 也能区分（第一阶段已满足）。
**Generation 层**：role-matched $G_{\text{content}}^{matched} = L_{\text{global\_async}} - L_{\text{correct}} > 0$ 且 $S_{\text{reference}} = L_{\text{wrong}} - L_{\text{correct}} > 0$。两者同时成立，才能第一次说“当前视频状态确实选择并利用了匹配的现实照片”。

## 第三阶段：AR 与长时（尚未开始）

等 R1-A 在 teacher forcing 下成立后，先做 3 / 6 个 AR block（10–15 秒）对比 No Memory / Global Mean / Correct / Wrong，再考虑 30s / 60s / 100s。人工 mild corruption 降级为 stress test，不再作为主要科学任务；真正的 drift 来自 self-rollout 自然积累。

## 本提交的边界

- 不训练 R1 视频模型、不改 R0 objective、不扩训练预算、不跑 50/200 updates。
- 数据不重新下载；新增 strict-online **retrieval-only** val manifest（`data/reality_val_online.jsonl`，83 行，schema 3，`causal_previous`），offline manifest 未改动。
- strict-online 仍**没有**视频训练结果、没有 global mean/role means（该配置不生成 role means）。
- 旧的 `1.000/0.940` 报告已标记 superseded，但仍保留在仓库中。
