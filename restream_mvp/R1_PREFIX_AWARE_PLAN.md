# R1 计划：Prefix-State-Aware Reality Memory

R0 作为基线**冻结**：不再加 loss、不扩训练步数。R0 的无更新反事实 probe 已定位问题——text-only query 主要学到通用视觉条件 prior，而不是“按当前世界选择现实照片”的能力（[R0_ACTIVE_ZERO_PROBE.md](R0_ACTIVE_ZERO_PROBE.md)）。

R1 分三阶段推进，每阶段都有独立、可证伪的 gate；本轮完成检索脚本修复、Top-1 路由、Train/Dev/Test source 固定、strict-online Train 与真实 LongLive 单次 forward/backward 验证。仅验证梯度连通性，optimizer updates 固定为 0。

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

**通过标准**：easy **和** hard 各自满足 pair accuracy > 70%、margin > 0、margin CI 下界 > 0；最终 `gate.pass = gate.easy.pass AND gate.hard.pass`。旧报告保留当时的 gate 字段，历史 aggregate 的重新审核见 `validation/reality_memory/r1_top1/gate_recheck.json`。

### 历史结果（全部 83 个 Dev target，旧文件名仍为 val，2026-09-10，query 帧 `[0, 10, 20]`）

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

## 第二阶段：R1-A = Frozen DINO Top-1 Router

结构（`restream/reality_router.py::FrozenStateRouter`，无可训练参数）：

```text
Visible Prefix [0,10,20] → frozen DINO → pooled query
Candidate references (correct async ×2 + hard wrong ×2) → pooled keys
cosine argmax → 一张 reference 的全部 8 个原始 DINO tokens
→ RealityMemory projector → gated context residual → frozen LongLive
```

只支持 `mode: top1`。`soft`/`topk` 显式拒绝：旧 top-k 的 raw-token scaling 会被 projector 首层 LayerNorm 基本消除；旧 soft 融合把不同图片同一 patch index 混合，缺乏跨视角空间对应。Top-1 不缩放、不融合 patch，temperature 只改变诊断用 softmax probabilities，不能改变选中 reference 或生成输入。全 mask 时返回零 memory、false mask 和 selected index −1。

测试不仅检查 router 输出：还比较 routed 与直接输入选中 reference 的 projector/context 结果完全一致，切换 reference 后 context 和 output-layer 梯度会改变。fresh zero-init 时输出必须等于基础 context，但 video loss 仍能向 output.weight 回传梯度；上游 projector/gate 首次 backward 梯度为零是零初始化的预期行为。

### Train / Dev / Test source 隔离

`data/r1_split_lock.json` 固定 seed、输入清单和历史诊断报告的 SHA-256、各 split 名单与 SHA-256：

- R1 Train：867 个 source 候选；strict-online 筛选后 728 个 target，`data/reality_train_online.jsonl`。
- R1 Dev：当前反复使用的 83 个 target。仍沿用 `data/reality_val_online.jsonl` 和 CLI `val` 名称以兼容已有工具；这些结果不能称为 untouched test。
- R1 Test：`data/r1_test_sources.txt` 固定 100 个 source，排除所有旧 Reality Train/Val（包括 global mean 的贡献者）及早期 ReAnchor、VAE、Reality 诊断报告显式出现的 source、video path 和内容 SHA。source 和已知内容 SHA 均不与 Train/Dev 相交。

**Test 的当前状态是 source 预留，不是已就绪的 100 个有效评测 target。** 现有 1068 个已下载 source 中未进入旧实验的部分主要是过去质量筛选未保留的 source；它们可能仍不满足固定的 shot/reference 规则。后续必须先固定资格规则再构造 Test target，记录淘汰率，不按模型结果替换 source。这里未解码 Test、未构造 Test features、未运行 Test 指标，不能据此声称最终 held-out test 已完成。

`freeze_r1_splits.py` 可幂等重放，但拒绝改写已有不同内容的冻结名单。builder、Dataset、特征缓存都验证冻结文件哈希、target membership 和全部 donor membership。既验证 source ID，也验证 video SHA；Test 不用于 architecture、temperature、top-k、loss weight 或 smoke 的选择。

Train 和 Dev 均为 `strict_online + causal_previous`。builder 用冻结的 source manifests 选 donor，并在写入前验证 split。`check_r1_data.py` 对 Train/Dev 逐条实际解码，确认所有 prefix 帧不超过 visible_until 且边界 sampled_time 与之相等。offline R0 manifests 与其统计结果保留为历史记录。

### 本轮验证与复现

在 `restream_mvp` 目录，已有本机模型和原始视频资产时执行：

```bash
.venv/bin/python scripts/freeze_r1_splits.py
.venv/bin/python scripts/build_reality_manifest.py --config configs/reality_memory_paired_online.yaml --splits train --stats-output data/reality_train_stats_online.json
.venv/bin/python scripts/cache_reality_features.py --config configs/reality_memory_r1_top1.yaml --device cuda --report data/reality_feature_stats_r1_online.json
.venv/bin/python scripts/check_r1_data.py
OMP_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/check_prefix_reference_retrieval.py --config configs/reality_memory_paired_online.yaml --cases 7 --target-seed 17 --null-samples 1000 --output validation/reality_memory/r1_top1/seeded_retrieval.json
.venv/bin/python scripts/check_r1_backward.py
```

seeded null regression 覆盖 83 行中选 7 个非连续 index、完整三套 similarity matrices 及 1000 个 derangement；局部 query row 与全局 dataset index 分离。实际 7-target Dev 运行的报告是工程回归证据，不能替代历史 83-target 统计。

`check_r1_backward.py` 使用真实训练 target 的可见 prefix、冻结 DINO、完整 mixed bank、Top-1、fresh zero-init RealityMemory 和真实 frozen LongLive teacher forcing。只做一次 video-loss forward/backward，不创建 optimizer、不接受 resume、不保存训练 checkpoint。报告记录模型初始化哈希、输入 provenance、选择结果、每个 adapter 参数梯度、主干冻结状态及 optimizer_steps=0。

本机验证结果见 [summary.json](validation/reality_memory/r1_top1/summary.json)：66 项 CPU 测试通过，728 Train + 83 Dev 的实际解码 invariant 全部通过，7-target seeded 检索完成 1000 个 null permutations。真实 A800 单次 backward 的 video loss = 0.328365，output.weight gradient norm = 0.325869，峰值显存 41.93 GiB，optimizer updates = 0。这只证明 routed memory → adapter 的 video-loss 梯度路径工作，不是效果结果。

R1 config 不含 paired/contrast objective。旧 `train_reality_memory.py` 显式拒绝 R1 config，避免误走 R0 输入路径；正式 R1 训练循环不在此提交中。

### 下一轮训练设计（尚未执行）

只比较三个 fresh matched runs：

| 分支 | 输入 |
|---|---|
| Global-Async Constant | 从本次 strict-online Train 的 async pool 去重计算的固定均值 |
| Correct-Only Oracle | 同 target 的 correct async references |
| Frozen-Router Mixed Bank | correct async ×2 + hard wrong ×2，经 Top-1 选完整一张 |

`restream/reality_r1.py::select_r1_memory` 提供三组输入；单步脚本也支持 `--branch correct_only` / `--branch global_async`。同一个 seed 在加载 backbone 后重置，再创建 adapter，以保持完全相同的 fresh initialization。R0 checkpoint 只做历史 baseline，不做 warm-start。

下一轮须保持相同 train targets、sample order、noise seeds、initialization、learning rate 和有效 optimizer update 预算。仅用 routed video loss + delta regularization，统一 20–30% No-Memory dropout；不加 retrieval loss、不单独给 wrong references GT video loss。先 10 / 30 / 100 updates，不能用本轮零更新 smoke 推断生成质量。

### 成功标准与停止条件

期望 `L_CorrectOnly < L_Global` 且 `L_Routed ≈ L_CorrectOnly < L_Global`；若 Correct-Only 也无法优于 Global，停止调 router，先审视视觉信息注入 LongLive 的位置。若 Oracle 有效而 Routed 落后，再定位 routing 问题。

role-matched `G_content = L_global_async − L_correct > 0` 且 `S_reference = L_wrong − L_correct > 0`，才能支持场景内容利用；在 Dev 上的架构选择不能包装为最终 Test 结论。

same-source/same-shot proxy 仍可能依赖 watermark、字幕、logo 或编码风格；shuffled-query null 无法排除 source fingerprint。论文前补充 **same source + different shot** hard negative，与现有 different-source/DINO-similar hard negative 并列。此诊断不在本轮范围，也不把当前结果称为 world-level retrieval。

## 第三阶段：AR 与长时（尚未开始）

等 R1-A 在 teacher forcing 下成立后，先做 3 / 6 个 AR block（10–15 秒）对比 No Memory / Global Mean / Correct / Wrong，再考虑 30s / 60s / 100s。人工 mild corruption 降级为 stress test，不再作为主要科学任务；真正的 drift 来自 self-rollout 自然积累。
