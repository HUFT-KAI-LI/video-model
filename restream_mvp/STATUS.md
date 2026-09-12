# ReStream 本轮审阅状态（2026-09-09）

## M1-B Expanded Oracle Teacher Distillation（2026-09-12）

M1-B 保持 M1-A 的 feature、controller 架构、mask-MSE loss 和训练超参数不变，
新增 seeds 1101-1140 的 160 个 Oracle teacher，并与锁定的 M1-A 40 条合并为
200-unit training set。LongLive/T5 始终冻结。held-out 使用完全隔离的 seeds
2001-2004，四个 edits、五个条件，共 80 个 final pairs。冻结协议见
[MASK_DISTILLATION_M1B_PROTOCOL.md](MASK_DISTILLATION_M1B_PROTOCOL.md)。

冻结判定为 **NO PASS**：Prompt+State 在 16 个 held-out units 上有 8 个严格 Pareto
wins，未达到 12/16；Prompt-only ablation 为 7/16。按 edit 的 Prompt+State wins 为
dress 2/4、jacket 2/4、darker 3/4、warm-to-cool 1/4。Prompt+State aggregate median
`(delta_R,D)=(0.19039,0.19508)`，优于 global `.5` 的 `(0.15803,0.22657)`，但逐
unit 泛化不稳定。有限 16-step SPSA reference 的 median 为 `(0.29727,0.17581)`，
仍显示可利用的 Oracle 余量；它不是数学 upper bound。

80/80 final pairs 通过 full-P0 exact、RNG、outside exact、30-layer routing、DINO、
boundary、hash 和 clean-provenance 检查。结论按预注册路线执行：不再仅靠增加
teacher 数量；下一阶段应改 controller representation 或训练 loss。机器可读结果为
`validation/mask_distillation_m1b_analysis_0fded5f.json` 和四个 held-out shard。

## M1-A Oracle Mask Distillation（2026-09-12）

M1-A 已按冻结协议完成：40 个 `lambda=.2` Oracle teacher（4 edits × seeds
606/707/801–808），训练 Prompt-only（6,110 参数）与 Prompt+State（6,513 参数）
controller，再在完全隔离的 seeds 1001/1002 上跑 48 个 final pairs。LongLive/T5
始终冻结，训练只使用 mask MSE；held-out 判定只使用真实 replay 的 `delta_R` 和
`D_drift`。协议见 [MASK_DISTILLATION_M1A_PROTOCOL.md](MASK_DISTILLATION_M1A_PROTOCOL.md)。

结果为 **NO PASS**：Prompt-only 为 5/8 Pareto wins，Prompt+State 为 4/8，均未达到
冻结的 6/8 门槛。Prompt+State aggregate median `(delta_R,D)=(0.42090,0.22375)`，
优于 global `.5` 的 `(0.19218,0.30963)`，但不能覆盖逐 unit 一致性失败。有限
16-step SPSA reference 在个别 unit 被 predictor 超过，因此只解释为同预算 search
reference，不称数学 upper bound。48/48 final pairs 的 routing/RNG/outside/DINO/
boundary/hash/clean-provenance invariants 全部通过。机器可读结果已归档到
`validation/mask_distillation_analysis_209160b.json` 和四个 held-out shard。

## M0/M1 Oracle Layer-wise Release Mask（2026-09-12）

D3 已定位到 value/content path 更敏感，但单一路径 attenuation 的 preservation cost
过高。本轮冻结 LongLive，为每个 `(edit, seed)` 用精确离散 proxy 和确定性 SPSA 单独
优化 30 个 layer release coefficients；seeds 606/707、chunk 4、四个 directional
edits、`lambda={.05,.2,.8}`，最终比较 full/global `.5`/current-only/三个 oracle 点。
协议见 [ORACLE_LAYER_MASK_PROTOCOL.md](ORACLE_LAYER_MASK_PROTOCOL.md)。

4xA800 实验已在 clean commit `312580ad8c0d` 完成，48/48 final pairs 通过冻结
invariants。判定为 **PASS**：8 个 edit-seed 单元中 6 个存在严格 Pareto 优于 global
`.5` 的非均匀 mask；dress、jacket、darker 均为 2/2 seeds，warm-to-cool 为 0/2。
global `.5` 的 median `(delta_R,D)=(0.13171,0.24872)`，`lambda=.2` oracle 为
`(0.22920,0.21773)`。这证明 per-case layer mask 有方法可行性，但同时显示明显
edit dependence；尚未训练或验证 prompt-conditioned controller。机器可读 analysis
和四个 clean-provenance shard 已归档到 `validation/`。

## History Attention-Path Decomposition D3（2026-09-12）

D1/D2 已停止继续拆 temporal position。D3 冻结为 score/access 与 value/content
attention-path screen：seed 505、chunk 4、四个 directional edits、七个条件，共
28 pairs / 56 replays。score 路径通过 history logits 加 `log(alpha)` 改变 odds；
value 路径只缩放 history V，保持 Q/K/logits/normalization 不变。另有 CUDA
`score_0 == native current-only` 与 full native endpoint bit-exact invariant。
协议见 [HISTORY_ATTENTION_PATH_PROTOCOL.md](HISTORY_ATTENTION_PATH_PROTOCOL.md)。

D3 4xA800 screen 已在 commit `1a10791f6627` 完成：28/28 pairs、56/56 replay
通过冻结 invariants。`score_.5` 的 median `delta_R=0.00735`，而 `value_.5` 为
0.44725，说明机制敏感性主要位于 history value/content path；但 value `.5` 的
median drift=0.75783，明显高于 Stage C global `.5` 的 0.15646，因此尚未形成可用的
selective release。`value_0` 的响应度在两个 edit 上超越 current-only，应解释为
保留 normalization mass 后的 representation-scale effect。机器可读 analysis 与四个
clean-provenance shard JSON 已归档到 `validation/`。

## History Subset Interaction D2（2026-09-12）

D1 已封存为 coarse temporal localization insufficient。下一步冻结为 chunk 4 的
`sink/old/recent` 完整 2^3 binary native-attention subset screen；使用全新 exploratory
seed 404，四个 edit 加 Stage C global `.5` reference，共 36 pairs。分析固定报告
pairwise/third-order inclusion-exclusion、locking set function 和 Shapley allocation；
不设置事后等价阈值、不自动判定 A/B/C/D，也暂不进入 soft alpha 或 score/value。
协议与启动方式见 [HISTORY_SUBSET_PROTOCOL.md](HISTORY_SUBSET_PROTOCOL.md)。

D2 4xA800 screen 已在 commit `daaf075311b1` 完成：36/36 pairs、72/72 replay
通过冻结 invariants。结果显示多数双组件 subset 已恢复接近完整 SOR locking，同时
singleton 结构随 edit 明显变化；当前口径为 distributed/redundant locking with
edit-dependent subset structure，不作显著性或自动 A/B/C/D 判断。机器可读 analysis
与四个 clean-provenance shard JSON 已归档到 `validation/`。

## History Component Decomposition D1（2026-09-12）

History-Constraint Release Stage C 已按 held-out 120-pair 结果封存为 GPU PASS。
下一阶段固定为纯推理的 sink / older-local / recent component screen：新 seed
303、4 edits、chunk {1,4}、6 conditions，共 48 个 P0/P1 pair。协议、manifest、
分析口径和 4×A800 启动方式见 [HISTORY_COMPONENT_PROTOCOL.md](HISTORY_COMPONENT_PROTOCOL.md)。
该阶段只做 exploratory localization，不训练 adapter，不加入 spatial mask/right
context，也不产生 confirmatory p-value。

## 新方向：Edit-Ready Video Generation 可行性 MVP（2026-09-10）

按 [EDIT_READY_VIDEO_FEASIBILITY_PLAN.md](EDIT_READY_VIDEO_FEASIBILITY_PLAN.md) 完成 Generation-Time Edit Cache 的纯推理可行性测试，**不训练任何模型、不改 R0/R1 逻辑**。完整报告见 [EDIT_READY_MVP.md](EDIT_READY_MVP.md)，机器可读汇总见 `validation/edit_ready_mvp/summary.json`。

**本段全部数字来自 commit `4c4d52f` 在 `git dirty = false` 的清洁工作树上的封存重跑**（`summary.json["run_provenance"]` 聚合了每个 run 自己记录的 commit / dirty / `code_tree_sha256`）。第一版 `251b0ed+dirty` 的结果已废弃。

- **Q1 缓存是否够用：够，而且逐位精确**。镜像 chunk 循环与上游 `inference()` 逐位一致（`max_abs=0.0`）；8 prompt × target {0,1,4} = 24 个 case 从 `S_{k-1}` 用原 prompt 重开全部 `torch.equal`，RNG state 24/24 精确重现 chunk 内 3 次噪声抽样，repeat-noise baseline 为 0。cache 体积 1.0499e9 B ≈ 0.978 GiB/chunk；保存全部 7 个 chunk 边界共 6.88 GiB（这是 *sufficient* 而非已证明 minimal）。
- **Q2 新 prompt 能否只影响重开 chunk：能，而且这次找到了"为什么弱"**。48 个编辑 case（2 seed）未编辑 chunk 编码前全部 `torch.equal`；沿用旧文本 K/V 时结果与同 prompt replay 逐位相同（k>0 32/32）。关键是新增的 **chunk 0 无历史 control**：`cache@chunk0 + P1 + 重绑定` 与 full regeneration 的 chunk 0 **逐位相同（16/16，directional 10/10）**，证明重绑定实现本身完全正确。
- **编辑强度随历史衰减**：`R_k = S_text_rebind / S_full` 在 chunk 0/1/4 分别为 **1.000 / 0.053 / 0.016**（directional，2 seed）。**1 个 chunk 的历史就压掉约 95% 的 prompt 作用。**
- **reverse 控制证明是历史在说话**：P1 历史 + **P0** 文本仍恢复 full regeneration 响应的 **94.7%（k=1）/ 97.4%（k=4）**，与 full-regen chunk 的 latent MSE 只有 0.037 / 0.009。这是本轮最有价值的结论。
- **history recache（审阅建议的 training-free 干预）被检验且为否定**：保持 P0 latent 不变、用 P1 重建 cache **不能**解锁编辑（`S` 与纯文本重绑定持平）。同时发现重建出的 cache 与 checkpoint cache 不同，差异全部落在被保护的 sink 区（k=1: token [0,1295]；k=4: [0,5183]）——**生成期 KV cache 是路径相关的，不是 (latent, timestep, prompt) 的纯函数**，这是 compact cache 的工程约束。
- **Q3 边界**：左边界几乎无损（k=1 +8.9%、k=4 +1.5%）；右边界在 **chunk 0 处严重破坏（+1258%）**，k≥1 为 +37%/+19%，与预期一致。
- **Q4 成本必须分三档**：device 端计算 `R=0.167`；端到端在 cache 驻留时 `R=0.367`（通过）；**冷存储读 1 GiB checkpoint 时端到端 `R=0.770`（不通过 Gate D）**。瓶颈是 cache 体积与 I/O，不是重算。
- **Gate B 口径收紧**：自动聚合只统计 `directional` prompt 且 k>0（15/20 方向正确，strong 1/20）；雨/微笑/推镜三类只报"探针一致的变化"（15/18），不计入任何"语义编辑成功"。
- **判定 `GO_WEAK_PROMPT_REBINDING`**：状态复用已解决、重绑定实现已证明正确，瓶颈是历史压制。**下一阶段先做 training-free 的"历史衰减 / 局部重算"与右 context 条件，再考虑训练 adapter**；同时必须做 cache 压缩与 staging。

本轮工程验证：CPU 测试 **103 项通过**（新增 32 项 Edit Cache 回归测试）。修复了两个真实协议缺陷——镜像循环曾把干净预测而非带噪 latent 送进下一次 forward（与上游 `max_abs_diff=11.06`）；full regeneration baseline 曾继承 base 的 RNG 流（噪声混淆），并被 chunk 0 control 抓出来。


## 第三轮：prefix-aware 检索 gate 与 matched control（R0 冻结为基线）

本轮**不训练任何视频模型、不改 paired objective、不扩训练预算**；按审阅把 R0 冻结为基线，只补诊断、控制变量与取证，并跑通 R1 第一阶段的 gate。

- **已修复的协议 bug**：早期检索脚本把可见 prefix 帧数写成 `4 * prefix_latents = 24`，6 latent 时 query 用了 `[0, 12, 23]`，其中 frame 23 越过可见边界（Wan 映射下 $L$ latent 只暴露 $4(L-1)+1=21$ 帧、索引 0..20）。新增 `restream/reality_temporal.py` 作为唯一真值，Dataset / builder / 检索脚本统一使用，并对 query 帧硬断言 `max(frame) <= 4*(prefix_latents-1)`；回归测试覆盖 latent=3/6/9 → 9/21/33 帧与边界 8/20/32。旧报告标记为 `summary_superseded_pixel_count_bug.json`，**不再作为 R1 依据**。
- **Prefix 检索 gate（修复后重跑）**：冻结 DINO 编码可见 prefix（3 帧：**0/10/20**），在**完整 83 个 val target** 上比较 correct / easy-wrong / DINO-hard 跨 source 负例。offline 与 strict-online 两份 manifest 结果一致：accuracy vs easy = **1.000**，vs hard = **0.940**；margin +0.545（CI [0.504, 0.588]，83/83 为正）；hard margin +0.274（CI [0.238, 0.311]）；AUROC 0.944；Recall@2 0.922、top-1 correct 0.976；**1000 个确定性 derangement null test**：0.492 / 0.436（真实值高出 null 约 0.50，null CI 很窄）。**gate 通过**：可见 prefix 本身能识别同源现实参考，且 strict-online 的因果筛选池给出同样结论。详见 [R1_PREFIX_AWARE_PLAN.md](R1_PREFIX_AWARE_PLAN.md)、[prefix_retrieval/summary.json](validation/reality_memory/prefix_retrieval/summary.json)、[prefix_retrieval_online/summary.json](validation/reality_memory/prefix_retrieval_online/summary.json)。
- **strict-online retrieval-only 数据**：新增 `configs/reality_memory_paired_online.yaml` 与 `data/reality_val_online.jsonl`（83 行、schema 3、`causal_previous`、所有正参考 ≤ `visible_until`），offline manifest 未改动；builder/cache 新增 `--splits`（缓存 1,328 个参考，其中 472 新编码）与 `--report/--stats-output`，避免覆盖 offline 统计。
- **R1-A 路由模块（结构实现，未训练）**：新增 `restream/reality_router.py::FrozenStateRouter`（无可训练参数）：prefix/reference DINO 相似度 → soft 或 top-k 权重 → 加权后的 reference token **直接作为 `RealityMemory` 输入**，使 retrieval 与 generation 结构耦合；含 CPU 前向测试（不同 prefix → 不同 routed memory → 不同 fused context）。未接训练循环、未加 retrieval loss。
- **Role-matched 控制**：global mean 升级为 schema 2，包含 `global_async` / `global_aligned` / `global_positive` 三个去重训练池均值（11,680 = 5,840 async + 5,840 aligned）；probe 增加对应变体与 `G_content_matched`（与 `correct_kind` 同角色）。加载器逐角色校验 manifest/keys/selection 摘要。
- **strict-online 运行时断言**：`RealityDataset` 在解码时断言 prefix 末帧 `sampled_time == visible_until`（不再只信 manifest 自洽），并检查 prefix 所有帧不越界；strict 检索运行的 83 个 target 全部通过该断言。
- **目标选择**：probe 支持 `--cases 0`（全部 target）与 `--target-seed` 确定性抽样，并把选中的 sample_id 写入验证报告；`--noise-seeds` 可覆盖。
- **统计**：`summarize_existing_probe` 以唯一 target 计算 `G_content_matched` 与 target-level bootstrap CI；通用统计函数抽到 `restream/reality_stats.py`。
- **文案修正**：R0 probe 的“直接否证”降级为“单 checkpoint / 16 target 的反方向证据”，并明确其局限。

CPU 测试 **59 项全部通过**。同一 checkpoint 的 **full-val matched-control 视频 probe**（83 个唯一 val target × 2 seeds）也已完成：`U_correct` +0.00595（CI [0.0037, 0.0088]，77/83）、`G_branch` +0.00228（CI [0.0016, 0.0031]）、`G_generic` +0.00469（CI [0.0028, 0.0071]），但 role-matched 的 **`G_content_matched` = −0.00117（CI [−0.0017, −0.0007]，19/83）**、`S_reference` CI 跨 0。也就是说：**可见 prefix 已经能识别正确参考（检索 gate 通过），但 R0 的生成路径仍未利用它**——瓶颈在 query，而不是参考数据。逐 target 统计见 `validation/reality_memory/full_val_controls/`。

## 第二轮审阅修补（ee73b8e → 本提交）

本提交针对 `ee73b8eaa061e0bbdaee462fee9fbf2cd4a00310` 的审阅意见继续收紧 strict-online 语义与溯源，并按审阅批准的下一步执行了**无参数更新的反事实 probe**（只加载 memory state_dict，不训练、不改模型）。probe 结论与数字见 [R0_ACTIVE_ZERO_PROBE.md](R0_ACTIVE_ZERO_PROBE.md)。

- **统一取帧语义**：新增 `data.temporal_sampling`；`offline_target_filtered` 固定 `first_at_or_after`，`strict_online` 固定 `causal_previous`（last frame at-or-before）。manifest builder、`VideoDataset`、target histogram 与 reference 采样共用同一策略；Dataset 额外返回 `sampled_times`。strict 行的 `visible_until` 必须等于 prefix 实际最后一帧，且落在 `[target_start, target_start + arrival]`。
- **strict 行强制身份**：strict 行必须带 `selection_schema == 3` 与匹配的 `selection_config_hash`（缺失不再按 legacy 放过），并校验 async+aligned 正参考的 split/source/shot 与时间上界。无身份字段的 legacy 行只有在 `references.allow_legacy_offline_manifest: true` 时才可加载（现有 offline 配置显式开启；新实验默认关闭）。
- **选择身份补全**：`data.selection_seed` 与训练 `seed` 分离（换训练 seed 不再隐式改变数据身份，换数据 seed 必然改变 hash）；`shot_filter_hash`（histogram_cut/pixel_jump/black_level/black_fraction）与显式 `references.pool_size` 进入 `selection_config_hash`（schema 3）。
- **Global mean 训练集证明**：`unique_train_reference_pool` 主动校验每行与每个正参考的 split、source_id、shot_id 一致，再写入 `source_split = train`，不再仅依赖输入文件可信。
- **诊断修正**：梯度分解采样改为 update 1,2,5,10…（修复 off-by-one）；global-constant 资产一致性改为三态 true/false/null，旧报告不再被误报为一致；`summarize_reality_paired.py` 相对路径 checkpoint 修复。
- **新工具**：`scripts/summarize_existing_probe.py` 以**唯一 target**为单位计算 `U_correct/G_branch/G_generic/G_content/S_reference/H_wrong` 与 target-level bootstrap 95% CI。

**本轮 probe 结果（2026-09-10，A800，无参数更新）**：特征缓存按新脚本重建（13,008 个唯一参考，9,884 新编码、3,124 复用），schema-1 global mean 覆盖 11,680 个去重训练正参考（digest 记录在 `data/reality_overfit_feature_stats.json`）。对 30-update checkpoint 的 16 个独立 val target × 3 seeds × clean/mild 做反事实 probe：`U_correct` 在 16/16 个 target 上为正（约 +0.0036/+0.0040，CI 不含 0），`G_generic` 为正（CI 不含 0），但 **`G_content ≤ 0`（clean 显著为负，mild 不显著）**、`S_reference ≈ 0`、`H_wrong < 0`。这落在事先约定的**情况 B**：收益主要来自通用真实图像统计，正确照片内容没有可证明的额外价值，wrong-source 同样优于 No Memory。详见 [R0_ACTIVE_ZERO_PROBE.md](R0_ACTIVE_ZERO_PROBE.md)。该结论仅针对这一个 checkpoint 与离线协议；不构成 Reality Memory 有效性结论，也不改变“strict-online 尚未闭合、暂不扩训练预算”的判断。

## 上一提交审阅修补（d134557 → ee73b8e）：控制变量与数据溯源收紧

本提交针对 `d134557bee7708370c37220417b516ca01812f80` 的审阅意见做代码级修补，**没有运行任何 GPU/训练**，因此本段不新增任何实测数字，也绝不用旧结果冒充新对照。

- **Global constant 数据来源**：均值改为**去重后的训练 split `reference_sets.async + aligned` 并集**（按 FeatureCache key 去重、按排序 key 计算），不再使用经过 mixture 随机选择的 `row["references"]`。因此它不再受 `no_memory/wrong` 概率、逐样本随机 K、donor 重复出现次数影响。构造逻辑位于 `scripts/cache_reality_features.py::global_mean_payload`；缓存脚本会一次性补齐训练/验证行的 async+aligned 全池编码（比上一轮只缓存 mixture 切片更重），否则 loader 会因为均值读取的池帧未缓存而报错。
- **Global mean 溯源**：`.pt` payload 增加 schema、`unique_reference_count`、`train_manifest_sha256`、`reference_keys_sha256`、`selection_config_hash`、`selection_protocol`，并按 tmp→replace 原子保存。`load_global_constant` 现在校验 cache identity、shape、finite、train split，并与**当前 manifest/选择策略的摘要比对**；旧格式或过期文件会被明确拒绝并要求重新生成。probe 报告写入 `global_constant` 溯源记录（含文件 sha256），`summarize_reality_paired.py` 校验 before/after 使用同一份 constant 资产。
- **strict-online 最后一帧**：候选 reference 构造时，target histogram 的边界采样改为 at-or-before 语义读取，`visible_until` 记录**实际消费的 pixel frame 起始时间**（≤ 理论边界），near-aligned 参考上限使用该实际值。`RealityDataset` 对 strict-online 行校验全部正参考池（async+aligned）时间 ≤ `visible_until`、schema 与 selection hash 一致。新增带 recording-decoder 的完整 `candidate()` 因果测试与 offline 对照测试。
- **共享选择身份**：新增 `restream/reality_selection.py`，manifest builder、Dataset、global-mean loader 共用同一 `selection_config_hash`（schema 2，覆盖 protocol/async_direction/min_gap/near_radius/boundary_margin/prefix_latents/frames/fps/scene_similarity/pool_size/min_count），不再各自手写。
- **Active-Zero 对照**：paired probe 增加 `active_zero`（K 个有效 mask + 全零 DINO 特征），与控制阶梯（No Memory / Active Zero / Global Mean / Pair Mean / Correct / Wrong Source）一起报告。
- **汇总统计**：`aggregate_pairs`/`summarize_reality_paired.py` 现在先在同一 target+noise 上做 paired 差值，再以唯一 target 聚合四个核心差值与 `S_reference`（`U_correct`、`G_branch`、`G_generic`、`G_content`），`target_means` 与主图覆盖全部变体。
- **梯度诊断开关**：`reality_memory_paired.yaml` 增加 `diagnostic_gradients: true` 与 `diagnostic_interval: 5`；训练循环只在 step 1/2 及间隔步记录分解梯度，长程/多卡运行可关闭，避免每步多余 `autograd.grad`。
- **旧 checkpoint 探测**：新增 `scripts/probe_existing_memory_checkpoint.py`，只加载 memory state_dict（不恢复 optimizer/scheduler/RNG/训练偏移），严格校验 stage、encoder identity、train/val manifest digest 与“仅 evaluation 字段差异”，随后跑全部反事实 probe。直接对新旧 checkpoint 跑新诊断前，需先用 `cache_reality_features.py` 重新生成 global-mean 文件（旧格式会被 loader 拒绝）。
- **数据与协议诚实性**：现有 offline manifest 未重建、未改名；未构造 strict-online 实验数据（真正测试 strict 时应新建 `reality_memory_paired_online.yaml` 与独立 `reality_*_online.jsonl`，不覆盖现有文件）；旧 30-update 配对 GPU 结果保持原样。本机 **38 项 CPU 单元／接口测试通过**。下一步值钱的是把 Active Zero + Global Mean + Correct 三层对照在 GPU 上跑干净，而不是继续改模型或扩训练预算。

## 新主路径：Reality Memory R0（等待审阅）

[新方案](REALITY_MEMORY_PLAN.md)与 [R0 实现说明](REALITY_MEMORY_R0.md)已加入仓库。新主路径将参考照片编码为外部记忆，再通过 gated residual 融合到文本 context；不替换生成 latent。新增独立训练／评估入口，复用原数据和 LongLive 组件。R1 按方案留待 R0 有正向实验信号后实现。

已准备 frozen DINOv2-S、730 train / 83 val 的镜头过滤与弱参考 manifest，以及 3,124 份缓存特征（含本轮 326 个固定对照参考）。短实验结果见 [R0_SHORT_EXPERIMENTS.md](R0_SHORT_EXPERIMENTS.md)：两步真实 AdamW 检查通过；单卡执行 10 个 batch step、其中 6 次有效 optimizer update；4×A800 NCCL 执行 3 步后重启恢复并完成第 4 步。固定噪声和单样本 AR 对照没有显示 correct-reference 优势，因此没有扩展到 50/200-step 效果实验。

本轮工程验证：23 项 CPU 单元／接口测试通过；冻结 DINO 编码与 projector 梯度、缓存精确重载通过；两进程 CPU/Gloo 混合空记忆和全空记忆的 DDP 梯度同步通过；真实 LongLive 的 R0 future backward、两步梯度传播和四卡 NCCL 保存／恢复通过。零初始化输出层首步先更新，第二步 projector/query/key/value/gate 均得到非零梯度；4 个 rank 的恢复后梯度范数一致，优化器步数为 **3 → 4**。这些工程证据仍不能替代 Reality Memory 效果结论。

第二轮 paired 诊断见 [R0_PAIRED_EXPERIMENTS.md](R0_PAIRED_EXPERIMENTS.md)：单卡 30 次有效更新完成。训练集 relevance score 为 32/32 正确高于 wrong-source，但验证集仅 2/8；训练／验证 video loss 没有形成 Correct 专属优势，故不扩展到 50/200 updates。

本提交还加入了 Constant/Mean Memory 反事实控制、唯一 target 统计字段、video／contrast 分离梯度与实际 `alpha * raw_delta` 干预量；manifest 选择协议区分 offline target-filtered 与 strict-online，未重建旧 manifest。

其中 pair mean 已明确降级为混合参考消融；真正的 global constant 必须由准备脚本从训练 split 固定生成并经 cache identity 校验。当前提交只修正工具和协议，没有把上一轮 GPU 结果冒充为新的 global-constant 实测。

下文为上轮 State Re-Anchoring 诊断结果。

上轮五项修改和真实 GPU 验证已完成。真实 teacher-forcing backward 可以运行；Oracle 的小样本改善有限，Hard Anchor 的平均误差反而增加，尚不满足 `Oracle < Hard < No Anchor` 的效果门槛。该段历史记录的 optimizer steps 为 **0**，没有启动 50/200/3000-step 训练。

## 已修改

- `longlive.patch` 覆盖 normal/infinity 两处 `_forward_train` 解锁、两种模型的零 padding 截断修复，以及 wrapper 不再向无缓存训练入口传递 `sink_recache_after_switch`。从 SHA-256 校验过的原始归档应用补丁，三份文件与工作源码逐字节一致。
- FPS 从 YAML 接入 manifest 构造、Dataset、训练、评估与视频输出。57 帧、16 FPS 对应 3.5 秒采样窗口；修改 FPS 后 manifest 不匹配会明确报错。
- `protected_sink_latents` 从 YAML 传入 corruption，拒绝负数和非整数；默认保护前三个 latent，与当前 sink size 一致。
- 增加真实 `check_real_backward.py`，损失不含 delta 正则，断言 Adapter 梯度存在、有限、总范数非零，冻结主干梯度为空。
- eval 增加 `oracle_gt_anchor`：仅将末尾 anchor latent 替换为 GT，同样重建全新缓存、重放历史、使用相同初始及后续噪声。输出完整未来与首个 future AR block 的 latent MSE。
- 增加 `check_anchor_latent_alignment.py`，比较 full/prefix/single，输出全局与逐通道 mean/std、MSE、cosine；实际块尾索引为 2/5/8/11，另有帧零对照。
- mask key 包含完整 `(T,C,H,W)`、设备、teacher-forcing 标记、每块帧数与空间 token 数。
- 保留 801 参数 Tiny Adapter，末层卷积零初始化，step 0 严格保持原输入。

## 真实 GPU backward

使用现有一条真实训练样本，57×256×432 RGB，15 个 latent，BF16，单卡 NVIDIA A800-SXM4-80GB。加载 Wan、LongLive 基座和官方配套 LoRA 后全部冻结。报告：[real_backward.json](validation/real_backward.json)。

| 检查 | 实测 |
|---|---:|
| Future flow loss（delta 正则为 0） | 0.18859188 |
| Adapter 梯度总范数 | 0.00175416 |
| Anchor latent / 实际时间 | 5 / 1.25 秒 |
| 主干梯度 | 全部为 None |
| 初始 Adapter 输出 | 与预测 latent 完全相同 |
| 峰值显存 | 42.31 GiB |
| Optimizer 更新 | 0 |

零初始化使首步 gate 和第一层梯度为 0，最后一层 weight/bias 梯度非零；这是预期行为。真实 FlexAttention 的 forward/backward 均通过，无需 activation checkpointing。此检查不等同于四卡 DDP、optimizer 或 checkpoint/resume 验证。

## Oracle / Hard / No Anchor（4 个验证样本）

报告包含全部逐样本结果、seed、实际 anchor 时间和配置：[pretrain_anchor_ablation.json](validation/pretrain_anchor_ablation.json)。误差越小越好。

| 变体 | 完整未来 latent MSE | 首个 future block MSE | 完整未来优于 No Anchor |
|---|---:|---:|---:|
| No Anchor | 0.65433649 | 0.40071913 | — |
| Real RGB Hard Anchor | 0.66431449 | 0.46481637 | 2/4 |
| Oracle GT Latent | 0.64248845 | 0.37374974 | 3/4 |

Oracle 完整未来误差降低约 1.81%，首块降低约 6.73%；Hard 的误差分别增加约 1.52% 和 16.00%。这是 4 个固定样本、人工漂移历史的诊断，不能认定效果显著或机制已被充分验证。Oracle 仅恢复一个锚点，其他受损历史仍保留；不能把它理解为整个历史都恢复为 GT，也不能在部署时取得该 latent。

本机已生成 4 组对比视频及各变体视频，共 20 个 MP4，全部完整解码验证为 57 帧、16 FPS。路径：`outputs/pretrain_anchor_ablation/case_000` 至 `case_003`。GitHub 收录 JSON 指标与视频哈希，视频保存在工作机器。未训练 learned adapter；LPIPS 未启用，值为 null。

## VAE latent 对齐（8 个验证视频）

32 个中途块尾锚点的平均结果如下；帧零对照不计入均值。报告含逐样本和逐通道统计：[anchor_latent_alignment.json](validation/anchor_latent_alignment.json)。

| 对比 | MSE | Cosine | Reference std | Candidate std |
|---|---:|---:|---:|---:|
| Single vs Full | 0.29088490 | 0.81317290 | 0.84683401 | 0.49335776 |
| Prefix vs Full | 0 | ≈1 | 0.84683401 | 0.84683401 |
| Single vs Prefix | 0.29088490 | 0.81317290 | 0.84683401 | 0.49335776 |

这些样本中，因果前缀编码与完整视频对应 latent 完全相同；独立单帧编码与时序 latent 有明显差距。这与 Hard Anchor 未稳定改善的结果相容，但尚不足以把误差变化全部归因于表示差异。

## 工程核验与资产

- 9 项 CPU 单元测试通过，覆盖配置改变后的采样时序和 sink 保护、首个 future block 损失、宽度变化的 mask 失效、零初始化与梯度、分组划分和解码。报告：[unit_tests.txt](validation/unit_tests.txt)。其中小型视频与合成 ID 只用于临时测试目录，未覆盖真实数据或统计。
- 原始归档 SHA-256、补丁重建和三份上游源码哈希核验通过：[longlive_patch.json](validation/longlive_patch.json)。
- 模型文件、LongLive 官方 SHA-256、真实视频路径、最低数据量、train/val source 隔离均通过：[readiness.json](validation/readiness.json)。
- 本轮配置、源码、manifest、报告与视频哈希：[run_manifest.json](validation/run_manifest.json)。
- 已下载 1068 个 Youku source，约 2.2606 GB，961 train / 107 val；本轮无需新增下载。
- 仓库工作目录为 `/workspace/video-model/restream_mvp`；本机 `.venv` 与模型链接到 `/workspace/restream_mvp` 的现有资产，视频 manifest 使用该资产目录的绝对路径。更换机器须重新构造 manifest。
- Python 3.11，torch 2.5.1+cu124，torchvision 0.20.1+cu124，flash-attn 2.8.3.post1，驱动 535.129.03。CPU 沙箱日志可能包含 CUDA 不可访问提示；真实 GPU 检查在可访问驱动的环境执行。

## 后续研究限制

R0 的 heuristic shot-cut 与场景相似度过滤已完成；人工视觉审核、每个长视频的多窗口采样尚未完成。旧 ReAnchor manifest 未按 R0 规则重新清洗，保留 `visual_review: pending`。此处的旧实验仍是人工漂移 GT 历史、单个中途 anchor 的短窗口诊断，没有长期 self-rollout 或多 anchor 实测。
