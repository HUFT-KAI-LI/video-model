# R0 第二轮：同目标配对与轻度退化 prefix

本轮针对 `add54daa797380315db289e6d04abb3b7929de6c` 的审阅意见，修改小实验目标与训练预算。原单边 wrong-gate 配置和第一轮报告保留；新实验使用独立的 [reality_memory_paired.yaml](configs/reality_memory_paired.yaml)。不实现 R1，不运行 200-step。

本提交新增两种反事实：`pair_mean_memory` 保留 K 个有效 reference 和同一可训练分支，但将当前 pair pool 的 DINO token 均值复制到每个 reference；`global_constant_memory` 读取准备阶段由**训练 split 去重后的 async+aligned 参考池**计算并保存的固定张量 `data/reality_global_mean_features.pt`。前者是混合参考消融，后者才是去除当前 target 特有内容的固定控制。文件保存 cache identity、schema、去重参考数、manifest/参考 key/选择策略摘要，并在加载时校验（见下文“溯源修补”）；同预算 constant-memory 训练基线仍是后续实验。

## 同一个 target 的配对目标

每次更新只解码、编码一个 target，共享 caption、GT future、prefix corruption、diffusion timestep 和 noise。该 target 同时读取 K=2 的同 source 过去参考与 K=2 的 wrong-source 参考。pair 两边均预测同一段 GT future，训练不奖励 wrong-source 的视频变差。

复用 R0 现有 attention，不增加参数。将 gate 输入中已有的余弦相似度明确输出为 relevance score：

$$
s_{rel}=\cos(\bar q,\bar r),\qquad
\alpha=G(\bar q,\bar r,s_{rel}).
$$

对 score 施加二分类对比目标，temperature=0.1：

$$
L_{contrast}=\operatorname{softplus}((s_w-s_c)/0.1),
$$

$$
L=\tfrac12(L_{video,c}+L_{video,w})+0.01L_{contrast}
  +10^{-5}L_{\Delta}.
$$

新配置关闭 `wrong_gate_weight` 和 reference dropout，使两边始终具有相同参考数量；仍允许 `contrast_weight=0` 做无 ranking 消融。不直接要求 correct gate 打开，也没有把 source ID、正确性标签、timestamp 或 GT future 输入 memory 模型。ranking 梯度会训练共享的 query/projector/key/value，可能间接影响 gate；它不是完全独立的两套网络。

R0 仍只按 prompt 查询，score 本身不能证明同世界识别，更不能识别当前视频 drift。correct 仍是 same-source past-only proxy，wrong-source 也不等于已认证的 wrong-world。被直接监督的 score 排序改善，需要与视频损失和验证目标分开判断。

## 轻度退化与对照

prefix 仍为 6 个 latent，保护前三个 GT latent；只向后三个添加标准差 0.05 的高斯噪声，固定强度、不做空间平移。GT 标签、prefix 后的未来和 reference 特征均不修改。照片始终只通过 soft context 影响生成。

训练前后，对 16 个固定训练 target 与 4 个独立 val target 分别执行 clean/mild 两种 prefix 对照，每个 target 使用两个固定 noise seed。每组比较 Base、No Memory、Correct、Wrong Source，共用同一个 history 与同一份未来噪声。报告保留每个 target 的 reference 元数据、history seed、noise seed、后缀实际 MSE、video loss、gate、score、有效 token 数和归一化 entropy。

No Memory 必须在两种 prefix 下严格等于对应 Base；前三个 latent 必须不变。两次 noise seed 是同一 target 的重复测量，不能算作两个独立样本。此轮是 first-future-block teacher-forcing 诊断，没有生成 AR 视频或长期 self-rollout。

## 有效更新预算与显存

`--max-steps` 保留历史 batch 语义；新增的 `--max-updates` 表示累计的有效 AdamW 更新数，两者必须且只能指定一个。恢复到 20 updates 后指定 `--max-updates 30`，只再更新 10 次。日志和 checkpoint 同时写 `batch_step` / `optimizer_step`，保留旧 `step` / `optimizer_steps` 字段供历史工具读取。update 模式下，warmup、定期保存与结束条件都按实际更新数推进；原混合训练的定期 AR 评估也按有效更新数触发。配对诊断只做训练首尾的固定探针。连续至少两轮且不少于 16 个 batch 没有更新则报错，避免全空记忆数据无限等待。日志的 `learning_rate` 是本次 scheduler 推进后的值，即下一次更新的学习率。

配对实验把所选 16 个目标全部变成 correct/wrong pair，因此旧 manifest 中的 No Memory 目标也会拥有两组参考。空记忆不变性通过训练前后的逐目标对照检查；旧混合训练中的空记忆 batch 仍经过正常 loss/gradient 检查并跳过 AdamW／weight decay，不占 update 预算。

两条 LongLive 反传图同时驻留可能超过单张 80 GB GPU。实现逐条计算 video loss 对 context 的梯度、释放冻结视频图，然后将平均梯度传回一次配对 memory 图，并加上 score 对比损失与 delta 正则。该一阶链式求导在零初始化和非零 output 两种情况下，与同时持有两条视频图的完整求导逐参数对照通过。没有使用近似 straight-through gradient。

## 复现

本机现有模型和视频已覆盖本轮数据。特征缓存需用新脚本重跑：它会补齐全部训练行的 async/aligned 池（去重）并重写 schema-1 的 global-mean 文件（一次性的完整池编码，比上一轮只缓存 mixture 切片更大）。迁移机器先按原准备流程建立资产，再运行下面的缓存命令：

```bash
cd /workspace/video-model/restream_mvp
# Now caches the full async/aligned pools (deduplicated) and writes the schema-1
# global mean; the loader refuses the older schema-less file, so this must run
# before any probe or training that reads the constant control.
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/cache_reality_features.py --overfit-samples 16
OMP_NUM_THREADS=2 .venv/bin/python -m unittest discover -s tests -v
CUDA_VISIBLE_DEVICES=0 RESTREAM_WORLD_SIZE=1 \
  bash scripts/train_reality_memory.sh \
  --config configs/reality_memory_paired.yaml --reviewed --max-updates 30 \
  --overfit-samples 16 --probe-overfit --output checkpoints/reality_memory_paired_review
MPLCONFIGDIR=/tmp/reality-paired-matplotlib \
  .venv/bin/python scripts/summarize_reality_paired.py
```

复现时先运行 `scripts/cache_reality_features.py` 生成并校验新格式全局训练均值，再使用新的 output 目录或显式 `--resume <checkpoint>`。已有 30-update checkpoint 不能按新配置 resume（签名变化是预期行为），改用 `scripts/probe_existing_memory_checkpoint.py` 只加载 memory state_dict 跑反事实 probe。改变 contrast、degradation、参考策略或数据划分时需新建实验；恢复签名拒绝语义不同的配置。新配对配置默认单卡，这次不重新声称它经过四卡 paired NCCL 验证；第一轮原目标的四卡保存／恢复证据仍在旧报告。旧 paired GPU 报告没有 active-zero／新溯源格式的 global-mean 变体，不能事后补写成新对照结果。

## 本轮实测

29 项 CPU 单元／接口测试通过。单卡 A800 完成 **30 次有效 AdamW 更新、30 个 batch step**，峰值显存 41.96 GiB，训练 batch 总耗时 92.19 秒，中位数 2.915 秒。所有 paired batch 都有有限梯度；首步 output/projector/query/key/value 非零，后续 gate 也获得视频损失梯度。`--max-updates` 的旧 checkpoint 恢复 smoke 另外确认了 `6 → 7` 的累计更新语义、AdamW state step=7 和 scheduler last_epoch=7。

固定对照报告：[summary.json](validation/reality_memory/paired_review/summary.json)，训练前后逐目标 JSON、[paired_controls.png](validation/reality_memory/paired_review/paired_controls.png)、[逐步日志](validation/reality_memory/paired_review/train_steps.jsonl)、[恢复预算报告](validation/reality_memory/paired_review/resume_update_budget.json)。

| split / prefix | Correct loss vs No Memory | Wrong Source loss vs No Memory | score correct > wrong | loss order `correct < none < wrong` |
|---|---:|---:|---:|---:|
| train / clean | −2.77% | −2.97% | 32/32 | 1/32 repeated-measure pairs |
| train / mild | −3.13% | −3.03% | 32/32 | 0/32 |
| val / clean | −0.86% | −0.85% | 2/8 | 1/8 |
| val / mild | −0.89% | −0.69% | 2/8 | 0/8 |

训练集的 relevance ranking 完全拟合，但验证集排序只为 2/8；训练集 Correct 和 Wrong 同时改善，说明视频目标没有把 score 的排序转化为正确参考专属收益。验证集两类参考也几乎同幅改善，不能宣称 `Correct < No Memory < Wrong`。轻度退化相对 clean 的 No Memory 变化为 train +0.12%、val −0.02%，验证集没有稳定的“救火”需求。

本轮结论是：配对 objective 和按有效更新计数的工程实现可运行，并能学习训练目标上的 relevance separation；它还没有泛化的 video utility 证据。暂不跑 50/200 updates、不扩四卡 paired 训练、不实现 R1。下一步若继续，应先换独立 source／view 的正确 memory 和更强但有明确语义的 degraded-history，再检查 score 排序是否在验证集和 video loss 上同时成立。

评价拆成三个描述性量：`U_correct = L_none - L_correct`（正确参考收益）、`S_reference = L_wrong - L_correct`（匹配额外收益）和 `H_wrong = L_wrong - L_none`（wrong-source 伤害）。`correct < none < wrong` 只保留为逐目标统计，不能作为继续研究的必要条件；wrong-source 可能仍提供类别级先验。

参考选择协议显式标注为 `offline_target_filtered`：当前 manifest 的过去图片筛选使用完整 target 的 histogram，适合离线整理，不等价于在线无前视。新增 `strict_online` 模式时只使用 target 起点到 prefix 可见时间的 histogram，并限制 near-aligned 参考不超过可见时间；单元测试保证它不会读取 target future。当前已有 manifest 未重建，因此报告继续按离线整理协议解释。

每条 paired 记录现在同时保存 video-gradient norm、contrast-gradient norm、`raw_delta_norm` 和实际 `applied_delta_norm = alpha * raw_delta_norm`。这些数值用于区分 score 学习和真正改变 LongLive context 的幅度；不能只依据 gate 或总梯度判断照片被使用。

## 溯源修补（d134557 之后的审阅轮次，未跑 GPU）

针对固定提交 `d134557` 的审阅，本提交只改代码、工具与协议约束，**不新增任何 GPU 数字**；原 30-update 报告仍按 d134557 代码与数据解释。

- **Global mean 去重与绑定**：均值改由训练 manifest 全部行的 `reference_sets.async + aligned` 按 cache key 去重（同一帧在多行/多池出现只计一次），不再从 mixture 选中的 `row["references"]` 计算。`.pt` payload 为 schema 1，含 `unique_reference_count`、`train_manifest_sha256`、`reference_keys_sha256`、`selection_config_hash`、`selection_protocol`、shape 元数据，tmp→replace 原子写。加载时逐项校验并与当前 manifest/选择策略摘要比对，旧格式或过期文件报错并要求 `scripts/cache_reality_features.py` 重新生成；**因此旧 `data/reality_global_mean_features.pt` 必须先重新生成再跑任何 probe**。
- **strict-online 因果闭合**：target histogram 边界帧以 at-or-before 语义读取，`visible_until` 取实际消费的 pixel frame 时间；near-aligned 参考采样上限与该实际值一致；`RealityDataset` 加载 strict 行时检查 `reference_sets.async+aligned` 全部正参考时间 ≤ `visible_until` 及 selection hash。`tests/test_reality_paired.py` 用 recording-decoder 直接驱动 `candidate()`，断言 histogram/正参考均不越界、协议与 hash 正确，并保留 offline 对照。
- **共享 selection identity**：`restream/reality_selection.py` 提供 schema 2 的 `selection_identity/selection_config_hash`，builder、Dataset、global-mean loader 使用同一实现，覆盖 protocol、async_direction、min_gap/near_radius/boundary_margin、prefix_latents、frames/fps、scene_similarity、pool_size、min_count。
- **Active Zero 与四层差值**：paired probe 增加 `active_zero`（K 个有效 reference + 全零 DINO 特征），控制阶梯为 No Memory / Active Zero / Global Mean / Pair Mean / Correct / Wrong Source。汇总先在同一 target+noise 做 paired 差值，再按唯一 target 聚合：

$$
U_{\text{correct}}=L_{\text{none}}-L_{\text{correct}},\quad
G_{\text{branch}}=L_{\text{none}}-L_{\text{active-zero}},
$$
$$
G_{\text{generic}}=L_{\text{active-zero}}-L_{\text{global}},\quad
G_{\text{content}}=L_{\text{global}}-L_{\text{correct}},\quad
S_{\text{reference}}=L_{\text{wrong}}-L_{\text{correct}}.
$$

  两个 noise seed 视为同一 target 的重复测量，不作为独立样本。before/after 报告都记录 global-constant 资产溯源，汇总器断言二者一致。
- **梯度诊断可关闭**：`diagnostic_gradients`/`diagnostic_interval` 控制分解梯度记录（step 1/2 与间隔步）；多卡/长程训练应关闭，不改变优化器数值。
- **旧 checkpoint 探测**：`scripts/probe_existing_memory_checkpoint.py` 只恢复 memory state_dict，校验 stage、encoder、manifest 摘要与语义（仅允许 evaluation 字段差异），对已有 30-update checkpoint 直接跑全部反事实 probe；恢复签名仍拒绝把旧 checkpoint 当新语义配置的正式 resume。
- 未来若真正测试 strict-online，应新建 `reality_memory_paired_online.yaml` 与独立的 `reality_*_online.jsonl`，不覆盖现有 offline manifest；当前报告继续按 `offline_target_filtered` 解释。
