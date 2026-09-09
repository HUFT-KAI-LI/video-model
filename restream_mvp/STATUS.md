# ReStream 本轮审阅状态（2026-09-09）

## 新主路径：Reality Memory R0（等待审阅）

[新方案](REALITY_MEMORY_PLAN.md)与 [R0 实现说明](REALITY_MEMORY_R0.md)已加入仓库。新主路径将参考照片编码为外部记忆，再通过 gated residual 融合到文本 context；不替换生成 latent。新增独立训练／评估入口，复用原数据和 LongLive 组件。R1 按方案留待 R0 有正向实验信号后实现。

已准备 frozen DINOv2-S、730 train / 83 val 的镜头过滤与弱参考 manifest，以及 3,124 份缓存特征（含本轮 326 个固定对照参考）。短实验结果见 [R0_SHORT_EXPERIMENTS.md](R0_SHORT_EXPERIMENTS.md)：两步真实 AdamW 检查通过；单卡执行 10 个 batch step、其中 6 次有效 optimizer update；4×A800 NCCL 执行 3 步后重启恢复并完成第 4 步。固定噪声和单样本 AR 对照没有显示 correct-reference 优势，因此没有扩展到 50/200-step 效果实验。

本轮工程验证：23 项 CPU 单元／接口测试通过；冻结 DINO 编码与 projector 梯度、缓存精确重载通过；两进程 CPU/Gloo 混合空记忆和全空记忆的 DDP 梯度同步通过；真实 LongLive 的 R0 future backward、两步梯度传播和四卡 NCCL 保存／恢复通过。零初始化输出层首步先更新，第二步 projector/query/key/value/gate 均得到非零梯度；4 个 rank 的恢复后梯度范数一致，优化器步数为 **3 → 4**。这些工程证据仍不能替代 Reality Memory 效果结论。

第二轮 paired 诊断见 [R0_PAIRED_EXPERIMENTS.md](R0_PAIRED_EXPERIMENTS.md)：单卡 30 次有效更新完成。训练集 relevance score 为 32/32 正确高于 wrong-source，但验证集仅 2/8；训练／验证 video loss 没有形成 Correct 专属优势，故不扩展到 50/200 updates。

本提交还加入了 Constant/Mean Memory 反事实控制、唯一 target 统计字段、video／contrast 分离梯度与实际 `alpha * raw_delta` 干预量；manifest 选择协议区分 offline target-filtered 与 strict-online，未重建旧 manifest。

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
