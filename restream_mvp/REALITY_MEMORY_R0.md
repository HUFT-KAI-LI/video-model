# Reality Memory R0：供审阅的实现

本轮审阅后的修补、两步真实优化器与 10-step 结果见 [R0_SHORT_EXPERIMENTS.md](R0_SHORT_EXPERIMENTS.md)。以下准备记录中的“未训练”描述对应原始 `b75a6f7` 交付时点。

[新方案原文](REALITY_MEMORY_PLAN.md)定义新的研究路线。本轮交付其第一批 R0 代码、过滤后的参考数据和冻结视觉特征，保留全部 State Re-Anchoring baseline 及历史报告。用户要求先审阅，因此本轮不执行 10-step overfit 或 50/200/500-step 训练；无优化器检查不作为实验成功证据。

## 本轮范围

R0 是 Memory-as-Context baseline。R1 的 state-conditioned retrieval、指定 Transformer 层 guidance、长时 rollout 和 RL 均不在本轮；配置会拒绝把 R1 静默当作 R0 运行。R0 是否有正向效果仍须由后续受控实验回答。

复用原有 Wan/LongLive/官方 LoRA 加载、目标视频解码、首个 future AR block 的 flow loss、缓存初始化与 AR replay。原来的 `train_reanchor.py`、`eval_reanchor.py`、`restream/anchor_*.py`、`ORIGINAL_PLAN.md`、历史 `validation` 报告和 `longlive.patch` 均保留。

## 具体接入方式

```text
K 张 RGB reference（timestamp 仅供解码与分析）
  → 冻结 DINOv2-S / patch features
  → 每张图 2×4 空间池化：8×384，离线缓存
  → trainable projector：384 → 256
  → prompt-conditioned memory attention + relevance gate
  → 零初始化 output projection：256 → 4096
  → residual fuse 到原始 UMT5 context
  → LongLive 原有 text projection / cross-attention
```

这是方案允许的 context **fusion** 路径。保留原有 512 个 context token，不追加 token：即便追加 token 的 value 是零，它们也会改变 attention softmax 的分母，不能保证初始基线一致。当前方案在固定长度 context 上增加零初始化残差，无须修改上游源码；UMT5 padding 位置也不修改。

R0 的 query 来自文本条件，所有 AR 块使用相同记忆；它不是 R1 的视频 hidden-state retrieval。gate 是 prompt/reference relevance 和 guidance-strength gate，不宣称是 drift estimator。

无 reference 或所有 reference 被 dropout 时，融合结果严格等于原始 context，gate 与残差为零；这一约束在训练后仍然成立。空记忆 rank 仍保留参数计算图，支持 DDP 混合空／非空记忆，不通过 `find_unused_parameters=True` 掩盖梯度断路。全局空记忆 batch 的零梯度跳过 AdamW 更新，避免 weight decay 改变参数。

可训练参数共 **2,472,321**：visual projector、context query、memory K/V、residual output、gate。Wan VAE、UMT5、LongLive、官方 LoRA、DINOv2 全部冻结，未加入新的 backbone LoRA。

## 数据准备

- 沿用现有 1068 个 source 的 train/val 分组与原始视频，不重新下载 Youku。
- `filter_continuous_shots.py` 解码每一帧，使用 RGB histogram jump、像素突变和黑帧检测拆分镜头；目标窗口和正确参考必须位于同一连续镜头，并留出边界间隔。
- `build_reality_manifest.py` 在合格镜头内重新选择 3.5 秒 target，检查 target 与 reference 的 histogram similarity。参考保存实际解码帧时间，同一参考集合不重复同一帧。
- 异步参考默认 `past_only`，来自目标窗口之前至少 1.5 秒，减少 future/action leakage。`both` 可作为后续数据消融；不是当前默认。
- Near-aligned 参考在固定 prefix 尾时刻附近 ±0.5 秒内采样，仍然只用于 soft context。
- No-reference 占约 15%；wrong-source 占约 15%，仅在本 split 的其他保留视频里选择。训练标签只用于 wrong-gate 正则，不输入模型。
- 训练参考数量 K=1–4，逐 reference dropout=0.2。No-reference 数据行已实现全记忆 dropout，不再额外叠加一次 15%。验证池为每类 8 个参考，支持 K=0/1/2/4/8。

过滤后：**730 train / 83 val**。训练分布为 async 365、aligned 146、none 109、wrong 110；验证分布为 async 42、aligned 17、none 12、wrong 12。全部源视频检查无解码错误，过滤、时间间隔和场景相似度要求淘汰了其余样本。完整统计见 [reality_stats.json](data/reality_stats.json)。

这些筛选是同视频、同连续镜头的 asynchronous proxy，不能证明跨时间／跨视角的 same-world identity。原始 source caption 也可能描述整个视频；保留 `visual_review: pending`，仍需人工抽查。

## 编码器与缓存

DINOv2-S 从 [ModelScope facebook/dinov2-small](https://modelscope.cn/models/facebook/dinov2-small) 固定版本 `3368852df502d7b60bf506eb3abde87533f55164` 下载。88.25 MB safetensors 文件的 SHA-256 与 [官方公布值](https://huggingface.co/facebook/dinov2-small/blob/main/model.safetensors) 一致：`ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1`。下载脚本同时固定 config/preprocessor 文件哈希。

`models/dinov2-small` 为本机资产；`data/reality_features` 缓存每张参考图的 8×384 float32 特征。本机已准备 **2,798** 份唯一参考特征，覆盖训练所用参考及全部验证对照池。模型权重和特征文件不进 Git，manifest、配置和核验报告进 Git。

缓存键包含视频 SHA-256、实际采样时间、编码器权重／配置哈希、预处理版本、图像大小和池化方式；重载校验键、形状和有限值。训练从缓存读取原始视觉特征，再执行可训练 projector，不缓存 projector 输出。

## 训练目标与恢复

目标视频采用原有 57 帧／16 FPS／256×432 处理，6 个干净 GT latent 作为 prefix；复用 `future_loss`，只监督后续第一个 3-latent AR block。没有当前 latent 替换、单帧 VAE reference 或 reference reconstruction loss。

总损失为 future flow loss + `1e-5 * ||raw context delta||²` + `0.01 * wrong gate²`。错误 gate 损失仅用于已知 wrong-source 且 dropout 后仍有记忆的样本，不要求 correct gate 接近 1。gate 和 projector 在零残差初始化时，来自 video loss 的首步梯度可能为零；output projection 非零梯度先启动分支，wrong loss 也可直接训练 gate。

新 checkpoint 只保存 memory、optimizer、scheduler、step、实际更新数、epoch、数据 offset、各 rank RNG 和恢复签名。恢复签名包括数据／模型／参考策略、编码器身份、train/val manifest 哈希、overfit 子集和 world size；允许继续增加 `--max-steps`，不允许静默更改训练语义。

## 准备与审核命令

```bash
cd /workspace/video-model/restream_mvp
bash scripts/prepare_reality_memory.sh
```

该命令只下载编码器、过滤与构造数据、检查冻结编码器、缓存特征和执行 CPU 回归检查。不会调用 LongLive 训练或评估。首次运行依赖现有 Wan/LongLive 权重、环境和 Youku 原始数据；迁移机器时先按 [README_REPRODUCE.md](README_REPRODUCE.md) 准备原有资产与绝对视频路径。

单独的无优化器工程验证：

```bash
.venv/bin/python -m unittest discover -s tests -v
GLOO_SOCKET_IFNAME=lo .venv/bin/python scripts/check_reality_ddp.py
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 \
  .venv/bin/python scripts/check_reality_backward.py
.venv/bin/python scripts/check_reality_ready.py
```

检查报告位于 [validation/reality_memory](validation/reality_memory)。其中 DDP 检查只验证两进程 CPU/Gloo 的模块梯度同步，不能替代四卡 NCCL 完整训练／恢复实测。

本轮实测：21 项 CPU 单元／接口测试通过；DINO 特征 `[1,1,8,384]`、projector 梯度与缓存精确重载通过。真实 R0 future loss 为 `0.32075986`，与初始化 base loss 完全相同；memory 梯度范数 `0.90910071`，冻结主干没有梯度，峰值显存 41.91 GiB。两个 CPU/Gloo rank 的混合空／非空记忆梯度同步一致，全空 batch 梯度为零。上述检查均没有 optimizer 更新。

## 审核后手动启动

先用单卡 8–16 个固定样本完成 10-step overfit。`--overfit-samples` 的子集包含 async/aligned/none/wrong 四类。

```bash
cd /workspace/video-model/restream_mvp
RESTREAM_WORLD_SIZE=1 bash scripts/train_reality_memory.sh \
  --reviewed --max-steps 10 --overfit-samples 16 \
  --output checkpoints/reality_memory_overfit

# overfit 成功后，以完整数据另建一个四卡短训练 run
bash scripts/train_reality_memory.sh --reviewed --max-steps 50
bash scripts/train_reality_memory.sh --reviewed --max-steps 200 \
  --resume checkpoints/reality_memory_r0/step_0050

# 训练后的受控消融；不传 checkpoint 只是在检验零初始化 baseline
.venv/bin/python eval_reality_memory.py --reviewed --cases 4 \
  --checkpoint checkpoints/reality_memory_r0/step_0200 \
  --counts 0 1 2 4 8 --visual-metrics
.venv/bin/python scripts/eval_memory_usage.py outputs/reality_memory/evaluation/metrics.json
```

默认上限为 500，入口要求显式 `--max-steps`。只有 overfit 和短实验有正信号后才扩规模，不能把当前参数可反传当作方法有效。

## 评估含义

同一 target、干净 prefix、prompt、初始及后续噪声，比较 Base、No Memory、Hard Anchor、Oracle GT State、Aligned Soft、Async Soft、Wrong Memory 和跨样本 Shuffled Memory。每个变体都调用现有 fresh-cache rollout。当前干净 prefix 下 Oracle 与 Base 相同，明确标记为接口对照；受损历史的 Oracle recovery 仍使用旧 `eval_reanchor.py` 与历史报告。

No Memory 同时检查 context 和整段 rollout 与 Base 完全相同。Shuffled Memory 指跨 source 交换 reference sets；R0 对同一集合内部的参考顺序应当不敏感，不能把顺序打乱当作有意义的性能消融。

输出 future/next-block latent MSE、correct/wrong-source gate、原始及按有效 token 数归一化的 attention entropy、correct-vs-wrong gap、视频，以及逐帧像素变化的 motion proxy。Wrong Source 仅表示来自不同 source，不保证是无关 world。启用 `--visual-metrics` 后以每隔 4 帧的未来帧计算 DINO reference-copy max similarity 和同世界相似度 proxy；未启用记为 null。`base_quality` 无统一质量评估器，明确为 null。这些指标不等于复制检测结论、相机轨迹自由度、物体持续性或长时世界一致性。

R0 的训练和正式生成评估尚未执行；没有 Reality Memory 效果结论。R1、不同偏移量的完整 asynchrony sweep、跨视角数据、深度／物体指标及 30/60/100 秒评估属于后续阶段。
