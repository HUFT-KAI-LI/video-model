# Reality Memory：稀疏异步真实世界记忆驱动的长时视频生成
## Codex 实现与实验计划（基于当前 ReStream MVP 重构）

> **项目核心转向**
>
> 当前 ReStream MVP 已经完成了 Hard Anchor / Oracle Anchor / 单帧 VAE latent 对齐等诊断。实测表明：
>
> - 真实 LongLive teacher-forcing backward 已打通；
> - `Oracle GT latent` 对未来有小幅正向改善；
> - `Hard RGB Anchor` 平均反而恶化；
> - 单帧 VAE latent 与时序视频 latent 存在明显 representation gap；
> - 因此不再把“真实照片”定义为必须对齐到当前时刻、用于替换生成 latent 的目标帧。
>
> **新的核心研究问题：**
>
> \[
> \boxed{
> \text{Sparse asynchronous real-world observations}
> \rightarrow
> \text{persistent world prior}
> \rightarrow
> \text{reduce long-horizon drift}
> }
> \]
>
> 更准确地说：
>
> > **给定 prompt 或初始视频，能否利用稀疏、甚至时间不对齐的真实世界照片作为持续的外部世界记忆，在不要求生成轨迹复现这些照片的情况下，抑制长时视频生成中的现实世界漂移？**
>
> 英文工作表述：
>
> > **Can sparse and potentially asynchronous real-world observations serve as a persistent world prior for long-horizon video generation, reducing generative drift without forcing the generated trajectory to reproduce those observations?**

---

# 0. 这份文档和旧 ReStream 计划的关系

## 0.1 不要删除旧实现

当前仓库中的以下代码必须保留，因为它们是非常重要的 baseline / diagnostic：

```text
restream/anchor_adapter.py
restream/anchor_injector.py
eval_reanchor.py
train_reanchor.py
validation/pretrain_anchor_ablation.json
validation/anchor_latent_alignment.json
validation/real_backward.json
```

当前的：

```text
No Anchor
Hard RGB Anchor
Oracle GT Latent
Tiny Learned ReAnchor
```

应继续保留。

但是它们的角色发生变化：

```text
过去：
Hard Anchor = 主方法雏形

现在：
Hard Anchor = 强状态锚定 baseline / 失败案例
Oracle Anchor = 状态纠偏诊断
Reality Memory = 新主方法
```

## 0.2 不建议覆盖 ORIGINAL_PLAN.md

`ORIGINAL_PLAN.md` 应作为最初 12h Re-Anchoring MVP 的历史记录保留。

建议新增：

```text
REALITY_MEMORY_PLAN.md
```

并在：

```text
README.md
STATUS.md
README_REPRODUCE.md
```

中加入链接。

---

# 1. 重新定义 Anchor

必须在代码、论文、README 中把两种概念明确拆开。

## 1.1 Strong State Anchor

只有当真实传感器观测和模型当前世界状态严格同步时使用。

例如：

```text
机器人此刻摄像头真实帧 I_real(t)
               ↓
当前 streaming state z_t
```

此时可以研究：

```text
Hard replacement
Strong residual correction
KV cache rebuild / recache
```

数学上：

\[
z_t^+ = U(z_t^-, I_t^{real})
\]

这个任务继续叫：

> **State Re-Anchoring**

当前仓库的 Hard Anchor 属于这一类。

---

## 1.2 Weak World Anchor / Reality Memory

新的主任务。

Reference images 可能：

- 来自同一个真实环境；
- 不同时间；
- 不同相机；
- 不同角度；
- 不与当前生成帧严格对应；
- 只是提供关于世界的证据。

它们不能直接：

\[
z_t \leftarrow VAE(I_k)
\]

而应该：

\[
\mathcal M = E_{real}(I_1,\dots,I_K)
\]

\[
r_t = Retrieve(h_t,\mathcal M)
\]

\[
\alpha_t = Gate(h_t,r_t)
\]

\[
h_t' = h_t + \alpha_t \Delta h_t
\]

其中：

- \(h_t\)：当前生成模型状态；
- \(\mathcal M\)：外部真实世界记忆；
- \(r_t\)：和当前状态相关的真实世界 evidence；
- \(\alpha_t\)：是否以及多大程度使用现实信息；
- \(\Delta h_t\)：soft reality guidance。

关键原则：

> **照片不是答案，照片只是证据。**

---

# 2. 新论文的核心 hypothesis

不是：

> “加入真实图片能提高视频生成。”

这个太弱。

我们真正要验证：

\[
\boxed{
\textbf{Real observations can function as world-level priors rather than frame-level targets.}
}
\]

必须同时强调四个属性：

```text
External
Sparse
Asynchronous
Non-forcing
```

### External

Memory 来自真实世界，而不是模型自己历史生成结果。

### Sparse

参考数量：

\[
K \ll T
\]

例如 100 秒只提供：

```text
1 / 2 / 4 / 8 / 16
```

张真实照片。

### Asynchronous

不要求：

\[
I_k \leftrightarrow x_t
\]

不存在必须匹配的视频 timestamp。

### Non-forcing

Reference 不要求目标视频：

```text
复制该视角
复制该相机轨迹
复制该人物动作
复制该像素内容
```

而只应约束：

```text
scene identity
environment layout
appearance statistics
object scale
materials
illumination
persistent objects
spatial relations
world plausibility
```

---

# 3. 当前实验给出的关键 motivation

当前验证结果必须保留，未来可以成为论文 motivation figure。

当前结果：

```text
No Anchor
完整 future MSE ≈ 0.6543
next block MSE ≈ 0.4007

Hard RGB Anchor
完整 future MSE ≈ 0.6643
next block MSE ≈ 0.4648

Oracle GT Latent
完整 future MSE ≈ 0.6425
next block MSE ≈ 0.3737
```

现象：

```text
Correct temporal state
→ 有一定 recovery signal

Single real image latent
→ 直接替换反而可能伤害 trajectory
```

VAE alignment：

```text
Prefix vs Full
MSE ≈ 0
Cosine ≈ 1

Single vs Full
MSE ≈ 0.2909
Cosine ≈ 0.813
Single std ≈ 0.493
Temporal std ≈ 0.847
```

因此：

> 强行把一个单帧 image latent 当作 current temporal video state，在 representation 和 trajectory 两方面都不合理。

这正是 Reality Memory 方法的出发点。

---

# 4. 新系统总体架构

目标架构：

```text
                    Prompt
                      │
                      ↓
Initial Video → LongLive / Wan → Generated State h_t → Future
                         ↑
                         │ soft / gated conditioning
                         │
              ┌──────────────────────┐
              │ Reality Memory       │
              │                      │
Real Image 1 ─┤                      │
Real Image 2 ─┤ Reality Encoder      │
Real Image 3 ─┤ + Memory Projector   │
...          ─┤ + Retriever          │
              │                      │
              └──────────────────────┘
                         ↑
                         │
                  Relevance / Gate
```

---

# 5. 第一版 Reality Memory 不再使用 Wan VAE 作为主 reference encoder

当前发现已经说明：

```text
single-image Wan VAE latent
≠
temporal video state latent
```

因此 Weak World Memory 的 reference encoder 应独立于生成 VAE。

## 推荐顺序

### 第一选择：DINO / DINOv2 visual features

特点：

- 对几何、结构、物体和 appearance 较稳定；
- patch token 容易作为 memory；
- 不需要把 reference 转成“目标视频 latent”。

### 第二选择：SigLIP / CLIP-style features

特点：

- semantic alignment 好；
- 对 text prompt / scene semantics 更方便。

### 第一版不要同时堆多个 encoder

MVP：

```text
one frozen visual encoder
+
small projector
```

即可。

只有主实验成立以后，再加：

```text
Depth
DINO + Depth
multi-scale features
```

作为后续 geometry-aware extension。

---

# 6. Reality Encoder

新增：

```text
restream/reality_encoder.py
```

建议接口：

```python
class RealityEncoder(nn.Module):
    def __init__(self, backbone, output_dim, num_memory_tokens):
        ...

    @torch.no_grad()
    def encode_visual(self, images):
        """
        images:
          B,K,3,H,W
        return:
          raw visual features
        """
        ...

    def project(self, features):
        """
        frozen visual features
            ↓
        trainable projector
            ↓
        B,M,D memory tokens
        """
        ...
```

原则：

```text
visual backbone frozen
projector trainable
```

第一版避免全量 fine-tune visual encoder。

---

# 7. Reality Memory

新增：

```text
restream/reality_memory.py
```

Memory 输入：

```text
K reference images
↓
patch / pooled visual features
↓
projector
↓
memory tokens M
```

建议第一版先不要构建复杂向量数据库。

单样本 K 很小：

```text
K = 1 / 2 / 4 / 8
```

直接保存：

\[
M\in\mathbb R^{B\times N_m\times D}
\]

即可。

---

# 8. 两阶段实现：先做简单 baseline，再做主方法

这是整个实现最重要的 scope 控制。

## Stage R0：Memory-as-Context Baseline

先验证：

> 不做 hard replace，只把真实 reference 作为 soft context，是否已经有正信号。

最少改动方案：

```text
Reality Encoder
↓
Visual Projector
↓
Memory Tokens
↓
append / fuse into LongLive conditioning context
↓
existing cross-attention
```

即：

```text
text context ─────┐
                  ├→ extended conditioning context
reality tokens ───┘
```

优点：

- 对 LongLive 改动小；
- 不需要修改每一层 self-attention；
- 很快验证 soft conditioning 是否优于 Hard Anchor；
- 适合作为 baseline。

缺点：

- 不是 state-aware retrieval；
- 所有时间块看到相同 memory；
- 无法实现真正 drift-aware guidance。

因此 R0 不是最终主方法。

## Stage R1：State-Conditioned Reality Memory（主方法）

主方法：

```text
Current hidden state h_t
         ↓
      Query
         ↓
retrieve relevant memory
         ↓
Reality Cross Attention
         ↓
Δh_t
         ↓
Gate α_t
         ↓
h'_t = h_t + α_t Δh_t
```

新增：

```text
restream/memory_retriever.py
restream/soft_guidance.py
```

---

# 9. Memory Retriever

对于 K 个 references：

\[
M=\{m_1,\dots,m_K\}
\]

当前 AR block state：

\[
q_t=Q(Pool(h_t))
\]

相关度：

\[
s_{t,k}
=
\frac{q_t^\top k_k}
{\sqrt d}
\]

然后使用 soft retrieval：

\[
w_{t,k}=\operatorname{softmax}(s_{t,k})
\]

\[
r_t=\sum_k w_{t,k}m_k
\]

第一版不要做复杂向量数据库，也不要先做离散 top-k。

---

# 10. Drift / Relevance Gate

不要第一版就训练复杂 RL policy。

先做：

\[
\alpha_t
=
\sigma(
MLP[
Pool(h_t);
Pool(r_t);
sim(h_t,r_t)
]
)
\]

输出：

```text
0 ≤ α_t ≤ 1
```

正常状态：

```text
α_t → 小
```

需要更多现实信息：

```text
α_t → 大
```

注意：

> 第一版不要声称 α 是严格“drift estimator”。

它首先是：

> **learned relevance / guidance-strength gate**

以后如果证明它和 drift 强相关，再称 Drift-Aware Gate。

---

# 11. Soft Guidance 模块

新增：

```text
restream/soft_guidance.py
```

第一版：

\[
\Delta h_t
=
CrossAttn(h_t,r_t)
\]

\[
h_t'
=
h_t+\alpha_t\Delta h_t
\]

建议只插入少量 Transformer block。

例如：

```text
layer 8
layer 16
layer 24
layer 31
```

不要全部层修改。

---

# 12. 参数冻结策略

第一阶段：

```text
Wan VAE                Frozen
UMT5                    Frozen
LongLive backbone       Frozen
LongLive official LoRA  Frozen
Reality visual encoder  Frozen

Train:
Reality projector
Memory query/projector
Reality cross-attn
Gate
```

不要一开始训练新的 backbone LoRA。

只有 R0/R1 已有明显 positive signal 后：

```text
Backbone LoRA r=8/16
```

作为后续 enhancement。

---

# 13. 数据集重新定义

当前 Youku 数据仍可以用于工程 MVP，但训练样本语义需要改变。

过去：

```text
GT video
+
one aligned anchor
```

现在：

```text
Target video window
+
a set of weak world references
```

Manifest 示例：

```json
{
  "video": "...",
  "target_start": 12.5,
  "target_sec": 3.5,
  "caption": "...",
  "references": [
    {"time": 2.0, "role": "same_world"},
    {"time": 7.5, "role": "same_world"},
    {"time": 21.0, "role": "same_world"}
  ],
  "source_id": "...",
  "split": "train"
}
```

训练时 **不要把 exact reference timestamp 输入模型**。

timestamp 只用于 dataset 构造和 analysis。

---

# 14. Reference 数据采样

推荐构造四种样本。

## A. Same-source asynchronous references：50%

从同一 source video 中 target window 外随机抽 K 张。

要求和 target window 保持最小时间间隔：

```text
>= 1.0–2.0s
```

MVP 可以先这样。

但是：

> 这只是 asynchronous proxy，不足以最终证明 same-world different-time。

## B. Near-aligned soft references：20%

从 target window 附近抽，但仍不允许 hard replacement。

用途：

> Aligned Soft Guidance baseline。

## C. No-reference：15%

Memory 为空。

要求：

```text
Ours(no reference)
≈
Base LongLive
```

## D. Wrong-reference：15%

从不同 source video 采 reference。

用途：

训练模型忽略错误 evidence。

期望：

\[
\alpha_{wrong}
<
\alpha_{correct}
\]

---

# 15. 当前 Youku 数据的限制

Youku 只能作为：

> pipeline / representation / short-term MVP

不能作为最终证明 asynchronous same-world memory 的唯一数据。

原因：

- 同一个 source 可能存在剪辑；
- 长视频可能包含不同场景；
- 同视频 reference 有 future/action leakage 风险；
- 缺乏真正 multi-view same-world identity。

因此必须增加：

```text
shot-cut filtering
scene consistency filtering
```

正式论文考虑：

```text
Ego-Exo4D
同场景多视角
长 continuous take
```

或者自己采集一个小型 real-world evaluation set。

---

# 16. Shot-cut filter 必须实现

新增：

```text
scripts/filter_continuous_shots.py
```

对于 target window 和 references 检测：

```text
hard cut
black frame
strong transition
```

MVP 可用：

```text
RGB histogram difference
frame embedding jump
```

输出：

```text
data/reality_train.jsonl
data/reality_val.jsonl
data/reality_stats.json
```

---

# 17. 第一阶段训练目标

不再训练：

> “reference frame 对应的 latent 应该变成什么。”

而训练：

> “使用世界 memory 后，生成未来是否更符合真实视频分布。”

基础：

\[
\mathcal L_{video}
=
\mathcal L_{flow}
\]

继续复用 LongLive 原 teacher-forcing future loss。

---

# 18. Memory Dropout

训练时随机：

```text
drop all memory
drop some references
```

建议：

```text
p_no_memory = 0.15
per_reference_dropout = 0.2
```

防止模型过度依赖 references。

---

# 19. Wrong Reference Suppression

对于 wrong reference：

\[
\alpha_t^{wrong}\rightarrow0
\]

可以用：

\[
\mathcal L_{wrong}
=
\alpha_t^2
\]

只对明确 wrong-source samples 使用。

不要要求 correct reference 的 gate 一定等于 1。

---

# 20. Guidance Magnitude Regularization

防止 Reality Memory 破坏 trajectory：

\[
\mathcal L_{\Delta}
=
\|\Delta h_t\|_2^2
\]

小权重：

```text
1e-5 ~ 1e-4
```

第一版总 loss：

\[
\boxed{
\mathcal L
=
\mathcal L_{video}
+
\lambda_\Delta\mathcal L_\Delta
+
\lambda_{wrong}\mathcal L_{wrong}
}
\]

---

# 21. 不使用 reference 时必须保持 baseline

对于 no-reference sample，结构上保证：

\[
h'_t=h_t
\]

而不是依赖模型“学会”不破坏。

---

# 22. Adapter 初始化

所有新增 residual branch：

```text
last projection zero-init
```

保证 step 0：

\[
\Delta h_t=0
\]

从 exact base LongLive behavior 开始。

---

# 23. Phase R0 最小实验

Baselines：

```text
Base LongLive
Hard RGB Anchor
Oracle GT State diagnostic
Aligned Soft Memory
Async Soft Memory
Wrong Memory
```

训练：

```text
Frozen LongLive
Frozen visual encoder
Train projector
Train soft memory adapter
```

顺序：

```text
1 step real backward
10 step overfit
50 step
200 step
500 step
```

不要直接 3000。

---

# 24. R0 Success Gate

200–500 step 后至少满足：

1. `No Memory` 与 Base 基本一致；
2. `Wrong Memory` 不明显破坏生成；
3. Correct Same-World Memory 优于 Wrong Memory；
4. Soft Memory 不产生 Hard Anchor 那种明显状态突变；
5. gate / memory attention 不是恒定值；
6. reference shuffle 会导致性能下降。

如果失败：

> 不升级 R1。

---

# 25. Phase R1 主方法

R0 有信号后实现：

```text
state-conditioned retrieval
+
dynamic gate
+
selected-layer cross attention
```

重点验证：

\[
\text{current generated state}
\rightarrow
\text{select relevant reality evidence}
\]

---

# 26. R1 训练阶段

## Stage 1：Short-window supervised

```text
3.5s / 57 frames
```

目标：学会 use / ignore Reality Memory。

## Stage 2：Longer-window teacher forcing

扩到：

```text
8s
16s
```

## Stage 3：Self-rollout

```text
generate
↓
accumulate drift
↓
query Reality Memory
↓
continue generation
```

这一步才逼近 long-horizon inference distribution。

---

# 27. 当前不做 RL

不要加入：

```text
GRPO
PPO
request-image policy
```

RL 留给未来：

> 模型主动决定什么时候向真实世界请求新 observation。

---

# 28. 实验 Baseline 表

| Method | Real References | Aligned | Hard Replace | State-aware Retrieval |
|---|---:|---:|---:|---:|
| Base LongLive | ❌ | — | ❌ | ❌ |
| Hard Anchor | ✅ | ✅ | ✅ | ❌ |
| Aligned Soft | ✅ | ✅ | ❌ | ❌ |
| Async Soft | ✅ | ❌ | ❌ | ❌ |
| Wrong Memory | ❌ same-world | ❌ | ❌ | ❌ |
| Ours Reality Memory | ✅ | ❌ | ❌ | ✅ |

---

# 29. Reference 数量实验

必须测：

```text
K = 0
K = 1
K = 2
K = 4
K = 8
```

希望看到：

```text
0 → baseline
1 → improvement
2 → more improvement
4 → saturation
8 → marginal gain
```

---

# 30. Asynchrony 实验

必须测：

```text
Aligned
±0.5s
±1s
±2s
random same-source
same-scene different-view
```

核心目标：

> 即使 reference 不与目标帧严格对齐，world-level guidance 仍有帮助。

---

# 31. Wrong Reference 实验

输入相同数量 reference，但来自：

```text
different source
different environment
```

观察：

```text
quality
world consistency
gate α
retrieval score
```

理想：

```text
Correct memory:
α high when useful

Wrong memory:
α low
```

---

# 32. 必须证明模型不是 Copy Reference

需要同时证明：

```text
world adherence ↑
```

但是：

```text
trajectory freedom ≈ Base
```

---

# 33. Trajectory Freedom 指标

## Camera / Motion Diversity

比较 Base 和 Ours：

```text
optical flow distribution
camera motion
motion magnitude
```

## Reference Copy Score

\[
C_{copy}
=
\max_{t,k}
sim(E(x_t),E(I_k))
\]

要求 Ours 不出现持续的近重复 reference frame。

---

# 34. World Consistency 指标

MVP：

```text
DINO same-world similarity
object detection persistence
depth temporal consistency
```

正式论文再组织为 World Consistency Score。

---

# 35. Long-Horizon Evaluation

最终：

```text
30s
60s
100s
```

绘制：

\[
Consistency(t)
\]

理想：

```text
Base 随时间明显下降
Reality Memory 下降更慢
```

---

# 36. repo 文件结构调整

建议新增：

```text
restream_mvp/
├── REALITY_MEMORY_PLAN.md
│
├── restream/
│   ├── anchor_adapter.py          # 保留 baseline
│   ├── anchor_injector.py         # 保留 baseline
│   ├── reality_encoder.py         # NEW
│   ├── reality_memory.py          # NEW
│   ├── memory_retriever.py        # NEW
│   ├── soft_guidance.py           # NEW
│   ├── reality_dataset.py         # NEW
│   └── reality_metrics.py         # NEW
│
├── configs/
│   ├── restream_mvp.yaml
│   ├── reality_memory_r0.yaml      # NEW
│   └── reality_memory_r1.yaml      # NEW
│
├── scripts/
│   ├── build_reality_manifest.py   # NEW
│   ├── filter_continuous_shots.py  # NEW
│   ├── check_reality_encoder.py    # NEW
│   ├── cache_reality_features.py   # NEW
│   └── eval_memory_usage.py        # NEW
│
├── train_reanchor.py
├── eval_reanchor.py
├── train_reality_memory.py         # NEW
└── eval_reality_memory.py          # NEW
```

---

# 37. 不要直接重写 LongLive 大量源码

优先：

```text
minimal hook
+
external modules
```

R0 尽量只修改 conditioning context 接口。

R1 才在少量指定 transformer blocks 增加 Reality Guidance hook。

所有上游修改继续进入：

```text
longlive.patch
```

保持可重建。

---

# 38. R0 配置建议

```yaml
reality_memory:
  enabled: true

  encoder:
    type: dinov2
    frozen: true

  references:
    min_count: 1
    max_count: 4
    no_memory_probability: 0.15
    wrong_memory_probability: 0.15
    per_reference_dropout: 0.20

  projector:
    trainable: true
    num_memory_tokens: 8

  guidance:
    mode: context
    zero_init: true

train:
  backbone_frozen: true
  max_steps: 500
  lr: 1.0e-4
```

具体 encoder checkpoint 由机器上可获得的可靠权重决定，不要为了某一个模型源阻塞实验。

---

# 39. R1 配置建议

```yaml
reality_memory:
  enabled: true

  retrieval:
    mode: soft
    query_dim: 256
    temperature: 0.1

  guidance:
    mode: state_conditioned
    layers: [8, 16, 24, 31]
    gate: learned
    zero_init: true

  regularization:
    delta_weight: 1.0e-5
    wrong_gate_weight: 1.0e-2
```

---

# 40. 4 × A800/A100 算力策略

当前真实 backward 峰值约：

```text
42 GiB / GPU
```

短窗口有较大空间。

第一阶段：

```text
per GPU batch = 1
DDP
BF16
no gradient checkpointing unless OOM
```

Reality visual features 最好 offline precompute。

---

# 41. 强烈推荐预缓存 Reference Features

新增：

```text
scripts/cache_reality_features.py
```

一次性：

```text
reference RGB
↓
frozen visual encoder
↓
feature file
```

训练时不重复 visual encoder forward。

缓存可用：

```text
.pt / safetensors
```

---

# 42. 训练时间策略

依次：

```text
1 step real backward
10 step overfit
50 step
200 step
500 step
```

### 10-step overfit

拿 8–16 个样本重复训练，验证：

```text
correct memory 是否真的能改变 future
wrong memory gate 是否下降
```

如果小样本都学不会，不扩大数据。

---

# 43. 第一周 Success Criteria

第一周必须回答：

### Q1

Soft Memory 是否比 Hard Anchor 更稳定？

### Q2

Same-world reference 是否优于 Wrong Reference？

### Q3

不提供 reference 时是否几乎恢复 Base？

### Q4

reference shuffle 是否降低效果？

### Q5

模型有没有明显 copy reference？

如果五个问题正向：

> 才进入 long-horizon。

---

# 44. 当前 ReAnchor 项目如何作为论文 motivation

不要浪费已经完成的工作。

可以形成：

### Observation 1

Naively injecting a real frame into the causal latent state can hurt generation.

### Observation 2

Single-frame and temporal video latents have a significant representation mismatch.

### Observation 3

Oracle temporal-state correction gives a positive but limited signal.

### Conclusion

Therefore, real observations should not be treated as target latent states.

Instead:

> **real observations are external world evidence.**

这会自然引出 Reality Memory。

---

# 45. 当前术语建议

主项目逐渐避免：

```text
Anchor
ReAnchor
```

主方法建议：

```text
Reality Memory
World Memory
Reality Guidance
External World Prior
```

Strong Anchor 只作为特殊子问题。

---

# 46. 暂定项目名

建议代码内部使用：

```text
RealityMemory
```

论文标题候选：

> **Reality as Memory: Grounding Long-Horizon Video Generation with Sparse Asynchronous Observations**

或者：

> **Stay in the Real World: Sparse Reality Memory for Long-Horizon Video Generation**

项目目录无需改：

```text
restream_mvp
```

新增代码统一：

```text
reality_*
```

即可。

---

# 47. 当前不要声称 Physics

Reality images 直接提供：

```text
geometry clues
scene layout
object scale
materials
appearance
illumination
spatial relationships
affordance clues
```

当前论文使用：

```text
real-world consistency
world plausibility
environment grounding
long-horizon drift
```

不要直接写：

```text
learn physical laws
physics-constrained generation
```

---

# 48. Codex 第一批具体任务

严格按顺序实现。

## Task 1：保留现有 ReAnchor baseline

不得删除或破坏：

```text
train_reanchor.py
eval_reanchor.py
validation/*
```

## Task 2：新增 Reality Memory 文档与配置

创建：

```text
REALITY_MEMORY_PLAN.md
configs/reality_memory_r0.yaml
```

## Task 3：Reality Reference Manifest

新增：

```text
scripts/build_reality_manifest.py
```

先从当前 Youku 数据构造：

```text
same-source async
near-aligned
no-reference
wrong-reference
```

## Task 4：Shot Filter

新增：

```text
scripts/filter_continuous_shots.py
```

## Task 5：Reality Encoder

新增：

```text
restream/reality_encoder.py
```

优先 frozen。

## Task 6：Feature Cache

新增：

```text
scripts/cache_reality_features.py
```

## Task 7：R0 Memory-as-Context

实现最小 soft-conditioning baseline。

目标：

```text
不用 hard replacement
不用改变当前 latent
```

## Task 8：单元测试

至少覆盖：

```text
No Memory -> exact zero residual
Wrong ref mask
Reference dropout
Feature cache reload
Variable K references
DDP batch=1
same-source / wrong-source manifest correctness
```

## Task 9：10-step Overfit

不启动大训练。

输出：

```text
memory gate
train loss
correct vs wrong memory
soft vs hard qualitative
```

## Task 10：50 / 200-step

只有 overfit 成功后执行。

---

# 49. 需要记录的新 metrics

每次 validation 输出：

```json
{
  "base_quality": null,
  "future_latent_mse": null,
  "memory_gate_mean": null,
  "correct_memory_gate": null,
  "wrong_memory_gate": null,
  "correct_vs_wrong_gap": null,
  "reference_copy_score": null,
  "memory_attention_entropy": null
}
```

---

# 50. 新的 STATUS.md 应该如何写

不要把 Reality Memory 写成已经验证成功。

应写：

```text
Current:
- ReAnchor diagnostic completed
- Hard RGB Anchor does not reliably improve future
- single-image / temporal latent mismatch confirmed
- Reality Memory is the next hypothesis
- no Reality Memory training result yet
```

直到 R0 真正跑完以前，不要写：

```text
Reality Memory works
```

---

# 51. 论文级路线

```text
Phase A
Hard Anchor diagnosis
DONE

Phase B
Soft Reality Context

Phase C
State-Conditioned Retrieval + Gate

Phase D
Long-horizon 30/60/100s

Phase E
Same-scene different-view / Ego-Exo4D

Phase F
Self-rollout training

Future
Geometry Memory
Active observation request
RL
```

---

# 52. 最终论文必须证明什么

不是：

> “我们的模块指标高。”

而是：

\[
\boxed{
\text{Sparse Reality Memory}
\rightarrow
\text{less long-term world drift}
}
\]

同时：

\[
\boxed{
\text{Trajectory Freedom}
\not\downarrow
}
\]

以及：

\[
\boxed{
\text{Exact temporal alignment is not required}
}
\]

这是整篇论文最核心的三个 empirical conclusions。

---

# 53. 一句话给 Codex

> **Do not continue optimizing Hard Anchor as the main method. Preserve it as a diagnostic baseline. The new main path is Reality Memory: treat sparse real images as asynchronous external world evidence, encode them with a frozen visual encoder, retrieve relevant memory from the current generation state, and inject only a gated residual guidance signal. The first milestone is not 100-second generation; it is to prove on short controlled experiments that correct same-world memory helps more than wrong memory, no-memory preserves the base model, and soft guidance avoids the discontinuity caused by hard latent replacement.**
