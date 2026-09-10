# Edit-Ready Video Generation：Generation-Time Edit Cache 可行性测试方案

> **用途**：交给 Codex 直接实现  
> **目标**：在不训练新模型的前提下，验证 LongLive/当前流式视频生成框架是否能够在首次生成时保存可复用的生成状态，并在之后仅重新打开指定 temporal chunk，用新 prompt 做局部重生成，同时保持未编辑 chunk 完全不变。  
> **当前阶段**：只做 feasibility / GO-NO-GO，不做自动 mask、不做双向传播、不做新 adapter 训练。  
> **资源约束**：4×A100，优先在 1–2 天内得到结论；整个方向最多 5 天完成第一轮实验。

---

## 1. 研究问题

当前流式视频生成通常执行：

\[
P_0,z \rightarrow C_1 \rightarrow C_2 \rightarrow \cdots \rightarrow C_N
\]

最终只保留 RGB/latent 视频，而生成某个 chunk 时使用过的内部状态被丢弃。

本测试验证：

\[
P_0,z
\rightarrow
\{C_1,\ldots,C_N\}
+
\mathcal{S}
\]

其中：

\[
\mathcal{S}=\{S_0,S_1,\ldots,S_{N-1}\}
\]

是每个 chunk 开始前可恢复的 **Generation-Time Edit Cache**。

之后，用户指定第 \(k\) 个 chunk 和新 prompt \(P_1\)，系统仅执行：

\[
C'_k = G(S_{k-1}, P_1)
\]

最终得到：

\[
V'=(C_1,\ldots,C_{k-1},C'_k,C_{k+1},\ldots,C_N)
\]

本阶段**不要求** \(C'_k\) 对后续 persistent state 做传播，只验证“局部重新打开生成决策”是否可行。

---

# 2. 本轮必须回答的 4 个问题

## Q1. 缓存状态是否足够复现原 chunk？

使用原 prompt、原 seed、原 noise/timestep 配置，从缓存的 \(S_{k-1}\) 重新生成 \(C_k\)。

期望：

\[
\hat C_k \approx C_k
\]

如果完全相同最好；若底层 CUDA/FlexAttention/BF16 导致非确定性，则记录可重复噪声基线。

这是**第一道硬 Gate**。

如果原 prompt 都无法从缓存状态稳定恢复同一 chunk，则说明保存的状态不完整，不能继续讨论编辑。

## Q2. 新 prompt 能否只影响被重新打开的 chunk？

例如：

**原 prompt**

> A woman in a red dress walks through a bright room.

**新 prompt**

> A woman in a blue dress walks through a bright room.

只重新生成指定 chunk \(C_k\)。

要求：

- 被编辑 chunk 对新 prompt 的响应增强；
- 未编辑 chunk 不重新运行生成器；
- 未编辑 chunk 的 latent/frame tensor 在拼接前保持完全一致。

即：

\[
V'_{\text{outside}}=V_{\text{base,outside}}
\]

这里优先做 **temporal chunk mask**，暂不做空间 mask。

## Q3. 局部重生成后的边界是否还能接得上？

检查两个边界：

### 左边界

\[
C_{k-1}\rightarrow C'_k
\]

因为 \(C'_k\) 从真实缓存的 \(S_{k-1}\) 开始，理论上应该相对容易保持连续。

### 右边界

\[
C'_k\rightarrow C_{k+1}
\]

这里 \(C_{k+1}\) 本轮被锁死，因此可能出现不连续。

本测试只做**局部 appearance / transient edits**，避免需要改变未来状态的编辑。

例如可做：

- red dress → blue dress（仅当人物在 target chunk 后不再明显出现，或只用于机制测试）；
- make this segment warmer / colder；
- make the lighting darker；
- add light rain in this segment；
- make the person smile briefly；
- slight camera zoom in。

暂不做：

- 删除一个未来还会出现的对象；
- 杯子摔碎；
- 门永久打开；
- 人物永久换衣；
- 新增持续存在的物体。

右边界不连续在本阶段是一个需要**量化**的风险，但不是立刻扩展成 causal propagation 的理由。

## Q4. 局部编辑是否比 full regeneration 便宜？

记录：

- full generation wall time；
- partial edit wall time；
- peak VRAM；
- 实际执行的 denoising/AR chunk 数；
- 可选：粗略 FLOPs proxy。

定义：

\[
R_{\text{cost}}
=
\frac{\text{partial edit cost}}
{\text{full regeneration cost}}
\]

目标是验证计算成本随编辑 chunk 数量而不是整条视频长度增长。

---

# 3. 本轮明确不做的内容

Codex **不要**在本提交中实现以下内容：

1. 自动 temporal mask predictor；
2. spatial mask；
3. SAM / segmentation；
4. bidirectional edit propagation；
5. future dependency predictor；
6. 新的 LoRA / adapter；
7. 新训练 loss；
8. bridge generation；
9. 右 context conditioning；
10. 多轮编辑；
11. Reality Memory；
12. 重新设计 LongLive backbone。

本轮是纯 inference/state-reuse feasibility test。

---

# 4. 核心假设

我们需要找到 LongLive 每个 AR chunk 开始前的**最小可恢复状态**。

第一版允许“缓存得比较重”，先证明功能。

候选需要检查并记录：

- 当前生成历史 latent；
- KV cache；
- attention sink / context cache；
- 当前 AR block index；
- RoPE / temporal position 状态；
- prompt/text conditioning；
- scheduler/timestep 状态；
- RNG state；
- CUDA RNG state；
- diffusion noise generator state；
- 任何 LongLive-specific cache metadata；
- VAE latent/history buffer；
- 当前 history truncation/window information。

第一轮**不要过早优化 cache 大小**。

原则：

> 先保存足够多的状态，证明“能 reopen”；之后再研究 compact edit cache。

---

# 5. 建议新增模块

建议不要侵入现有训练代码。

可以新增：

```text
restream_mvp/
├── restream/
│   ├── edit_cache.py
│   └── edit_replay.py
├── scripts/
│   ├── check_edit_cache_replay.py
│   ├── run_chunk_edit.py
│   └── summarize_edit_ready_mvp.py
├── configs/
│   └── edit_ready_mvp.yaml
└── validation/
    └── edit_ready_mvp/
```

文件名允许根据当前 repo 结构调整，但建议保持**独立实验入口**，不要污染已有 R0/R1 逻辑。

---

# 6. Edit Cache 数据结构

建议先定义明确 schema，例如：

```python
@dataclass
class EditCheckpoint:
    schema_version: int
    sample_id: str
    chunk_index: int

    # Provenance
    model_hash: str
    config_hash: str
    prompt_hash: str
    seed: int

    # Reproducibility state
    torch_rng_state: Tensor
    cuda_rng_state: Tensor
    generator_state: Tensor | None

    # Streaming generation state
    latent_history: Tensor | None
    kv_cache: Any | None
    sink_cache: Any | None
    position_state: Any | None
    scheduler_state: Any | None

    # Optional debugging
    previous_chunk_latent: Tensor | None
    next_noise: Tensor | None
```

注意：

- 第一版允许使用 pickle/torch.save；
- 保存的对象必须可 `map_location` 恢复；
- 每个字段必须记录 shape/dtype/device provenance；
- 恢复后 GPU tensor 统一迁移到目标 device；
- 不允许默默重新初始化缺失的随机状态。

---

# 7. 实验 A：Same-Prompt Replay Exactness

## 目的

判断缓存是否真正包含了重新生成 chunk 所需的状态。

## 流程

首次生成：

```text
P0
↓
C1
↓ save S1
C2
↓ save S2
...
CN
```

随机选定 chunk \(k\)。

然后：

```text
load S_{k-1}
+
same P0
+
same generation settings
↓
regenerate C_k
```

比较原始：

\[
C_k
\]

和 replay：

\[
\tilde C_k
\]

## 必须记录

### Latent-level

\[
\text{MSE}_{latent}
\]

\[
\text{cosine}_{latent}
\]

\[
\max |\Delta latent|
\]

### Pixel-level

decode 后记录：

\[
\text{MSE}_{RGB}
\]

可选：

- PSNR；
- LPIPS（若 repo 已有，没必要为此加重依赖）。

### Internal state

如果方便：

- cache tensor difference；
- next-block first-step model output difference。

## 重复性噪声基线

考虑 CUDA/BF16/FlexAttention 可能不是逐位确定。

对相同缓存状态、相同 prompt、相同 seed 连续 replay 2 次：

\[
D_{\text{repeat}}
=
D(\tilde C_k^{(1)},\tilde C_k^{(2)})
\]

然后比较：

\[
D_{\text{original-replay}}
\]

是否与 \(D_{\text{repeat}}\) 同量级。

## Gate A

**PASS 条件：**

优先：

\[
\text{original-replay exact}
\]

若不能逐位 exact，则：

\[
D_{\text{orig,replay}}
\le 3D_{\text{repeat}}+\epsilon
\]

并且肉眼/latent 层面不存在明显结构变化。

**FAIL 条件：**

- replay 与原 chunk 明显不同；
- 同 prompt 同 seed 仍出现大幅 drift；
- 必须重新跑前面多个 chunk 才能恢复目标 chunk。

若 Gate A FAIL：

> 停止后续 edit 实验，先定位缺失状态。

---

# 8. 实验 B：Prompt-Changed Local Replay

Gate A 通过后执行。

## 流程

假设 target chunk = \(k\)。

原始：

\[
P_0,S_{k-1}\rightarrow C_k
\]

编辑：

\[
P_1,S_{k-1}\rightarrow C'_k
\]

最终拼接：

```text
C1 ... C{k-1} C'k C{k+1} ... CN
```

## 第一版 prompt 类型

优先选 **局部 appearance / transient edits**：

### 颜色

```text
red dress → blue dress
red car → black car
warm lighting → cool lighting
```

### 短时表情

```text
neutral → smiling briefly
```

### 局部天气/光照

```text
clear → light rain in this segment
daylight → darker lighting
```

### 镜头微调

```text
slight zoom in
```

不要第一版做复杂动作/拓扑改变。

---

# 9. Baselines

至少做以下 4 组：

### B0. Original

原始视频：

\[
V(P_0)
\]

### B1. Full Regeneration

从头使用新 prompt \(P_1\) 重生成整条视频：

\[
V_{\text{full}}(P_1)
\]

作为 edit-success upper reference，但它会改变整条视频。

### B2. Cached Local Replay — Same Prompt

仅 replay \(C_k\)，但还是 \(P_0\)。

验证 state replay 本身。

### B3. Cached Local Edit — New Prompt

仅 replay \(C_k\)，使用 \(P_1\)。

这是 MVP 方法。

---

# 10. 核心指标

## 10.1 Replay Fidelity

\[
F_{\text{replay}}
=
D(C_k,\tilde C_k)
\]

越低越好。

## 10.2 Edit Responsiveness

至少同时做一个 text-image/video alignment 指标和人工 case review。

例如：

\[
S_{\text{edit}}
=
sim(E(C'_k),E(P_1))
-
sim(E(C'_k),E(P_0))
\]

编码器可优先复用已有 CLIP/DINO/VLM 基础设施，避免引入新大依赖。

如果颜色编辑可用简单颜色/区域 proxy，也可以额外记录。

## 10.3 Outside Exact Preservation

拼接前，所有未编辑 chunk：

\[
C'_j=C_j,\quad j\ne k
\]

代码中直接 assert：

```python
torch.equal(original_chunk, final_chunk)
```

若最终经过统一视频压缩，MP4 字节不要求 exact；以**编码前 frame/latent tensor**为准。

## 10.4 Boundary Continuity

分别定义：

\[
D_L
=
D(\text{end}(C_{k-1}),\text{start}(C'_k))
\]

\[
D_R
=
D(\text{end}(C'_k),\text{start}(C_{k+1}))
\]

同时计算原视频：

\[
D_L^{base},D_R^{base}
\]

报告相对增加：

\[
\Delta D_L=D_L-D_L^{base}
\]

\[
\Delta D_R=D_R-D_R^{base}
\]

第一版可以用：

- pixel/latent MSE；
- DINO feature distance；
- optical-flow jump（如果已有）。

不要为了这个 MVP 专门引入复杂 metric。

## 10.5 Edit Cost

报告：

```text
full_generation_seconds
partial_edit_seconds
full_generated_chunks
partial_regenerated_chunks
peak_vram
cache_disk_bytes
cache_bytes_per_chunk
```

并计算：

\[
R_{time}
=
T_{partial}/T_{full}
\]

---

# 11. 最小实验规模

## Smoke

先：

```text
2 videos
3–5 chunks/video
1 editable chunk/video
2 prompt edits/video
```

确认所有代码正确。

## Main MVP

建议：

```text
8–16 videos/prompts
5–8 chunks/video
2 target chunks/video
2 edit prompts/target
1 fixed seed first
```

总量已经足够判断 feasibility。

如果时间足够，再：

```text
2–3 seeds
```

但不作为首要 blocker。

4 张 A100 可按 sample/edit case 完全并行。

---

# 12. GO / NO-GO 标准

## Gate A — Replay（必须通过）

同 prompt replay 必须几乎复现原 chunk。

如果 replay 自己都不稳定：

> **NO-GO：当前 LongLive state 不适合作为直接 edit cache，先定位恢复状态。**

## Gate B — Editability

至少一组受控编辑中：

\[
S_{\text{edit(local)}} > S_{\text{edit(original)}}
\]

且效果方向与新 prompt 一致。

如果新 prompt 基本无法改变 replay chunk：

> **NO-GO：需要额外 edit adapter / prompt-rebinding post-training。**

## Gate C — Preservation

未编辑 chunk 编码前必须：

\[
\text{exact unchanged}
\]

这个本轮属于设计保证，不接受“差不多一样”。

## Gate D — Cost

对于 \(N\ge5\) 的视频、只修改 1 个 chunk：

目标：

\[
T_{\text{partial}}
<
0.5\,T_{\text{full}}
\]

更理想：

\[
T_{\text{partial}}
\approx O(1/N)
\]

如果必须大规模 replay 前文才能编辑某一 chunk，则记录真实复杂度，不要掩饰。

---

# 13. 结果解释矩阵

| Replay | Edit Success | Boundary | 结论 |
|---|---|---|---|
| FAIL | - | - | Edit cache 状态不完整，先修 state replay |
| PASS | FAIL | - | 状态能 reopen，但新 prompt 无法重绑定，下一步做轻量 edit post-training |
| PASS | PASS | 差 | 核心可行，下一步研究 bridge / local propagation |
| PASS | PASS | 好 | **强 GO**，进入 Edit-Ready Video Generation |
| PASS | PASS | 右边界差、左边界好 | 预期且有研究价值，下一步做 selective forward propagation |

---

# 14. 必须保存的 provenance

所有实验 JSON 至少包含：

```text
git_commit
config_sha256
model_checkpoint_sha256 / identifier
prompt_old
prompt_new
seed
noise seed / generator state hash
target_chunk
chunk_count
cache schema
cache file sha256
original video sha256
replay video sha256
edited video sha256
wall time
peak VRAM
```

防止后续无法确认结果来自哪个 generation state。

---

# 15. 建议输出文件

```text
validation/edit_ready_mvp/
├── replay_smoke.json
├── replay_main.json
├── local_edit_main.json
├── timing.json
├── summary.json
├── videos/
│   ├── case_x_original.mp4
│   ├── case_x_same_prompt_replay.mp4
│   ├── case_x_local_edit.mp4
│   └── case_x_full_regen.mp4
└── cache_manifest.json
```

---

# 16. `summary.json` 建议 schema

```json
{
  "status": "passed",
  "replay_gate": {
    "passed": true,
    "latent_mse_mean": 0.0,
    "repeat_noise_baseline": 0.0
  },
  "edit_gate": {
    "passed": true,
    "cases": 16,
    "successful_cases": 12
  },
  "preservation_gate": {
    "passed": true,
    "outside_exact_fraction": 1.0
  },
  "boundary": {
    "left_delta_mean": 0.0,
    "right_delta_mean": 0.0
  },
  "efficiency": {
    "full_seconds_mean": 0.0,
    "partial_seconds_mean": 0.0,
    "speedup": 0.0,
    "cache_bytes_per_chunk": 0
  },
  "decision": "GO"
}
```

数值字段用真实结果填写，不要预填虚假值。

---

# 17. Codex 实现顺序

严格按以下顺序，不要并行扩功能：

### Step 1
阅读当前 LongLive generation loop，画出：

```text
chunk start
→ history/cache state
→ diffusion/flow generation
→ append history
→ next chunk
```

在报告里明确哪些对象必须保存。

### Step 2
实现 `save_edit_checkpoint()` / `load_edit_checkpoint()`。

### Step 3
实现 **same-prompt single-chunk replay**。

### Step 4
只跑 1 个真实 case。

若 replay 不通过，停止，不做 prompt edit。

### Step 5
replay 通过后，实现 `--new-prompt`。

### Step 6
实现拼接和 outside exact assertion。

### Step 7
加入 timing / VRAM / cache size。

### Step 8
跑 2-video smoke。

### Step 9
跑 8–16 video 主 MVP。

### Step 10
输出统一 `summary.json` 和 Markdown 小结。

---

# 18. 禁止的“为了让结果好看”行为

1. same-prompt replay 失败后，不允许直接换成重新跑前面所有 chunks 并假装是 cache replay；
2. 不允许编辑失败就手选成功案例作为 aggregate；
3. 不允许根据结果修改 target chunk 后覆盖原设计；
4. 不允许将 full regeneration 的改变范围与 local method 的 edit fidelity 直接混为一个指标；
5. 不允许为了边界好看偷偷重生成邻接 chunk，除非单独标记为新 baseline；
6. 不允许把未编辑区域“感知相似”写成 exact preservation，必须真的 `torch.equal`；
7. 不允许在本轮引入新训练模块来掩盖 replay feasibility 问题。

---

# 19. 本轮最终要回答的一句话

本轮不是证明完整的 Edit-Ready Video Generation。

只回答：

> **Can the generator reuse its own generation-time state to reopen and revise one previously generated video chunk under a new prompt, without recomputing or modifying the rest of the video?**

中文：

> **视频生成器能否复用首次生成时保存的内部状态，在不重算、不改变其他视频片段的前提下，用新 prompt 重新打开并修改一个已经生成过的 chunk？**

如果答案是 YES，下一阶段才进入：

\[
\text{automatic temporal mask}
\]

\[
\text{persistent semantic edit}
\]

\[
\text{selective forward propagation}
\]

以及最终的：

\[
\text{spatiotemporal edit-ready state}
\]

---

# 20. 下一阶段仅在 GO 后讨论

若本 MVP 强通过，下一轮研究顺序建议：

1. 新 prompt 是否需要轻量 prompt-rebinding post-training；
2. temporal mask 自动预测；
3. 修改状态的 persistence 分类；
4. selective forward propagation；
5. spatial mask / token-level partial recomputation；
6. compact edit-cache compression；
7. 多轮编辑；
8. backward retrospective editing / bridge generation。

**本提交不要提前实现以上内容。**
