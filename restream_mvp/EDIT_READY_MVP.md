# Edit-Ready Video Generation：Generation-Time Edit Cache 可行性 MVP 结果

> 对应方案：[EDIT_READY_VIDEO_FEASIBILITY_PLAN.md](EDIT_READY_VIDEO_FEASIBILITY_PLAN.md)
> 本轮范围：**纯 inference / state reuse**。不训练、不加 adapter、不做 mask、不做双向传播、不做多轮编辑。
> 机器：4 × A800-SXM4-80GB，PyTorch 2.5.1+cu124，LongLive v1.0（NVlabs/LongLive `e52d9ef6865d843282a6b5e9d46d03b35f88929d`）+ 官方 `longlive_base.pt` + 冻结的官方 rank-256 LoRA。

**本轮封存的所有实验 JSON 由 commit `4c4d52f2d6b244da7566f4518edf5bcead4b1640` 在 `git dirty = false` 的工作树上产生**，源码树内容摘要 `code_tree_sha256 = f4482c5e5b95ef48a3aba38cccce80b18908a8bcf0f2f2f39aa16de50b45ef00`。每个 run 自己记录这份 provenance，`summary.json["run_provenance"]` 做了聚合。第一版报告里的 `251b0ed...+dirty` 结果已全部废弃重跑。

---

## 0. 一句话结论

**机制层面强通过**：LongLive 每个 AR chunk 开始前的内部状态可以被保存并单独恢复——同 prompt 重开是**逐位精确**的（24/24，latent 与解码后像素都 `torch.equal`），未编辑 chunk 编码前**逐位不变**（48/48），RNG state 完整（24/24）。

**并且我们找到了"新 prompt 为什么改不动 chunk"的干净答案**：不是 rebinding 实现有问题，而是**已 committed 的视觉历史压过了文本条件**。两个控制给了直接证据：

- **chunk 0（无历史）**：`cache@chunk0 + P1 + 重绑定` 与 full regeneration 的 chunk 0 **逐位相同**（10/10 有向 case，2 seed）——重绑定本身完全正确。
- **reverse 控制（P1 历史 + P0 文本）**：在 chunk 1/4 上仍能拿到 full regeneration 响应的 **94.7% / 97.4%**，与原 chunk 的 latent MSE 仅 **0.037 / 0.009**——**决定 chunk 的是历史，不是 prompt**。

于是"编辑强度"随历史增长而衰减：`R_k` = 100%（k=0）→ **5.3%**（k=1）→ **1.6%**（k=4）。

按方案 §13 矩阵，判定 **`GO_WEAK_PROMPT_REBINDING`**。

同时有两个必须写清楚的负面结果：

1. **training-free 的 history recache 不能解锁编辑**（`S` 与单纯的文本重绑定持平），而且它连 cache 都复现不出来——差异全部落在被保护的 sink 区，说明 **KV cache 是路径相关的**。
2. **成本只在"cache 已驻留"时便宜**：device 端计算 `R=0.167`、驻留内存时端到端 `R=0.367`，但**从冷存储读 1 GiB checkpoint 时端到端 `R=0.770`，Gate D 不通过**。瓶颈是 cache 体积与 I/O，不是重算。

---

## 1. 本轮回答的 4 个问题

| 问题 | 结论 | 关键证据 |
|---|---|---|
| Q1 缓存状态是否足以复现原 chunk | **是，逐位精确** | 24/24 `torch.equal`；RNG 重放 24/24 精确；镜像循环与上游 `max_abs=0.0` |
| Q2 新 prompt 能否只影响被重开的 chunk | **能，且已证明实现正确；但被历史压制** | 48/48 未编辑 chunk `torch.equal`；chunk 0 与 full regen **逐位相同**；`R_k` 100%→5.3%→1.6% |
| Q3 边界是否接得上 | **左边界好；右边界在 chunk 0 处很差，k≥1 尚可** | 左 +8.9%/+1.5%（k=1/4）；右 **+1258%**（k=0）、+37%（k=1）、+19%（k=4） |
| Q4 局部编辑是否更便宜 | **只在 cache 驻留时** | compute 0.167 / warm e2e 0.367 / cold e2e **0.770**（冷读不通过 Gate D） |

---

## 2. 本轮相对上一版修掉的两个协议缺陷

这一节很重要，因为上一版的 `2.7%` 与 `R_time=0.401` 都是在这两个缺陷下测的。

### 2.1 full regeneration baseline 没有对齐 RNG（数值混淆）

上一版里 base video 与 full regeneration 是紧接着生成的，第二次生成继承了第一次剩下的 RNG 流，于是**两次生成的 chunk 内逐步 re-noising 噪声不同**。base 侧 replay 会恢复 checkpoint 的 RNG，所以 replay 是逐位的；但 `S_local` vs `S_full` 的比较里混入了噪声差异。

修法：每次完整生成前都重新 seed 并重抽初始噪声；run 里记录 `base_and_full_regen_share_rng_stream` 与 `share_initial_noise`，本轮 **48/48 全为 True**。

这个 bug 是被 **chunk 0 control** 抓出来的：如果两次生成的 RNG 流一致，chunk 0 必须逐位相同；第一次跑出来"接近但不精确"，才顺着查到 RNG 流不对齐。

### 2.2 full regeneration baseline 付了 cache 写盘的钱（成本虚高）

上一版 full regeneration 为了给 reverse 控制提供 P1 checkpoint，写入了目标 chunk 的 cache，于是 baseline 被抬到 ~10 s，`R_time` 看起来只有 0.307。

修法：计时用的 full regeneration **不写 cache**（真实用户只想重生成时不会付这份钱），另起一次不计时的生成专门产出 reverse 控制所需的 P1 checkpoint，并断言两者 latent 逐位相同。现在 baseline = **3.519 s**。

---

## 3. Step 1：LongLive 生成循环与必须保存的状态

### 3.1 循环结构（`code/LongLive/pipeline/causal_inference.py`）

```
文本编码            inference() L80-82      conditional_dict = text_encoder(prompts)
缓存分配            L109-132               kv_cache1[30] / crossattn_cache[30]
                                            kv_cache_size = local_attn_size(12) * frame_seq_length(432) = 5184

for chunk k in 0..N-1:                      L144-209
    noisy = noise[:, k*B : (k+1)*B]         L150-151
    for step in denoising_step_list:        L154-188
        forward(kv_cache, crossattn_cache, current_start = k*B*frame_seq_length)
        若非最后一步: noisy = add_noise(x0, randn_like(x0), next_timestep)   L173-178
    output[:, k*B:(k+1)*B] = x0            L190
    用 context_noise 重跑一次写回干净 KV    L192-200   ← chunk 结束时的 cache 只含历史
```

- 单步 forward：`wan/modules/causal_model.py::_forward_inference` L891-1050。**cache 更新是延迟的**：各 block 只返回 `cache_update_info`，30 个 block 全部算完后由 `_apply_cache_updates` 一次性写回（L1043-1044）。
- 自注意力 cache：L228-311，局部窗口滚动 + 前 `sink_size * frame_seqlen` 个 token 固定保护。
- 文本 cross-attention cache：`wan/modules/model.py::WanT2VCrossAttention` L161-181，`is_init=False` 时用当前 context 计算 K/V 并写入，之后复用。

### 3.2 必须保存的对象（"sufficient recoverable state"，不是已证明的 minimal）

| 对象 | 位置 | 本轮是否必须 | 说明 |
|---|---|---|---|
| `kv_cache1`（30 × {k, v, global_end_index, local_end_index}） | pipeline | **必须** | 自回归视觉历史；sink token 就是它的前 `sink_size*frame_seq_length` 个槽位，**没有独立 sink 对象** |
| `crossattn_cache`（30 × {k, v, is_init}） | pipeline | **必须** | 文本绑定；`is_init` 决定新 prompt 能否生效 |
| `current_start_frame` | 循环变量 | **必须** | 决定 RoPE 帧偏移与 cache 写指针 |
| chunk 初始噪声 | 全局 noise | **必须** | 整条视频开始时抽好，无法从 chunk 边界 RNG 反推 |
| chunk 内 3 次 `randn_like` 抽样 | 循环内 | 必须（可验证） | 记录后可直接比对 RNG state 是否完整 |
| CPU/CUDA RNG state | 全局 | **必须** | 恢复后精确重现上述抽样 |
| prompt 条件（`prompt_embeds`） | 文本编码器 | 必须 | 同 prompt replay 直接复用；换 prompt 重算 |
| `latent_history` | output | 需要 | 拼接、边界指标、体积溯源 |
| scheduler / RoPE / position 状态 | scheduler、model.freqs | 不需保存，只记录溯源 | 由 config 完全决定，无 per-chunk 可变状态 |

一个 chunk 的 cache 是 **1,049,887,200 B ≈ 0.978 GiB**；保存全部 7 个 chunk 边界共 **6.88 GiB**。本轮**没有**证明这是最小状态，只证明了它足够。

---

## 4. 实现

### 4.1 新增文件

```text
restream_mvp/
├── EDIT_READY_VIDEO_FEASIBILITY_PLAN.md   # 方案原文（sha256 09a6cb6d…）
├── EDIT_READY_MVP.md                       # 本文件
├── configs/edit_ready_mvp.yaml             # 8 个 prompt 对 + directional/qualitative 标注 + gate 阈值
├── restream/edit_cache.py                  # EditCheckpoint schema、save/load、capture/restore、RNG、provenance
├── restream/edit_replay.py                 # 镜像 chunk 循环、replay_chunk、history recache、cache 比对、拼接
├── restream/edit_metrics.py                # latent/pixel/边界指标 + 颜色/亮度代理 + 可选 DINO
├── restream/edit_media.py                  # PyAV 视频写出与并排对比
├── restream/edit_experiment.py             # case 展开/shard、确定性噪声、provenance、page-cache 驱逐、严格 JSON
├── scripts/check_edit_streaming_equivalence.py  # 镜像循环 vs 上游 + 单 chunk 重开自检
├── scripts/check_edit_cache_replay.py      # 实验 A（Gate A）
├── scripts/run_chunk_edit.py               # 实验 B（replay / text_rebind / history_recache / reverse / control / full regen）
├── scripts/check_edit_vae_locality.py      # 解码后编辑局部性诊断
├── scripts/summarize_edit_ready_mvp.py     # summary.json / timing.json / cache_manifest.json
├── scripts/08_edit_ready_mvp.sh            # smoke / all-chunks / main / main-edit / rerun
└── tests/test_edit_cache.py                # 32 项 CPU 回归测试（全套 103 项通过）
```

独立入口，未改动任何 R0/R1 训练逻辑；`restream/runtime.py` 只加了一处向后兼容改动（几何参数可来自 `data` 或 `generation`）。

### 4.2 Checkpoint schema（`restream/edit_cache.py`）

`EditCheckpoint` 覆盖方案 §6 全部字段，并额外记录：

- `model_hash` / `config_hash` / `prompt_hash` / `git_commit` / `git_dirty` / `code_tree_sha256`；`git_state()` + `source_tree_sha256()` 让 provenance 同时有"提交号"和"源码内容摘要"。
- `provenance.tensors`：递归记录每个 tensor 的 shape/dtype/device。
- `denoise_noise`：chunk 内 3 次噪声抽样，用于**直接验证 RNG state 完整性**。
- `sink_cache=None` 并显式说明原因（sink 在 `kv_cache1` 内）。
- `scheduler_state` / `position_state`：记录 timesteps/sigmas 与 RoPE 摘要，说明它们无 per-chunk 状态。

保存用 tmp→`os.replace`，返回 sha256/bytes；加载 `weights_only=True` 并校验 sha256。**恢复时总是新建设备副本**，同一个 checkpoint 可以反复 replay 而不被污染（有专门回归测试）。`replay_chunk` 现在强制执行 `verify_checkpoint_identity`（model/config）与 `verify_checkpoint_against_pipeline`（block 大小 / token 几何 / 注意力窗口 / cache 长度），checkpoint 作为独立 artifact 使用时不会再被静默错配。

### 4.3 六个变体，以及它们各自的因果含义

| 变体 | visual history | text binding | 用途 |
|---|---|---|---|
| `replay` | P0 KV | P0 | 状态复用是否逐位精确 |
| `text_rebind` | P0 KV | **P1（重绑定）** | MVP 方法 |
| `history_recache` | 由**未改动的 P0 latent** 在 P1 下重建 | P1 | 检验"visual KV 本身带着 P0 条件"这一假设 |
| `reverse` | P1 轨迹 | **P0（重绑定）** | 检验"是历史还是 prompt 决定 chunk" |
| `crossattn_control` | P0 KV | P1 文本但保留旧 text K/V | 证明新 prompt 的唯一入口 |
| `full_regeneration` | P1 轨迹 | P1 | 上界参考（不写 cache） |

设计要点：`crossattn_cache["is_init"]=True` 时 forward 直接复用缓存的文本 K/V，**根本不会看新的 `prompt_embeds`**。因此 local edit 必须清零 `is_init` 与 K/V。实测 control 在 **32/32** 个 k>0 case 上与 replay `torch.equal`：新 prompt 的唯一入口就是 text cross-attention，编辑的全部可见差异都来自重绑定。

chunk 0 例外且已验证：那里 checkpoint 还没有任何文本绑定，control 与 edit **逐位相同**（16/16），所以 Gate B 在 chunk 0 不把 `local > control` 当作否决条件。

---

## 5. 实验设置

- 视频：21 latent 帧（81 像素帧 @16 fps，256×432），AR block = 3 帧 ⇒ **7 个 chunk**。
- 目标 chunk：**`[0, 1, 4]`**。chunk 0 无历史，是校准点；chunk 4 在中后段。
- prompt：8 组，其中 **5 组 `directional`**（红→蓝、红→黑、亮→暗、暖→冷、绿→黄，探针可给出有定义的方向），**3 组 `qualitative`**（雨、短暂微笑、轻推镜，探针只能检测"变了"）。
- 随机性：seed 1 + seed 2 复现（`--seed-stride 1`），共 48 个编辑 case、24 个 replay case。
- 4 卡按 prompt 分片；实验 A 与实验 B 分开跑，避免进程竞争污染计时。
- 编辑响应 `S_proxy`：颜色占比 / 亮度 / 外观变化的**方向性代理**（正值 = 朝新 prompt 方向）。**不是** text-video alignment——本机无法访问 HuggingFace 取 CLIP，报告里不冒充有该指标。

---

## 6. 结果

### 6.1 镜像循环等价性与单 chunk 重开自检

`validation/edit_ready_mvp/streaming_equivalence.json`

| 检查 | 结果 |
|---|---|
| `stream_generate` vs 上游 `inference()` | `exact=True`，`max_abs=0.0` |
| 从 `S_{k-1}` 重开 chunk（同 prompt） | `exact=True` |
| checkpoint 内 RNG state 重现 chunk 内 3 次噪声 | `rng_exact=True` |
| 每 chunk cache 体积 | `1,049,887,200` B ≈ 0.978 GiB |

### 6.2 实验 A：Gate A（24 个 case，8 prompt × target {0,1,4}）

`validation/edit_ready_mvp/replay_main_shard*.json`

| 指标 | 实测 |
|---|---|
| 逐位精确 replay | **24 / 24** |
| RNG state 精确重现 | **24 / 24** |
| latent MSE / cosine | **0.0 / 1.0** |
| repeat-noise baseline（同 cache 重放两次） | **0.0**（无需容忍噪声） |
| 解码后像素 | `exact=True`，PSNR = inf |

**Gate A：PASS（exact 级别）。**

### 6.3 全部 chunk 边界缓存（heavy-cache 路径）

`validation/edit_ready_mvp/replay_all_chunks_shard0.json`

7 个 chunk 全部保存 = **6.88 GiB**（逐 chunk 从 1,054,833,658 B 随 `latent_history` 增长到 1,055,995,293 B）；对 chunk 0/1/4 的 replay 仍然逐位精确。按方案 §4，本轮不做压缩。

### 6.4 校准：chunk 0 的 no-history control（本轮最重要的 sanity check）

chunk 0 没有 visual history，因此

```
cache@chunk0 + P1 + 重绑定   ==   full regeneration(P1) 的 chunk 0
```

必须**逐位**成立。实测：

| 项 | 结果 |
|---|---|
| `chunk0_text_rebind_equals_full_regeneration` | **16 / 16（directional 10/10）** |
| `R_k` @ chunk 0 | **1.000** |
| `S_text_rebind` vs `S_full` @ chunk 0 | 0.2140 vs 0.2140 |

结论：**prompt rebinding 的实现本身完全正确**。后续 chunk 编辑变弱，只能由已 committed 的历史解释。

### 6.5 编辑强度随历史衰减：`R_k` 曲线

`R_k = S_text_rebind(k) / S_full_regeneration(k)`（只统计 directional prompt，每档 10 个 case：5 prompt × 2 seed）

| target chunk | 有历史 | `R_k` 均值 | `S_text_rebind` | `S_full_regeneration` |
|---|---|---:|---:|---:|
| **0** | 无 | **1.000** | 0.2140 | 0.2140 |
| **1** | 1 个 chunk | **0.053** | 0.0091 | 0.2124 |
| **4** | 4 个 chunk | **0.016** | 0.0024 | 0.2083 |

另有 1 个 case（`car_red_to_black` seed43 chunk1）达到 `R_k = 0.362`，是 k>0 里唯一 ≥25% 的；其余 9 个 k=1 case 都在 ±0.05 以内。

> **Editability decays as generated state becomes committed into history.**
> 这条曲线比原来那句"2.7%"信息量大得多：**1 个 chunk** 的历史就把 prompt 的作用压掉了约 95%。

### 6.6 为什么？— reverse 控制证明是历史在说话

`reverse` = P1 轨迹的历史 + **P0** 文本（重绑定）。如果 chunk 由 prompt 决定，它应该接近 base；如果由历史决定，它应该接近 full regeneration。

| target chunk | `S_reverse / S_full` | `reverse` 与 full-regen chunk 的 latent MSE |
|---|---:|---:|
| 0（无历史） | 0.000 | 0.521（即等于 base，符合预期） |
| **1** | **0.947** | **0.0370** |
| **4** | **0.974** | **0.0092** |

也就是说：**在一个已经 committed 的 P1 历史上，即使把文本换回 P0，chunk 仍然基本是 P1 的 chunk**。k=4 时与原 full-regen chunk 的差异只有 0.009 MSE。

这是本轮最有价值的科学结论，也直接回答了审阅提出的问题：

> **为什么一个已经 committed 的视觉历史如此难被新 prompt 改写？**
> 因为在 reopen 的那一步，视觉历史对 chunk 的约束远强于文本条件；prompt 只在**没有历史**（chunk 0）时能完全决定结果。

### 6.7 history recache：假设被检验，并且是否定的

审阅建议的 training-free 控制：保持 P0 latent 内容完全不变，用 P1 重做一遍 history context-update，得到 `KV(H, P1)`，再生成目标 chunk。

实测（`validation/edit_ready_mvp/local_edit_*_shard*.json`）：

| 项 | 结果 |
|---|---|
| recache 后的 cache 与 checkpoint cache 逐位相同？ | **只在 chunk 0（16/16）**；k=1、k=4 全部不同 |
| 差异落在哪 | k=1: token **[0, 1295]**；k=4: token **[0, 5183]**（cache 长 5184）；60 个 tensor 里 58/60 不同，`max_abs` 可达 5.0 |
| recache 结果与 `text_rebind` 逐位相同？ | 16/48（正好是 chunk 0 那 16 个） |
| recache 的编辑响应 `S_proxy` | 与 `text_rebind` **持平**（k=1: 0.0080 vs 0.0091），没有解锁编辑 |

两条结论：

1. **"stale prompt-conditioned visual KV" 不是本 backbone 的瓶颈机制。** 自注意力的 K/V 只由 latent `x` 决定，文本只走 cross-attention；用同样的 latent 在 P1 下重建，不会把 P0 的语义"洗掉"。
2. 但 recache **确实**复现不出 checkpoint 的 cache，而且差异精确地落在被保护的 sink 区。原因是自注意力 K/V 是**逐 block 顺序产生**的：后一个 block 写入的 K/V 取决于前一个 block 在当前 cache 内容下的注意力输出。生成路径里 sink 槽位由"带噪 latent 的第一次去噪 forward"写入并被 recompute 路径保护，而 recache 路径由一次干净的 context forward 写入。所以：

> **生成期 KV cache 是路径相关的（path-dependent），尤其在 sink 区；它不是 `(latent, timestep, prompt)` 的纯函数。**

这条性质对后续做 compact cache / cache 复用是重要的工程约束，本轮把它测出来并定位到了具体 token 区间。

### 6.8 Gate B：诚实版的可编辑性

**自动聚合只统计 `directional` prompt 且 `k > 0`**（chunk 0 是校准，不参与 pass/strong）：

| 项 | 结果 |
|---|---|
| directional 且 k>0 的 case | **20** |
| 方向正确（`S_text_rebind > 0` 且 > replay 且 > control） | **15 / 20** |
| 达到 full-regeneration 响应 ≥25%（"strong"） | **1 / 20**（strong fraction 5%） → 判定 `GO_WEAK_PROMPT_REBINDING` |
| `S_text_rebind` 均值 / `S_full_regeneration` 均值 | **0.0080 / 0.2612**（≈3%） |
| `crossattn_control` 均值 | **0.000** |

**qualitative（雨 / 短暂微笑 / 轻推镜）单独报告，不计入任何"语义编辑成功"**：

| 项 | 结果 |
|---|---|
| qualitative cases | **18** |
| 探针一致的"确实变了" | **15 / 18** |
| 含义 | 只能说明 prompt 引起了可测的变化，**不能**说明"真的下雨了 / 真的笑了" |

指标命名一律用 **"proxy-consistent prompt-induced change"**，不使用 "semantic edit success"。

### 6.9 Gate C：外部保持与解码后的泄漏

| 项 | 结果 |
|---|---|
| 未编辑 chunk 编码前 `torch.equal` | **48 / 48** |
| 解码后 chunk 之外的 max abs 变化（均值，按 target） | k=0: 0.868 / k=1: 0.521 / k=4: 0.399 |

Gate C 按方案 §10.3 判定在**编码前张量**上。解码后的泄漏是 Wan VAE 时间因果解码的性质，单独报告。`check_edit_vae_locality.py` 的单 case 逐帧诊断显示：编辑点**之前**为 0，紧邻编辑点之后的帧 mean abs 0.047，随距离单调衰减到片尾 ≈0.0005；同 prompt replay 解码后**仍逐位精确**。

### 6.10 边界连续性

| target chunk | 左边界 base→edited（相对） | 右边界 base→edited（相对） |
|---|---|---|
| 0 | —（无左邻） | 0.066 → 大幅上升（**+1258%**） |
| 1 | 0.069 → 0.075（**+8.9%**） | 0.066 → 0.090（**+37%**） |
| 4 | 0.066 → 0.067（**+1.5%**） | 0.066 → 0.079（**+19%**） |

- **左边界几乎无损**，且历史越长越无损（生成时确实是从真实缓存继续）。
- **右边界在 chunk 0 处严重破坏（+1258%）**：编辑 chunk 0 后，被锁死的 chunk 1 与它不匹配。k≥1 时右边界只有 +19%~+37%，属于方案预期的"右边界风险"，不是灾难。
- 全局 2× 规则下：左 31/32 通过，右 29/48 通过——**大部分不通过来自 chunk 0**。

### 6.11 成本：必须分成三个 regime 才说得清

| regime | 比值 | 说明 |
|---|---:|---|
| **device 端计算**（restore+encode+denoise）/ full generation | **0.167** | 纯算力上确实接近 O(1/N) |
| **端到端 · cache 已驻留**（去掉冷读）/ full e2e | **0.367** | 通过 Gate D |
| **端到端 · 冷存储**（含 1 GiB checkpoint 冷读）/ full e2e | **0.770** | **不通过 Gate D** |

明细（均值，2 seed，48 case）：

| 项 | 值 |
|---|---|
| full generation（7 chunk，不写 cache） | **3.519 s** |
| disk → CPU（page cache 已用 `posix_fadvise` 驱逐） | **1.791 s**（≈1.82 s/GiB） |
| CPU → GPU | 0.118 s |
| restore 进 pipeline | 0.012 s |
| 文本编码 P1 | 0.060 s |
| 单 chunk 去噪 | 0.514 s |
| partial 端到端 | **3.422 s** |
| cache 体积 / 峰值显存 | 0.978 GiB/chunk / 20.65 GB |

结论：**"能不能更便宜"的答案取决于 cache 在哪里。** 算力不是瓶颈；**cache I/O 与体积才是**。第一版 0.978 GiB/chunk 的"足够状态"离可交互还很远，压缩与常驻 staging 是必须的下一步。

---

## 7. Gate 判定

| Gate | 判定 | 依据 |
|---|---|---|
| A Replay | **PASS** | 24/24 逐位精确，RNG 24/24，repeat baseline 0 |
| B Editability | **PASS（很弱）** | 15/20 方向正确；strong 1/20（5%）；chunk 0 校准 10/10 逐位 |
| C Preservation | **PASS** | 48/48 `torch.equal`（编码前） |
| D Cost | **按 regime 分裂**：compute 0.167 PASS；warm e2e 0.367 PASS；**cold e2e 0.770 FAIL** | 见 §6.11 |

`summary.json`：

```json
{
  "status": "passed",
  "decision": "GO_WEAK_PROMPT_REBINDING",
  "next_step": "State reuse is proven; add light prompt-rebinding post-training."
}
```

`run_provenance`（`summary.json` 自带，来自各 run 自己记录的值）：

```json
{"git_commits": ["4c4d52f2d6b244da7566f4518edf5bcead4b1640"],
 "git_dirty_values": [false], "clean": true,
 "code_tree_sha256": ["f4482c5e5b95ef48a3aba38cccce80b18908a8bcf0f2f2f39aa16de50b45ef00"]}
```

---

## 8. 下一阶段建议（按证据强度排序，**不建议直接上 adapter**）

既然 chunk 0 已经证明重绑定实现正确，"编辑弱"就纯粹是**历史压制**问题。所以在训练之前，先做两个仍然 training-free、且能直接证伪/证实"必须训练"的实验：

1. **历史衰减 / 局部重算（最便宜、最该先做）**：重开目标 chunk 时对历史 cache 施加可调衰减，或只把最近 1 个历史 chunk 在 P1 下重算（而不是全量 recache）。本轮已证明"全量 recache 无效"，但"局部重算 + 衰减"是完全不同的干预。若 `R_k` 从 5% 抬到 30%+，说明历史权重就是主要旋钮，不必训练。
2. **右 context / bridge 条件**：让被编辑 chunk 能看到被锁死后继 chunk 的一个 lookahead 锚点（方案里的后续阶段）。这同时针对 §6.10 的右边界问题。
3. **compact edit cache + staging**：0.978 GiB/chunk、冷读 1.82 s/GiB 已经让冷路径 Gate D 不通过。任何"交互式编辑"的说法在此之前都不成立。
4. 若 1+2 都不能把 `R_k` 抬起来，再进入 **轻量 prompt-rebinding post-training**；那时它是有充分动机的，而不是替代诊断的手段。
5. 之后才是：自动 temporal mask、selective forward propagation、多轮编辑。

另外，§6.7 的路径相关性会被任何"compact cache / cache 复用"方案直接撞上，建议在压缩前先决定 sink 区的处理策略。

---

## 9. 诚实声明与已知限制

1. **没有 text-video alignment 指标**。`S_proxy` 是颜色占比 / 亮度 / 外观变化的可解释代理，不是 CLIP 相似度；`qualitative` 的三类编辑**没有被语义验证**，只能标为"探针一致的变化"。人工审阅 mp4 在 `validation/edit_ready_mvp/videos/`（`examples/` 另附三个代表样例）。
2. **只做局部 appearance / transient 编辑**，没有对象删除、永久换衣、拓扑改变、多轮编辑、右 context conditioning、bridge generation。
3. **缓存很重且未压缩**：0.978 GiB/chunk，冷读 1.82 s/GiB；这是 "sufficient" 而不是 "minimal"，本报告不把措辞往 "minimal" 上靠。
4. **解码后非逐位保持**：VAE 因果解码会让编辑点之后产生小幅、随距离衰减的泄漏；Gate C 的判定保持在编码前张量上（与方案 §10.3 一致）。
5. **边界指标只用 latent/pixel MSE + 可选 DINO 特征距离**，没有引入光流等新依赖。
6. 分辨率 256×432、21 latent 帧、2 个 seed；不是多分辨率、多长度、多 seed 的完整实验。
7. 所有数值来自实际运行；缺失输入写 `null` 并让 `status` 变为 `incomplete`。派生文件（`summary.json` / `timing.json` / `cache_manifest.json`）由 `scripts/summarize_edit_ready_mvp.py` 从原始 shard JSON 重新计算，其自身 provenance 记录的是分析时刻的 commit，而原始实验 JSON 的 provenance 固定为 `4c4d52f`。

---

## 10. 复现命令

```bash
cd /workspace/video-model/restream_mvp
.venv/bin/python -m unittest discover -s tests          # 103 项 CPU 测试

# 镜像循环 vs 上游 + 单 chunk 重开自检（秒级）
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/check_edit_streaming_equivalence.py --reviewed --gpu 0

# 全部入口
bash scripts/08_edit_ready_mvp.sh smoke                 # 2 prompt × target {0,1}
bash scripts/08_edit_ready_mvp.sh all-chunks            # 保存全部 chunk 边界
SHARDS=4 bash scripts/08_edit_ready_mvp.sh main         # 8 prompt × target {0,1,4}
SHARDS=4 bash scripts/08_edit_ready_mvp.sh rerun        # 清洁树封存重跑（等价 + A + B 两个 seed + all-chunks）

# 解码局部性诊断
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/check_edit_vae_locality.py --reviewed --gpu 0
```

输出：`validation/edit_ready_mvp/{summary,summary_seed1,summary_seed2,summary_smoke,timing,cache_manifest,vae_locality}.json` 与各 shard JSON；视频在 `validation/edit_ready_mvp/videos/`（体积原因不入 Git，清单见 `videos_manifest.json`，代表性样例在 `examples/`）。

`rerun` / `smoke` 模式只写 git-ignored 的 `validation/edit_ready_mvp/rerun*/`，跑完再把产物拷进受版本控制的位置——这样 `git status --porcelain` 在整个运行期间为空，记录下来的 `git_dirty` 才有意义。
