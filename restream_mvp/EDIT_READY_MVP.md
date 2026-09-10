# Edit-Ready Video Generation：Generation-Time Edit Cache 可行性 MVP 结果

> 对应方案：[EDIT_READY_VIDEO_FEASIBILITY_PLAN.md](EDIT_READY_VIDEO_FEASIBILITY_PLAN.md)
> 本轮范围：**纯 inference / state reuse**。不训练、不加 adapter、不做 mask、不做双向传播、不做多轮编辑。
> 机器：4 × A800-SXM4-80GB，PyTorch 2.5.1+cu124，LongLive v1.0（NVlabs/LongLive `e52d9ef6865d843282a6b5e9d46d03b35f88929d`）+ 官方 `longlive_base.pt` + 冻结的官方 rank-256 LoRA。

---

## 0. 一句话结论

**YES（机制层面）**：LongLive 每个 AR chunk 开始前的内部状态可以被完整保存并在之后单独恢复——用原 prompt 重开目标 chunk 是**逐位精确**的（16/16 个 case，latent 与解码后像素都 `torch.equal`），未编辑 chunk 在编码前也**逐位不变**（32/32），单 chunk 编辑的生成耗时只有整条重生成的 **16.5%**。

**但（能力层面）**：新 prompt 对重开 chunk 的作用**方向正确却明显偏弱**——32 个受控编辑里 23 个方向正确，但平均只恢复了 full regeneration 响应幅度的 **2.7%**，只有 1 个 case 达到 ≥25%。

按方案 §13 矩阵，本轮判定为 **`GO_WEAK_PROMPT_REBINDING`**：状态复用问题解决，下一步该做轻量 prompt-rebinding（而非 bridge / propagation）。

---

## 1. 本轮回答的 4 个问题

| 问题 | 结论 | 证据 |
|---|---|---|
| Q1 缓存状态是否足以复现原 chunk | **是，逐位精确** | 16/16 case `torch.equal`；RNG state 重放 16/16 精确复现 chunk 内噪声 |
| Q2 新 prompt 能否只影响被重开的 chunk | **能，但作用弱** | 32/32 case 未编辑 chunk `torch.equal`；新 prompt 只在重置 text K/V 后生效（control 32/32 与原 prompt 逐位相同） |
| Q3 边界是否接得上 | **左边界好、右边界有风险** | latent 左边界断裂 MSE 0.0764→0.0817（+7.0%），右边界 0.0698→0.0950（+36.0%）；解码后右边界仍有余波 |
| Q4 局部编辑是否更便宜 | **是** | 单 chunk 生成 0.578 s vs 整条重生成 3.507 s，`R_time = 0.165`；端到端（含 1 GiB cache 冷读 + 解码）0.401 |

---

## 2. Step 1：LongLive 生成循环与必须保存的状态

### 2.1 循环结构（`code/LongLive/pipeline/causal_inference.py`）

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

- 单步 forward：`wan/modules/causal_model.py::_forward_inference` L891-1050。注意 **cache 更新是延迟的**：各 block 只返回 `cache_update_info`，全部 30 个 block 算完后由 `_apply_cache_updates` 一次性写回（L1043-1044）。
- 自注意力 cache 逻辑：L228-311，局部窗口滚动 + `sink_size * frame_seqlen` 个 sink token 固定保留。
- 文本 cross-attention cache：`wan/modules/model.py::WanT2VCrossAttention` L161-181，`is_init=False` 时用当前 context 计算 K/V 并写入，之后一直复用。

### 2.2 必须在 chunk 边界保存的对象

| 对象 | 位置 | 本轮是否必须 | 说明 |
|---|---|---|---|
| `kv_cache1`（30 × {k, v, global_end_index, local_end_index}） | pipeline | **必须** | 自回归视觉历史；sink token 就是它的前 `sink_size*frame_seq_length` 个槽位，**不存在独立 sink 对象** |
| `crossattn_cache`（30 × {k, v, is_init}） | pipeline | **必须** | 文本绑定；`is_init` 决定新 prompt 能否生效 |
| `current_start_frame` | 循环变量 | **必须** | 决定 RoPE 帧偏移与 cache 写指针 |
| chunk 初始噪声 | 全局 noise | **必须** | 它在整条视频开始时就抽好了，无法从 chunk 边界 RNG 反推 |
| chunk 内 `randn_like` 的 3 次抽样 | 循环内 | 必须（可验证） | 记录后可直接比对 RNG state 是否完整 |
| CPU/CUDA RNG state | 全局 | **必须** | 恢复后应精确重现上面 3 次抽样 |
| prompt 条件（`prompt_embeds`） | 文本编码器 | 必须 | 同 prompt replay 直接复用；换 prompt 时重算 |
| `latent_history` | output | 需要 | 拼接、边界指标、体积溯源 |
| scheduler / RoPE / position 状态 | scheduler、model.freqs | 不需保存，只需记录溯源 | 由 config 完全决定，无 per-chunk 可变状态 |
| `sink_cache` | — | 不存在 | 本 backbone 中 sink 就在 `kv_cache1` 内 |

---

## 3. 实现

### 3.1 新增文件

```text
restream_mvp/
├── EDIT_READY_VIDEO_FEASIBILITY_PLAN.md   # 方案原文（sha256 09a6cb6d…）
├── EDIT_READY_MVP.md                       # 本文件
├── configs/edit_ready_mvp.yaml             # 8 个 prompt 编辑对 + gate 阈值
├── restream/edit_cache.py                  # EditCheckpoint schema、save/load、capture/restore、RNG
├── restream/edit_replay.py                 # 镜像 chunk 循环、replay_chunk、拼接、解码
├── restream/edit_metrics.py                # latent/pixel/边界指标 + 颜色/亮度代理 + 可选 DINO
├── restream/edit_media.py                  # PyAV 视频写出与并排对比
├── restream/edit_experiment.py             # case 展开/shard、确定性噪声、provenance、严格 JSON
├── scripts/check_edit_streaming_equivalence.py  # 镜像循环 vs 上游 + 单 chunk 重开自检
├── scripts/check_edit_cache_replay.py      # 实验 A（Gate A）
├── scripts/run_chunk_edit.py               # 实验 B（Gate B/C/D + 对照组）
├── scripts/check_edit_vae_locality.py      # 解码后编辑局部性诊断
├── scripts/summarize_edit_ready_mvp.py     # summary.json / timing.json / cache_manifest.json
├── scripts/08_edit_ready_mvp.sh            # smoke / all-chunks / main 入口
└── tests/test_edit_cache.py                # 23 项 CPU 回归测试
```

独立入口，未改动任何 R0/R1 训练逻辑；`restream/runtime.py` 只加了一处向后兼容改动（几何参数可来自 `data` 或 `generation`）。

### 3.2 Checkpoint schema（`restream/edit_cache.py`）

`EditCheckpoint` 覆盖方案 §6 全部字段，并额外记录：

- `model_hash` / `config_hash` / `prompt_hash` / `code_commit`：拒绝跨模型、跨配置误用（`verify_checkpoint_identity`）。
- `provenance.tensors`：递归记录每个 tensor 的 shape/dtype/device。
- `denoise_noise`：chunk 内 3 次噪声抽样，用于**直接验证 RNG state 完整性**。
- `sink_cache=None` 并显式说明原因（sink 在 `kv_cache1` 内）。
- `scheduler_state` / `position_state`：记录 timesteps/sigmas 与 RoPE 的摘要，说明它们无 per-chunk 状态。

保存用 `torch.save` 写 tmp 再 `os.replace`，返回 sha256/bytes 的 manifest entry；加载用 `weights_only=True` 并校验 sha256。**恢复时会为设备新建 tensor 副本**，因此同一个 checkpoint 可以反复 replay 而不会被前一次 replay 污染（这条有专门的回归测试）。

### 3.3 编辑机制：为什么必须重置 text cross-attention

`crossattn_cache["is_init"]=True` 时，forward 直接复用缓存的文本 K/V，**根本不会看新的 `prompt_embeds`**。因此：

- same-prompt replay：保留 cache → 逐位复现；
- local edit：清零 `is_init` 与 K/V，让新 prompt 重新绑定 → 编辑生效；
- **control 组**：给新 prompt 但保留旧 K/V → 结果应与 replay **逐位相同**。

实测 control 在 32/32 个 case 上与 replay `torch.equal`，`S_proxy` 恒为 0.000：这既证明新 prompt 的唯一入口就是 text cross-attention，也证明 local edit 的全部差异都来自 prompt 重绑定，而不是重置动作本身。

### 3.4 关键 bug（已在提交前修复，值得记录）

第一版镜像循环把**上一拍的干净预测** `denoised_pred` 送进下一次 forward，而上游送的是 `add_noise` 之后的**带噪 latent**。结果：镜像循环自身完全自洽（因此 replay 看起来"精确"），但与上游 `inference()` 的 `max_abs_diff = 11.06`、cosine 0.31。

没有用"看起来对"糊过去：新增 `scripts/check_edit_streaming_equivalence.py` 把镜像循环与上游逐步对比（forward 输入/输出、KV cache、cross-attn cache），定位到第一次 `add_noise` 之后；修复后 `stream_generate` 与上游 **逐位相同**。这也是本报告里 `upstream_equivalence` 那一行的来源。

---

## 4. 实验设置

- 视频：21 latent 帧（81 像素帧 @16 fps，256×432），AR block = 3 帧 ⇒ **7 个 chunk**。
- 目标 chunk：`[1, 4]`（避开 chunk 0 无左边界、chunk 6 为末块）。
- prompt 对：8 组局部外观/瞬时编辑（换色、明暗、冷暖、短暂微笑、轻微推镜、局部降雨）。
- 随机性：固定 seed；主实验另跑 `--seed-stride 1` 做**第二 seed 复现**（共 32 个编辑 case）。
- 4 卡按 prompt 分片（每卡 1 进程，避免同时跑两个实验污染计时）。
- 新增指标：
  - `S_proxy`：颜色占比 / 亮度 / 外观变化的**方向性**代理（正值 = 朝新 prompt 方向变化）。**不是** text-video alignment（本机无法访问 HuggingFace 取 CLIP，报告里不冒充有该指标）。
  - `ratio_to_full_regeneration = S_proxy(local) / S_proxy(full regen)`：编辑幅度相对整条重生成的恢复比例。
  - 冻结 DINOv2-S 特征距离用于边界外观连续性（可选 `--dino`）。

---

## 5. 结果

### 5.1 镜像循环等价性与单 chunk 重开自检

`validation/edit_ready_mvp/streaming_equivalence.json`（2 chunk 小视频）

| 检查 | 结果 |
|---|---|
| `stream_generate` vs 上游 `inference()` | `exact=True`，`max_abs=0.0` |
| 从 `S_{k-1}` 重开 chunk（同 prompt） | `exact=True` |
| checkpoint 内 RNG state 重现 chunk 内 3 次噪声 | `rng_exact=True` |
| 每 chunk cache 体积 | `1,049,887,200` B（≈0.978 GiB） |

### 5.2 实验 A：Gate A（16 个 case，8 prompt × 2 target）

`validation/edit_ready_mvp/replay_main_shard*.json`

| 指标 | 实测 |
|---|---|
| 逐位精确 replay | **16 / 16** |
| RNG state 精确重现 | **16 / 16** |
| latent MSE / cosine / max\|Δ\| | 0.0 / 1.0 / 0.0 |
| repeat-noise baseline（同 cache 重放两次） | 0.0（本来就无需容忍噪声） |
| 解码后像素 | `exact=True`，PSNR = inf |

**Gate A：PASS（exact 级别，不是"在噪声范围内"）。**

### 5.3 全部 chunk 边界缓存（heavy-cache 路径）

`validation/edit_ready_mvp/replay_all_chunks_shard0.json`

7 个 chunk 全部保存 = **6.9 GB**，逐 chunk 体积随 `latent_history` 增长从 1,054,833,658 B 略增到 1,055,995,293 B；对 chunk 1 与 chunk 4 的 replay 仍然逐位精确。按方案 §4"先保存足够多的状态，之后再研究 compact cache"，本轮不做压缩。

### 5.4 实验 B：Gate B/C/D（32 个编辑 case）

`validation/edit_ready_mvp/summary.json`、`timing.json`、`cache_manifest.json`

| Gate | 实测 | 判定 |
|---|---|---|
| B 可编辑性（方向正确） | 23 / 32；`S_proxy` 均值 **0.00577** vs full regeneration **0.21382**（2.7%）；≥25% 幅度的只有 1 个（`car_red_to_black` seed43 chunk1，ratio 0.416） | PASS（但弱） |
| C 外部保持 | **32 / 32** `torch.equal`（编码前 latent） | PASS |
| D 成本 | 整条 3.507 s vs 单 chunk 0.578 s，`R_time_generation = 0.165`，端到端 0.401 | PASS |

逐 case 明细（按 ratio 排序，`rep`/`ctl`/`pres` 为 `torch.equal` 结果）：

| case | S_proxy(local) | S_proxy(full) | ratio | rep | ctl | pres | R_time |
|---|---:|---:|---:|:--:|:--:|:--:|---:|
| car_red_to_black seed43 c1 | +0.10466 | 0.2516 | **0.416** | 1 | 1 | 1 | 0.165 |
| dress_red_to_blue seed50 c4 | +0.00931 | 0.0725 | 0.128 | 1 | 1 | 1 | 0.165 |
| park_light_rain seed54 c4 | +0.00179 | 0.0270 | 0.066 | 1 | 1 | 1 | 0.168 |
| dress_red_to_blue seed50 c1 | +0.00489 | 0.0914 | 0.054 | 1 | 1 | 1 | 0.164 |
| lighting_darker seed44 c1 | +0.00870 | 0.2001 | 0.043 | 1 | 1 | 1 | 0.164 |
| …（其余 27 个 case 见 shard JSON） | | | ≤0.043 | 1 | 1 | 1 | ≈0.165 |
| 最弱：lighting_warm_to_cool seed45 c4 | −0.00001 | 0.4575 | −0.000 | 1 | 1 | 1 | 0.166 |

两个 seed 分开看：seed 1 → 10/16 方向正确、1 个 strong；seed 2 → 13/16 方向正确、0 个 strong。**"方向对但幅度弱"在第二个 seed 上复现。**

可以明确说清楚的因果结论：
1. 重开 chunk 与原始 chunk **逐位相同**（同 prompt）；
2. 未编辑 chunk 编码前 **逐位不变**；
3. 新 prompt 若沿用旧文本 K/V，结果与同 prompt replay **逐位相同**——所以可见的编辑效果 100% 来自 prompt 重绑定；
4. 重绑定后 chunk 确实变了（latent MSE 1e-5 ~ 0.43），但幅度相对整条重生成平均只有 2.7%。

对本轮最重要的一句话解释：**kV cache 中的视觉历史对重开 chunk 的约束远强于新 prompt 的文本条件**，所以"能 reopen"不等于"能大幅改写"。

### 5.5 边界连续性

| 边界 | base MSE | edited MSE | 相对增加 | 满足 2× 规则的 case |
|---|---:|---:|---:|---:|
| 左 `C_{k-1} → C'_k` | 0.07641 | 0.08175 | **+7.0%** | 31 / 32 |
| 右 `C'_k → C_{k+1}` | 0.06981 | 0.09496 | **+36.0%** | 29 / 32 |

方向与方案 §Q3 的预期一致：**左边界几乎无损，右边界明显更差**（被锁死的 `C_{k+1}` 与新的 `C'_k` 不匹配）。绝对量级不大，但右边界相对增幅是左边界的 5 倍，且有 3 个 case 超出 2× 规则。

### 5.6 解码后的编辑局部性（`check_edit_vae_locality.py`）

Gate C 保证的是**编码前** latent 逐位不变；Wan VAE 解码器是时间因果的，因此解码后的像素仍可能"漏"到后面。以最强的 `car_red_to_black` seed43 chunk1（像素帧 9–20）为例：

| 区间 | mean abs | p99 | max |
|---|---:|---:|---:|
| 同 prompt replay（全片） | 0.0 | 0.0 | 0.0 |
| chunk **之前**的所有帧 | **0.0** | 0.0 | 0.0 |
| chunk 之内 | 0.0657 | 0.764 | 1.0 |
| chunk **之后**的所有帧 | **0.00453** | 0.092 | 0.996 |

逐帧看，紧邻的 frame 21 为 0.047，随后单调衰减到片尾的 ≈0.0005。也就是说：
- **向后完全没有泄漏**；
- 向前有一处**小但非零、随距离衰减**的泄漏，少数像素可以变化很大（max≈1.0）。

这是本轮新发现的一个工程风险：即使 latent 逐位保持，**解码后的视频在编辑点之后也不是逐位保持的**。

### 5.7 成本

| 项 | 值 |
|---|---|
| full generation（整条 7 chunk，新 prompt，无 cache 写） | 3.507 s |
| partial edit 生成（读 cache 0.071 s + 生成 0.507 s） | 0.578 s |
| `R_time_generation` | **0.165** |
| `R_time_end_to_end`（含 1 GiB cache 冷读 + 解码） | **0.401** |
| cache 冷读 | 0.245–0.314 s / GiB |
| cache 体积 | 1.0499e9 B ≈ 0.978 GiB / chunk |
| 峰值显存 | 18.30 GB |
| **写 checkpoint 的代价** | 带 2 个 cache 写的首轮生成 7.93–8.13 s，比无写的 3.51 s 多 ≈2.2 s/个 |

注意最后一行：**首轮生成时的 cache 落盘开销（≈2.2 s/1 GiB）与 chunk 生成本身同量级**。这是"重 cache 第一版"的真实代价，本报告不掩饰；压缩/异步写盘属于后续工作。

---

## 6. Gate 判定

| Gate | 判定 | 依据 |
|---|---|---|
| A Replay | **PASS** | 16/16 逐位精确，RNG 16/16 |
| B Editability | **PASS（弱）** | 23/32 方向正确；均值仅恢复 full regen 的 2.7%；strong 1/32 |
| C Preservation | **PASS** | 32/32 `torch.equal`（编码前） |
| D Cost | **PASS** | `R_time = 0.165 < 0.5` |

`summary.json`：

```json
{
  "status": "passed",
  "decision": "GO_WEAK_PROMPT_REBINDING",
  "next_step": "State reuse is proven; add light prompt-rebinding post-training."
}
```

对应方案 §13 矩阵：`Replay PASS / Edit PASS / 边界偏差` ⇒ **核心可行**；但由于编辑幅度弱，本轮更精确的落点是矩阵中"状态能 reopen，但新 prompt 无法强力重绑定"这一档。

---

## 7. 诚实声明与已知限制

1. **没有 text-video alignment 指标**。本机无法访问 HuggingFace 取 CLIP，因此 `S_proxy` 是颜色占比 / 亮度 / 外观变化的**可解释代理**，不是 CLIP 相似度。人工审阅用的 mp4 已保存在 `validation/edit_ready_mvp/videos/`。
2. **只做局部 appearance / transient 编辑**，没有对象删除、永久换衣、拓扑改变、多轮编辑、右 context conditioning、bridge generation。
3. **缓存很重**：0.978 GiB/chunk，且落盘约 2.2 s/个。第一版按方案要求"先证明能 reopen"，不做压缩。
4. **解码后非逐位保持**：见 §5.6，这是 VAE 因果解码的性质，不是 state 复用的问题；Gate C 的判定保持在编码前张量上（与方案 §10.3 一致）。
5. **边界指标只用 latent/pixel MSE + 可选 DINO 特征距离**，没有引入光流等新依赖。
6. 分辨率 256×432、21 latent 帧、单 seed 主实验 + 单 seed 复现；不是多分辨率、多长度、多 seed 的完整实验。
7. 本轮的 `summary.json` 数值全部来自实际运行；没有任何预填值，缺失输入会写 `null` 并让 `status` 变为 `incomplete`。

---

## 8. 复现命令

```bash
cd /workspace/video-model/restream_mvp
.venv/bin/python -m unittest discover -s tests          # 94 项 CPU 测试

# 镜像循环 vs 上游 + 单 chunk 重开自检（秒级）
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/check_edit_streaming_equivalence.py --reviewed --gpu 0

# 全部入口
bash scripts/08_edit_ready_mvp.sh smoke        # 2 prompt × 2 target
bash scripts/08_edit_ready_mvp.sh all-chunks   # 保存全部 chunk 边界
SHARDS=4 bash scripts/08_edit_ready_mvp.sh main        # 8 prompt × 2 target，4 卡
SHARDS=4 EDIT_TAG=edit_seed2 EDIT_PREFIX=validation/edit_ready_mvp/local_edit_seed2 \
  bash scripts/08_edit_ready_mvp.sh main-edit --seed-stride 1

# 解码局部性诊断
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/check_edit_vae_locality.py --reviewed --gpu 0
```

输出：`validation/edit_ready_mvp/{summary,summary_seed1,summary_seed2,summary_smoke,timing,cache_manifest,vae_locality}.json` 与各 shard JSON；视频在 `validation/edit_ready_mvp/videos/`（体积原因不入 Git，清单见 `videos_manifest.json`，另附一个代表性对比在 `examples/`）。

---

## 9. 下一阶段建议（仅在 GO 之后讨论）

按本轮证据排序：

1. **轻量 prompt-rebinding post-training**（最值钱）：状态复用已经被证明是逐位精确的，瓶颈是"新文本对重开 chunk 的影响太弱"。最小实验是在冻结主干上用注入的 edit prompt 训练一个极小的 prompt→context 适配器，使 `S_proxy(local)` 恢复到 full regeneration 的量级。
2. **右边界 bridge / selective forward propagation**：右边界相对增幅 +36%、解码后还有前向余波；若要让编辑"接得上"，需要让被编辑 chunk 有条件地影响其后继。
3. **自动 temporal mask 与 compact edit cache**：0.978 GiB/chunk + 2.2 s 落盘不适合交互；可先做 kv/跨注意力缓存压缩与异步写盘。
4. 多轮编辑、空间 mask、Reality Memory 结合，均等 1–3 有结论后再谈。
