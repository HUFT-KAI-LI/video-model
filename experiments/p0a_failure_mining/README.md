# P0-A：LongLive 自然失败挖掘

**进度更新：** [Gate 1 Canary](canary/README.md) 记录四条真实 60 秒轨迹；[实际结果](canary/RESULTS.md) 与旧的准备阶段验证分开。Mini-Pilot / 全量实验尚未放行。

本目录实现用户提供的 P0-A 设计：20 个单 prompt × 4 个 seed，默认每条自然连续生成 60 秒，分析 15/30/45/60 秒前缀。四类主体各 5 个 prompt；难度为 Easy 8、Medium 8、Stress 4。只做自然 rollout、自动候选筛选和人工确认，不训练、不注入 corruption、不在生成途中重采样。

**以下为 da0174d 准备阶段记录；最新 GPU 实测见上述 Canary 报告。** CPU 协议测试、上游观察回调和小型随机权重 VAE 解码测试见 [validation](validation/)。没有完成真实 GPU rollout、视觉指标模型推理、自然失败人工标注或 GO/NO-GO 判断。真实 480×832、60 秒的耗时和显存仍需在目标 GPU 上测量。

## 审阅入口

| 文件 | 审阅重点 |
|---|---|
| [prompts.jsonl](prompts.jsonl) | 类别、难度、目标属性、对照描述，不扩写或中途切换 prompt |
| [generation_config.yaml](generation_config.yaml) | 固定 LongLive v1.0 / 1.3B，官方冻结 LoRA，480×832 / 16 FPS，seed 0–3 |
| [generate_pool.py](generate_pool.py) / [backend.py](backend.py) | 默认仅建计划；显式执行时生成一次；模型哈希、实际帧数、随机状态、完成标记 |
| [extract_features.py](extract_features.py) | DINOv2 外观余弦、CLIP target-minus-alternatives margin、逐 block 中位数 |
| [detect_candidates.py](detect_candidates.py) | 相对参考下降、持续性、局部变化、pool 内 Top 20%、随机阴性审计 |
| [summarize.py](summarize.py) | 人审校验、缺失标签边界、首次失败目录、正常对照匹配、分层统计 |
| [materialize_snapshot.py](materialize_snapshot.py) | 原轨迹逐块重放一致性验证；捕获真实 KV / cross-attention cache 和 RNG |

相对上游仅增加两个可选推理参数：`block_callback` 在每块 clean-context recache 后观察状态；`decode_video=False` 返回 latent 供连续缓存解码。默认原推理行为保留。改动同步到 `restream_mvp/longlive.patch`，不调用旧 Reality Memory、anchor、corruption 或训练路径；仅复用冻结官方模型加载器。

## 环境与资产

在仓库根目录执行。建议使用已安装 LongLive、PyTorch/CUDA 和 flash-attn 的专用环境；基础安装见 [现有复现说明](../../restream_mvp/README_REPRODUCE.md)。之后安装本实验额外依赖：

```bash
python -m pip install -r experiments/p0a_failure_mining/requirements.txt
python experiments/p0a_failure_mining/check_ready.py
python -m unittest discover -s experiments/p0a_failure_mining/tests -v
```

`generation_config.yaml` 的模型路径相对于仓库根目录，也接受绝对路径。需要已有完整 Wan2.1-T2V-1.3B（含 UMT5 tokenizer/text encoder/VAE）和 LongLive-1.3B 官方 base/LoRA。代码不会自动下载生成权重。生成前强制校验官方 base/LoRA SHA-256，同时记录 Wan 权重哈希、上游源码哈希和运行环境。`check_ready.py --verify-weights` 可提前执行完整检查。

默认 DINOv2-small 和 CLIP ViT-B/32 在第一次特征提取时通过 Transformers 下载；离线运行需预先缓存。配置可指定本地模型目录或固定 HF revision，输出记录解析后的 commit；不要在同一次实验内改变模型、reference 或 crop 协议。whole-frame 模式只是粗筛，不能解释为主体 identity ground truth。`--boxes boxes.json` 支持外部主体检测/跟踪后的 crop：`{video_id: {"frame_index": [left, top, right, bottom]}}`，坐标为原图像素，每个采样帧都必须有合法框。当前不包含自动检测/跟踪器。

生成阶段按上游方式动态换入 text encoder；VAE 在生成结束后移入 GPU，并把 generator 移回 CPU。连续保留 VAE temporal cache，直接编码 MP4，避免整条 RGB 视频驻留显存。设备调度和真实 60 秒吞吐尚未做 GPU 验证，不能据此承诺 RTX 3090 或其他显卡上的运行性能。

## 执行顺序

以下命令中 `RUN` 必须保持一致，建议指向空间充足的数据盘。为可审阅性，默认建计划不会生成视频。

```bash
RUN=experiments/p0a_failure_mining/runs/pilot60
python experiments/p0a_failure_mining/generate_pool.py --run-dir "$RUN"

# 审阅后先生成一条完整长轨迹，测量成本/验证模型和 VAE。
python experiments/p0a_failure_mining/generate_pool.py --run-dir "$RUN" --execute --video-id P001_S00

# 再执行全部计划；只跳过哈希核验通过的已完成轨迹。
python experiments/p0a_failure_mining/generate_pool.py --run-dir "$RUN" --execute
python experiments/p0a_failure_mining/extract_features.py --run-dir "$RUN"
python experiments/p0a_failure_mining/detect_candidates.py --run-dir "$RUN"
```

如果 60 秒成本过高，复制配置，将 `duration_sec` 改为 45，使用新的 `--run-dir` 与 `--config` 建立独立实验；不会把 45 秒结果计入 60 秒分母。不可通过更改 FPS 把短轨迹当作长视频。

本阶段不并发写入同一 run。执行失败会保留 `started.json` / `failed.json`，再次执行不会隐式重试该轨迹。先调查原因、归档失败 run，再在新 run 中明确重跑；不能筛选多次随机尝试中的“最好视频”。同一 config/prompt 池、模型和源码版本固定在 run 内，输出损坏或版本改变直接报错。

## 时间 / block 定义

所有 block/frame 索引均从 **0** 开始。Wan VAE 输出帧数为 `4 × latent_frames − 3`；每个 AR block 是 3 个 latent 帧，首块 9 个像素帧，后续每块 12 帧。block 的像素帧区间为 `[start,end)`：

- block 0：frame `[0,9)`，`[0,0.5625)` 秒；
- block 1：frame `[9,21)`，`[0.5625,1.3125)` 秒；
- 60 秒需要 243 latent 帧 / 81 blocks，解码为 969 帧后只保留前 960 帧；
- 45 秒需要 183 latent 帧 / 61 blocks，解码为 729 帧后只保留前 720 帧。

最后一个 block 的可见区间按实际保留帧数截断，完整 latent 仍保存用于状态重建。不要把 `num_output_frames`（上游的 latent 数）当成像素帧数。`block_map.json` 是后续分析和复核的统一映射。

## 自动筛选与人工复核

每块均匀取 3 帧，默认用最初 4 个 block 的归一化 DINO 特征均值作为参考；如果早期生成已错误或参考不稳定，必须在人审 notes 中记录，可标 `UNCERTAIN`，不要把分数正常当作健康历史。CLIP 对目标和多个替代描述计算余弦 margin，不用 softmax 概率。

逐 block 中位数经早期参考相对下降、局部变化两种方法评分。连续 3 blocks 的最小下降值用于排除孤立低分；分别对主体、属性及其变化分量选池内前 20%，保留并列值，取并集。全零/无下降不选。自动建议 onset 是高分持续窗口起点，**并不声称是第一次真实错误**；第一次错误必须人工定位。参数只服务候选挖掘，没有调参或校准结果。

从未命中视频中按固定 seed 均匀抽取 25% 审计样本。可直接打开 `RUN/review.html`，或本机启动 `python -m http.server --directory "$RUN" 8000`，通过 `http://localhost:8000/review.html` 访问。页面提供视频、block 定位、预览和表单；更换视频时暂存，关闭前点击导出，替换 run 内 `human_review.csv`。再次运行筛选不会覆盖人工填写的 CSV。

复核 `candidate` 与 `audit`；要发布完整 cohort 的 failure rate / recall，还需复核所有剩余视频。人工填写：

- `failure_confirmed`：YES / NO / UNCERTAIN；空白 = 未复核。
- YES 必须有固定 taxonomy、`onset_kind`、`pre_onset_normal` 和置信度。
- abrupt 必须标第一个明显错误 block；可额外指定该 block 内的精确帧。
- gradual_drift 可留空 onset，但不得作为已定位 abrupt state 进入 P0-B。
- NO 必须给 `reviewed_until_sec`；只有完整看过且无失败的视频才可作正常 control。
- block 0 已失败、pre-onset 不正常或不确定，都不能作为可恢复的首次失败状态。

```bash
python experiments/p0a_failure_mining/summarize.py --run-dir "$RUN"
```

报告输出四张主图和两张分数曲线示例，以及 `failure_catalog.csv`、`matched_controls.csv` 和 `summary.json`。还没有真实标注时不会生成伪造实验图或把失败率填写为零。

## 统计边界

`P(first failure before T)` 是累计首次事件比例，本来就不会随 T 减少；其上升本身不能证明瞬时 / 单 block failure hazard 随时间增大。当前报告用于比较增加观察时长带来的已确认失败收益，不自动作出该因果或 hazard 结论。多 seed 共享 prompt，后续推断应按 prompt 处理聚类，不把 80 条当成 80 个独立 prompt。

未复核、UNCERTAIN、复核范围不足和跨过前缀边界的 block onset 均保留 unknown。报告给出 `[confirmed/N, (confirmed+unknown)/N]` **识别边界，不是置信区间**；仅所有 eligible 视频在该前缀都已确定时给单一 failure rate。仅标 block 时 onset 是时间区间，不能伪装成精确秒数。未完成生成的视频单独报告，不能偷偷算正常。

precision 仅在该检测器的全部命中项已有确定标签时发布；完整 recall 仅在全 cohort 确定时发布。`reviewed_subset_recall` 明确是被选择子集上的描述值，不能当作全池召回率。审计阴性有助于发现漏检，但当前不以小样本抽检估计替代正式 recall。三个检测器（subject、attribute、combined）的评价目标均为“任一种确认失败”，不是逐 taxonomy 的专用识别器。

控制组按类别、难度、主体、motion_group、同一 block 时间匹配，优先同 prompt 的另一 seed，不重复使用同一正常视频。目标 control 数为失败数的 25%（约为失败+正常状态总数的 20%），不足时报告 shortfall，不放宽匹配规则或伪造正常状态。

## P0-B 状态准备

每条成功轨迹保存 `replay.pt`：完整原始噪声、全部生成 latent、推理前及每块后的 CPU/CUDA RNG。仅保存视频、seed 或 clean-prefix 重放不足以保证同一原始 cache 状态，所以未验证的 replay capsule **不会**被标作 `p0b_eligible`。

人工确认某条 onset block 为 24 时，重放到 pre-onset block 23：

```bash
python experiments/p0a_failure_mining/materialize_snapshot.py \
  --run-dir "$RUN" --video-id P001_S00 --after-block 23
python experiments/p0a_failure_mining/summarize.py --run-dir "$RUN"
```

重放使用原始噪声与 RNG，逐 block 要求 latent **逐位相等**且随机状态相同，然后保存真实 KV / cross-attention cache、prefix latent、未来原始噪声、下一个 latent 位置和 RNG。模型 / 源码 / runtime 必须完全一致。GPU 算子非确定性可能导致核验失败；失败时不放宽判据、不标作已恢复状态。此路径只重建原始状态，不采样新 future 或做恢复实验。匹配 control 也用相同命令重建指定 block。

只有人工确认 abrupt、pre-onset 正常、存在前块且 snapshot 哈希验证通过，才标 `p0b_eligible=True`。30–50 个有效失败状态是实验目标，不是本次准备工作的已完成成果。快照体积可能达数 GB；默认只保存轻量 replay capsule，对入选状态按需重放，避免对 80×81 个 block 全量存储 KV。

## 输出与已知限制

所有运行产物写入 `--run-dir`：`plan.json` / `block_map.json` / `provenance.json`、`outputs/<video_id>/`、`outputs/feature_cache/`、`manifests/`、`human_review.csv`、`review.html`、`figures/`。`runs/`、权重、视频、latent、cache 不提交 Git。单条 BF16 noise+latent 约 93 MiB，80 条约 7.4 GiB，另外需要 MP4、feature、选择性 KV snapshot 和足够模型内存。

真实 GPU rollout、GPU replay 位级一致性、CLIP/DINO 下载及真实特征前向、候选高召回、人工复核负担和失败率随时长的表现均待实测。whole-frame 特征容易受背景与视角影响；对难例可提供 subject boxes 另跑明确标注的 crop 协议。当前为串行单 GPU 准备版本，不支持同目录多进程调度或实验中途改变协议。
