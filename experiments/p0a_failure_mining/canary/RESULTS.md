# Gate 1 Canary 实测：生成通过，人工校准待完成

2026-09-22，按审阅指定只运行 **P001 / P003 / P005 / P016，seed=0，各 60 秒**。四条均成功，未 OOM，未途中重试、未人为 corruption。没有合并 main，没有启动 Mini-Pilot 或完整 80 条。工程生成关已通过；由于完整视频人工标注尚未完成，**不能宣称 Gate 1 整体通过，也不能报告 detector recall 或自然失败率**。

## 真实 GPU 与文件核验

生成使用一张 NVIDIA RTX 3090 24 GiB，Torch 2.5.1+cu124，BF16，LongLive 官方 base + 冻结 LoRA。运行前校验 base/LoRA SHA-256；全部 Wan 资产和实际执行源码哈希见 [provenance](evidence/provenance.json)。

| Video | 难度 / 主体 | 生成并保存 latent | 生成 + 解码 | 运行峰值 allocated / reserved |
|---|---|---:|---:|---:|
| P001_S00 | Easy / red jacket | 251.0 s | 364.1 s | 7.82 / 8.43 GiB |
| P003_S00 | Medium / red jacket | 251.6 s | 368.8 s | 7.81 / 8.33 GiB |
| P005_S00 | Stress / red jacket | 251.9 s | 365.1 s | 7.81 / 8.33 GiB |
| P016_S00 | Easy / blue car | 251.8 s | 369.4 s | 7.81 / 8.33 GiB |

首次模型加载 96.2 秒；表中耗时不含它及权重哈希；峰值统计从模型加载完成后开始，不涵盖加载瞬间，因此不能把 8.43 GiB 当作加载阶段最低显存要求。原始 [generation.log](evidence/generation.log) 保留全部进度。

四条全部通过 [文件核验](evidence/generation_validation.json)：MP4 实际解码 **960 帧、16 FPS、480×832、60 秒**，逐帧 PTS 连续；noise/latent 均为 `[1,243,16,60,104]`，latent 为有限 BF16；每条 81 个 block 的 RNG boundary 完整。MP4 与 replay.pt 的 SHA-256 均匹配完成记录。每条 replay 约 93 MiB，不是已验证的 KV snapshot。

另外使用第一条真实 latent 的前 6 帧和官方 VAE，在独立 GPU 上比较一次性解码与连续缓存分块解码：**21 帧 = 9 + 12 帧，逐位相等，max error 0**。[实测结果](evidence/real_vae_decode_check.json)。这只验证 6 latent 帧的解码一致性，没有运行 P0-B 或重放长序列 KV snapshot。

## 自动候选与早期参考

DINOv2-small + CLIP ViT-B/32，whole frame，每 block 3 帧，共 972 个采样帧 / 324 个 block。特征提取在另一张 RTX 3090 上完成；[特征 provenance](evidence/feature_provenance.json) 记录全部本地模型文件哈希。CLIP 镜像权重已匹配[官方 SHA-256](https://huggingface.co/openai/clip-vit-base-patch32/blob/main/pytorch_model.bin) `a6308213…ff1576f`，DINO 使用已有官方哈希核验副本。

| Video | 筛选状态 | 自动建议 block / 秒 | 最初 4 blocks 的属性 margin 中位数 | 人工失败 / onset |
|---|---|---|---:|---|
| P001_S00 | unselected | — | -0.01722 | 待复核 |
| P003_S00 | candidate | 39 / 29.0625 s | -0.01881 | 待复核 |
| P005_S00 | candidate | 44 / 32.8125 s | -0.00519 | 待复核 |
| P016_S00 | audit | — | +0.03399 | 待复核 |

[完整分数](evidence/block_scores.csv) · [候选](evidence/candidate_failures.csv) · [human/auto 对照表](evidence/human_auto_comparison.csv)

自动建议只是最强持续窗口附近的位置，不能当作首次真实失败。四条 pool 的 Top 20% 每个分量实际只取前 1 条，再取并集，样本很小。审计随机选中 P016，但 **Canary 四条都必须完整观看**，包括未命中 P001。

## 预览线索：需先区分初始 mismatch 与后期 drift

以下是智能体查看每 2 秒采样图和原始 VAE 首帧的观察，**不是人工 Ground Truth，也没有写入 human_review.csv**：

- P001 的红夹克要求没有出现在首帧；首帧明显为黄色上衣。采样帧总体维持黄色，故不能因为后期仍非红色就记为 late clothing-color drift。
- P003 采样帧也从黄色上衣开始，后段有黄绿外观与视角/尺度变化。是否存在真正的后期颜色漂移，需排除光照后逐段确认；whole-frame DINO 降幅也可能来自背景和构图。
- P005 开始为偏橙色外套，与 bright red 的边界需要人工判定；未凭截图标注 failure。
- P016 采样帧中的车辆持续蓝色，但角度、光照和背景变化；不能据此声明完整视频无其他错误。

前三条初始 CLIP margin 为负是需检查早期参考的信号，不是独立的错误证明。当前相对下降筛选对“从一开始就不符合 prompt”的轨迹不一定敏感；这是 **prompt adherence 与 temporal drift 的区分**，不能据此直接计算漏检率。本轮没有为获得红色而重抽 seed 或挑选视频。

| Video | 预览 | 分数 |
|---|---|---|
| P001_S00 | [每 2 秒预览](previews/P001_S00_overview_2s.jpg) / [原始 VAE 首帧](previews/P001_S00_first_frame_raw_vae.png) | [曲线](previews/P001_S00_scores.png) |
| P003_S00 | [每 2 秒预览](previews/P003_S00_overview_2s.jpg) | [曲线](previews/P003_S00_scores.png) |
| P005_S00 | [每 2 秒预览](previews/P005_S00_overview_2s.jpg) | [曲线](previews/P005_S00_scores.png) |
| P016_S00 | [每 2 秒预览](previews/P016_S00_overview_2s.jpg) | [曲线](previews/P016_S00_scores.png) |

## 人工复核入口与下一步

原始 MP4、replay 与交互复核页保存在本机：

```text
/data/lk/深度学习/video-model-p0a/experiments/p0a_failure_mining/runs/canary60_20260922/
├── review.html
├── human_review.csv
└── outputs/P001_S00|P003_S00|P005_S00|P016_S00/video.mp4
```

从仓库根目录执行 `python -m http.server --bind 127.0.0.1 --directory experiments/p0a_failure_mining/runs/canary60_20260922 8000`，本机浏览器打开 `http://127.0.0.1:8000/review.html`。完整看完四条、导出 CSV 后重新运行 `summarize.py`。预览图不能替代观看原视频。视频与权重未写入 Git；GitHub 提供报告、曲线、预览和核验记录。

全体人审为空时 cumulative incidence 仅有 [0,1] 的缺失标签边界、precision/recall 与 hazard 均不可估计（N/A）。[hazard 表](evidence/failure_hazard.csv) 保留真实的未知状态，没有制造“风险上升”曲线。四段 hazard 与 human/auto onset 表代码已经补齐，待真实标注即可生成。

只有确认生成质量和自动指标对应关系可接受后，才考虑 5 prompts × 2 seeds Mini-Pilot，并对 10 条全部人审；目前没有启动。

## 可重现性与验证

生成过程中只新增分析/报告代码，生成核心没有改动。`generation_source.tar.gz` 留存所有 92 个生成时源码文件（已逐一核对记录哈希）；可在基线 `da0174d` 上应用 [generation_code.patch](evidence/generation_code.patch) 恢复当时的运行代码。后续精确 snapshot 重放应使用生成时源码，不能把当前增加的分析代码与旧全目录哈希混为一谈。

CPU/torch 测试 **16/16 通过**；独立绘图环境 **14 通过、2 项 torch 检查跳过**。新增测试覆盖 interval hazard 的 at-risk 分母、精确 15 秒边界与缺失标注。实时 MP4 / 官方 VAE / 特征模型 GPU 检查单独记录，不能用单元测试替代。原实验准备阶段的 `validation/` 保持为历史记录。
