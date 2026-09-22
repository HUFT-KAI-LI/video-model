# Gate 1：4 条 60 秒真实 Canary

本轮只生成 P001_S00（clothing/easy）、P003_S00（clothing/medium）、P005_S00（clothing/stress）、P016_S00（vehicle/easy），不合并 main，不进入 10 条 Mini-Pilot 或完整 80 条。保留 explicit persistence prompt；车辆/箱包只保留自然的颜色对照，去掉 `without a body/shell`。

在仓库根目录、LongLive Python 环境执行。以下保留本轮运行目录名；如需独立复现，请给 RUN 换一个新目录，不要在已完成目录重新生成：

```bash
RUN=experiments/p0a_failure_mining/runs/canary60_20260922
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python -u experiments/p0a_failure_mining/generate_pool.py \
  --config experiments/p0a_failure_mining/canary/generation_config.yaml \
  --prompts experiments/p0a_failure_mining/canary/prompts.jsonl --run-dir "$RUN" --execute
python experiments/p0a_failure_mining/canary/verify_outputs.py --run-dir "$RUN"
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 python experiments/p0a_failure_mining/extract_features.py \
  --run-dir "$RUN" --dino-model restream_mvp/models/dinov2-small \
  --clip-model restream_mvp/models/clip-vit-base-patch32
python experiments/p0a_failure_mining/detect_candidates.py --run-dir "$RUN"
python experiments/p0a_failure_mining/summarize.py --run-dir "$RUN"
python experiments/p0a_failure_mining/canary/make_review_artifacts.py --run-dir "$RUN"
```

实际机器有 4×RTX 3090，生成只用一张；特征提取用另一张。报告绘图可使用有 NumPy/Matplotlib/PyYAML 的 CPU 环境。`--dino-model` / `--clip-model` 只提供同一模型的本地副本，要求 provenance 声明匹配计划中的模型 ID，并记录文件哈希；同名声明本身不能证明权重官方性，需另核验官方 SHA-256。本次两者权重已与官方记录核对。

本次修正了原冻结加载器没有执行 `component_devices` 的问题：P0-A 的 text encoder 和 VAE 先留 CPU，generator 在 GPU；text encoder 使用上游 DynamicSwap，解码阶段再卸载 generator / 移入 VAE。默认不传 component_devices 的旧路径不变。没有修改 checkpoint、去噪调度、prompt 条件或生成中途 retry。

## 必须完成的人审

完整观看四条 60 秒视频，不能只看采样图、候选片段。打开 run 的 `review.html`，导出并保存 `human_review.csv`。所有视频都需要复核，包括 `unselected`。智能体对图片的观察只记作预览线索，不能写入人工 Ground Truth。

特别检查初始 0–3 秒是否满足目标属性。如果开始就不符合 red jacket，记录初始 mismatch，不能当成 late drift，也不能标成可恢复 pre-failure state。相对早期 reference 的指标可能对“始终错误”的轨迹给出正常的变化分数，这和漂移漏检不是同一件事。

人审后 `human_auto_comparison.csv` 给出每条视频的 human failure、auto candidate、human onset 区间、auto onset。候选只在四条内部排名，不能把这个 cohort 的 recall 当成已校准的完整 80 条表现。Gate 1 的人审与自动对齐未确认前不进入 Mini-Pilot。

## Hazard

新增 `failure_hazard.csv` / `failure_hazard.png`，按 `[0,15)`、`[15,30)`、`[30,45)`、`[45,60)` 统计首次失败数除以进入区间时尚未失败的视频数；精确 15 秒的事件计入第二个区间。未标注、未看完整、跨边界的 block onset 均保留未知，只有风险集和结局都确定时才报告 hazard。否则显示 N/A，避免把缺失标签当正常。该值是离散区间风险，不是每秒瞬时 hazard，也不自动给出“历史变长导致风险增加”的因果结论。

## 生成证据冻结

生成过程中不修改 generator、backend 或加载代码。本轮在视频运行期间补充了报告/分析功能，故 `provenance.json` 中的全部源码哈希不等于最终分析提交。为可复现性，run 内 `generation_source.tar.gz` 保存全部生成时源码文件（92 个），逐文件与 provenance 核对；`generation_code.patch` 是相对 da0174d 的生成入口/加载器差异。后续精确 replay 必须用该冻结源码和原计划/环境；不要重写原 provenance 以绕过校验。本轮未做 P0-B 或故障恢复分支。

## 结果

实际数据、图表和限制见 [RESULTS.md](RESULTS.md)。视频和 replay 留在本机被忽略的 run 目录；审阅报告、指标和预览随实验分支上传。
