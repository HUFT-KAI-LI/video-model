# ReStream MVP

代码、配置、数据清单和复现说明已整理到本仓库。大型模型权重、原始视频、缓存、运行日志和输出视频不提交到 Git；按 [README_REPRODUCE.md](README_REPRODUCE.md) 下载。

## Edit-Ready Video Generation（Generation-Time Edit Cache）可行性 MVP

不训练新模型，验证"首次生成时保存每个 AR chunk 开始前的内部状态，之后用新 prompt 只重开其中一个 chunk，且其余 chunk 逐位不变"。

- 方案：[EDIT_READY_VIDEO_FEASIBILITY_PLAN.md](EDIT_READY_VIDEO_FEASIBILITY_PLAN.md)
- 实现与实测：[EDIT_READY_MVP.md](EDIT_READY_MVP.md)
- 汇总：[validation/edit_ready_mvp/summary.json](validation/edit_ready_mvp/summary.json)
- 复现：`bash scripts/08_edit_ready_mvp.sh smoke|all-chunks|main`

结论：同 prompt 重开 chunk **逐位精确**（24/24），未编辑 chunk 编码前 **逐位不变**（48/48）；**chunk 0（无历史）与新 prompt 的 full regeneration 逐位相同**，证明 prompt 重绑定实现正确；编辑强度随已 committed 的历史衰减（`R_k` 100% → 5.3% → 1.6%），reverse 控制显示是历史而非 prompt 决定 chunk；history recache 是无效的 training-free 干预；成本分三档（compute 0.167 / cache 驻留 0.367 / 冷存储 0.770）。判定 `GO_WEAK_PROMPT_REBINDING`。

## Reality Memory / R1

R1-A Top-1 路由、strict-online Train/Dev、Test source 预留与单步验证见 [R1_PREFIX_AWARE_PLAN.md](R1_PREFIX_AWARE_PLAN.md)。

最新 R1 matched 训练与 Test 资格封存见 [R1_MATCHED_EXPERIMENTS.md](R1_MATCHED_EXPERIMENTS.md)。
