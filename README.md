# video-model

Reality Memory：基于冻结 Wan / LongLive，用稀疏真实图像作为外部记忆的实验。原有 State Re-Anchoring 保留为 baseline 与诊断。

新方向 **Edit-Ready Video Generation（Generation-Time Edit Cache）** 的可行性 MVP 已完成，结论为 `GO_WEAK_PROMPT_REBINDING`：

- [可行性方案](restream_mvp/EDIT_READY_VIDEO_FEASIBILITY_PLAN.md)
- [MVP 实现与实测结果](restream_mvp/EDIT_READY_MVP.md)
- [机器可读汇总 summary.json](restream_mvp/validation/edit_ready_mvp/summary.json)

要点：同 prompt 重开 chunk **逐位精确**（24/24）；未编辑 chunk 编码前**逐位不变**（48/48）；**chunk 0（无历史）与新 prompt 的 full regeneration 逐位相同**，证明重绑定实现正确；编辑强度随历史衰减 `R_k` 100%→5.3%→1.6%，reverse 控制显示历史主导（P1 历史 + P0 文本仍恢复 full regen 响应的 95%）；history recache 是无效的 training-free 干预；冷存储读 1 GiB cache 时端到端成本不优于整条重生成（compute 0.167 / warm 0.367 / cold 0.770）。

- [Reality Memory 新方案](restream_mvp/REALITY_MEMORY_PLAN.md)
- [R0 实现、数据和审阅后训练命令](restream_mvp/REALITY_MEMORY_R0.md)
- [R0 修补与短实验结果](restream_mvp/R0_SHORT_EXPERIMENTS.md)
- [R0 同目标配对与轻度退化实验](restream_mvp/R0_PAIRED_EXPERIMENTS.md)

- [当前审阅状态与实测结果](restream_mvp/STATUS.md)
- [环境、模型、数据与复现命令](restream_mvp/README_REPRODUCE.md)
- [本轮验证报告](restream_mvp/validation/)

模型和原始视频已保存在工作机器，GitHub 收录代码、manifest 和验证报告。
