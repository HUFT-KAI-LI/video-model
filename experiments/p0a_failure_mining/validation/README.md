# 准备阶段验证（2026-09-22）

- `tests_torch.txt`：已有 LongLive Python 3.11 / torch 2.5.1+cu124 环境，15/15 通过。覆盖 prompt 分层、45/60 秒 frame/block 映射、持续漂移与瞬时低分、参考特征、pool ties、人审校验、前缀未知边界、不能从候选集声称 recall、匹配正常 control、快照证据篡改、默认仅准备与不兼容恢复。
- 两项上游相关检查实际运行 torch：使用替身 generator 执行真正的 `CausalInferencePipeline.inference`，比较原始默认路径与新增观察/latent 路径的 latent 和 RNG；使用真正 WanVAE 类、缩小尺寸的随机模型比较一次性解码和连续缓存分块解码（9 + 12 帧）。这不是已下载官方权重的 GPU 验证。
- `tests_cpu_plots.txt`：系统 Python 有 NumPy/Matplotlib、无 torch，13 项通过，2 项明确跳过。端到端测试在临时目录创建合成分数/人工标签，验证未知率、失败目录、正常匹配和全部六张 PNG；临时测试样本不属于自然生成数据，也未提交为实验结果。
- `patch_roundtrip.json`：完整四文件补丁无 fuzz 反向应用、重新生成文本一致、正向恢复逐字节一致；重建检查脚本同步纳入 causal inference 文件。本次未重新核验固定上游归档：本机只有 checksum 不符的 partial 文件，独立下载也超时。不能把补丁回环检查称为上游归档 SHA-256 复核。
- `preflight.json`：模型文件存在；当前验证进程 CUDA 不可用，LongLive 环境缺 Matplotlib（另一个 CPU 环境已验证绘图）。因此 `ready_to_generate=false`、`gpu_rollout_validated=false`。使用目标环境前按主 README 安装额外依赖并重新预检。
- 已在本机被 Git 忽略的 runs 目录分别创建 45 秒与 60 秒、各 80 条轨迹的计划。没有启动真实视频生成，没有下载 DINO/CLIP，没有人审标签或实验发现。

本次没有进行完整旧 Reality Memory 回归训练；新增接口默认关闭，受影响的上游 inference 路径已用专门测试覆盖。
