## 真实 Canary 结果与变更

按审阅只完成 P001/P003/P005/P016 seed 0 的四条 60 秒自然轨迹，不合并 main、不扩展 Mini-Pilot 或 80 条。每条实际验证 960 帧 / 16 FPS / 480×832、完整 81 个 block RNG 与有限 latent；单张 RTX 3090 每条生成加解码约 6.1 分钟，运行阶段峰值 allocated 7.82 GiB。官方 VAE 的 6 latent 帧连续/分块解码逐位相同。

修复模型加载器未执行 CPU component placement，删除 object 类不自然的无 body/shell 负例，增加同模型本地 feature 文件的哈希记录、分区间 first-failure hazard、human/auto onset 对照表和逐视频复核曲线。

DINO/CLIP 已完成全四条特征。P003 和 P005 是自动候选，不能当作人工标签。预览发现前两条红夹克 prompt 从首帧即呈黄色，需区分初始 prompt mismatch 与 late drift。完整视频人审尚未完成，recall / hazard 不可估计，Gate 1 的语义校准仍待审阅。

## 审阅与验证

主入口：`experiments/p0a_failure_mining/canary/RESULTS.md`。报告附生成核验、分数、曲线、预览和源码 provenance；大视频 / replay / 权重保留本机。

- torch 环境 16 项测试全部通过；CPU 绘图环境 14 项通过、2 项依赖 torch 的测试跳过。
- MP4 全部帧数 / 尺寸 / PTS / 哈希，以及 replay shape / finite / RNG 实测验证。
- 四条真实 DINOv2 / CLIP GPU 特征前向完成；模型权重与官方 SHA-256 匹配。
- Node 检查复核页启动、四条视频切换、onset 时间映射及带引号 CSV 导出。
- 没有重试采样、人工 corruption、训练或 P0-B branching。
