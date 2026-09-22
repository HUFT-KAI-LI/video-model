## 问题与改动

新增 P0-A LongLive 自然失败挖掘准备管线。固定 20 个 prompt（四类主体、8/8/4 难度）× 4 seeds × 60 秒自然单 prompt rollout，支持显式改为 45 秒。默认命令只建立计划；生成、DINOv2/CLIP 特征、持续性候选筛选、离线人审表单、失败目录、匹配正常对照与报告均有独立入口。

按 Wan `4L−3` 解码规则建立精确的 block/frame/time 映射。保留原轨迹噪声、latent 和逐块 RNG；仅在原始轨迹重放逐位一致后捕获 cache 并允许标记为 P0-B state。LongLive 上游仅增加可选的 block 观察回调和跳过 RGB 解码开关，完整补丁与重建检查同步更新。

未复核样本保留 unknown，报告识别边界；不从候选集标注声称全池 recall。随机抽检未命中视频，完整 recall 需要完整 cohort 标签。累计首次失败比例上升不解释为 per-block hazard 上升。

## 验证

- LongLive Python 环境下 15 项测试通过，含上游观察回调的轨迹/RNG 不变性和小型随机权重 VAE 连续/缓存解码一致性。
- CPU/绘图环境下 13 项通过、2 项依赖 torch 的测试跳过；临时合成数据贯通候选→人审→目录→六张图，合成数据不作为实验结果提交。
- 真实长视频 rollout、真实 DINO/CLIP 特征推理、GPU 位级重放及人工标注仍未执行；预检记录当前验证进程无法访问 CUDA。

主要审阅入口：`experiments/p0a_failure_mining/README.md`。运行输出与大文件不提交。本 PR 仅请求审阅准备代码，不宣称已经观察到自然失败或达到 GO 条件。
