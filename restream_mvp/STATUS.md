# ReStream 审核前准备状态

用户要求：尽快准备完成，不设 12 小时硬截止；模型、数据和代码完成后先审核，暂不训练。

## 硬件与位置

- 工作目录：`/workspace/restream_mvp`。
- 4 × NVIDIA A800-SXM4-80GB；驱动 535.129.03，CUDA 12.4。
- 沙箱内无法访问 GPU，沙箱外 `nvidia-smi` 已确认四卡空闲。
- Python 3.11；已有 torch 2.5.1+cu124 / torchvision 0.20.1+cu124 / flash-attn 2.8.3.post1。

## 已完成

- 从 GitHub 官方 codeload 下载 LongLive v1.0 源码，提交 `e52d9ef6865d843282a6b5e9d46d03b35f88929d`。
- Wan2.1-T2V-1.3B 官方 ModelScope 全套权重（含 VAE、UMT5 与 tokenizer）。
- LongLive 基座与配套 LoRA 从 ModelScope 下载，本地 SHA-256 与 Hugging Face 官方公布值相同。
- Adapter、corruption、时间对齐、视频加载、teacher-forcing 未来损失、缓存重建与 AR 推理代码。
- 四卡 Adapter-only 训练入口、checkpoint/resume、A/B/C 评估与视频/指标输出代码。
- 准备入口不启动训练，实验入口要求显式 `--reviewed`。
- 7 项 CPU 单元检查通过（最后一次日志 `logs/unit_tests_final.log`）。
- GPU 权重加载检查通过：基座严格匹配，配套 LoRA 键和形状匹配，全部参数冻结。
- GPU VAE 检查通过：9×64×64 RGB 像素帧编码为 `[1,3,16,8,8]`，单帧编码为 `[1,1,16,8,8]`，值有限。峰值显存约 26.4 GiB，报告见 `logs/model_load_check.json`。
- 新增 Adapter 只有 801 个参数。未做真实 transformer backward / AR rollout / DDP 实验，留待审核后。

## 正在准备

- Youku caption/train 小子集已逐条下载和完整解码校验：1068 usable source videos、2.2606 GB、961 train / 107 val；最终数字见 `data/dataset_stats.json`。下载在达到约 2 GB 的最低目标后停止，保留 8 GB 上限。
- 最终 train/val 划分、统计和 readiness 检查已通过；`logs/readiness_check.log` 全部为 true。

## 接口核对

- VAE：`code/LongLive/utils/wan_wrapper.py` → `WanVAEWrapper.encode_to_latent`，输入 B,C,T,H,W，输出 B,T,C,H,W。
- teacher forcing：同文件 `WanDiffusionWrapper.forward(clean_x=...)` → `wan/modules/causal_model.py::_forward_train`。
- 未来损失复用 `utils/loss.py::FlowPredLoss`，只对 anchor 所在块之后计算。
- 原始 cache refresh：`pipeline/causal_inference.py::CausalInferencePipeline.inference` 的 clean-context forward。
- 本项目使用 Fallback B：`restream/runtime.py::rollout` 分配全新 cache、重放修正后的 prefix，然后生成 future。
- 动态计算空间 token 数，替代 pipeline 原先固定的 1560；真实时间向后对齐 AR 块尾。
- 上游两处 `[:-padded_length]` 在 padding=0 时会变为空张量，改为按原 token 长度截取；见 `longlive.patch`。

## 审核需要知道

- 数据集包含水印、字幕、剪辑、动画和真人混合内容。关键词只能弱过滤；初始缩略图仍发现动画样本，人工视觉审核未完成。
- 单帧真实图像 VAE latent 与内部时序 latent 的分布差异需要实际 Hard Anchor 实验检验。
- 训练使用人为漂移的真实历史与 teacher forcing，尚未实现长期 self-rollout 训练；本轮只实现单个中途 anchor。
- 官方配套 rank 256 LoRA 加载后冻结，只有新增 ReAnchor Adapter 训练。
- LPIPS 未计算时会写 null；它需要审核后显式启用，可能另外下载 AlexNet 权重。
- `.venv` 继承机器已有 torch/CUDA 包。全局 `pip check` 会报告与继承的 Gradio、LLaMAFactory、TRL 的版本冲突；本项目不使用这些工具，也未更改系统环境。ReStream 依赖导入和真实模型加载检查通过，详见 `logs/pip_check.log`。
- 训练步数：0；正式推理/实验视频/指标/checkpoint：尚未生成，按用户要求留待审核后。
