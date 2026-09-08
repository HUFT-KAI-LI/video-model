# ReStream：审核前准备

本次交付范围由用户最新要求决定：尽快完成模型、数据和代码准备；没有 12 小时硬截止；审核后再启动推理、训练与正式评估。`scripts/prepare.sh` 和 `run_night.sh` 都只做准备。

任务目录：`/workspace/restream_mvp`。当前机器是 4 × A800-SXM4-80GB，驱动可在沙箱外访问。项目使用独立 `.venv`，继承机器已有 PyTorch 2.5.1+cu124 / torchvision 0.20.1+cu124 / flash-attn 2.8.3.post1。环境准备不会更改驱动。模型和数据不会写入附件原先的 `/data/restream_mvp`。

## 先审核这些文件

- `STATUS.md`：实际完成状态与仍未验证的事项。
- `configs/restream_mvp.yaml`：默认 57 像素帧 → 15 latent 帧，每个 AR block 为 3 帧；分辨率 256 × 432。
- `restream/objective.py`：原生 teacher-forcing 与上游 flow loss；只监督 anchor 后的未来块。
- `restream/runtime.py`：加载官方基座与冻结的配套 LoRA；推理时重新分配缓存并重放修正后的历史。
- `restream/anchor_injector.py`：时间映射、独立真实图像编码和强制状态重建接口。
- `data/train.jsonl`、`data/val.jsonl`：按完整 source ID 分组划分，固定 seed 42。
- `longlive.patch`：上游只有两处 padding 截断修复；原始源码归档完整保留。

## 模型来源与核验

Wan 来源：<https://modelscope.cn/models/Wan-AI/Wan2.1-T2V-1.3B>。

LongLive 下载源：<https://modelscope.cn/models/Efficient-Large-Model/LongLive-1.3B>。不能仅凭同名组织认定模型由官方上传，因此本任务使用官方 Hugging Face 公布的 SHA-256 核验二进制文件，结果记录在 `models/LongLive-1.3B/official_sha256_verified.json`：

| 文件 | 官方 SHA-256 |
|---|---|
| models/longlive_base.pt | `10a2aa8fcf89c77d9033f4c117405412a690e289625766619d293f0c5a208ee7` |
| models/lora.pt | `c4e43b87d62d4b0614b496773639f1ab170a7ee486dc23407901e9d3a5ebc07a` |

核验依据：[官方基座页面](https://huggingface.co/Efficient-Large-Model/LongLive-1.3B/blob/main/models/longlive_base.pt)、[官方 LoRA 固定版本页面](https://huggingface.co/Efficient-Large-Model/LongLive-1.3B/blob/17516b597d2675e53a056eb7e8f66160e2714103/models/lora.pt)。配套 LoRA 是基线模型已有的 rank 256 模块，加载后冻结；本实验不添加或训练新的 backbone LoRA。

上游源码为 NVlabs/LongLive `v1.0`，归档记录的提交为 `e52d9ef6865d843282a6b5e9d46d03b35f88929d`。因 Git 协议网络失败，通过 GitHub 官方 codeload 取得归档。`code/longlive-v1.0.tar.gz` 是原始副本，`code/LongLive` 是适配工作目录。

## 数据下载策略

来源：ModelScope `modelscope/Youku-AliceMind`，caption/train。已下载 1068 个可用 source video（约 2.26 GB；961 train / 107 val）。元数据使用 streaming；视频通过同一 ModelScope SDK 的 OSS 通道逐个读取，写入前检查对象大小，每个文件读取也有字节上限，避免 SDK 在 yield 样本前无限下载。

下载器只保留至少 4 秒、能完整解码的片段。明显动画描述会被排除，并筛选含人物描述的样本；这只是弱过滤，不能保证全为真实拍摄。manifest 的 `visual_review: pending` 明确保留人工审核状态。不能把这些数据直接当作已经人工清洗的真实视频集。

每个源视频只选一个 3.8–8 秒窗口，统一 resize、中心裁切和 RGB `[-1,1]`。使用 PyAV 解码，训练时不调用外部 ffmpeg。真实图片直接取同一处理后视频帧，独立进行单帧 Wan VAE 编码，不使用包含未来图像的 GT 视频 latent 冒充真实观测。

## 可复现的准备命令

```bash
cd /workspace/restream_mvp
bash scripts/prepare.sh
```

需要下载认证时，通过环境变量 `MODELSCOPE_TOKEN` 提供，脚本不会打印 token。下载可按已完成 manifest 续传；未提交到 manifest 的孤立文件会明确报错，避免静默覆盖。

单独检查，无需 GPU：

```bash
cd /workspace/restream_mvp
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/check_ready.py
```

## 审核后才执行的实验命令

这些命令写在这里供审核，本次准备不执行。

```bash
cd /workspace/restream_mvp
bash scripts/05_smoke_infer.sh --reviewed
bash scripts/07_eval_mvp.sh --reviewed --cases 1 --output outputs/hard_anchor_smoke
bash scripts/06_train_mvp.sh --reviewed --max-steps 50
bash scripts/06_train_mvp.sh --reviewed --max-steps 200 --resume checkpoints/step_0050
# 查看 step_time / 显存 / validation 后，再决定主训练长度
bash scripts/06_train_mvp.sh --reviewed --resume checkpoints/step_0200
bash scripts/07_eval_mvp.sh --reviewed --checkpoint checkpoints/step_3000 --lpips
```

恢复保存 Adapter、optimizer、scheduler、step、epoch、数据位置、配置、训练 manifest 的 SHA-256 和各 rank RNG；不重复存储 backbone。`checkpoints/latest.txt` 指向最新目录。当前实现固定每卡 batch=1；精确恢复要求相同 world size、数据配置和 manifest 内容。不要在运行或恢复过程中修改 manifest。

评估使用同一 GT、prompt、anchor、corruption seed、初始噪声及后续重加噪随机流，比较 no/hard/learned。推理始终使用新缓存重放历史（方案 Fallback B）。这是人为漂移历史的 recovery 实验；它不是已验证的长期 self-rollout 训练。

输出包含三组视频、GT 对比视频、实际 anchor 时间和未来 latent MSE。LPIPS 通过 `--lpips` 开启，首次运行可能需要下载其 AlexNet 权重；没开启时记录 `null`，不会伪造为零。0.5/1/2 秒指标在采样间隔内没有未来 latent 时也记录 `null`。

## 工程检查与研究限制

CPU 单元测试可检查 tensor shape、BF16、未来梯度、替换与重建回调、同源隔离和解码；这些检查不证明正式 LongLive rollout 或四卡训练已成功。GPU 完整验证以 `STATUS.md` 的实际记录为准。

原生 teacher-forcing 能让冻结 backbone 保留关于输入的梯度；上游 recache 的 `no_grad` 路径仅用于推理。15 latent 帧的短窗口足以容纳一个中途 anchor 和未来块，未扩展多 anchor、RL、geometry、5B 或全参训练。

真实图片单帧 VAE latent 与视频内部时序 latent 的分布可能不同，Hard Anchor 的未来影响与 Adapter 的改善幅度必须在审核后的实验中检验。时间映射采用向后对齐，不会提前注入尚未到达的观测，但会引入块级等待；评估记录实际时间。
