# ReStream / Reality Re-Anchoring MVP：今晚 12 小时执行手册

> **目标**：在 `4 × NVIDIA A100 80GB` 上，从零开始搭建一个最小可用的「真实帧重锚定（Reality Re-Anchoring）」流式视频生成实验，今晚完成 **模型下载 → 小数据集下载 → LongLive 基线复现 → 真实帧注入 → 小模块 SFT → A/B 评估 → checkpoint 与视频结果打包**。
>
> **今晚不是最终论文训练。** 今晚唯一必须回答的问题是：
>
> \[
> \boxed{\text{真实世界 Anchor 到来后，能否把已经发生漂移的 streaming video state 拉回正确轨迹？}}
> \]
>
> 如果能，明天再扩展 5B、长时间窗、自回滚训练、geometry、adaptive anchoring、RL。
>
> **硬截止：12 小时。** 所有设计都以“不超时”为第一优先级。

---

## 0. Codex 执行原则

Codex 请把这份文档当作 **autonomous implementation brief**。

### 必须遵守

1. **不要等待大数据集。**
   - 今晚训练数据硬上限：**15GB**
   - 推荐：**6–10GB**
   - 下载慢时，允许降到 **2–5GB**
   - 只要能够形成数百到上千条真实视频训练 clip 即可。

2. **今晚不要上 LongLive 2.0 5B 主训练。**
   - 今晚使用：**LongLive 1.3B / Wan2.1-T2V-1.3B**
   - 原因：先验证 mechanism，不烧掉 12 小时。

3. **今晚不要做：**
   - Full fine-tuning
   - RL / GRPO
   - 100s 训练
   - geometry / depth / 3D
   - 5B 全参
   - FSDP（除非实测真的需要）
   - NVFP4（A100 不走这条）
   - 大规模 benchmark

4. **A100 使用 BF16。**
   - 4 卡优先 DDP。
   - Backbone 默认冻结。
   - 新增 Re-Anchor 模块全量训练。
   - LoRA 今晚默认关闭；只有 Adapter 工作后且有时间才开启 `r=16`。

5. **先做能工作的 baseline，再做 learnable module。**
   - Hour 4 之前必须得到 `Hard Anchor Injection` 可运行结果。
   - 如果 learnable SFT 来不及，至少保留 Hard Anchor baseline 和 A/B 视频。

6. **每一步写日志。**
   - 所有 shell command 输出写入 `logs/`
   - 每 500 step checkpoint
   - 每 500 step 做少量固定 validation
   - 任何改动不要覆盖原 LongLive 源码；优先新增文件和小 patch。

7. **不要因为某一步不完美卡住。**
   - 下载失败 → 换源 / 缩小数据
   - 训练慢 → 降分辨率 / 帧数 / step
   - Adapter 接口难改 → 使用 Hard Anchor + KV-recache 作为 fallback
   - 12h 时必须有可复现实验产物。

---

# 1. 今晚最终必须交付什么

12 小时后，目录中至少要有：

```text
/data/restream_mvp/
├── code/
│   └── LongLive/
├── models/
│   ├── Wan2.1-T2V-1.3B/
│   └── LongLive-1.3B/
├── data/
│   ├── raw/
│   ├── train.jsonl
│   ├── val.jsonl
│   └── dataset_stats.json
├── restream/
│   ├── anchor_adapter.py
│   ├── anchor_injector.py
│   └── dataset.py
├── configs/
│   └── restream_mvp.yaml
├── scripts/
│   ├── 00_check_env.sh
│   ├── 01_setup_env.sh
│   ├── 02_download_models.sh
│   ├── 03_download_youku_subset.py
│   ├── 04_build_manifest.py
│   ├── 05_smoke_infer.sh
│   ├── 06_train_mvp.sh
│   ├── 07_eval_mvp.sh
│   └── run_night.sh
├── checkpoints/
│   ├── step_0500/
│   ├── step_1000/
│   └── latest/
├── outputs/
│   ├── baseline/
│   ├── hard_anchor/
│   ├── learned_anchor/
│   ├── comparisons/
│   └── metrics.json
├── logs/
├── STATUS.md
└── README_REPRODUCE.md
```

### 必须出现的实验结果

固定 8–16 个 validation case，至少生成三组：

1. `baseline_no_anchor.mp4`
2. `hard_real_anchor.mp4`
3. `learned_reanchor.mp4`（如果 SFT 成功）

每个 case 最好再生成一个 side-by-side：

```text
GT | No Anchor | Hard Anchor | Learned ReAnchor
```

### 最低成功标准

至少满足其中两项：

- Anchor 注入后，后续生成在视觉上明显更接近真实 GT；
- 在固定 validation 上，anchor 后的未来 1–2 秒 LPIPS / latent error 优于 no-anchor；
- Learned ReAnchor 比 Hard Anchor 的未来连续性更好；
- Adapter gate / delta 学到非零且训练稳定；
- 在人为制造 drift 的输入下，模型能够 recovery。

---

# 2. 技术路线：今晚只做最小版本

论文完整版未来是：

```text
Prompt + streaming history + sparse real observations
                    ↓
             Reality Re-Anchoring
                    ↓
           corrected world state
                    ↓
          continue causal generation
```

但今晚不要一上来做复杂 cross-attention。

## 2.1 今晚 MVP：Latent Re-Anchoring

LongLive 是 causal / autoregressive video generation。

在真实 Anchor 到达时间 `t_a`：

```text
当前模型历史 latent / state
          ↓
      已经有 drift
          +
真实世界图片 I_real(t_a)
          ↓
        Wan VAE
          ↓
     z_anchor_real
          ↓
   ReAnchor / Hard Replace
          ↓
      corrected latent
          ↓
      KV recache / state refresh
          ↓
继续生成 future video
```

---

## 2.2 第一版：Hard Anchor（必须先完成）

先完全不训练 Adapter：

\[
z^{+}_{t_a}=z^{real}_{t_a}
\]

即在 anchor timestamp：

1. 真实图片用 frozen Wan VAE 编码；
2. 用真实 anchor latent 替换当前对应时刻的生成 latent；
3. 调用 LongLive 已有的历史/cache rebuild 或 KV-recache 路径；
4. 从纠正后的 state 继续往后生成。

### 为什么先做这个

这是最便宜、最重要的 sanity check：

> 如果连 Hard Anchor + cache refresh 都无法影响后续 trajectory，今晚不应该继续烧 SFT。

---

## 2.3 第二版：Learned Gated Re-Anchor

Hard Anchor 成功后再增加小模块。

推荐最小结构：

```python
z_pred   # 当前模型预测/漂移的 anchor latent
z_real   # 真实图片 VAE latent

x = cat([z_pred, z_real], channel_dim)
delta = AnchorAdapter(x)
g = sigmoid(trainable_gate)
z_corr = z_pred + g * delta
```

如果 latent 为 `(B, C, T, H, W)`，使用很小的 `Conv3d`：

```text
2C
↓
1×1×1 Conv
↓
SiLU
↓
1×1×1 Conv
↓
C
```

### gate 初始化

建议：

```python
gate_logit = -2.0
```

初始 `sigmoid(-2) ≈ 0.12`，使模型从接近原模型开始，不会第一步就把 backbone 行为破坏。

### 今晚不要先做复杂 transformer Adapter

今晚优先级：

1. Hard latent replacement
2. Tiny gated residual adapter
3. 若仍有时间，再考虑 temporal attention LoRA

---

# 3. 基础模型选择

## 3.1 今晚使用 LongLive v1.0 / 1.3B

LongLive 1.0 是官方 ICLR 2026 工作，1.3B checkpoint 基于 Wan2.1-T2V-1.3B。

官方 v1.0 分支包含：

```text
inference.py
interactive_inference.py
train.py
train_init.sh
train_long.sh
configs/
model/
pipeline/
trainer/
wan/
```

今晚 clone **v1.0**，不要 main 分支。

```bash
mkdir -p /data/restream_mvp/code
cd /data/restream_mvp/code

git clone \
  --single-branch \
  --branch v1.0 \
  --depth 1 \
  https://github.com/NVlabs/LongLive.git
```

### 为什么不是 5B

5B 是论文最终主模型候选，但今晚 12h：

- 1.3B 更适合调接口；
- 更容易 DDP；
- 更容易跑 2K–4K step；
- 一旦 architecture 错误，损失小得多；
- 今天的任务是 hypothesis validation，不是最终 VBench。

---

# 4. 环境检查

创建：

```text
scripts/00_check_env.sh
```

内容至少执行：

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "===== GPU ====="
nvidia-smi

echo "===== CUDA ====="
nvcc --version || true

echo "===== STORAGE ====="
df -h /data || df -h

echo "===== MEMORY ====="
free -h

echo "===== PYTHON ====="
python3 --version

echo "===== GPU COUNT ====="
python3 - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
```

### 必须确认

```text
GPU count = 4
GPU = A100 80GB
/data 可用空间最好 >= 50GB
```

若空间低于 40GB：

- 数据目标立刻缩到 2–5GB；
- 模型只保留必要 checkpoint；
- 不缓存大量 decoded frames / latent。

---

# 5. Python / CUDA 环境

LongLive 1.3B 官方 model card 给出的测试环境为：

- Python 3.10
- CUDA 12.4
- PyTorch 2.5.0 + cu124
- torchvision 0.20.0
- flash-attn 2.7.4.post1

优先按官方环境。

创建：

```text
scripts/01_setup_env.sh
```

建议：

```bash
#!/usr/bin/env bash
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | grep -q "^restream "; then
  conda create -n restream python=3.10 -y
fi

conda activate restream

python -m pip install -U pip setuptools wheel

pip install \
  torch==2.5.0 \
  torchvision==0.20.0 \
  torchaudio==2.5.0 \
  --index-url https://download.pytorch.org/whl/cu124

cd /data/restream_mvp/code/LongLive
pip install -r requirements.txt

pip install flash-attn==2.7.4.post1 --no-build-isolation

# Project extras
pip install \
  modelscope \
  huggingface_hub \
  accelerate \
  av \
  decord \
  opencv-python-headless \
  einops \
  safetensors \
  tensorboard \
  lpips \
  pandas \
  pyarrow \
  tqdm \
  pyyaml
```

### flash-attn 安装失败时

今晚不要为 flash-attn 卡几个小时。

处理策略：

1. 保留完整 error log；
2. 尝试 repo requirements 自带版本；
3. 若 PyTorch SDPA 可以跑，先完成 smoke test；
4. 后续再优化。

---

# 6. 模型下载

## 6.1 Wan2.1-T2V-1.3B：优先 ModelScope

这个模型在 ModelScope 有官方仓库，今晚中国网络环境优先使用魔搭。

创建：

```text
scripts/02_download_models.sh
```

核心：

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/restream_mvp
mkdir -p \
  "$ROOT/models/Wan2.1-T2V-1.3B" \
  "$ROOT/models/LongLive-1.3B"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate restream

echo "===== Download Wan2.1 base from ModelScope ====="
modelscope download \
  --model Wan-AI/Wan2.1-T2V-1.3B \
  --local_dir "$ROOT/models/Wan2.1-T2V-1.3B"

echo "===== Download LongLive checkpoint ====="
huggingface-cli download \
  Efficient-Large-Model/LongLive-1.3B \
  --local-dir "$ROOT/models/LongLive-1.3B"

echo "===== Disk usage ====="
du -sh "$ROOT/models/"*
```

### 注意

目前能够确认 **Wan2.1-T2V-1.3B 有 ModelScope 官方下载**。

LongLive 1.3B 官方 checkpoint 我们优先使用其官方 Hugging Face repository。

**不要为了强行全部使用 ModelScope 去下载来源不明的 third-party LongLive 权重。**

如果 Hugging Face 下载特别慢：

- 先继续做数据与代码准备；
- 不要阻塞整个任务；
- 可以使用你机器已有的 HF 镜像配置，但不要使用无法验证来源的模型文件。

---

# 7. 模型基线必须先复现

## 7.1 修改 LongLive 原配置路径

Codex 先查看：

```bash
cd /data/restream_mvp/code/LongLive

grep -R "Wan2.1-T2V-1.3B" -n .
grep -R "checkpoint" -n configs *.sh | head -100
grep -R "longlive_models" -n .
```

**不要猜 checkpoint filename。**

使用实际下载出来的文件结构：

```bash
find /data/restream_mvp/models/LongLive-1.3B -maxdepth 3 -type f | sort
find /data/restream_mvp/models/Wan2.1-T2V-1.3B -maxdepth 3 -type f | sort
```

然后只修改 config / launch script 中的路径。

---

## 7.2 Smoke inference

创建：

```text
scripts/05_smoke_infer.sh
```

先调用 repo 官方 `inference.sh` 或等价命令。

目标：

```text
outputs/baseline/smoke.mp4
```

### 验收

必须确认：

- 能 load Wan base；
- 能 load LongLive weight；
- 能成功 decode mp4；
- 没有 NaN；
- GPU 显存正常；
- 一个短 clip 能生成。

### 这一步的时间上限

**90 分钟。**

90 分钟仍不能 baseline inference：

- 停止所有新功能；
- 先修 base environment；
- 不允许在 base inference 未通时写训练逻辑。

---

# 8. 今晚数据集：不要等 Ego-Exo4D

## 8.1 为什么今晚不使用 Ego-Exo4D

Ego-Exo4D 很适合最终论文，但官方 license credential 通常不是即时到账；官方文档提醒可能需要约 48 小时。

因此：

```text
Ego-Exo4D = 后续正式实验
NOT tonight blocker
```

可以今晚提交 access request，但 Codex 不应等待。

以后得到权限后，只下载 selected UIDs 的：

```text
downscaled_takes/448
```

而不是整套数百 GB。

---

# 9. 今晚数据集：ModelScope Youku-mPLUG 小子集

推荐：

```text
modelscope/Youku-AliceMind
subset_name = caption
split = train
use_streaming = True
```

它的优点：

- ModelScope；
- 真实视频；
- 自带 caption；
- 可以 streaming iterate；
- 不需要一次下载整个大数据集；
- 今晚可以按 bytes / item 数硬截断。

---

## 9.1 数据量

今晚目标：

```text
推荐：6–10 GB
硬上限：15 GB
最低可接受：2 GB
```

样本数：

```text
目标：500–1500 usable videos
最低：300 train + 40 val
```

不要为了达到整数继续下载。

---

# 10. 写小数据下载器

创建：

```text
scripts/03_download_youku_subset.py
```

要求：

### CLI

```bash
python scripts/03_download_youku_subset.py \
  --output /data/restream_mvp/data/raw/youku \
  --max-gb 8 \
  --max-items 1200 \
  --min-duration 4.0 \
  --token "$MODELSCOPE_TOKEN"
```

### 逻辑

使用：

```python
from modelscope.hub.api import HubApi
from modelscope import MsDataset

api = HubApi()
api.login(token)

ds = MsDataset.load(
    "Youku-AliceMind",
    namespace="modelscope",
    subset_name="caption",
    split="train",
    use_streaming=True,
)
```

dataset item 大致包含：

```text
video_id:FILE
golden_caption
```

程序：

1. iterate streaming dataset；
2. 获取本地 cache video path；
3. `ffprobe` / PyAV 检测 duration；
4. 过滤：
   - duration < 4s
   - 无法 decode
   - 文件过小/损坏
5. 通过 hardlink 优先复制到：
   ```text
   data/raw/youku/videos/
   ```
6. 写：
   ```text
   data/raw/youku/raw_manifest.jsonl
   ```
7. 累计 file size；
8. 达到 `max_gb` 或 `max_items` 立刻停止。

### 不要重复复制 cache 文件

优先：

```python
os.link(src, dst)
```

失败再：

```python
shutil.copy2(src, dst)
```

这样避免缓存 + 数据目录占双倍空间。

---

# 11. 构造 train / val

创建：

```text
scripts/04_build_manifest.py
```

## 11.1 split 规则

按 **source video id** 分组：

```text
90% train
10% val
```

不要把同一 source video 的不同 crop 分别放到 train / val。

避免 data leakage。

---

## 11.2 训练 window

Youku clips 可能比较短，因此今晚动态选择：

```text
duration >= 8s  →  window 6–8s
duration 4–8s   →  window 3.5–min(duration-0.2, 6)s
```

最终论文才做：

```text
16s / 32s train
60s / 100s test
```

今晚不要追 100 秒。

---

## 11.3 视频帧数

**不要硬编码 16 frames。**

Wan / LongLive 的 VAE / temporal block 可能要求特定合法帧数（常见形式类似 `4k+1`）。

Codex 必须查看当前 LongLive config，优先使用：

> **代码已经支持的最小 temporal shape**

目标只是减少计算。

若允许：

```text
17 / 33 frames
```

优先。

若官方训练 config 的最小 shape 更高：

- 保持合法 shape；
- 通过降 spatial resolution 和 training steps 来控制时间。

---

# 12. Anchor schedule：今晚版

今晚只需要 1–3 个中途 anchor。

每个 window：

```text
initial frame = optional scene initialization
anchor #1 = window 的 30–50%
anchor #2 = 可选，70–85%
```

例如：

```text
4s window:
  0.0s
  1.4s
  3.0s

8s window:
  0.0s
  1.2s
  3.0s
  6.5s
```

但时间位置必须 random jitter，不能所有训练样本固定一样。

manifest：

```json
{
  "video": "/data/restream_mvp/data/raw/youku/videos/000123.mp4",
  "caption": "一个人在室内做某项活动。",
  "duration": 7.41,
  "window_start": 0.35,
  "window_sec": 6.0,
  "anchor_sec": [0.0, 2.1, 5.0],
  "source_id": "000123",
  "split": "train"
}
```

---

# 13. Artificial Drift：今晚非常重要

如果历史一直是完美 GT，模型可能没有动力真正使用 anchor。

所以训练中必须人为模拟 drift。

在 anchor 之前的历史 latent，随机使用一种或两种 corruption：

```text
Gaussian latent noise
small spatial torch.roll
channel scaling
drop / replace one history frame
small temporal offset
```

建议：

```python
p_corrupt = 0.8
sigma = Uniform(0.02, 0.12)
```

不要一开始做严重破坏。

### 核心训练样本变成

```text
corrupted generated-like history
        +
real anchor
        ↓
future GT supervision
```

这更接近我们真正的研究目标：

> 已经发生漂移后，真实观测能不能 recovery。

---

# 14. Codex 如何找到 LongLive 的注入点

不要假设文件名。

先搜索：

```bash
cd /data/restream_mvp/code/LongLive

grep -R "recache" -n .
grep -R "kv_cache" -n model pipeline trainer
grep -R "cache" -n model pipeline trainer | head -200
grep -R "vae" -n pipeline trainer model
grep -R "encode" -n pipeline trainer wan | head -200
```

确认三个位置：

1. **Wan VAE image/video encode path**
2. **历史 latent / chunk 被送入 causal model 的位置**
3. **LongLive KV-recache / historical state rebuild 的位置**

在 `STATUS.md` 记录实际文件和函数名。

---

# 15. 实现 Hard Anchor Injector

新增：

```text
restream/anchor_injector.py
```

建议接口：

```python
class HardAnchorInjector:
    def __init__(self, vae, ...):
        ...

    @torch.no_grad()
    def encode_anchor(self, image):
        # image -> Wan VAE latent
        ...

    def inject(
        self,
        current_latent,
        anchor_latent,
        anchor_index,
        state_or_cache,
    ):
        # replace/blend the relevant temporal latent
        # trigger existing cache rebuild / recache if required
        return corrected_latent, corrected_state
```

### Hard Anchor 目标

不追求优雅。

只需要证明：

```text
同一 corrupted history
          ↓
No Anchor    → future A
Real Anchor  → future B
```

且 B 更靠近 GT。

---

# 16. 如果 cache 接口太难：fallback

如果在第 4–5 小时仍无法把 anchor 直接注入 internal KV：

### Fallback A

把真实 anchor 作为新的 history frame / first frame of next chunk：

```text
history chunk
   ↓
real anchor latent = next chunk clean first state
   ↓
continue autoregressive generation
```

然后让 LongLive 自己从该 frame 继续。

### Fallback B

直接重建当前 short history window：

```text
recent generated frames
+
latest real anchor
↓
re-encode short window
↓
rebuild cache
```

虽然较慢，但今晚可作为 proof-of-concept。

### Fallback C

如果训练完全来不及：

只提交：

```text
Hard Re-Anchor inference baseline
+
No Anchor comparison
+
evaluation
```

不要交“没有结果的 300 行未完成 Adapter”。

---

# 17. Learnable Anchor Adapter

Hard Anchor 成功后新增：

```text
restream/anchor_adapter.py
```

示例结构：

```python
import torch
import torch.nn as nn

class GatedAnchorAdapter(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(channels * 2, channels, 1),
            nn.SiLU(),
            nn.Conv3d(channels, channels, 1),
        )
        self.gate_logit = nn.Parameter(torch.tensor(-2.0))

    def forward(self, z_pred, z_anchor):
        x = torch.cat([z_pred, z_anchor], dim=1)
        delta = self.net(x)
        gate = torch.sigmoid(self.gate_logit)
        return z_pred + gate * delta
```

### 注意 shape

不要假设 latent 一定 5D。

先打印真实 VAE latent shape。

如果是：

```text
B,C,T,H,W
```

使用 Conv3d。

如果 anchor 是单帧：

- 显式扩 temporal dimension；
- 或使用 Conv2d 后重新 reshape。

---

# 18. Backbone 参数策略

今晚默认：

```text
VAE            frozen
Text encoder   frozen
LongLive/Wan   frozen
AnchorAdapter  trainable
```

先完成这个。

## 有时间才启用 LoRA

如果 step < 8 秒并且 Adapter 已明显有效，可增加：

```text
temporal/self attention q/k/v/o
LoRA rank = 16
```

但不要因此让主实验错过 12h deadline。

---

# 19. 训练 objective

尽量复用 LongLive 原训练 objective。

今晚不要自己重写 diffusion / flow loss。

在原 teacher-forcing loss 的基础上，仅把 prefix/history 的某个位置变成：

```text
corrupted prediction/state
   ↓
re-anchor module
   ↓
corrected state
```

然后未来帧继续使用 LongLive 原监督。

形式上：

\[
\mathcal L =
\mathcal L_{\text{LongLive-future}}
+
\lambda_{\Delta}\mathcal L_{\text{adapter-reg}}
\]

今晚 `adapter-reg` 可以只是很弱的 delta L2：

\[
\mathcal L_{\Delta}=\|\Delta z\|_2^2
\]

例如：

```text
lambda_delta = 1e-4
```

防止 adapter 产生巨大无约束改动。

### 不要使用

```text
L(z_corr, z_anchor)
```

作为主要损失，否则模型很可能学成简单强制复制 anchor，而不是优化 anchor 后未来 generation。

---

# 20. 训练配置

创建：

```text
configs/restream_mvp.yaml
```

建议：

```yaml
project:
  root: /data/restream_mvp
  seed: 42

model:
  backbone: longlive_1_3b
  freeze_backbone: true
  freeze_vae: true
  freeze_text_encoder: true

reanchor:
  mode: learned
  adapter: gated_residual
  gate_init_logit: -2.0
  artificial_drift_prob: 0.8
  drift_sigma_min: 0.02
  drift_sigma_max: 0.12

lora:
  enabled: false
  rank: 16
  alpha: 16

data:
  train_manifest: /data/restream_mvp/data/train.jsonl
  val_manifest: /data/restream_mvp/data/val.jsonl
  target_short_side: 256
  num_workers: 8
  prefetch_factor: 2

train:
  precision: bf16
  max_steps: 3000
  per_gpu_batch: 1
  grad_accum: 1
  lr_adapter: 1.0e-4
  weight_decay: 0.01
  warmup_steps: 100
  grad_clip: 1.0
  save_every: 500
  eval_every: 500
  log_every: 10
  gradient_checkpointing: false

distributed:
  mode: ddp
  world_size: 4
```

### spatial resolution

`target_short_side: 256` 只是目标。

如果 LongLive 当前代码要求固定 shape：

- 使用最接近的合法 shape；
- 记录在 `STATUS.md`；
- 不要为了 256 破坏 architecture。

---

# 21. DDP 启动

创建：

```text
scripts/06_train_mvp.sh
```

预期：

```bash
#!/usr/bin/env bash
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate restream

export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

cd /data/restream_mvp

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=4 \
  train_reanchor.py \
  --config configs/restream_mvp.yaml \
  2>&1 | tee logs/train_$(date +%Y%m%d_%H%M%S).log
```

---

# 22. 训练前必须跑 50-step smoke test

不要直接 3000 steps。

启动参数：

```text
max_steps = 50
save_every = 50
eval_every = 50
```

验证：

- 四卡都在工作；
- loss finite；
- gradient finite；
- Adapter 有 gradient；
- frozen backbone 没 gradient；
- checkpoint 能保存；
- validation 能生成。

### 自动检查 trainable params

打印：

```python
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())

print(trainable, total, trainable / total)
```

今晚期望：

```text
trainable << total
```

---

# 23. 200-step 性能基准：决定今晚跑多少

50-step smoke 成功后，跑 200 steps。

忽略前 20 step warm-up。

记录：

```text
mean sec/step
p50 sec/step
peak VRAM / GPU
GPU utilization
data loader waiting
```

### 决策表

| 实测速度 | 今晚 max_steps |
|---|---:|
| `<= 7 s/step` | 4000 |
| `7–10 s/step` | 3000 |
| `10–14 s/step` | 2200 |
| `14–18 s/step` | 1500–1800 |
| `> 18 s/step` | 立刻优化，不允许傻跑 |

---

# 24. 如果 >18s/step，按这个顺序优化

## 第一刀：spatial size

例如：

```text
384/480 → 256
```

保持合法尺寸。

## 第二刀：temporal length

改成 LongLive/Wan 支持的更小合法 frame count。

不要破坏 temporal VAE 要求。

## 第三刀：关闭 LoRA

今晚只 Adapter。

## 第四刀：减少 validation frequency

```text
500 → 750
```

## 第五刀：DataLoader

增加：

```text
num_workers
persistent_workers
prefetch
```

并检查 CPU 解码是否是瓶颈。

## 第六刀：gradient checkpointing

**不要为了速度主动开启。**

只有 OOM 才开，因为 checkpointing 会换显存、牺牲速度。

---

# 25. 数据解码性能

视频模型很容易 GPU 等 CPU decode。

训练日志必须定期显示：

```text
data_time
forward_time
backward_time
optimizer_time
```

如果：

```text
data_time / step_time > 0.2
```

说明数据 decode 成为瓶颈。

解决：

- `decord` / PyAV；
- `persistent_workers=True`；
- `pin_memory=True`；
- `prefetch_factor=2~4`；
- 避免每 step 调用外部 ffmpeg；
- 视频 metadata 提前写好，不在训练时反复 ffprobe。

---

# 26. 今晚评估方法

不要跑完整 VBench。

我们只回答 recovery。

固定 8–16 个 val source video：

1. 取同一段 GT；
2. 在 anchor 前制造相同 artificial drift；
3. 使用相同随机 seed；
4. 分别跑：

```text
A: no anchor
B: hard anchor
C: learned re-anchor
```

---

## 26.1 简单 metrics

至少输出：

```json
{
  "future_latent_mse_no_anchor": 0.0,
  "future_latent_mse_hard_anchor": 0.0,
  "future_latent_mse_learned_anchor": 0.0,
  "future_lpips_no_anchor": 0.0,
  "future_lpips_hard_anchor": 0.0,
  "future_lpips_learned_anchor": 0.0,
  "recovery_win_rate_hard": 0.0,
  "recovery_win_rate_learned": 0.0
}
```

### Future recovery window

分别测：

```text
anchor 后 ~0.5s
anchor 后 ~1s
anchor 后 ~2s
```

若视频太短，至少测 1s 左右。

---

# 27. qualitative 结果更重要

为每个 validation case 生成 contact sheet / 视频：

```text
GT
No Anchor
Hard Real Anchor
Learned ReAnchor
```

在画面上标记：

```text
ANCHOR ARRIVES HERE
```

我们今晚最想看到：

```text
anchor 前：No / Ours 都发生 drift
anchor 时：输入相同真实图片
anchor 后：Ours 更快回到 GT scene / identity / motion
```

---

# 28. 今晚 12 小时时间表

## 00:00–00:30：机器检查

完成：

```text
GPU
disk
CUDA
Python
tmux
目录创建
```

创建 `STATUS.md`：

```markdown
# STATUS
Start time:
Deadline:
GPU:
Disk free:
Current phase:
Blockers:
```

---

## 00:30–01:30：环境 + 并行下载

并行运行两个 tmux pane：

### Pane A

```text
clone LongLive v1.0
install environment
```

### Pane B

```text
download Wan2.1 ModelScope
download LongLive checkpoint
```

下载模型时不要空等。

---

## 01:00–02:30：数据同步下载

第三个 pane：

```text
ModelScope Youku subset
target 6–8GB
```

如果网络慢：

```text
max 3–5GB
```

数据数量优先于 bytes。

---

## 01:30–03:00：Baseline inference

必须完成：

```text
LongLive official short inference
```

输出：

```text
outputs/baseline/smoke.mp4
```

如果失败，先修它。

---

## 02:30–04:00：数据 manifest + loader

完成：

```text
train.jsonl
val.jsonl
dataset_stats.json
```

至少：

```text
300 train
40 val
```

---

## 03:30–05:00：Hard Anchor

找到：

```text
VAE encode
history state
KV/cache refresh
```

实现 Hard Anchor。

Hour 5 前至少输出：

```text
no_anchor.mp4
hard_anchor.mp4
```

这是今晚第一个关键 milestone。

---

## 05:00–06:00：Learned Adapter

实现：

```text
GatedAnchorAdapter
Artificial Drift
Adapter-only optimizer
```

跑 unit test。

---

## 06:00–06:30：50-step smoke

必须验证：

```text
loss finite
4 GPU
trainable params correct
save/reload
```

---

## 06:30–07:15：200-step benchmark

记录 `sec/step`。

根据真实速度自动修改 max_steps。

---

## 07:15–10:30：主训练

目标：

```text
1500–3000 steps
```

checkpoint：

```text
500
1000
1500
2000
...
```

如果 1000 step 后 validation 已明显 improvement：

优先保住 checkpoint，不要冒险重构 architecture。

---

## 10:30–11:30：最终 evaluation

固定 validation：

```text
No Anchor
Hard Anchor
Learned Anchor
```

生成：

```text
metrics.json
comparison videos
```

---

## 11:30–12:00：打包

Codex 必须：

1. 保存 latest checkpoint；
2. 保存 git diff；
3. 写 `README_REPRODUCE.md`；
4. 更新 `STATUS.md`；
5. 写清：
   - 完成了什么
   - 没完成什么
   - 真实 sec/step
   - 最佳 checkpoint
   - 数据量
   - metrics
   - 下一步该做什么。

---

# 29. 一键 nightly runner

创建：

```text
scripts/run_night.sh
```

但不要写成不可控的一个巨型脚本。

建议阶段化：

```bash
#!/usr/bin/env bash
set -euo pipefail

bash scripts/00_check_env.sh
bash scripts/01_setup_env.sh
bash scripts/02_download_models.sh

python scripts/03_download_youku_subset.py \
  --output /data/restream_mvp/data/raw/youku \
  --max-gb 8 \
  --max-items 1200 \
  --min-duration 4

python scripts/04_build_manifest.py

bash scripts/05_smoke_infer.sh

# Only if all above succeeded
bash scripts/06_train_mvp.sh

bash scripts/07_eval_mvp.sh
```

实际今晚推荐各阶段独立运行，防止后面的错误隐藏前面成功产物。

---

# 30. 训练恢复必须支持

`train_reanchor.py` 必须支持：

```bash
--resume /data/restream_mvp/checkpoints/step_1000
```

12 小时任务最怕中间机器/进程断。

checkpoint 至少保存：

```text
Adapter state_dict
optimizer
scheduler
global_step
random state
config
```

如果 backbone frozen，不要每 500 step 复制一份 1.3B backbone。

只保存：

```text
small adapter checkpoint
```

节省磁盘与 IO。

---

# 31. 实验 reproducibility

固定：

```text
seed = 42
val source IDs 固定
artificial drift seed 固定
generation seed 固定
```

A/B/C 比较必须使用同一个：

```text
GT clip
prompt
seed
anchor timestamps
corruption
```

否则不能判断是 anchor 作用还是 stochasticity。

---

# 32. 今晚不要宣称“物理约束”

本实验最多证明：

```text
reality grounding
state recovery
long-term consistency 的雏形
```

还不能说：

```text
physics-constrained video generation
```

物理约束后续需要：

```text
depth
camera pose
flow
3D pointmap
object state
```

这不是今晚任务。

---

# 33. 今晚不要做 RL

今晚问题：

> 给了 anchor 后，模型会不会用？

这是监督学习问题。

RL 未来解决：

> 什么时候应该请求 / 使用 anchor？

即：

```text
continue
vs
request_real_anchor
```

今晚如果有人试图加入 GRPO：

**直接停止 scope expansion。**

---

# 34. 成功后第二阶段怎么升级

今晚成功后，明天/后续按：

## Stage A

```text
LongLive 1.3B
Adapter-only
short real clips
```

↓

## Stage B

```text
LongLive 2.0 / Wan2.2-TI2V-5B
Anchor Adapter full train
Backbone LoRA r=16/32
16–32s windows
```

↓

## Stage C

```text
self-rollout streaming fine-tuning
让模型先自己 drift，再 real-anchor recovery
```

↓

## Stage D

```text
Ego-Exo4D selected long takes
60s / 100s evaluation
```

↓

## Stage E

```text
geometry-aware anchors
RGB + depth + camera pose
```

↓

## Stage F

```text
Adaptive Anchor Scheduler
RL / GRPO
```

---

# 35. 后续正式数据集

## Ego-Exo4D

正式论文很适合。

官方提供：

```text
downscaled_takes/448
```

整套约数百 GB，所以以后也不要整套下载。

拿到 credential 后：

1. 先下载 metadata；
2. 挑 selected take UIDs；
3. 只下载这些 UID 的 `downscaled_takes/448`；
4. 目标仍可控制在 50–100GB。

今晚不等待该数据集。

---

# 36. ModelScope 数据策略总结

今晚：

```text
Wan base model:
ModelScope ✅

small real-video dataset:
ModelScope ✅

LongLive official 1.3B checkpoint:
Hugging Face official repo
```

这是更可靠的组合。

不要为了“全魔搭”使用未经验证的 LongLive third-party checkpoint。

---

# 37. Codex 需要自己生成的文件

除了修改 LongLive 必要的小 hook，新增代码尽量放在 `/data/restream_mvp`：

```text
restream/
  dataset.py
  corruption.py
  anchor_adapter.py
  anchor_injector.py
  metrics.py

train_reanchor.py
eval_reanchor.py
```

LongLive patch 要尽量小。

最后：

```bash
cd /data/restream_mvp/code/LongLive
git diff > /data/restream_mvp/longlive.patch
```

---

# 38. 单元测试

至少创建简单 tests：

## `test_anchor_adapter.py`

检查：

```text
input shape == output shape
gradient exists
gate finite
BF16 works
```

## `test_dataset.py`

检查：

```text
video decodes
anchors in bounds
same source not across train/val
tensor finite
```

## `test_injector.py`

检查：

```text
hard anchor really changes target latent
cache/state rebuild does not crash
```

---

# 39. 数据统计文件

`dataset_stats.json`：

```json
{
  "download_gb": 0,
  "total_source_videos": 0,
  "usable_videos": 0,
  "train_sources": 0,
  "val_sources": 0,
  "mean_duration_sec": 0,
  "median_duration_sec": 0,
  "min_duration_sec": 0,
  "max_duration_sec": 0
}
```

---

# 40. STATUS.md 最终模板

```markdown
# ReStream MVP Status

## Hardware
- GPUs:
- CUDA:
- PyTorch:
- Peak VRAM:

## Data
- Source:
- Download size:
- Train source videos:
- Val source videos:
- Window duration:
- Frame shape:

## Models
- Wan base:
- LongLive:
- Checkpoint paths:

## Baseline
- Inference success:
- Output:

## Hard ReAnchor
- Success:
- Injection implementation:
- Cache strategy:

## Learned ReAnchor
- Trainable params:
- Steps:
- sec/step:
- Training wall time:
- Best checkpoint:

## Metrics
- No Anchor:
- Hard Anchor:
- Learned Anchor:

## Qualitative
- Best comparison video:

## Known Problems
1.
2.

## Next Steps
1.
2.
3.
```

---

# 41. 12h Deadline 的降级策略

如果到第 6 小时：

### 状况 A：模型还没跑通

放弃训练，集中修 baseline。

### 状况 B：Baseline 通，但 Hard Anchor 不通

只研究 state/cache injection。

### 状况 C：Hard Anchor 通，Adapter 还没写完

Hard Anchor + A/B evaluation 已经是有效 MVP。

### 状况 D：Adapter 能训练但速度慢

只跑：

```text
800–1500 steps
```

不要为了 3000 steps 超时。

### 状况 E：Adapter 训练有 improvement

立即保存！

不要在最后 2 小时大改 architecture。

---

# 42. 今晚的科学结论应该是什么

我们不是证明：

> “我们已经解决 100s 物理一致性视频生成。”

而是验证：

\[
\boxed{
\text{Sparse real-world visual observations can causally re-anchor a drifting autoregressive video state.}
}
\]

如果这个现象成立，后面才有理由扩展：

```text
100s
adaptive anchors
geometry
world model
RL
```

---

# 43. 实验最小对照表

最终至少整理：

| Variant | Real Anchor | Learnable | Drift Input | Future Recovery |
|---|---:|---:|---:|---:|
| Base LongLive | ❌ | ❌ | ✅ | baseline |
| Hard Anchor | ✅ | ❌ | ✅ | ? |
| ReStream MVP | ✅ | ✅ | ✅ | ? |

后续论文才加入：

```text
First-frame only
Multi-image offline conditioning
LoRA-only
Self-rollout
Adaptive anchoring
Geometry anchor
```

---

# 44. 今晚实际训练参数建议

首轮：

```text
Model: LongLive 1.3B
GPU: 4 × A100 80GB
Precision: BF16
Distributed: DDP
Backbone: frozen
VAE: frozen
Text encoder: frozen
Adapter: full train
LoRA: OFF
Batch/GPU: 1
Grad accumulation: 1
Gradient checkpointing: OFF unless OOM
Dataset: 2–10 GB ModelScope real videos
Train sources: 300–1200+
Val sources: 40–100
Window: 3.5–8s
Frames: smallest legal LongLive/Wan temporal shape
Spatial short side: target 256, fallback to legal config
LR Adapter: 1e-4
Warmup: 100
Save: 500
Eval: 500
Target steps: decided after 200-step benchmark
Expected: 1500–3000 steps under 12h
```

---

# 45. 如果今晚 Adapter 效果不明显

不要立即宣布 idea 错误。

依次检查：

1. real anchor 是否真的进入 model state；
2. cache 是否在 anchor 后 rebuild；
3. Adapter output magnitude；
4. gate 是否一直接近 0；
5. corruption 是否过轻；
6. corruption 是否过重；
7. future loss 是否真的只监督 anchor 后；
8. frozen model 是否完全忽略 corrected state；
9. anchor time 是否和 latent time 对齐；
10. VAE anchor frame 是否做了相同 normalization。

这是第一晚最可能出现的 bug。

---

# 46. 很重要：pixel time ↔ latent time

真实图片 timestamp：

```text
t_anchor_sec
```

必须正确映射到：

```text
video frame index
↓
VAE temporal latent index
↓
AR chunk index
```

不要直接：

```python
latent_idx = int(seconds * fps)
```

Wan VAE 有 temporal compression。

Codex 必须查看当前 VAE encode 后：

```text
input video T
output latent T'
```

写一个统一函数：

```python
timestamp_to_latent_index(...)
```

并加 unit test。

**这很可能是整个 MVP 最重要的工程细节之一。**

---

# 47. 数据 normalization 一致性

real anchor 必须经过和 GT video 完全相同：

```text
resize
crop
RGB normalization
VAE preprocessing
dtype
```

不能：

```text
video 用 [-1,1]
anchor 用 [0,1]
```

这种 bug 会让结果完全失真。

---

# 48. 不要下载完整 Youku dataset

必须 streaming。

程序终止后检查：

```bash
du -sh /data/restream_mvp/data
du -sh ~/.cache/modelscope 2>/dev/null || true
```

若 ModelScope cache 占太多：

- 确认 hardlink / copy 后；
- 只清理本次不再使用的无关 cache；
- 不要盲删整个用户 cache。

---

# 49. 网络下载与训练并行

如果 dataset 已经有：

```text
300 usable videos
```

就可以先训练。

不要等 1200 个全部下载完。

理想 pipeline：

```text
download dataset
      ↘
      build partial manifest
             ↘
             training
```

后续下载的数据可留给下一轮。

---

# 50. 官方参考（执行前可核对）

## LongLive official code

https://github.com/NVlabs/LongLive

使用：

```text
branch: v1.0
```

## LongLive 1.3B checkpoint

https://huggingface.co/Efficient-Large-Model/LongLive-1.3B

## Wan2.1-T2V-1.3B ModelScope

https://modelscope.cn/models/Wan-AI/Wan2.1-T2V-1.3B

下载：

```bash
modelscope download \
  --model Wan-AI/Wan2.1-T2V-1.3B \
  --local_dir /data/restream_mvp/models/Wan2.1-T2V-1.3B
```

## ModelScope Youku-mPLUG

https://modelscope.cn/datasets/modelscope/Youku-AliceMind

使用：

```text
subset_name=caption
split=train
use_streaming=True
```

## Ego-Exo4D（未来正式数据）

https://docs.ego-exo4d-data.org/

未来只选特定 UID + `downscaled_takes/448`，不下载整套。

---

# 51. 最终一句话给 Codex

> **今晚不要试图完成论文。今晚必须完成一个可复现的、能够通过真实 Anchor 对 streaming video state 进行 causal correction 的 1.3B MVP。先跑通 LongLive，先完成 Hard Anchor，随后只训练一个 tiny gated adapter。数据只取 ModelScope 小子集，绝不因为下载或大模型拖过 12 小时。最终必须留下 checkpoint、A/B 视频、metrics、完整日志和复现脚本。**
