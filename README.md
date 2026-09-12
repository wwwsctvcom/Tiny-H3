# Tiny-H3

**一个与 [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) 工作方式相同的文生音视频（T2AV）模型 —— 完整流程、四种训练方式，一块 RTX 5090 就能训练。**

MiniMax-H3 把文本、视频、音频放进同一条 packed 序列，由一个 omni-modal 扩散 Transformer 一起去噪。Tiny-H3 复刻了这套流程的每一个环节：官方发布的视频/音频 VAE、同款调度器、同款 packed 序列布局与 flow 约定，链路从数据一直通到**带同步声音的可播放 MP4**。区别在规模：官方 DiT 有 24.4B 参数（需要多机多卡），Tiny-H3 的 DiT 缩小到约 64M，**单卡 RTX 5090（32GB）即可完成从数据到成片的全部过程**。

| | 官方 MiniMax-H3 | Tiny-H3 |
|---|---|---|
| DiT | 24.4B（5376 隐层 × 50 层） | **63.7M**（512 × 10 层，另有 140M / 285M preset） |
| 文本编码器 | Qwen3-VL-8B（66GB） | 冻结的 **Qwen3-0.6B**（或 t5-small） |
| 视频/音频 VAE | 官方 | **官方原版，冻结** |
| 潜空间 / packed 序列 / 调度 / flow 符号 | — | **与官方一致** |
| 训练基础设施 | SGLang + Ray + 多机 FSDP2 | 单进程 PyTorch，一块 GPU |

---

## 训练与效果

一块 RTX 5090 上全参训练 4000 步（约 40 分钟，峰值显存 <1GB）：

![loss 曲线](assets/loss_curve.png)

验证集（64 条未参与训练的片段）loss 从 5.04 降到 0.61。

用训练好的模型生成的音视频（与训练数据同一风格：几何图形运动 + 与画面事件同步的音效）：

<p align="center">
  <a href="assets/demos/generated_1_bouncing.mp4">弹跳</a> ·
  <a href="assets/demos/generated_2_pulsing.mp4">脉冲</a> ·
  <a href="assets/demos/generated_3_orbiting.mp4">环绕</a> ·
  <a href="assets/demos/generated_4_swinging.mp4">摆动</a>
  &nbsp;|&nbsp;
  <a href="assets/demos/training_sample_bouncing.mp4">训练集片段对照</a>
</p>

<p align="center">
  <video src="assets/demos/generated_1_bouncing.mp4" controls width="200"></video>
  <video src="assets/demos/generated_2_pulsing.mp4" controls width="200"></video>
  <video src="assets/demos/generated_3_orbiting.mp4" controls width="200"></video>
  <video src="assets/demos/generated_4_swinging.mp4" controls width="200"></video>
</p>

提示词（从左到右）：*a red circle bouncing on a dark background, with rhythmic thumps* · *two squares, one cyan and one yellow, pulsing on a purple background, with steady beats at a fast tempo* · *three rings in purple, white and yellow, orbiting on a blue background, with a rising and falling tone* · *a white ring swinging on a warm background, with soft ticks at a slow tempo*。

---

## 快速开始

前置条件：一块 NVIDIA GPU、Python 3.10+、约 60GB 磁盘。ffmpeg 无需安装（项目自带）。

### 1. 拉取代码

```bash
git clone <本仓库地址> tiny-h3 && cd tiny-h3
```

### 2. 环境（国内镜像）

```bash
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple   # AutoDL 等自带镜像可不设
export HF_ENDPOINT=https://hf-mirror.com                        # env.sh 默认已设置
bash scripts/setup_env.sh
```

看到 `MiniMax-H3 classes importable` 和 `torch cuda available: True` 即成功。裸环境（无 torch）加 `--with-torch`。

### 3. 下载模型组件（约 12GB，走 hf-mirror）

```bash
source scripts/env.sh
bash scripts/download_assets.sh
```

| 组件 | 大小 | 用途 |
|---|---|---|
| H3 视频 VAE + 音频 VAE | 11 GB | 编码训练数据、解码生成结果 |
| Qwen/Qwen3-0.6B | 1.2 GB | 冻结文本编码器（默认） |
| t5-small | 240 MB | 轻量备选编码器 |

### 4. 准备数据

训练数据由程序合成：几何图形按四种运动模式运动，音效与画面事件帧级同步，提示词自动生成，无需下载任何数据集。

```bash
bash scripts/prepare_data.sh synth      # 512 训练 + 64 验证片段（CPU，几分钟）
bash scripts/prepare_data.sh latents    # 用官方 VAE 编码 latent 缓存（GPU，几分钟）
```

### 5. 训练

```bash
python -m tiny_h3.train.train_full --latents $TINY_H3_DATA/synth/latents --out runs/full --steps 4000
```

5090 上约 0.6 秒/步（4000 步约 40 分钟）。loss 实时打印并写入 `runs/full/metrics.jsonl`，每 500 步保存 diffusers 格式 checkpoint。

### 6. 生成

```bash
bash scripts/generate_demo.sh runs/full/final
```

输出标准 MP4（H.264 + AAC 立体声）。自定义提示词（训练数据为英文模板，建议用英文）：

```bash
python -m tiny_h3.pipeline --checkpoint runs/full/final \
  --prompts "a green square bouncing on a dark background, with rhythmic thumps" \
  --out outputs/my_first_demo
```

想先快速验证全流程：`bash scripts/run_all.sh --fast`（约 15 分钟跑完小规模的"数据→训练→生成"）。

---

## 四种训练方式

共享同一个数据集、同一个 DiT、同一个 flow-matching 损失（`src/train/core.py`）：

| 方式 | 命令 | 说明 |
|---|---|---|
| 全参微调 | `python -m tiny_h3.train.train_full --latents $TINY_H3_DATA/synth/latents --out runs/full` | 默认起点 |
| LoRA | `python -m tiny_h3.train.train_lora --latents ... --out runs/lora --rank 32` | 只训 3.6M 适配器参数；`--base <checkpoint>` 可继续叠加 |
| FSDP | `torchrun --standalone --nproc_per_node=1 -m tiny_h3.train.train_fsdp --latents ... --out runs/fsdp` | 多卡改 `nproc_per_node`；单卡也跑通完整流程 |
| Flow-GRPO | `python -m tiny_h3.train.train_flow_grpo --base runs/full/final --prompt-data $TINY_H3_DATA/synth/val.jsonl --out runs/grpo` | 强化学习对齐：采样→解码→离线 reward（提示词符合 0.55 + 音画同步 0.30 + 质量 0.15）→ PPO 更新 |

所有 checkpoint 直接交给 `python -m tiny_h3.pipeline --checkpoint <目录>` 生成视频。

---

## 数据下载

默认数据程序化生成，零下载。使用真实素材：

```bash
python tools/download_assets.py --group data
```

| 数据集 | 规模 | 说明 |
|---|---|---|
| `rockdu/WISA-80K-Practical-Dynamics-254` | 254 条真实片段 | 动态场景 |
| concerts_audiovideo_dataset 子集 | ~1GB | 带同步音频的真实演出 |

用自己的 footage：准备目录 + `train.jsonl`（每行 `{"prompt": "...", "metadata": {"video": "clips/x.mp4"}}`），然后 `python tools/prepare_latents.py --data-dir <目录> --out <目录>/latents`。帧数需在 `17n+5` 网格（22、39、56…）、宽高为 32 的倍数。

单文件下载慢时，多线程断点续传：

```bash
python tools/fetch_file.py --url <resolve 链接> --out <本地文件> --workers 10 --sha256 <摘要>
```

---

## 常见问题

**训练 loss 正常，生成却是噪声？**
检查 flow 符号（H3 预测 `v = x0 - noise`，与常见约定相反）和 VAE 归一化，两者都封装在 `src/vae.py`，不要绕开手写。

**显存够吗？**
DiT 训练峰值 <1GB；VAE 编解码约 11GB（float32），但训练用 latent 缓存、全程不碰 VAE。

**换文本编码器？**
支持 `Qwen3-0.6B`（默认）、`Qwen2.5-0.5B`、`t5-small`。换编码器会改变 `text_dim`，需重新执行 `tools/prepare_latents.py`。

---

## 许可

项目代码：**Apache-2.0**（见 [LICENSE](LICENSE)）。模型与 VAE 来自 [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)（遵循其自身许可），模型类与调度器来自 [diffusers](https://github.com/huggingface/diffusers)，RL 公式化参考 [miles-diffusion](https://github.com/radixark/miles_diffusion) 与 [Flow-GRPO](https://github.com/yifan123/flow_grpo)。
