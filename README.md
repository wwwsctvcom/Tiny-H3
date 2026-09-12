# Tiny-H3

**一个与 [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) 工作方式完全相同的文生音视频（T2AV）模型 —— 完整流程、四种训练方式，一块 RTX 5090 就能训练。**

MiniMax-H3 把文本、视频、音频放进同一条 packed 序列，由一个 omni-modal 扩散 Transformer 一起去噪。Tiny-H3 复刻了这套流程的每一个环节：官方发布的视频/音频 VAE、同款调度器、同款 packed 序列布局与 RoPE、同款 flow 约定，链路从数据一直通到**带同步声音的可播放 MP4**。唯一的区别是规模：官方 DiT 有 24.4B 参数（需要多机多卡），Tiny-H3 的 DiT 缩小到约 64M，**单卡 RTX 5090（32GB）即可完成从数据到成片的全部过程**。

| | 官方 MiniMax-H3 | Tiny-H3 |
|---|---|---|
| DiT | 24.4B（5376 隐层 × 50 层） | **63.7M**（512 × 10 层，另有 140M / 285M preset） |
| 文本编码器 | Qwen3-VL-8B（66GB） | 冻结的 **Qwen3-0.6B**（或 t5-small） |
| 视频/音频 VAE | 官方 | **官方原版，冻结**（这是单卡能出真实效果的关键） |
| 潜空间 / packed 序列 / 调度 / flow 符号 | — | **与官方逐位一致** |
| 训练基础设施 | SGLang + Ray + 多机 FSDP2 | 单进程 PyTorch，一块 GPU |

项目专为刚入门的学习者准备：

* **命令即所得**：环境、数据、训练、生成各只有一个入口命令，每步都有预期输出和耗时；
* **只 loss 一个损失函数**（flow-matching），训练曲线一眼看懂；
* **每一步都可验证**：数据检查 → VAE 重建 → 冒烟测试 → 训练 → 生成，环环有工具；
* **模块解耦、代码可读**：打包、模型、VAE、采样、训练互不纠缠，读代码只需顺着一条链往下走。

---

## 真实训练记录与效果

一块 RTX 5090 上全参训练 4000 步（约 40 分钟，batch 2×4、bf16、峰值显存 <1GB）：

![loss 曲线](assets/loss_curve.png)

总 loss 从 5.0 收敛到 0.6；红色圆点是**验证集**（64 条未参与训练的片段）上的评估，从 5.04 降到 0.61——模型学到的是泛化能力，不是背诵训练集。

用训练好的模型生成的音视频（几何图形运动 + 与画面事件同步的真实音效，与训练数据同一风格）：

<p align="center">
  <a href="assets/demos/generated_1_bouncing.mp4">弹跳 + 节奏砰声</a> ·
  <a href="assets/demos/generated_2_pulsing.mp4">脉冲 + 稳定节拍</a> ·
  <a href="assets/demos/generated_3_orbiting.mp4">环绕 + 起伏音调</a> ·
  <a href="assets/demos/generated_4_swinging.mp4">摆动 + 轻响</a>
</p>

<p align="center">
  <video src="assets/demos/generated_1_bouncing.mp4" controls width="200"></video>
  <video src="assets/demos/generated_2_pulsing.mp4" controls width="200"></video>
  <video src="assets/demos/generated_3_orbiting.mp4" controls width="200"></video>
  <video src="assets/demos/generated_4_swinging.mp4" controls width="200"></video>
</p>

* 生成提示词（从左到右，与训练数据同一模板）：
  1. *a red circle bouncing on a dark background, with rhythmic thumps*
  2. *two squares, one cyan and one yellow, pulsing on a purple background, with steady beats at a fast tempo*
  3. *three rings in purple, white and yellow, orbiting on a blue background, with a rising and falling tone*
  4. *a white ring swinging on a warm background, with soft ticks at a slow tempo*
* 训练集原始片段对照：[training_sample_bouncing.mp4](assets/demos/training_sample_bouncing.mp4)（这就是模型学习的目标样式）
* 以上 demo **未做任何挑选**，用下文的默认命令生成。如实说明：63M 小模型 + 4000 步，颜色、背景、运动模式已与提示词一致，画面清晰度受模型容量限制——换更大 preset（`--preset tiny_h3_xl`）或训练更久即可提升。

---

## 快速开始（五步跑通）

前置条件：一块 NVIDIA GPU（显存越大越宽松，5090/4090/A100 均可）、Python 3.10+、约 60GB 磁盘。ffmpeg 无需安装（项目自带静态版）。

### 第 1 步：拉取代码

```bash
git clone <本仓库地址> tiny-h3 && cd tiny-h3
```

### 第 2 步：一键搭环境（国内镜像）

```bash
# pip 走清华镜像（AutoDL 等环境自带阿里镜像，可不设）
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
# HuggingFace 走 hf-mirror（scripts/env.sh 默认已设置）
export HF_ENDPOINT=https://hf-mirror.com

bash scripts/setup_env.sh
```

脚本依次完成：安装 Python 依赖 → 安装 pinned 版 diffusers（H3 模型类在 diffusers main 分支，脚本优先从国内 git 镜像 clone，失败自动回落 GitHub）→ `pip install -e .` 注册命令 → 打印自检结果（看到 `MiniMax-H3 classes importable` 和 `torch cuda available: True` 即成功）。
裸环境（还没装 torch）加 `--with-torch`。

### 第 3 步：下载模型组件（约 12GB，自动走 hf-mirror）

```bash
source scripts/env.sh
bash scripts/download_assets.sh
```

| 组件 | 大小 | 用途 |
|---|---|---|
| H3 视频 VAE + 音频 VAE | 11 GB | 官方冻结 VAE：训练前编码数据、生成后解码成片 |
| Qwen/Qwen3-0.6B | 1.2 GB | 冻结文本编码器（默认） |
| t5-small | 240 MB | 轻量备选编码器 |

官方的 66GB DiT 和 66GB 文本编码器**故意不下载**——Tiny-H3 训练自己的小 DiT。

### 第 4 步：准备数据（默认零下载）

训练数据由程序合成：几何图形按四种运动模式（弹跳/脉冲/环绕/摆动）运动，音效与画面事件帧级同步，提示词自动生成——所以 reward 可以离线验证，新手也无需任何数据集。

```bash
bash scripts/prepare_data.sh synth      # 生成 512 训练 + 64 验证片段（CPU，几分钟）
bash scripts/prepare_data.sh latents    # 用官方 VAE 编码 latent 缓存（GPU，几分钟）
```

### 第 5 步：训练，然后生成

```bash
# 训练（5090 上约 0.6 秒/步，4000 步约 40 分钟）
python -m tiny_h3.train.train_full --latents $TINY_H3_DATA/synth/latents --out runs/full --steps 4000

# 用训练好的模型生成（4 个预设提示词）
bash scripts/generate_demo.sh runs/full/final

# 或用自定义提示词（训练数据是英文模板，建议用英文提示词）
python -m tiny_h3.pipeline --checkpoint runs/full/final \
  --prompts "a green square bouncing on a dark background, with rhythmic thumps" \
  --out outputs/my_first_demo
```

生成结果是标准 MP4（H.264 视频 + AAC 立体声）外加一份 `manifest.json`。想先快速验证全流程？`bash scripts/run_all.sh --fast` 用 15 分钟跑一遍小规模的"数据→训练→生成"。

---

## 四种训练方式

四种方式共享同一个数据集、同一个 DiT、同一个 flow-matching 损失（定义在 `src/train/core.py`），只是"训练哪些参数"和"怎么启动"不同：

| 方式 | 命令 | 什么时候用 |
|---|---|---|
| 全参微调 | `python -m tiny_h3.train.train_full --latents $TINY_H3_DATA/synth/latents --out runs/full` | 默认起点，效果最完整 |
| LoRA | `python -m tiny_h3.train.train_lora --latents ... --out runs/lora --rank 32` | 只训 3.6M 适配器参数；`--base <checkpoint>` 可在已有模型上继续叠加 |
| FSDP | `torchrun --standalone --nproc_per_node=1 -m tiny_h3.train.train_fsdp --latents ... --out runs/fsdp` | 多卡并行（把 `nproc_per_node` 改成卡数）；单卡也能跑通完整封装/保存路径 |
| Flow-GRPO | `python -m tiny_h3.train.train_flow_grpo --base runs/full/final --prompt-data $TINY_H3_DATA/synth/val.jsonl --out runs/grpo` | 在已训练模型上做强化学习对齐：采样→官方 VAE 解码→离线可验证 reward（提示词符合度 0.55 + 音画同步 0.30 + 质量分 0.15）→ 组内归一化优势 → PPO 裁剪更新 |

训练产物：`metrics.jsonl`（loss 曲线数据）、diffusers 格式 checkpoint 目录（`from_pretrained` 可直接加载）、GRPO/LoRA 的 PEFT adapter。所有 checkpoint 都可以直接交给 `python -m tiny_h3.pipeline --checkpoint <目录>` 生成视频。

---

## 数据下载

**默认路径零数据集下载**（第 4 步的程序化数据）。想做真实素材时：

### 可选数据集（走 hf-mirror 自动下载）

```bash
python tools/download_assets.py --group data
```

| 数据集 | 规模 | 说明 |
|---|---|---|
| `rockdu/WISA-80K-Practical-Dynamics-254` | 254 条真实片段 | 动态场景，按 H3 的服务网格组织 |
| concerts_audiovideo_dataset 子集 | ~1GB | 带同步音频的真实演出画面 |

### 用自己的 footage

准备一个目录，里面放 `train.jsonl`（每行 `{"prompt": "...", "metadata": {"video": "clips/x.mp4"}}`）和片段文件，然后：

```bash
python tools/prepare_latents.py --data-dir <你的目录> --out <你的目录>/latents
```

要求：帧数在 `17n+5` 网格上（22、39、56…）、宽高为 32 的倍数，否则 VAE 会补边导致音画时钟漂移。

### 下载慢？

所有 HuggingFace 下载默认走 `hf-mirror.com`。单个大文件仍慢时，用多线程断点续传（按整文件 sha256 校验）：

```bash
python tools/fetch_file.py --url <resolve 链接> --out <本地文件> --workers 10 --sha256 <摘要>
```

---

## 常见问题

**训练 loss 正常，生成出来却是噪声？**
先查 flow 符号：H3 预测 `v = x0 - noise`（与常见约定相反），再查 VAE 归一化（ImageNet 像素统计 + latents_mean/std）。两者都封装在 `src/vae.py`，出现这类问题通常是自己绕开了它手写了一版。

**显存够吗？**
63M 的 DiT 训练峰值 <1GB；真正的大头是 VAE 的编解码（float32 约 11GB），而且训练用 latent 缓存、全程不碰 VAE。生成时 VAE 解码约 12GB。

**`python -m tiny_h3.…` 是什么写法？**
包内模块用相对导入，`-m` 提供包上下文。`pip install -e .` 之后也可以用等价的短命令：`tiny-h3-full`、`tiny-h3-lora`、`tiny-h3-fsdp`、`tiny-h3-grpo`、`tiny-h3-generate`。`tools/` 下的脚本则可以直接 `python tools/xxx.py` 运行。

**VAE 参与训练吗？**
不参与，默认必须冻结——它是单卡能出真实效果的根基。若你的领域离自然视频太远（用 `tools/reconstruct_data.py` 重建测试能看出明显退化），`tools/finetune_vae.py` 只微调解码器，之后需重新编码数据并重训 DiT。

**换个文本编码器行不行？**
行：`Qwen3-0.6B`（默认）、`Qwen2.5-0.5B`、`t5-small`。注意换编码器会改变 `text_dim`，需要重新执行 `tools/prepare_latents.py`（训练器检测到缓存不匹配会拒绝训练）。`--text-layer N` 可以像官方 H3 读第 50 层那样读中间层。

---

## 许可与致谢

项目代码：**Apache-2.0**（见 [LICENSE](LICENSE)）。

站在巨人的肩膀上：[MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)（模型与 VAE，遵循其自身许可）· [diffusers](https://github.com/huggingface/diffusers)（模型类、调度器、LoRA）· [miles-diffusion](https://github.com/radixark/miles_diffusion) 与 [Flow-GRPO](https://github.com/yifan123/flow_grpo)（RL 公式化）。可选数据集 `rockdu/WISA-80K-Practical-Dynamics-254` 与 `alejandroparedeslatorre/concerts_audiovideo_dataset` 版权归原作者所有。
