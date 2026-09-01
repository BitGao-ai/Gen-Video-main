# 真实服务器训练运行命令（L-COCF / COCF-SS-DCA）

本文件记录在真实 GPU 服务器（`/data/suqiang/Text2Video-main`）上完成三阶段训练的完整命令：
Stage A（反事实教师数据生成）→ Stage B（插件联合训练）→ Stage C（端到端轻量微调）。

> 前置依赖：`pip install torch scipy scikit-image einops pyyaml lmdb diffusers>=0.34 transformers accelerate`
>
> 本地权重目录（离线服务器，勿用 HF 在线 id）：
> - Wan2.2-A14B：`/data/suqiang/Text2Video-main/weights/Wan2.2`
> - SAM：`/data/suqiang/Text2Video-main/weights/SAM`
> - DINOv2：`/data/suqiang/Text2Video-main/weights/DINO`
> - CLIP：`/data/suqiang/Text2Video-main/weights/Clip`
> - RAFT（光流，torchvision）：放进 torch hub 缓存，见下节「前置：RAFT 光流权重准备」

---

## 前置：RAFT 光流权重准备（离线服务器必做）

`--real-models` 会构建**两条互相独立的 RAFT**，架构不同、不能互换：

| 用途 | 构建位置 | variant | torchvision 权重 |
|---|---|---|---|
| 感知（`motion_phase` / `s_A` / affinity 的 flow·IoU 项） | `cocf/tubes/model_perception.py:571` | `raft_large` | `Raft_Large_Weights.DEFAULT` = `C_T_SKHT_V2` |
| 损伤指标（两条 damage 轴） | `cocf/data/metrics.py:455` | `raft_small` | `Raft_Small_Weights.DEFAULT` = `C_T_V2` |

`ModelMetricExtractor` 的 `share_from` 只复用 DINOv2/CLIP（`metrics.py:325`），RAFT 是重新加载的；
**Stage A 和 Stage C 都同时构建这两条**，所以两个权重必须都备齐。

而 `--raft-weights` 只能接一个文件路径，同一个值会被同时喂给 large 和 small
（`cocf/common/vram.py:320` 与 `:336`）——给了 `raft_large.pth`，small 那边就会
`load_state_dict` 报 size mismatch，`--real-models` 下直接抛 `RaftUnavailable`。
因此**离线机器的正确做法是预置 torch hub 缓存，然后不传 `--raft-weights`**：
torchvision 仅在缓存文件不存在时才联网（`torch/hub.py` 的 `if not os.path.exists(cached_file)`），
文件名必须与 URL 的 basename 完全一致（含哈希后缀），命中缓存时不校验哈希、不发网络请求。

在**有网**的机器上下载，再整个目录拷到服务器同名路径：

```bash
DIR="$(python -c 'import torch; print(torch.hub.get_dir())')/checkpoints"  # 默认 ~/.cache/torch/hub/checkpoints
mkdir -p "$DIR"
wget -P "$DIR" https://download.pytorch.org/models/raft_large_C_T_SKHT_V2-ff5fadd5.pth  # 20.1 MB
wget -P "$DIR" https://download.pytorch.org/models/raft_small_C_T_V2-01064c6d.pth       # 3.8 MB
```

自检（在仓库根目录跑，两行都应打印 `... loaded and verified at load`；`load_raft` 内置
合成位移探针，会拒绝加载成功但输出零流场的坏权重）：

```bash
python - <<'PY'
import torch
from cocf.common.raft import load_raft
dev = "cuda" if torch.cuda.is_available() else "cpu"
for v in ("large", "small"):
    print(v, load_raft(dev, variant=v, required=True) is not None)
PY
```

- **不要**用 princeton-vl 官方 RAFT 仓库的 `raft-things.pth` / `raft-sintel.pth`：那是
  DataParallel 存的（`module.` 前缀 + 不同模块命名），塞不进 torchvision 的 `raft_large`。
- `--raft-weights` 只在**单 variant** 场景下可用（例如只加 `--real-perception` 而不加
  `--real-metrics`），此时传 `raft_large.pth` 才是对的。全流程请走缓存预置。
- 缓存目录可用 `TORCH_HOME` 环境变量改写；8 个分片进程共读同一份缓存没有问题。

---

## Stage A：反事实教师数据生成（真实模型）

真实 backbone（Wan2.2-A14B）+ 真实感知（SAM/DINOv2/CLIP/RAFT）+ 真实损伤指标。
只有 `--real-models` 生成的数据才可用于训练插件。

> 跑之前先做完上一节的 RAFT 缓存预置，否则 `build_perception_and_metrics` 会在加载感知模型
> 时直接抛 `RaftUnavailable`。下面所有命令都**不传** `--raft-weights`。

### 单卡运行（a14b-t2v，40 GB 显卡，49×384×640）

```bash
python scripts/data/generate_counterfactual_data.py \
    --openvid-csv datasets/OpenVidHD.csv \
    --data-root datasets --video-subdir videos \
    --only-existing-videos \
    --processed-root ./LCOCF_OpenVid1M_Processed \
    --backbone wan22 --wan-variant a14b-t2v \
    --model-path /data/suqiang/Text2Video-main/weights/Wan2.2 \
    --backbone-dtype bfloat16 \
    --num-frames 49 --height 384 --width 640 \
    --vae-tile 256 --metric-frame-chunk 4 \
    --real-models --sam-points-per-crop 16 --perception-dtype bfloat16 \
    --sam-model /data/suqiang/Text2Video-main/weights/SAM \
    --dino-model /data/suqiang/Text2Video-main/weights/DINO \
    --clip-model /data/suqiang/Text2Video-main/weights/Clip \
    --limit 2600 --device cuda
```

- 不加 `--use-real-video`：轨迹锚定在 caption 上（文本到视频），会持久化 `z_init`，Stage C 可复用缓存 baseline。
- `--limit 2600` 仅限前 2600 个 clip（冒烟/小批量）；全量跑时删除该参数。
- 中断恢复：该脚本可断点续跑（`counterfactual_lmdb/_progress.s00.jsonl` 记录已完成 clip），直接重跑同一条命令即可跳过已处理 clip。
- **必须完整跑到底**：§1.6 finalize（写出 `splits/`、`sample_index.csv`、`manifest.json`、`norm_stats.json`）只在所有 clip 处理完后自动执行；日志出现 `§1.6 finalize: indexed ...` 和 `Stage A complete` 才算完成，否则 Stage B 的 preflight 会报 store 未就绪。

### 多卡分片并行（例如 8 张 GPU，每张卡一个进程）

> ⚠️ **先看宿主内存，不是显存。** 每个分片进程稳态约 37 GB 常驻 CPU 内存（换出的 26 GB
> 空闲专家 + 11 GB umT5），8 进程约 300 GB。先 `free -g`；不足就把 `--num-shards` 降到 4。
>
> ⚠️ **离线机器先完成「前置：RAFT 光流权重准备」。** 缺了 RAFT，光流恒为零，`motion_phase`
> 和因果动作强度 `s_A` 会一路是 0。`--real-models` 下现在会直接抛 `RaftUnavailable` 而不是
> 静默降级——把 raft_large + raft_small 两个权重都预置进 torch hub 缓存，然后**不要**传
> `--raft-weights`（单个路径喂不了两个 variant）。

```bash
# 8 个进程使用完全相同的参数，仅 CUDA_VISIBLE_DEVICES 和 --shard-index 不同（0..7）
CUDA_VISIBLE_DEVICES=0 python scripts/data/generate_counterfactual_data.py \
    --openvid-csv datasets/OpenVidHD.csv --data-root datasets --video-subdir videos \
    --only-existing-videos \
    --backbone wan22 --wan-variant a14b-t2v \
    --model-path /data/suqiang/Text2Video-main/weights/Wan2.2 \
    --backbone-dtype bfloat16 \
    --num-frames 49 --height 384 --width 640 \
    --vae-tile 256 --metric-frame-chunk 4 \
    --real-models --sam-points-per-crop 16 --perception-dtype bfloat16 \
    --sam-model /data/suqiang/Text2Video-main/weights/SAM \
    --dino-model /data/suqiang/Text2Video-main/weights/DINO \
    --clip-model /data/suqiang/Text2Video-main/weights/Clip \
    --limit 2600 --device cuda \
    --num-shards 8 --shard-index 0
# ... --shard-index 1 / 2 / ... / 7 同理
```

### 全部分片完成后，合并索引（只执行一次）

```bash
python scripts/data/generate_counterfactual_data.py \
    --openvid-csv datasets/OpenVidHD.csv --data-root datasets --video-subdir videos \
    --processed-root ./LCOCF_OpenVid1M_Processed \
    --finalize-only
```

### VRAM 调优提示

- 如仍 OOM：降低 `--vae-tile` 至 128，再加 `--metric-frame-chunk 2`，再降渲染分辨率。
- `--no-baseline`：跳过占空间的 `full_baseline` 桶（Stage B 不读取；但 Stage C 需要它，若跑 Stage C 则不要加）。
- 40 GB 卡跑 a14b-t2v：单专家常驻 + text encoder 独占换入换出（默认开启 offload），无需额外参数。

---

## Stage B：插件联合训练

不加载任何大模型权重（仅从 store 的 `stage_a_env.json` 读取 `token_dim` 确定插件网络宽度），在缓存的标签上联合训练 L-COCF predictor + strength 权重、STA 平滑、RAEC certificate、CMSC 对齐。

```bash
python scripts/train/train_stage_b.py \
    --processed-root ./LCOCF_OpenVid1M_Processed \
    --batch_size 32 --num_epochs 10 --mixed-precision \
    --num_workers 8 \
    --device cuda \
    --checkpoint_save ./checkpoints/stage_b_final.pt
```

8 卡数据并行（每个 rank 拿每个动作桶的一个跨步切片，动作 1:1:1:1 在 rank 内依然成立；
梯度每步 all-reduce，只有 rank 0 写 checkpoint）：

```bash
torchrun --standalone --nproc_per_node=8 scripts/train/train_stage_b.py \
    --processed-root ./LCOCF_OpenVid1M_Processed \
    --batch_size 32 --num_epochs 10 --mixed-precision \
    --num_workers 8 \
    --checkpoint_save ./checkpoints/stage_b_final.pt
```

- Stage B 只训练约 7 M 参数的插件，**瓶颈是样本存储的随机读而不是算力**。先把
  `--num_workers` 调上去；样本量在百万级以内时单卡通常就够。
- 调整学习率：加 `--lr 1e-4`
- 设备自动检测（有 GPU 用 cuda，否则 cpu），可用 `--device` 显式指定
- 小批量冒烟：`--batch_size 8 --num_epochs 1`（训练集不足 batch_size 时脚本自动 clamp）

---

## Stage C：端到端轻量微调

加载 Stage B checkpoint，将插件嵌入完整加速推理管线并对可微插件做端到端微调，目标为 Stage A 持久化的 `Y_full` 基线。骨干（Wan2.2-A14B）全程冻结。

```bash
python scripts/train/train_stage_c.py \
    --processed-root ./LCOCF_OpenVid1M_Processed \
    --checkpoint_load ./checkpoints/stage_b_final.pt \
    --checkpoint_save ./checkpoints/stage_c_final.pt \
    --backbone wan22 --wan-variant a14b-t2v \
    --model-path /data/suqiang/Text2Video-main/weights/Wan2.2 \
    --num-frames 49 --height 384 --width 640 \
    --vae-tile 128 --grad-window-steps 1 --decode-grad-frames 4 \
    --real-models --metric-frame-chunk 2 \
    --sam-model /data/suqiang/Text2Video-main/weights/SAM \
    --dino-model /data/suqiang/Text2Video-main/weights/DINO \
    --clip-model /data/suqiang/Text2Video-main/weights/Clip \
    --batch_size 1 --num_epochs 3 \
    --device cuda
```

8 卡数据并行（硬样本优先采样按 rank 分片；某个 rank 的 batch 失败会由 `all_agree`
让全体一起跳过，不再让其余 rank 挂在 all-reduce 上）：

```bash
torchrun --standalone --nproc_per_node=8 scripts/train/train_stage_c.py \
    --processed-root ./LCOCF_OpenVid1M_Processed \
    --checkpoint_load ./checkpoints/stage_b_final.pt \
    --checkpoint_save ./checkpoints/stage_c_final.pt \
    --backbone wan22 --wan-variant a14b-t2v \
    --model-path /data/suqiang/Text2Video-main/weights/Wan2.2 \
    --num-frames 49 --height 384 --width 640 \
    --vae-tile 128 --grad-window-steps 1 --decode-grad-frames 4 \
    --real-models --metric-frame-chunk 2 \
    --sam-model /data/suqiang/Text2Video-main/weights/SAM \
    --dino-model /data/suqiang/Text2Video-main/weights/DINO \
    --clip-model /data/suqiang/Text2Video-main/weights/Clip \
    --batch_size 1 --num_epochs 3
```

- 几何、教师步数与 `flow_shift` 自动从 `stage_a_env.json` 读取，显式传参若与 store 不一致会警告
  （缓存 baseline 不命中，每 batch 重算基线）。
- Stage C 的 `--real-models` 同样同时构建 raft_large（感知）+ raft_small（损伤指标），
  两个权重都要在 torch hub 缓存里，见开头「前置：RAFT 光流权重准备」。
- Stage C 会优先复用 Stage A 落盘的 `text_embeds/<video_id>.pt`，不再每个 batch 重跑 umT5
  （默认残留策略下那是每步约 74 GiB 的 PCIe 往返）。
- 宿主内存同 Stage A：8 rank 约 300 GB。
- `--use_lora` 在 a14b-t2v 上不可行（双专家无法共驻，需 ≥64 GB），40 GB 卡勿加。

---

## 一键串联（单卡全流程）

```bash
# 1. Stage A：数据生成（必须完整跑完，含 §1.6 finalize）
python scripts/data/generate_counterfactual_data.py \
    --openvid-csv datasets/OpenVidHD.csv --data-root datasets --video-subdir videos \
    --only-existing-videos \
    --processed-root ./LCOCF_OpenVid1M_Processed \
    --backbone wan22 --wan-variant a14b-t2v \
    --model-path /data/suqiang/Text2Video-main/weights/Wan2.2 \
    --backbone-dtype bfloat16 \
    --num-frames 49 --height 384 --width 640 \
    --vae-tile 256 --metric-frame-chunk 4 \
    --real-models --sam-points-per-crop 16 --perception-dtype bfloat16 \
    --sam-model /data/suqiang/Text2Video-main/weights/SAM \
    --dino-model /data/suqiang/Text2Video-main/weights/DINO \
    --clip-model /data/suqiang/Text2Video-main/weights/Clip \
    --limit 2600 --device cuda

# 2. Stage B：联合训练
python scripts/train/train_stage_b.py \
    --processed-root ./LCOCF_OpenVid1M_Processed \
    --batch_size 32 --num_epochs 10 --mixed-precision \
    --device cuda \
    --checkpoint_save ./checkpoints/stage_b_final.pt

# 3. Stage C：端到端微调
python scripts/train/train_stage_c.py \
    --processed-root ./LCOCF_OpenVid1M_Processed \
    --checkpoint_load ./checkpoints/stage_b_final.pt \
    --checkpoint_save ./checkpoints/stage_c_final.pt \
    --backbone wan22 --wan-variant a14b-t2v \
    --model-path /data/suqiang/Text2Video-main/weights/Wan2.2 \
    --num-frames 49 --height 384 --width 640 \
    --vae-tile 128 --grad-window-steps 1 --decode-grad-frames 4 \
    --real-models --metric-frame-chunk 2 \
    --sam-model /data/suqiang/Text2Video-main/weights/SAM \
    --dino-model /data/suqiang/Text2Video-main/weights/DINO \
    --clip-model /data/suqiang/Text2Video-main/weights/Clip \
    --batch_size 1 --num_epochs 3 \
    --device cuda
```