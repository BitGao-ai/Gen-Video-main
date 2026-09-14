# COCF-SS-DCA

**基于轻量级反事实因果算力场与语义管锚定的高效视频扩散加速**
*(Counterfactual Causal Compute Field with Semantic-Tube Anchoring)*

一个面向视频扩散 Transformer（DiT）的**即插即用加速层**。COCF 包裹一个**冻结的骨干模型**（Wan2.2 主力 / Wan2.1 / HunyuanVideo），在去噪的每一步按跨帧**语义管**动态分配算力——因果上稳定的区域廉价近似，因果上关键的区域保持全精度——并用可撤销的误差证书兜底。

> 📐 **算法架构图**（系统总览、单步 8 步闭环、四大创新内部、三阶段训练、六级数据管线，共 8 张 Mermaid 图）见 [**ARCHITECTURE.md**](./ARCHITECTURE.md)
> 🖥️ **真实 GPU 服务器的三阶段训练命令**见 [**run.md**](./run.md)
> 📄 原始设计文档（`.docx`）在 [`files/`](./files/)，本文档的 § 号与其对应

---

## ✨ 核心特性

- **不是生成器，是加速层**：骨干全程冻结，只训练百万～千万级的轻量插件（默认 mock 骨干上 0.09 M）。
- **四大创新协同**：因果算力场（L-COCF §3）+ 语义管锚定（STA §4）+ 可撤销锚定证书（RAEC §5）+ 跨模态语义守恒（CMSC §6）。
- **多骨干兼容**：「适配器 + 注册表」结构，新增骨干只需实现 `_load` 与 `_run_transformer` 两个方法。
- **诚实的效率口径**：`compute_ratio`（适配器自报的真实算力）与 `mask_ratio`（分配计划占用率）严格分离，后者**永远不会**被当成加速比上报。
- **CPU 可跑通**：骨干 / VAE / SAM / DINOv2 / CLIP / RAFT 全部依赖注入，内置确定性 Mock，无 GPU、无权重也能构建、运行、测试。

---

## 🧠 核心思想

视频去噪中并非每个区域每一步都值得全量计算。COCF 把这一直觉工程化为**每去噪步的四步闭环**：

1. **分组**（STA）：SAM 区域 + 5 维亲和度 + 匈牙利匹配 → 跨帧**语义管**，每管维护 7 维状态。
2. **定级**（L-COCF）：由文本因果三元组与管状态算出**因果强度** `s = α·s_E + β·s_A + γ·s_T`（仅 3 个可学习参数），并由损伤预测器给出各动作的**反事实损伤** `(μ, σ)`。
3. **分配**（调度器）：在**动态单步预算** `B_t` 与硬风险约束 `E_cert ≤ τ_r` 下，用贪心多选背包为每管选一个动作：`FULL / LOWFREQ / INTERP / ANCHOR`。
4. **兜底**（RAEC）：转移**之后**用真实跳算残差 δ 算**误差证书**，按三档触发**回滚 / 边界修复 / KV 刷新**——跳算在证书通过前从不是最终的。

全程由 **CMSC** 的跨模态语义守恒约束加速结果不偏离全算力教师。

---

## 🏗️ 架构

```
                    ┌──────────────────────────────────────────────────────────┐
                    │                  Accelerator (core/)                      │
                    │    冻结骨干（plain attribute，不入 parameters()）          │
                    │  + 4 个可学习插件 + 预算调度器 + 分配器 + 动作执行器        │
                    └────┬──────────┬──────────┬──────────┬────────────────────┘
                         │          │          │          │
                ┌────────┘   ┌──────┘    ┌─────┘    ┌─────┘
                ▼            ▼           ▼          ▼
          ┌──────────┐ ┌──────────┐ ┌────────┐ ┌──────────┐
          │  L-COCF  │ │   STA    │ │  RAEC  │ │   CMSC   │
          │   §3     │ │   §4     │ │  §5    │ │   §6     │
          └──────────┘ └──────────┘ └────────┘ └──────────┘
                 由 InferenceEngine (engine/) 逐步编排 8 步闭环
```

| 创新 | 包 | 可学习量 | 职责 |
|------|----|---------|------|
| **L-COCF** | `cocf/lcocf` | 强度 3 参 + 损伤预测器 + 残差修复网 | 因果强度场 + 反事实损伤 (μ,σ) |
| **STA** | `cocf/tubes` | 管平滑损失（训练期） | 逐帧感知 → 跨帧语义管 + 7 维状态 |
| **RAEC** | `cocf/raec` | 证书 6 系数（softplus 保正） | 误差证书 → 风险三档 → 回滚/修复/刷新 |
| **CMSC** | `cocf/cmsc` | 文本↔管对齐投影 W | 文本↔管对齐 + 6 项守恒损失 |

---

## 📦 安装

```bash
git clone <repo-url> && cd pro_011

# 核心
pip install torch                                 # ≥ 2.0
pip install scipy scikit-image einops pyyaml      # 算法 / 配置
pip install lmdb                                  # 训练主库（缺失自动回退 .pt 分片）

# 真实骨干与真实感知（CPU/mock 冒烟不需要）
pip install 'diffusers>=0.34' transformers accelerate
```

> `diffusers>=0.34` 是**硬性要求**：低版本 VAE 没有 `enable_tiling()`，适配器会在加载期直接报错而非在数小时后 OOM。

---

## 🚀 快速开始（CPU，mock 骨干，无需权重）

```bash
python scripts/inference/infer_single_video.py \
    --prompt "a cat jumping over a fence" \
    --backbone mock --steps 30 --quality balanced --device cpu
```

或在 Python 中：

```python
import torch
from cocf.common.config import Config
from cocf.core.accelerator import Accelerator
from cocf.engine import InferenceEngine
from cocf.common.types import TokenGrid

config = Config()
config.backbone.name = "mock"
accelerator = Accelerator.from_config(config)          # 冻结骨干 + 4 插件（全 CPU）
engine = InferenceEngine(accelerator, config.engine, config.trigger)

prompt = "a cat jumping"
cond   = accelerator.backbone.encode_text([prompt])
grid   = TokenGrid(t=13, h=8, w=8)
z_init = torch.randn(1, grid.num_tokens, accelerator.token_dim)

result = engine.generate([prompt], z_init, grid, cond, accelerator.backbone)
print(result.summary())
# {'steps': 30, 'mean_compute_ratio': ..., 'mean_mask_ratio': ...,
#  'rollbacks': ..., 'repairs': ..., 'cf_repairs': ...}
```

**`mean_compute_ratio` 是唯一可以当作加速比引用的数字**；`mean_mask_ratio` 只是分配计划的 token 占用率。

---

## 🎓 三阶段训练

一个 `processed_root`（六级处理存储 `LCOCF_OpenVid1M_Processed`）贯穿三阶段：A 写，B/C 读。
完整的真机命令（含 40 GB / 80 GB 显卡分档、多卡分片）见 [**run.md**](./run.md)。

### 阶段 A — 反事实教师数据生成（离线，§1）

冻结骨干作教师，对每个 (语义管, 代表时间步, 动作) 做**单跳**反事实干预，测量对最终视频的多维退化。

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
    --clip-model  /data/suqiang/Text2Video-main/weights/Clip \
    --limit 2600 --device cuda
```

单机 8×40 GB 的实际跑法 —— 每卡一个分片，跑完再单进程合并索引：

```bash
COMMON=(--openvid-csv datasets/OpenVidHD.csv --data-root datasets --video-subdir videos
        --only-existing-videos --processed-root ./LCOCF_OpenVid1M_Processed
        --backbone wan22 --wan-variant a14b-t2v
        --model-path /path/to/Wan2.2-T2V-A14B-Diffusers --backbone-dtype bfloat16
        --sam-model  /path/to/...
        --dino-model /path/to/...
        --clip-model /path/to/...
        --num-frames 49 --height 384 --width 640
        --vae-tile 256 --metric-frame-chunk 4
        --real-models --sam-points-per-crop 16 --perception-dtype bfloat16
        --limit 2600 --device cuda --seed 1234)

for i in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$i python scripts/data/generate_counterfactual_data.py \
    "${COMMON[@]}" --num-shards 8 --shard-index $i > logs/stage_a.s$i.log 2>&1 &
done
wait
python scripts/data/generate_counterfactual_data.py "${COMMON[@]}" \
    --num-shards 8 --shard-index 0 --finalize-only
```

> ⚠️ **`--wan-variant` 必须与权重目录一致。** A14B 是双专家（有 `transformer_2/` 子目录），
> 5B 是单专家。搞反了会让低噪专家根本不加载、所有噪声档都路由到高噪专家，产出的标签
> off-distribution 而运行速度看起来完全正常。适配器现在会在**加载期直接 raise** 并给出
> 应该传的 variant，不再静默降级。
>
> ⚠️ **CPU 内存是多分片并行的实际瓶颈**，不是显存。每个进程稳态约 40 GB（离线的 28 GB
> 空闲专家 + 11 GB umT5），8 进程约 320 GB。先 `free -g`，不足就减少分片数。

关键开关：

| 开关 | 作用 |
|------|------|
| `--real-models` | 同时打开真实感知（SAM/DINOv2/CLIP/RAFT）与真实损伤指标。**只有真实模型生成的数据才可用于训练插件。** |
| `--num-frames / --height / --width` | 渲染几何。**必须与阶段 C 一致**（C 的目标就是这里落盘的 `Y_full`）。校验 4k+1 与 16 整除，并写入 `metadata/stage_a_env.json` 供 B/C 回读 |
| `--num-shards N --shard-index i` | 按 md5(video_id) 稳定分片，N 个进程各跑一片，同一存储追加写 |
| `--finalize-only` | 全部分片跑完后执行一次，构建合并的 `sample_index.csv` / `splits/` / 归一化统计 |
| `--limit N` | 每个 CSV 的取样上限。配 `--only-existing-videos` 时限制的是**保留数**而非扫描行数，所以小 N 也会一直扫到找满 |
| `--use-real-video` | 教师轨迹锚定在真实 mp4 上而非纯文生视频。**会导致不落盘 `z_init`，阶段 C 因此无法复用 `Y_full`**（见下方「已知限制」），只适合不喂给阶段 C 的存储 |
| `--vae-tile` | 分块解码的输出边长。峰值 ∝ tile²、块数 ∝ 1/tile²，所以标签期的阶段 A 取大（256，384×640 下 8 块）、可微解码的阶段 C 取小（128） |
| `--sam-points-per-crop` | SAM 点提示网格边长，实际提示数是其平方。SAM 每解码帧跑一次，是非 DiT 部分的主要开销 |
| `--perception-dtype` | DINOv2/CLIP/SAM 的权重精度（RAFT 恒为 fp32，其全对相关体积在半精度下不稳） |
| `--no-baseline` | 跳过 `full_baseline` 桶。省 ~72 MB/clip，但**会关掉阶段 C 的基线缓存**，使 C 每个 batch 重跑两条完整轨迹 |
| `--metric-frame-chunk 2` / `--vae-tile 128` | OOM 时的降档旋钮 |

每个 clip 处理完写一行 `_progress` 日志，中断后重跑自动跳过，不会重做教师前向。运行期每
clip 会打两行进度（§1.3-1.4 完成、样本写完），带耗时与显存高水位 —— 真实骨干上单 clip 是
十几分钟量级，没有这两行时「很慢」和「卡死」无法区分。

### 阶段 B — 插件联合训练（§4.1）

骨干冻结，在离线反事实样本上联合训练四个插件：
`L = L_cocf + λ_sta·L_tube + λ_cert·L_cert + λ_cmsc·L_cmsc + λ_cost·L_budget`

```bash
python scripts/train/train_stage_b.py \
    --processed-root ./LCOCF_OpenVid1M_Processed \
    --batch_size 32 --num_epochs 10 --mixed-precision \
    --device cuda \
    --checkpoint_save ./checkpoints/stage_b_final.pt
```

不需要传骨干相关参数：本阶段只在阶段 A 的离线标签上训练插件，**不加载任何 Wan 权重**（显存
< 3 GiB）。但��件的残差修复网宽度由骨干的 token 维度决定，所以脚本会从
`metadata/stage_a_env.json` 回读阶段 A 的几何（A14B → `token_dim=64`）。没有该文件时会退回默认
的 mock 几何（32）并告警 —— 那样存出来的 checkpoint 装不进跑真骨干的阶段 C。旧存储补一次
`--finalize-only` 即可生成。

采样遵循 §4.1：batch 内**动作均衡 1:1:1:1** + 6 类场景均衡 + 早/中/晚去噪步分层（全部由轻量 `sample_index.csv` 规划，组 batch 不读 payload）。每个 epoch 在 `val` 划分上评估**退化预测 MAE / 证书违规率 / 预算命中率 / 管级动作平滑度**，按 MAE 早停并保存最佳。

### 阶段 C — 端到端轻量微调（§4.2，可选 LoRA）

跑完整加速引擎，对 Stage A 持久化的 `Y_full` 基线做端到端微调：
`L = λ_pixel·L_pixel + λ_quality·L_sem + L_reg(调度正则)`

```bash
python scripts/train/train_stage_c.py \
    --processed-root ./LCOCF_OpenVid1M_Processed \
    --checkpoint_load ./checkpoints/stage_b_final.pt \
    --checkpoint_save ./checkpoints/stage_c_final.pt \
    --backbone wan22 --wan-variant a14b-t2v \
    --model-path /path/to/Wan2.2-T2V-A14B-Diffusers \
    --backbone-dtype bfloat16 \
    --vae-tile 128 --grad-window-steps 1 --decode-grad-frames 4 \
    --real-models --metric-frame-chunk 2 --perception-dtype bfloat16 \
    --batch_size 1 --num_epochs 3 --device cuda
```

**骨干参数必传**：与阶段 B 不同，阶段 C 真跑扩散（`L_pixel` 要用加速引擎渲一遍再和 `Y_full`
比像素，梯度穿过可微解码回到插件），所以 Wan 权重必须在场。`--backbone` 的默认值是 `mock`，
只用于 CPU 冒烟跑（`--backbone mock --device cpu --num_epochs 1`）；不传就会拿一个 32 维假
DiT 去拟合 Wan2.2 生成的 `Y_full`，训练不报错但没有意义。`--wan-variant` 默认 `a14b-t2v`，若阶段 A
用的是 5B，必须显式传 `--wan-variant ti2v-5b`；A14B 可沿用默认值。

#### 显存档位与推荐命令（A14B @ 49×384×640）

激活预算 = 卡容量 − 冻结常驻。A14B 双专家 bf16 常驻 **54.1 GiB**，只驻一个约 **27 GiB**
（umT5 默认停在 CPU）。一次 `engine.generate` 渲染整个 batch，没有梯度累积可换，所以
**`--batch_size` 恒为 1**；一个 clip 在此几何下需要约 20 GiB 激活余量：
DiT 检查点栈（40 个 block 边界 × 128 MiB）≈5 GiB + block 内重算 ≈3 GiB + 可微 VAE 解码图
8–14 GiB。脚本会按 `free ÷ 20 GiB` 钳制 batch 并打印原因，而不是让你在几分钟后撞 OOM。

**① 单卡 80 GB（A100/H100 80G）—— 双专家常驻，可开 LoRA**

```bash
python scripts/train/train_stage_c.py \
    --processed-root ./LCOCF_OpenVid1M_Processed \
    --checkpoint_load ./checkpoints/stage_b_final.pt \
    --checkpoint_save ./checkpoints/stage_c_final.pt \
    --backbone wan22 --wan-variant a14b-t2v \
    --model-path /path/to/Wan2.2-T2V-A14B-Diffusers \
    --backbone-dtype bfloat16 \
    --vae-tile 128 --grad-window-steps 1 --decode-grad-frames 2 \
    --real-models --metric-frame-chunk 2 --perception-dtype bfloat16 \
    --sam-model /path/to/SAM --dino-model /path/to/DINO --clip-model /path/to/Clip \
    --no-offload-idle-expert \
    --batch_size 1 --num_epochs 3 --use_lora --device cuda
```

`--no-offload-idle-expert` 是 `--use_lora` 的前提：LoRA 注入每个专家，只常驻一个时，被换出
专家的 adapter 会在激活还挂在 autograd 图上时跨设备搬运（脚本会告警）。代价是 54.1 GiB 常驻，
只剩 25.1 GiB 给激活 —— 刚好够一个 clip，所以这里 `--decode-grad-frames` 取 2 而非 4：
可微解码是最大的一项，且分块解码在 autograd 下**不省显存**（每块中间量都要留到 backward，
重叠还多留 `(128/96)² ≈ 1.78` 倍）。仍然 OOM 就把它降到 1。

**② 单卡 40 GB（A100 40G / L40S 48G）—— 单专家常驻，不能开 LoRA**

```bash
python scripts/train/train_stage_c.py \
    --processed-root ./LCOCF_OpenVid1M_Processed \
    --checkpoint_load ./checkpoints/stage_b_final.pt \
    --checkpoint_save ./checkpoints/stage_c_final.pt \
    --backbone wan22 --wan-variant a14b-t2v \
    --model-path /path/to/Wan2.2-T2V-A14B-Diffusers \
    --backbone-dtype bfloat16 \
    --vae-tile 128 --grad-window-steps 1 --decode-grad-frames 1 \
    --real-models --metric-frame-chunk 2 --perception-dtype bfloat16 \
    --sam-model /path/to/SAM --dino-model /path/to/DINO --clip-model /path/to/Clip \
    --offload-idle-expert \
    --batch_size 1 --num_epochs 3 --device cuda
```

40 GB 上 A14B 只剩约 11 GiB 激活余量，因此**必须**换出空闲专家、**不能**加 `--use_lora`，
`--decode-grad-frames` 也要降到 1。此时主损失到骨干的唯一通路是残差修复网（见下方 ⚠️），
多数 batch 实际只训练插件与调度正则 —— 40 GB 上这是能跑与跑不动之间的取舍。
若阶段 A 用的是 **ti2v-5b**（单专家，常驻约 11 GiB），把 `--wan-variant` 换成 `ti2v-5b`
即可保留 `--use_lora` 与 `--decode-grad-frames 2`，这是 40 GB 卡上更合理的组合。

**③ 单机 8 卡 × 40 GB —— `torchrun` 数据并行**

阶段 C 支持数据并行：每个 rank 独占一张卡跑自己的分片，`backward()` 之后对 **7.0M 可训练
参数**做一次 all-reduce（约 28 MB/step，与 27B 的冻结主干无关，因为它永远没有梯度）。用
`torchrun` 起 8 个进程即可，不需要任何额外开关：

```bash
torchrun --standalone --nproc_per_node=8 scripts/train/train_stage_c.py \
    --processed-root ./LCOCF_OpenVid1M_Processed \
    --checkpoint_load ./checkpoints/stage_b_final.pt \
    --checkpoint_save ./checkpoints/stage_c_final.pt \
    --backbone wan22 --wan-variant a14b-t2v \
    --model-path /path/to/Wan2.2-T2V-A14B-Diffusers \
    --backbone-dtype bfloat16 \
    --vae-tile 128 --grad-window-steps 1 --decode-grad-frames 1 \
    --real-models --metric-frame-chunk 2 --perception-dtype bfloat16 \
    --sam-model /path/to/SAM --dino-model /path/to/DINO --clip-model /path/to/Clip \
    --offload-idle-expert \
    --batch_size 1 --num_epochs 3 --device cuda
```

单卡命令加 `torchrun --nproc_per_node=N` 就是分布式版本，其余参数一字不改：`--device cuda`
会被自动钉到本 rank 的 `cuda:$LOCAL_RANK`，数据由 `DistributedSampler` 切分
（`drop_last=True`，保证各 rank 的 batch 数相同 —— 否则先跑完的 rank 会把其他 rank 卡死在
all-reduce 里），日志与 checkpoint 只由 rank 0 输出/写盘。有效 batch = `--batch_size × N`。

实现细节都在 `cocf/training/distributed.py`：为什么不是 `DistributedDataParallel`（它要包住
**一个** module 的前向，而阶段 C 的一步是整条加速引擎加若干个损失）、为什么梯度平均放在裁剪
之前（各 rank 必须裁同一个梯度才能走出同一步）、为什么开跑前要从 rank 0 广播一次参数（因形状
不符被丢弃、转而随机初始化的层，或新注入的 LoRA，否则会在各 rank 上分叉成 8 个不同的模型）。未经
`torchrun` 启动时，其中每个函数都是 no-op，单机单卡的行为与从前逐字节一致。

> ⚠️ **主机内存是 8 rank 的真正门槛**：每个进程把换出的空闲专家（28 GB）和 umT5（11 GB）
> 放在 CPU 上，稳态约 40 GB/进程，8 个就是 ~320 GB。先 `free -g`；不够就降到
> `--nproc_per_node=4`，或者改用下面的配对方案。

**配对方案（4 rank，把空闲专家停在邻卡而不是主存）**

`--offload-device` 可以让被换出的组件停在**另一张 GPU** 上：28 GB 的专家换页变成 P2P 拷贝而不
是绕主存往返，同时彻底消除上面的主机内存压力。8 张卡配成 4 组（偶数卡算、奇数卡停）：

```bash
cat > run_rank.sh <<'EOF'
#!/usr/bin/env bash
exec python scripts/train/train_stage_c.py "$@" \
    --device cuda:$((2*LOCAL_RANK)) --offload-device cuda:$((2*LOCAL_RANK+1))
EOF
chmod +x run_rank.sh

torchrun --standalone --nproc_per_node=4 --no-python ./run_rank.sh \
    --processed-root ./LCOCF_OpenVid1M_Processed \
    --checkpoint_load ./checkpoints/stage_b_final.pt \
    --checkpoint_save ./checkpoints/stage_c_final.pt \
    --backbone wan22 --wan-variant a14b-t2v \
    --model-path /path/to/Wan2.2-T2V-A14B-Diffusers \
    --backbone-dtype bfloat16 --offload-idle-expert \
    --vae-tile 128 --grad-window-steps 1 --decode-grad-frames 2 \
    --real-models --metric-frame-chunk 2 --perception-dtype bfloat16 \
    --sam-model /path/to/SAM --dino-model /path/to/DINO --clip-model /path/to/Clip \
    --batch_size 1 --num_epochs 3
```

吞吐是 4 而不是 8，换来的是几乎免费的专家换页和零主机内存占用 —— 在换页频繁（`--steps` 大、
噪声边界被反复跨越）时通常更划算。`--offload-device` 若指向计算卡本身或本机没有的设备，会告警
并退回 CPU，不会在运行数小时后炸在一次前向里。**注意每个 rank 必须停在不同的卡上**：直接给
`torchrun` 传一个固定的 `--offload-device cuda:1`，会让 8 个 rank 把 8 份 28 GB 全堆到同一张卡上。

其余两个阶段不变：阶段 A 用 `--num-shards 8 --shard-index i` 分片并行（见上文），阶段 B 单卡
即可（只训 1.26M 插件）。

| 档位 | 常驻 | 激活余量 | LoRA | `--decode-grad-frames` | `--batch_size` |
|------|------|----------|------|------------------------|----------------|
| 80 GB | 54.1 GiB（双专家） | ~25 GiB | ✅ | 2 | 1 |
| 40 GB（A14B） | ~27 GiB（单专家） | ~11 GiB | ❌ | 1 | 1 |
| 40 GB（ti2v-5b） | ~11 GiB | ~28 GiB | ✅ | 2 | 1 |
| 8×40 GB（torchrun） | 同 40 GB，每 rank 一张卡 | 同 40 GB | ❌ | 1 | 1/rank（有效 8） |
| 4×(40+40) GB（配对） | 计算卡 27 GiB，邻卡托管空闲专家 | ~11 GiB | ❌ | 2 | 1/rank（有效 4） |

渲染几何不用再传 —— 脚本从 `metadata/stage_a_env.json` 读取阶段 A 的
`num_frames/height/width` 与 teacher 步数并对齐；显式传 `--height` 等覆盖时会告警，因为几何一
旦不匹配，`_cached_baseline` 会**静默**放弃缓存并每个 batch 重算基线。

| 开关 | 作用 |
|------|------|
| `--grad-window-steps` | 按**已计算步**计数的截断 BPTT 段长。激活峰值与它线性相关：12,480 token 下每保留一个计算步约 5 GiB，40 GB 卡上只能取 1。`0` 表示完整 BPTT，真实骨干必 OOM |
| `--decode-grad-frames` | 放到 autograd 图上的潜空间时间槽数。分块解码只能限制**前向**瞬时，可微解码会把每块中间激活留到 backward，所以这里必须另外设窗 |
| `--use_lora` | 在每个专家的最后若干 DiT block 注入 LoRA。需要被注入的专家全部同时常驻，**A14B 在 64 GB 以下不可用**（只有一个专家能常驻，而在激活还挂在图上时搬运模块并不成立）|
| `--batch_size` | 整个 batch 由一次 `engine.generate` 渲染，激活随之线性增长且没有梯度累积可换。A14B @ 49×384×640 下**任何档位都只能是 1**（一个 clip 约需 20 GiB 激活余量；不足时脚本按 `free ÷ 20 GiB` 自动钳制并打印原因）|

梯度范围严格遵循 §4.2：像素/语义项经**可微解码**回传到残差修复网与 LoRA；调度正则项回传到强度场与损伤预测器。骨干主体不在优化路径上。两个专家现在都会开启激活检查点（此前只对 `backbone.module` 即高噪专家生效，且只在 `--use_lora` 下才调用）。

> ⚠️ **关掉 LoRA 时，主损失到骨干的唯一通路是残差修复网**，而它只在 RAEC 触发修复时才动。
> 配 `--grad-window-steps 1`，修复恰好落在最后一个计算步的概率不高，多数 batch 实际只有正则
> 项在训练插件。引擎会打印 `decode_grad=True but the render carries no autograd graph` 提示这
> 一情况。想拿回这条通路需要 `--grad-window-steps 2`（峰值 +5 GiB，40 GB 放不下）或更低的分
> 辨率。

> ⚠️ 脚本默认让引擎步数取 `config.teacher.num_inference_steps`，这样 Stage A 持久化的 `Y_full` 才是有效目标。手动传 `--steps` 覆盖成别的值会导致基线缓存失效、每个 batch 额外重跑一条完整轨迹。

---

## 🎬 推理

```bash
python scripts/inference/infer_single_video.py \
    --prompt "a cat jumping" \
    --checkpoint ./checkpoints/stage_c_final.pt \
    --backbone wan22 --model-path /path/to/weights \
    --steps 30 --quality balanced \
    --num-frames 49 --height 480 --width 832 \
    --output ./output.mp4 --device cuda
```

- `--quality` 映射到预算下限 `b_min`：`fast=0.30` / `balanced=0.50` / `quality=0.80`。
- checkpoint 接受 Stage-B 裸 `state_dict` 与 Stage-C `{"accelerator", "lora"}` 两种布局，LoRA 会按存储的几何自动重新注入。
- 视频写盘按 `imageio → torchvision → cv2 → PNG 序列 → .npy` 逐级降级，无编解码环境也能落地产物。

---

## 💾 显存策略

训练/生成期的显存控制分三层，全部可从配置或 CLI 关掉：

**架构层（最有效）** — 骨干冻结，优化器只见插件；Stage A 全程 `inference_mode`，不建任何激活图。

**常驻层**（`BackboneConfig`，默认全开）

| 开关 | 省下 | 说明 |
|------|------|------|
| `offload_text_encoder` | ~11 GB | umT5 只在每条 prompt 用一次，其余时间停在 CPU |
| `offload_idle_expert` | ~28 GB | Wan2.2-A14B 双专家 MoE，只驻留当前噪声档的那个 |
| `vae_tiling` + `vae_tile_size=128` | 7.7 GiB → ~0.4 GB | 未分块的 49×480×832 解码需要单块 7.71 GiB；Stage A 每 clip 解码约 90 次 |

加载完成后会打印一行 `frozen stack resident X GiB / Y GiB` 报告，可直接从日志首行核对策略是否真的生效。

**瞬时层** — RAFT 相关体积（整段 49 帧约 7 GiB 单块）与 DINO/CLIP 前向按 `--metric-frame-chunk` 分块；每 clip 的参考特征只抽一次而非每次 rollout 重抽；`text_embed` 按 clip 存一份 fp16 去 padding 版本（8 MiB/样本 → 0.3 MiB/clip）；锚点库只存管自身的 token 并在重分割后回收；Stage C 用**按已计算步计数的截断 BPTT**（`engine.grad_window_steps`）限制激活峰值。

---

## 📁 目录结构

```
cocf/
├── common/        类型 / 分层配置 / 省显存工具 / 日志
├── backbones/     适配层：base · diffusers_base · wan22 · wan21 · hunyuan · mock · transition(动作执行器)
├── tubes/         STA（§4）：regions → affinity → matching → builder → state → smoothing
├── lcocf/         L-COCF（§3）：triplets → strength → mapping → predictor → counterfactual
│   ├── damage.py  多维反事实损伤（8 个 VBench 风格轴）
│   └── data.py    ★ 单跳反事实样本生成器（§1.5）
├── raec/          RAEC（§5）：certificate → trigger → repair → anchor_store
├── cmsc/          CMSC（§6）：alignment（文本↔管） + losses（6 项守恒）
├── scheduler/     §2.2 / §7.3：budget（动态 B_t） + allocator（贪心多选背包）
├── engine/        §7.2 加速推理引擎：8 步闭环 + EngineState/StepTrace
├── core/          Accelerator：唯一装配枢纽（维度全部从适配器探测）
├── data/          §1/§2：OpenVid 清单 · 四级过滤 · 六级存储布局 · LMDB 读写 · 分层 batch 采样 · 指标 · 视频写盘
└── training/      §7.1：pipeline · stage_a/b/c · teacher_forward · lora · checkpoint
scripts/           CLI：数据生成 / Stage-B / Stage-C / 推理 / 索引重建
tests/             237 个单元 + 集成测试（含 P0–P4 回归套件）
```

---

## ⚙️ 配置

所有超参集中在 `cocf/common/config.py`（分层 `dataclass`，每项标注来源章节），可从 YAML/JSON 载入，未知键忽略：

```python
from cocf.common.config import Config
config = Config.load("my_experiment.yaml")
config.save("dump.yaml")
```

| 项 | 默认值 | 出处 |
|----|--------|------|
| 强度分层阈值 θ₁/θ₂ | `0.66 / 0.33` | §3.3.3 |
| 风险阈值 τ_low/τ_high | `0.40 / 0.80` | §5.3.2 |
| 锚点写入门限 τ_anchor | `0.60`（**刻意不复用 τ_low**） | §5.3.2 |
| 证书系数 κ/λ_res/λ_bnd/λ_age/λ_cmsc | `1.96 / 0.10 / 0.05 / 0.01 / 0.20` | §5.3.1 |
| 亲和度权重 w_id/flow/iou/txt/pos | `0.40/0.30/0.15/0.10/0.05`，σ_p=16 | §4.3.1 |
| CMSC 权重 align/id/motion/spatial/ocr/bnd | `0.30/0.20/0.20/0.15/0.10/0.05`，τ=0.07 | §6.3.2 |
| 动作成本乘子 | `FULL/LOWFREQ/INTERP/ANCHOR = 1.0 / 0.25 / 0.02 / 0.0` | 见下注 |
| 单步预算 b_min/b_max | `0.30 / 1.00` | §7.3 |
| 截断 BPTT 窗口 / 可微解码槽数 | `4` / `8`（0 = 全部） | §4.2 |
| 整步跳过阈值 / 未测量步上限 | `0.0`（关闭）/ `3` | §5.3.1 |
| 帧数 / 分辨率桶 | `49 (4k+1)` / `(480×832)`、`(720×1280)` | §7.1 |
| 教师步数 / 引擎步数 | `20` / `30`（Stage C 自动对齐到教师步数） | §1.3 / §7.2 |

> **动作成本注**：文档 §2.2 给的是 `1.0/0.45/0.15/0.0`。代码里 LOWFREQ 由 `engine.lowfreq_stride` 推导（stride-2 → 1/4），ANCHOR 不动任何 token 故为 0。INTERP 不跑去噪器，但要对 `|g_k|` 个 token 做 gather + 混合，是访存受限的**非零**成本 —— 取 0.02。这个非零值同时是贪心阶梯保持严格全序的前提：若与 ANCHOR 并列，被分层映射判为 LOW 档的管将永远无法升档（详见 `ActionAllocator._warn_if_ladder_collapses`）。

---

## 🚧 已知限制

上一轮 code review 的 10 条问题中，1–7 已修复（见 [`FIX_PLAN.md`](./FIX_PLAN.md) 的批次 1–3 与 `tests/unit/test_p4_*.py` 共 72 个回归用例）。以下是**仍然存在**的部分。

**1. 真实骨干上的端到端加速比仍接近 0（最大的一块）**
现有视频 DiT 没有任意 token 稀疏注意力，`mask_ratio` 再低也要付一次稠密前向。唯一真实的节省是**整步跳过**，由 `engine.dense_step_skip_below` 控制。它此前因为「促发后被降级的管测不到跳算残差 δ、证书无法定价」而只能关闭；该缺口现已由 `engine.max_unmeasured_steps`（任一管连续未测量步数达到上限即否决促发）**变成有界**，所以打开它是一个有依据的取舍而不是盲赌。默认仍为 0 —— 「一个管最多能有多少步不被证书覆盖」是质量策略，属于运行方的决定。

即便打开，只要该步存在任何被强制 FULL 的管（不稳定管、RAEC 回滚钉住），促发就会被否决——这是既有的正确安全规则。§10 预期的 20–30% 延迟下降仍需要先实现 `_run_transformer_sparse` 这个已留好的稀疏 kernel 接口（建议形式：**query 稀疏** —— K/V 保留全部 token，Q 只取活跃子集，注意力 `N²→|active|·N`、MLP `∝|active|`）。

**2. §6.3.2 六项里 `L_spatial` / `L_bnd` 仍为零**
`L_align` / `L_id` / `L_motion` / `L_ocr` 已全部接线并在训练中生效（`L_id` 由每管 DINO 身份特征驱动，可回传到渲染）。另两项是**结构性**为零，原因已写进 `build_cmsc_observation` 的 docstring：`L_spatial` 在两侧共用同一套语义管时恒等（引擎只在加速渲染上分割一次），要让它有信息量必须对全算力渲染单独分割；`L_bnd` 则是框架里根本没有边界描述子可读（§6.3.2 本身也标为可选，权重 0.05）。

**3. §7.1.1「相邻时间步标签插值」未实现**
只在 3 个代表步生成样本，中间步依赖 `step_frac` 的正弦编码泛化；文档预期的「减少 60%+ 推理量」尚未兑现。注意 `DamageLabelInterpolator` 曾被 P3 当死代码删除，重新引入应放在 `finalize_processed_store` 的流式扫描里，并在 `sample_index.csv` 打 `interpolated=1` —— 否则合成标签会流进验证集，早停指标失真。

**4. 其他与文档的出入**
`tube_refresh_every=0` 默认下语义管整条轨迹只建一次（§7.2 写的是每步更新 `G_t`，代码每步只刷新状态不重分割）；§1.3 要求落盘的 KV cache 与关键帧注意力热力图未持久化；§4.2 的「硬样本优先 / 长度动态采样」与 §4.1 的「多线程异步预读取」未实现（后者需要先让 LMDB 句柄按 worker 惰性打开）；§9 的评测协议（VBench / EvalCrafter / T2V-CompBench、700 prompt × 3 seed、配对 bootstrap）无对应脚本。

**5. `--use-real-video` 改变了教师参照系**
该模式下 `Y_full` 是真实片段的 VAE 往返重建、`z_t` 由前向加噪得到，而非 §1.3 的「完整步数无加速推理」。这是有意的成本权衡（代码已注释），但 run.md 推荐的命令默认带该开关，论文口径需明确说明。

---

## 🧪 测试

```bash
python -m pytest tests/ -q        # 237 passed
```

- `tests/unit/` — 骨干适配器、Wan2.2、L-COCF 损伤/预测、模型感知、数据生成、LMDB 存储、引擎
- `tests/unit/test_p0_regressions.py` … `test_p4_*.py` — 分优先级的回归套件，每条锁住一个已修复的具体缺陷
- `tests/integration/test_pipeline.py` — mock 骨干上的全流水线端到端冒烟

---

## 📋 依赖

- Python ≥ 3.9，PyTorch ≥ 2.0
- `scipy` / `scikit-image` / `einops`（算法）、`pyyaml`（配置）
- `lmdb`（训练主库，缺失自动回退 `.pt` 分片）
- `diffusers>=0.34` / `transformers` / `accelerate`（**仅**真实骨干，惰性导入）
- `imageio` 或 `torchvision` 或 `opencv-python`（写 mp4，可选）
- `easyocr`（OCR 守恒轴，`--enable-ocr` 才需要）

---

*章节号（§1~§9）对应 `files/` 下的设计文档。本项目是训练 + 推理加速层，不含骨干模型权重。*
