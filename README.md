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
- **CPU 可跑通**：骨干 / VAE / SAM / DINOv2 / CLIP / RAFT 全部依赖注入，内置确定性 Mock，无 GPU、无权重也能构建、运行、测试（`165 passed`）。

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
    --only-existing-videos --use-real-video \
    --processed-root ./LCOCF_OpenVid1M_Processed \
    --backbone wan22 --wan-variant ti2v-5b \
    --model-path /path/to/Wan2.2-TI2V-5B-Diffusers \
    --real-models --device cuda
```

关键开关：

| 开关 | 作用 |
|------|------|
| `--real-models` | 同时打开真实感知（SAM/DINOv2/CLIP/RAFT）与真实损伤指标。**只有真实模型生成的数据才可用于训练插件。** |
| `--use-real-video` | 教师轨迹锚定在真实 mp4 上（VAE 编码 + 前向加噪）而非纯文生视频 |
| `--num-shards N --shard-index i` | 按 md5(video_id) 稳定分片，N 个进程各跑一片，同一存储追加写 |
| `--finalize-only` | 全部分片跑完后执行一次，构建合并的 `sample_index.csv` / `splits/` / 归一化统计 |
| `--no-baseline` | 跳过占空间的 `full_baseline` 桶。**Stage B 不读它，但 Stage C 需要它**（见下方「已知限制」） |
| `--vae-tile 96` / `--metric-frame-chunk 2` | OOM 时的降档旋钮 |

每个 clip 处理完写一行 `_progress` 日志，中断后重跑自动跳过，不会重做教师前向。

### 阶段 B — 插件联合训练（§4.1）

骨干冻结，在离线反事实样本上联合训练四个插件：
`L = L_cocf + λ_sta·L_tube + λ_cert·L_cert + λ_cmsc·L_cmsc + λ_cost·L_budget`

```bash
python scripts/train/train_stage_b.py \
    --processed-root ./LCOCF_OpenVid1M_Processed \
    --batch_size 32 --num_epochs 10 --mixed-precision \
    --checkpoint_save ./checkpoints/stage_b_final.pt
```

采样遵循 §4.1：batch 内**动作均衡 1:1:1:1** + 6 类场景均衡 + 早/中/晚去噪步分层（全部由轻量 `sample_index.csv` 规划，组 batch 不读 payload）。每个 epoch 在 `val` 划分上评估**退化预测 MAE / 证书违规率 / 预算命中率 / 管级动作平滑度**，按 MAE 早停并保存最佳。

### 阶段 C — 端到端轻量微调（§4.2，可选 LoRA）

跑完整加速引擎，对 Stage A 持久化的 `Y_full` 基线做端到端微调：
`L = λ_pixel·L_pixel + λ_quality·L_sem + L_reg(调度正则)`

```bash
python scripts/train/train_stage_c.py \
    --processed-root ./LCOCF_OpenVid1M_Processed \
    --checkpoint_load ./checkpoints/stage_b_final.pt \
    --checkpoint_save ./checkpoints/stage_c_final.pt \
    --batch_size 4 --num_epochs 3 --use_lora --device cuda
```

梯度范围严格遵循 §4.2：像素/语义项经**可微解码**回传到残差修复网与 LoRA；调度正则项回传到强度场与损伤预测器。骨干主体不在优化路径上。

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
tests/             165 个单元 + 集成测试（含 P0–P3 回归套件）
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
| 动作成本乘子 | `FULL/LOWFREQ/INTERP/ANCHOR = 1.0 / 0.25 / 0.0 / 0.0` | 见下注 |
| 单步预算 b_min/b_max | `0.30 / 1.00` | §7.3 |
| 帧数 / 分辨率桶 | `49 (4k+1)` / `(480×832)`、`(720×1280)` | §7.1 |
| 教师步数 / 引擎步数 | `20` / `30` | §1.3 / §7.2 |

> **动作成本注**：文档 §2.2 给的是 `1.0/0.45/0.15/0.0`。代码改为「该动作真正重算的 token 比例」：LOWFREQ 由 `engine.lowfreq_stride` 推导（stride-2 → 1/4），INTERP/ANCHOR 完全不跑去噪器故为 0。这样成本标签与执行器的实际行为一致，但也带来一个已知副作用，见下节。

---

## 🚧 已知限制

按影响排序，均已定位到具体文件/行为，可直接作为下一轮迭代的输入。

**1. 分配器的动作阶梯在 INTERP/ANCHOR 处并列（`scheduler/allocator.py`）**
`action_cost` 里 INTERP 与 ANCHOR 同为 `0.0`，贪心背包按成本排序后两者并列最低。后果：
(a) 被强度场判为 LOW 档、种子动作是 INTERP 的管，**在任何预算下都无法被升档**（`_upgrade_into_budget` 里 `extra <= 0` 直接 `continue`，位置永远停在 0），§2.2 的「预算富余 → 升档」通路对这类管失效；
(b) MID/HIGH 档管在预算紧张时会降到 ANCHOR 而不是破坏性更小的 INTERP，与 §3.3.3 的严重度顺序相反；
(c) 一整步全是 INTERP 时 `predicted_cost` 报 0，预算约束对其形同虚设。
修法：给 INTERP 一个非零成本（如文档的 0.15），或在排序阶梯里合并等成本档位。

**2. 真实骨干上的端到端加速比目前接近 0**
现有视频 DiT 没有任意 token 稀疏注意力，所以 `mask_ratio` 再低也要付一次稠密前向。唯一真实的节省是**整步跳过**，而它由 `engine.dense_step_skip_below`（默认 `0.0`，即关闭）控制，再叠加 `background_refresh_every=4` 每 4 步强制一次全 mask。代码在 `compute_ratio` 上如实反映了这一点（不虚报），但 §10 预期的 20–30% 延迟下降需要先接上 `_run_transformer_sparse` 这个已留好的稀疏 kernel 接口。

**3. 走 `TrainingPipeline` 时 Stage C 的基线缓存 100% 不命中**
`TeacherConfig.num_inference_steps=20` 与 `EngineConfig.num_inference_steps=30` 默认不等，而 `_cached_baseline` 要求两者一致才肯复用 Stage A 的 `Y_full`。CLI 脚本 `train_stage_c.py` 已做对齐（`--steps` 缺省即取教师步数），但 `TrainingPipeline._run_stage_c` 没有——该路径下每个训练 step 都要额外跑一条完整 30 步轨迹 + 一次解码，算力与显存直接翻倍。建议把这段对齐逻辑下沉到 `StageCConfig`，让两个入口行为一致。

**4. §6.3.2 的六项守恒损失只训练了三项**
`CMSCLoss.forward`（含 spatial / ocr / boundary）与整个 `cocf/training/stage_c_losses.py`（257 行，已实现 `build_cmsc_observation` / `cmsc_quality_loss` / `stage_c_regularizers`）**当前没有任何调用者**。Stage C 实际用的是 `FinettuneStage._semantic_loss` 里手写的 3 项（id / appearance / motion），Stage B 只用 `alignment_conservation` 一项。接上 `stage_c_losses` 即可补齐。

**5. 证书的 λ_cmsc 项在推理闭环里恒为 0**
`engine._step` 调 `raec.certify(...)` 时不传 `local_cmsc`（加速循环内没有逐管 CLIP 视觉嵌入），而专为此写的 `CMSCLoss.local_conservation` 无调用者。所以推理期 `E_cert` 六项里实际只有五项生效（Stage B 侧已通过 `_local_cmsc_violation` 给该系数梯度）。

**6. 几处省显存工具已实现但未接线**
`set_gradient_checkpointing()` / `checkpointed()` / `peak_memory()` 零调用；`MemoryConfig.gradient_checkpointing`、`cache_latents` 与整套 `LatentCacheWriter/Dataset` 无使用者；`MemoryConfig.offload_backbone_to_cpu` 传不到 `AnchorStore`（`engine` 调 `new_anchor_store()` 未带 `memory` 参数）。其中最值得接的是 Stage C 解码路径的 checkpointing：VAE 分块只限制了前向峰值，`decode_grad=True` 时每块 tile 的激活仍要留到 backward。

**7. Stage C 的语义损失只覆盖 batch 的第 0 个样本**
`_semantic_loss(y_accel, y_full, captions[0])` 与 `_to_fchw` 里的 `video[0]`，`batch_size=4` 时另外 3 个渲染只进入了像素项。

**8. §7.1.1「相邻时间步标签插值」未实现**
只在 3 个代表步生成样本，中间步依赖 `step_frac` 的正弦编码泛化；文档预期的「减少 60%+ 推理量」这一条尚未兑现（配置注释已声明）。

**9. 其他与文档的出入**
`tube_refresh_every=0` 默认下语义管整条轨迹只建一次（§7.2 写的是每步更新 `G_t`，代码每步只做状态刷新不重分割）；§1.3 要求落盘的 KV cache 与关键帧注意力热力图未持久化；§4.2 的「硬样本优先 / 长度动态采样」与 §4.1 的「多线程异步预读取」未实现；§9 的评测协议（VBench / EvalCrafter / T2V-CompBench、700 prompt × 3 seed、配对 bootstrap）无对应脚本。

**10. `--use-real-video` 改变了教师参照系**
该模式下 `Y_full` 是真实片段的 VAE 往返重建、`z_t` 由前向加噪得到，而非 §1.3 的「完整步数无加速推理」。这是有意的成本权衡（代码已注释），但 run.md 推荐的命令默认带该开关，论文口径需明确说明。

---

## 🧪 测试

```bash
python -m pytest tests/ -q        # 165 passed
```

- `tests/unit/` — 骨干适配器、Wan2.2、L-COCF 损伤/预测、模型感知、数据生成、LMDB 存储、引擎
- `tests/unit/test_p0_regressions.py` … `test_p3_hygiene.py` — 分优先级的回归套件，锁住已修复的具体缺陷
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
