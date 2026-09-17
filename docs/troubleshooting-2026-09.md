# COCF-SS-DCA 问题排查与修复总结

> 记录 2026-09-15 ~ 2026-09-17 期间在 Wan2.2-A14B（40GB 单卡 / 8×A100）上定位并修复的问题。
> 每条修复都有对应的对照实验证据；未闭环的问题在「已定位、暂缓」一节。

## 一、已修复的问题（均有实验证据闭环）

### 1. 文本编码被裁剪（影响全局，最先修）

- **现象**：加速管线生成的视频与官方 Wan 基线系统性不一致。
- **根因**：`cocf/backbones/wan21.py` 基类的 `_trim_text` 把 umT5 的 512 token 文本裁到实际
  长度（~21 token），改变了 cross-attention 的 softmax 分母，与官方 pipeline 的全长
  padding 行为不一致。
- **修复**：`encode_text` 加 `prompt_clean`、padding 位置显式置零；覆写 `_text_kwargs`
  保持全长 512、不传 attention mask。
- **证据**：`scripts/diagnose/compare_wan_paths.py` 对照实验确认文本路径与官方一致。

### 2. MoE 专家边界 ε 缓存失效（修复 A）

- **根因**：Wan2.2-A14B 是双专家 MoE，高噪专家产生的 ε 缓存被跨边界喂给低噪专家，
  缓存语义错误。
- **修复**：`cocf/backbones/wan22.py` 在专家切换边界丢弃 ε 缓存。
- **证据**：日志出现 `dropping ε cache at the MoE boundary`，触发时机正确
  （step 10，高噪→低噪切换点）。

### 3. 单帧 INTERP / 无锚 ANCHOR 被冻结（修复 B）

- **根因**：`cocf/backbones/transition.py` 对单帧 INTERP 管直接冻结 latent，不随去噪推进。
- **修复**：改为骑 z_full 的拼接 ε Euler 步（token 粒度跟随全量计算结果推进）。
- **证据**：日志 `rode the spliced-ε step`，实测每步仅 1~2 个 token，代价可忽略。

### 4. LOWFREQ 马赛克（最核心的根因定位）

- **现象**：加速视频出现大块马赛克；修复 C（值粗化）只把它变成细碎彩色噪声，未治愈。
- **定罪过程**（三模式对照，逐位 md5 验证）：
  - `lowfreq_full`（仅把 LOWFREQ 换成 FULL）与 `allfull`（全量参照）md5 相同
    → **LOWFREQ 是唯一病灶**；
  - `no_cache` 与 accelerated md5 相同 → **ε 缓存无辜**；
  - 结论：填充重建把**异位置的 latent 值**写进当前位置，污染全局注意力输入，
    19 步复利成马赛克。机制上：不同位置噪声不同，邻居的更新量完不成本位置的去噪。
- **最终方案**：ride-through（LOWFREQ hole 不做重建，直接骑 z_full，token 粒度
  TeaCache）+ **周期刷新**（`lowfreq_refresh_every`，每 N 步把 LOWFREQ 管提升 FULL 一步）。
  `EngineConfig.lowfreq_fill` 默认 `False`，`--lowfreq-fill` 留作消融。
- **证据**（PSNR vs allfull 参照）：无刷新 19.81 dB → refresh=4 22.20 dB →
  **refresh=2 32.47 dB**，闪烁 0.79（基准 0.71）。默认值已改为 2
  （`cocf/common/config.py`）。

## 二、已定位、暂缓的问题

### 5. 定价失真（b_min 失效）

- **现象**：`b_min=0.80` 但实际 mask=0.27，budget 最低只到 0.476，质量旋钮完全失效；
  RAEC 证书从不越阈，rollback / repair 恒为 0。
- **后果**（Stage C 冒烟实锤）：无 LoRA 时 pixel 主损失到可训练参数的唯一通路是
  repair net 改 z，repair 不触发 → `pixel grad=False`，修复网络零训练。
  quality + regularizer 仍有梯度，插件调度部分照常训练。
- **根本修法（论文级，顺序固定）**：改 `cocf/lcocf/data.py` 的 LOWFREQ 标签为 ride 语义
  → **重跑 Stage A store** → 重训 Stage B。**必须先改标签代码再跑 A，否则白跑**。
- **现状**：暂缓。refresh=2 已兜底 LOWFREQ 定价失真对画质的影响；Stage C 不依赖
  store 旧标签语义，可以先跑。

### 6. RAEC 残差参考系盲点

ride 语义下 LOWFREQ hole 的 δ≡0，证书对该动作是盲的；dense 前向其实算出了新鲜 eps
但被 splice 丢弃（可作真参考）。论文级修复，与 #5 一起做。

### 7. 文档过时（已更正）

`run.md` 曾称 `--raft-weights` 只能传单文件；代码（`cocf/common/raft.py` 的
`_resolve_weights`）实际已支持**目录**（按文件名 large/small 子串分发给两个消费方）。
run.md 的 RAFT 一节已同步更正。

## 三、各阶段产出与当前状态

| 阶段 | 状态 | 关键数字 |
|---|---|---|
| Stage A store | ✅ 完成（修复前生成，可用） | 160 clips，full_baseline 齐，z_init 已持久化 |
| Stage B | ✅ 完成 | val nonfull MAE 0.0053；test pearson 0.86 (all) / 0.81 (nonfull) |
| Stage C | 🔄 冒烟通过，待 8 卡全量 | 1.26M 插件参数可训；骨干 14B 冻结；~2.5 min/batch，8 卡 3 epochs ≈ 2.5h |

- 基础设施：`scripts/train/run_stage_c.sh`（预检 RAFT / full_baseline / checkpoint，
  自动归档日志和 ckpt）。
- 硬件：8×A100 40GB 够用（单卡 40GB 即设计目标；宿主内存 614G 可用 ≫ 8 rank 约 300G 需求）。
- **LoRA 不可行**：需双专家共驻 ≥64GB 单卡，与卡数无关（数据并行不汇聚单副本显存）；
  非必需——无 LoRA 是 §4.2 的设计内路径。
- 诊断工具备查：`scripts/diagnose/trace_accelerated.py`（逐步 latent 统计 + 中途解码 +
  动作地图，服务器上尚未用过）。

## 四、下一步

1. `bash scripts/train/run_stage_c.sh` 跑 8 卡全量 Stage C（~2.5h）；
2. 用 Stage C checkpoint 跑 `scripts/inference/run_stage_b_probe.sh` 对比微调前后画质；
3.（论文级闭环，择机）改 LOWFREQ 标签语义 → 重跑 Stage A → 重训 Stage B → 重训 Stage C。
