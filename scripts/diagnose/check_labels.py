#!/usr/bin/env python
"""Sanity-check Stage A's counterfactual labels before Stage B trains on them.

There is no external ground truth for a damage label, so correctness has to be
established from *internal consistency* instead. Four properties are strong enough
to catch the failure modes that actually occur, and all four are checkable offline:

1. **The FULL anchor.** ``action=FULL`` skips nothing — its rollout re-runs the same
   transition the teacher ran — so its damage must be ≈ 0. It is the only label in
   the store whose correct value is known a priori, which makes it the calibration
   point for every other label: if FULL is not ≈ 0, the rollout diverges from the
   teacher for reasons unrelated to the intervention (different noise, a schedule
   mismatch), and every other damage value carries that same offset.
2. **Action monotonicity.** FULL ≤ LOWFREQ ≤ INTERP ≤ ANCHOR by construction — the
   actions are ordered by how much compute they remove. An inversion means the
   damage metric is not measuring degradation.
3. **Non-degenerate axes.** An axis with zero variance across the store is a metric
   backend that silently fell back (the classic one is RAFT: without it
   ``raft_motion`` and ``motion_smoothness`` are computed from a descriptor diff and
   ``s_A`` is identically 0).
4. **Design coverage.** The (tube × action × step) enumeration must actually cover
   all four actions and all three representative steps, with no duplicate sample ids.

Usage::

    python scripts/diagnose/check_labels.py ./LCOCF_OpenVid1M_Processed
    python scripts/diagnose/check_labels.py ./LCOCF_OpenVid1M_Processed --limit 5000
"""

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ``python scripts/diagnose/check_labels.py`` puts *this file's directory* on
# sys.path[0] — never the repo root, and never the cwd. ``import cocf`` therefore
# resolves through site-packages, where a stale editable install can serve a
# completely different checkout than the one being edited. Pinning the sibling
# package first makes the tree this script ships with the tree that runs; the
# assertion below turns a silent substitution into a startup error, because a
# diagnostic that validates the wrong codebase is worse than one that refuses.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if (_REPO_ROOT / "cocf" / "__init__.py").is_file():
    sys.path.insert(0, str(_REPO_ROOT))

import cocf

_LOADED = Path(cocf.__file__).resolve().parent.parent
if (_REPO_ROOT / "cocf" / "__init__.py").is_file() and _LOADED != _REPO_ROOT:
    raise SystemExit(
        f"import cocf 解析到了 {_LOADED}/cocf,而不是本仓库的 {_REPO_ROOT}/cocf。\n"
        "site-packages 里有一个指向别处的 editable 安装,你在本仓库里的所有修改都不会生效。\n"
        "修复:  pip uninstall -y cocf  &&  pip install -e "
        f"{_REPO_ROOT}\n"
        f"临时绕过:  PYTHONPATH={_REPO_ROOT} python ..."
    )

from cocf.common.types import Action, TUBE_STATE_FIELDS
from cocf.data.processed_layout import ProcessedLayout
from cocf.data.sample_store import _store_is_lmdb, iter_lmdb_records
from cocf.lcocf.damage import DAMAGE_DIMENSIONS, DEFAULT_DAMAGE_WEIGHTS

# A FULL rollout reproduces the teacher transition, so its damage is bounded by
# sampler noise alone. Above this the store's reference and counterfactual sides are
# not comparable and no other label in it can be trusted.
_FULL_ANCHOR_MAX = 0.05
# Below this an axis carries no signal (constant across the whole store).
_DEGENERATE_STD = 1e-6
# Share of a damage axis allowed to sit exactly at 0 or exactly at 1 before the axis
# is reported as saturated rather than informative.
_SATURATED_FRAC = 0.95

_W = np.array([DEFAULT_DAMAGE_WEIGHTS[a] for a in DAMAGE_DIMENSIONS], np.float32)


class Report:
    """Collects check outcomes so the exit code can reflect the worst one."""

    def __init__(self) -> None:
        self.rows: List[tuple] = []

    def add(self, level: str, name: str, detail: str) -> None:
        self.rows.append((level, name, detail))
        mark = {"PASS": "✅", "WARN": "⚠️ ", "FAIL": "❌"}[level]
        print(f"{mark} [{level}] {name}\n      {detail}\n")

    @property
    def worst(self) -> str:
        levels = {r[0] for r in self.rows}
        return "FAIL" if "FAIL" in levels else ("WARN" if "WARN" in levels else "PASS")


def _vec(payload: Dict, key: str, dim: int) -> Optional[np.ndarray]:
    v = payload.get(key)
    if v is None:
        return None
    a = np.asarray(v, dtype=np.float32).reshape(-1)
    return a if a.size == dim else None


class Scan:
    """Lightweight aggregates from one streaming pass over the store.

    The checks only read a handful of small fields, so full payloads (tensors and
    all) are never resident at once: a whole store used to be materialised into a
    list, which is tens of GiB of Python objects on a real run.
    """

    def __init__(self) -> None:
        self.damage: List[np.ndarray] = []
        self.actions: List[int] = []
        self.uncertainty: List[np.ndarray] = []
        self.tube_features: List[np.ndarray] = []
        self.step_fracs: List[float] = []
        self.video_ids: set = set()
        self.keys: set = set()
        self.n = 0
        self.n_dup = 0
        self.n_missing_damage = 0


def _iter_records(layout: ProcessedLayout):
    """Stream ``(sample_id, payload)`` — at most one record (LMDB) or one shard
    (.pt fallback) resident at a time."""
    if _store_is_lmdb(layout.lmdb_dir):
        yield from iter_lmdb_records(layout.lmdb_dir)
    else:
        import torch

        for shard in sorted(Path(layout.lmdb_dir).glob("shard_*.pt")):
            for rec in torch.load(shard, map_location="cpu", weights_only=False):
                yield rec["sample_id"], rec["payload"]


def scan_samples(layout: ProcessedLayout, limit: Optional[int]) -> Scan:
    n_dim = len(DAMAGE_DIMENSIONS)
    scan = Scan()
    for key, payload in _iter_records(layout):
        if limit is not None and scan.n >= limit:
            break
        if key in scan.keys:
            scan.n_dup += 1
        scan.keys.add(key)
        d = _vec(payload, "damage_label", n_dim)
        if d is None:
            d = np.zeros(n_dim, np.float32)
            scan.n_missing_damage += 1
        scan.damage.append(d)
        scan.actions.append(int(payload.get("action", -1)))
        u = _vec(payload, "uncertainty", n_dim)
        if u is not None:
            scan.uncertainty.append(u)
        tf = _vec(payload, "tube_features", len(TUBE_STATE_FIELDS))
        if tf is not None:
            scan.tube_features.append(tf)
        scan.step_fracs.append(round(float(payload.get("step_frac", 0.0)), 4))
        vid = payload.get("video_id")
        if vid:
            scan.video_ids.add(str(vid))
        scan.n += 1
    if not scan.n:
        raise SystemExit(f"{layout.lmdb_dir} 里没有样本 —— Stage A 可能没跑完")
    print(f"读取 {scan.n} 条样本 (来自 {layout.lmdb_dir})\n")
    return scan


# --------------------------------------------------------------------------- #
# the checks
# --------------------------------------------------------------------------- #

def check_full_anchor(rep: Report, dmg: np.ndarray, act: np.ndarray) -> None:
    """§1: the one label whose correct value is known independently."""
    sel = act == int(Action.FULL)
    if not sel.any():
        rep.add("FAIL", "FULL 锚点",
                "存储里没有任何 action=FULL 的样本 —— 无法校准,且说明动作枚举有缺陷"
                "(见 P0-4:_balanced_triplets 在奇数管数下会漏掉动作)。")
        return
    scal = (dmg[sel] * _W).sum(1)
    m, p95 = float(scal.mean()), float(np.percentile(scal, 95))
    detail = (f"n={int(sel.sum())}  damage 均值={m:.4f}  p95={p95:.4f}  "
              f"(阈值 {_FULL_ANCHOR_MAX})")
    if m <= _FULL_ANCHOR_MAX:
        rep.add("PASS", "FULL 锚点", detail + "\n      FULL 不跳过任何计算,其损伤接近 0,"
                                              "说明反事实 rollout 与教师轨迹同源可比。")
    else:
        rep.add("FAIL", "FULL 锚点", detail +
                "\n      FULL 本应复现教师轨迹却测出了显著损伤 —— 反事实侧与参考侧不可比。"
                "\n      常见原因:rollout 与教师用了不同噪声/不同 σ 调度,或 Y_full 本身是坏的。"
                "\n      在这条不通过之前,其余所有标签都带着同样的偏移,不要拿去训练。")


def check_monotonicity(rep: Report, dmg: np.ndarray, act: np.ndarray) -> None:
    """§2: damage must grow as the action removes more compute."""
    means, missing = {}, []
    for a in Action:
        sel = act == int(a)
        if sel.any():
            means[int(a)] = float((dmg[sel] * _W).sum(1).mean())
        else:
            missing.append(a.name)
    line = "  ".join(f"{Action(k).name}={v:.4f}" for k, v in sorted(means.items()))
    if missing:
        rep.add("FAIL", "动作单调性",
                f"缺失动作 {missing} —— 无法检验单调性。这正是 P0-4 的症状。\n      实测: {line}")
        return
    order = [means[int(a)] for a in (Action.FULL, Action.LOWFREQ, Action.INTERP, Action.ANCHOR)]
    inversions = [i for i in range(3) if order[i] > order[i + 1] + 1e-4]
    if not inversions:
        rep.add("PASS", "动作单调性",
                f"{line}\n      损伤随动作激进程度单调递增,符合 FULL≤LOWFREQ≤INTERP≤ANCHOR 的构造。")
    else:
        names = [f"{list(Action)[i].name}→{list(Action)[i+1].name}" for i in inversions]
        rep.add("WARN", "动作单调性",
                f"{line}\n      在 {names} 处出现反转 —— 损伤度量可能没有真正测到退化。"
                "\n      若同时 FULL 锚点也不通过,先修锚点再看这条。")


def check_axes(rep: Report, dmg: np.ndarray) -> None:
    """§3: a constant axis is a metric backend that silently degraded."""
    dead, saturated, healthy = [], [], []
    for i, name in enumerate(DAMAGE_DIMENSIONS):
        col = dmg[:, i]
        std = float(col.std())
        at_bounds = float(((col <= 1e-6) | (col >= 1 - 1e-6)).mean())
        if std < _DEGENERATE_STD:
            dead.append(f"{name}(恒={col.mean():.3f})")
        elif at_bounds > _SATURATED_FRAC:
            saturated.append(f"{name}({at_bounds:.0%} 在边界)")
        else:
            healthy.append(f"{name}(σ={std:.3f})")
    detail = "存活: " + ", ".join(healthy) if healthy else "无存活轴"
    if dead:
        rep.add("FAIL", "damage 轴有效性",
                f"恒定轴: {', '.join(dead)}\n      {detail}"
                "\n      恒定轴 = 对应度量后端已静默退化。若含 raft_motion/motion_smoothness,"
                "查 RAFT 是否加载成功(P0-5);若含 ocr_accuracy,查 easyocr。"
                "\n      注意 mock 度量后端会让多个轴同时恒定 —— 确认跑了 --real-models。")
    elif saturated:
        rep.add("WARN", "damage 轴有效性",
                f"饱和轴: {', '.join(saturated)}\n      {detail}"
                "\n      标签挤在 0/1 两端,回归目标几乎没有梯度可学。")
    else:
        rep.add("PASS", "damage 轴有效性", f"8 个轴全部有方差。{detail}")


def check_uncertainty(rep: Report, scan: Scan) -> None:
    """Multi-seed variance: zero everywhere means the seeds never differed."""
    if not scan.uncertainty:
        rep.add("WARN", "多 seed 不确定度", "样本里没有 uncertainty 字段。")
        return
    u = np.stack(scan.uncertainty)
    nz = float((u > 1e-8).any(1).mean())
    if nz > 0.5:
        rep.add("PASS", "多 seed 不确定度",
                f"{nz:.0%} 的样本方差非零,均值={float(u.mean()):.5f} —— seeds_per_prompt 生效。")
    else:
        rep.add("WARN", "多 seed 不确定度",
                f"仅 {nz:.0%} 的样本方差非零 —— 多 seed 可能没起作用"
                "(seeds_per_prompt=1,或每个 seed 用了相同的随机源)。"
                "\n      后果:Stage B 的异方差加权退化为等权。")


def check_coverage(rep: Report, act: np.ndarray, scan: Scan) -> None:
    """§4: the sampling design actually covers what it claims to."""
    counts = Counter(int(a) for a in act)
    dist = "  ".join(f"{Action(k).name}={counts.get(int(k),0)}" for k in Action)
    n = len(act)
    share = [counts.get(int(a), 0) / max(1, n) for a in Action]
    skew = max(share) / max(1e-9, min(share)) if min(share) > 0 else float("inf")

    steps = sorted(set(scan.step_fracs))
    dup = scan.n_dup

    parts = [f"动作分布: {dist}  (最大/最小 = {skew:.2f}×)", f"step_frac 取值: {steps}"]
    level = "PASS"
    if counts.get(int(Action.ANCHOR), 0) == 0:
        level = "FAIL"
        parts.append("ANCHOR 完全缺失 —— P0-4 的经典症状(管数为奇数时动作枚举退化)。")
    elif skew > 2.0:
        level = "WARN"
        parts.append("动作分布严重不均 —— Stage B 的 StratifiedBatchSampler 会反复重采小桶,"
                     "造成过拟合。")
    if dup:
        level = "FAIL"
        parts.append(f"发现 {dup} 个重复 sample_id —— P1-7(.pt 后端不去重)。"
                     "重复样本会在一个 epoch 里被访问多次。")
    if len(steps) < 3:
        level = "WARN" if level == "PASS" else level
        parts.append(f"只覆盖了 {len(steps)} 个代表步(配置期望 3 个)—— "
                     "step_frac 接近常数会让预测器学不到去噪阶段的条件依赖。")
    if level == "PASS":
        parts.append("4 个动作与 3 个代表步均衡覆盖,无重复 id。")
    rep.add(level, "采样设计覆盖度", "\n      ".join(parts))


def check_tube_features(rep: Report, scan: Scan) -> None:
    """The 7-dim state is consumed as if every component were in [0,1]."""
    if not scan.tube_features:
        rep.add("WARN", "管状态值域", "样本里没有 tube_features 字段。")
        return
    tf = np.stack(scan.tube_features)
    bad = [(TUBE_STATE_FIELDS[i], float(tf[:, i].min()), float(tf[:, i].max()))
           for i in range(tf.shape[1])
           if tf[:, i].min() < -1e-6 or tf[:, i].max() > 1.0 + 1e-6]
    if not bad:
        rep.add("PASS", "管状态值域", f"7 个分量全部落在 [0,1](n={len(tf)})。")
    else:
        desc = ", ".join(f"{n}∈[{lo:.3f},{hi:.3f}]" for n, lo, hi in bad)
        rep.add("WARN", "管状态值域",
                f"越界分量: {desc}"
                "\n      未归一化的分量会主导 DamagePredictor 前几层的输入尺度。"
                "\n      已知项:interaction 是 IoU 之和而非均值(P2-4),上界为 (K-1)×帧数。")


def check_text_embeds(rep: Report, layout: ProcessedLayout, scan: Scan) -> None:
    """A missing per-clip embedding silently zeroes the whole batch's CMSC loss."""
    vids = scan.video_ids
    if not vids:
        rep.add("WARN", "text_embed 完整性", "样本里没有 video_id,无法校验。")
        return
    missing = [v for v in sorted(vids) if not layout.text_embed_path(v).exists()]
    if not missing:
        rep.add("PASS", "text_embed 完整性", f"{len(vids)} 个 clip 的 prompt embedding 齐全。")
    else:
        rep.add("FAIL", "text_embed 完整性",
                f"{len(missing)}/{len(vids)} 个 clip 缺失 text_embeds/*.pt,例如 {missing[:3]}"
                "\n      P1-9:collate 是「全有才组装」,任何一个样本缺失都会让**整个 batch** 的"
                "CMSC 损失静默变 0,而 λ_cmsc 因此拿不到梯度。")


def main() -> int:
    ap = argparse.ArgumentParser(description="校验 Stage A 生成的反事实标签是否可用于训练")
    ap.add_argument("root", type=Path, help="Stage A 输出根目录")
    ap.add_argument("--limit", type=int, default=None, help="只读前 N 条样本(默认全部)")
    args = ap.parse_args()

    layout = ProcessedLayout(args.root)
    env = layout.read_stage_a_env() if layout.stage_a_env.exists() else {}
    if env:
        print(f"Stage A env: backbone={env.get('backbone')} extra={env.get('backbone_extra')}")
        print(f"             {env.get('num_frames')}x{env.get('height')}x{env.get('width')}, "
              f"{env.get('teacher_steps')} steps, token_dim={env.get('token_dim')}\n")

    scan = scan_samples(layout, args.limit)
    if scan.n_missing_damage:
        print(f"注意: {scan.n_missing_damage} 条样本缺少 damage_label,已按全零计入。\n")
    dmg = np.stack(scan.damage)
    act = np.array(scan.actions)

    rep = Report()
    check_full_anchor(rep, dmg, act)
    check_monotonicity(rep, dmg, act)
    check_axes(rep, dmg)
    check_uncertainty(rep, scan)
    check_coverage(rep, act, scan)
    check_tube_features(rep, scan)
    check_text_embeds(rep, layout, scan)

    n_fail = sum(1 for r in rep.rows if r[0] == "FAIL")
    n_warn = sum(1 for r in rep.rows if r[0] == "WARN")
    print("─" * 70)
    print(f"结论: {rep.worst}   ({n_fail} FAIL, {n_warn} WARN, "
          f"{len(rep.rows) - n_fail - n_warn} PASS)")
    if rep.worst == "FAIL":
        print("存在 FAIL 项 —— 这批标签不适合训练 Stage B,先按上面的提示定位。")
    return 1 if rep.worst == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
