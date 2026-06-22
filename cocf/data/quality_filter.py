"""Four-level video quality filter (§2).

Filters the parsed OpenVid records (`:class:`~cocf.data.openvid_manifest.OpenVidRecord``)
from raw ingestion down to the §2.4 final training set, in four escalating levels
that lean on the dataset's own metadata so the pass is cheap and decode-free:

    L1  basic hard filter (§2.1, Table 0)  — resolution / duration / (opt) blur·watermark
    L2  semantic filter   (§2.2)           — caption length, aesthetic %ile, dedup
    L3  task-fitness       (§2.3)           — drop pure-static, force-keep hard samples
    L4  final sampling     (§2.4)           — target count, HD ≥60%, video-disjoint split

The leakage-safe train/val/test partition (§2.4 / §4.2 "按视频 ID 切分…不跨集") is
decided here at the **video level** and recorded as a ``split`` column in
``filtered_final.csv``; Stage A then materialises the §3 level-6 sample-id lists
(``splits/*.txt``) by inheriting each generated sample's split from its video. This
keeps the partition decision in one place and guarantees no source video appears in
two splits.

Optional ``perception`` / ``metric_extractor`` hooks enable the decode-dependent
gates (Laplacian blur, CLIP image-text alignment) when real backends are supplied;
without them those gates are skipped (the metadata gates still apply), so the filter
runs on this CPU box and in tests.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from cocf.common.config import FilterConfig
from cocf.common.logging import get_logger
from cocf.data.openvid_manifest import OpenVidRecord, SCENE_TYPES, scene_histogram
from cocf.data.processed_layout import ProcessedLayout

_log = get_logger(__name__)

# Scenes that §2.3 force-keeps as "hard" (multi-subject / occlusion / text / face)
# plus fast motion; these are never dropped and seed the hard-sample test list.
_HARD_SCENES = ("multi", "occlusion", "text", "face")
# strip a trailing OpenVid segment suffix ``_<start>_<end>`` so clips cut from the
# same source video share a base id and never split across train/val/test.
_SEGMENT_SUFFIX = re.compile(r"_\d+_\d+$")


def base_video_id(video_id: str) -> str:
    """Source-video id for leakage-safe splitting (clip stem minus segment range)."""
    return _SEGMENT_SUFFIX.sub("", video_id)


@dataclass
class FilterReport:
    """Per-level drop counts and final-set statistics (§2.4 report)."""

    total_in: int = 0
    kept_l1: int = 0
    kept_l2: int = 0
    kept_l3: int = 0
    kept_final: int = 0
    dropped: Dict[str, int] = field(default_factory=dict)
    scene_hist: Dict[str, int] = field(default_factory=dict)
    hd_frac: float = 0.0
    complex_frac: float = 0.0
    mean_aesthetic: float = 0.0
    n_train: int = 0
    n_val: int = 0
    n_test_hard: int = 0

    def as_dict(self) -> Dict[str, object]:
        return dict(self.__dict__)


@dataclass
class FilterResult:
    kept: List[OpenVidRecord]
    split_by_video: Dict[str, str]   # video_id -> "train" | "val" | "test_hard"
    report: FilterReport


def _is_hard(r: OpenVidRecord, fast_motion: float) -> bool:
    return r.scene_type in _HARD_SCENES or r.motion >= fast_motion


def _complexity(r: OpenVidRecord, fast_motion: float) -> str:
    if r.scene_type in ("multi", "occlusion") or r.motion >= fast_motion:
        return "complex"
    if r.scene_type == "static":
        return "simple"
    return "medium"


class QualityFilter:
    """The §2 four-level filter over OpenVid records."""

    def __init__(
        self,
        cfg: FilterConfig,
        *,
        perception=None,
        metric_extractor=None,
    ) -> None:
        self.cfg = cfg
        self.perception = perception
        self.metric_extractor = metric_extractor
        # "fast motion" threshold for the hard/complex predicate — well above the
        # static cut-off so only genuinely dynamic clips qualify.
        self.fast_motion = max(0.2, cfg.static_motion_max * 10.0)

    # ------------------------------------------------------------------ #
    # public entry
    # ------------------------------------------------------------------ #

    def apply(self, records: Sequence[OpenVidRecord], *, seed: int = 1234) -> FilterResult:
        report = FilterReport(total_in=len(records), dropped={})
        pool = list(records)

        pool = self._level1_hard(pool, report)
        report.kept_l1 = len(pool)
        pool = self._level2_semantic(pool, report)
        report.kept_l2 = len(pool)
        pool = self._level3_task_fitness(pool, report)
        report.kept_l3 = len(pool)
        pool = self._level4_final_sampling(pool, report)
        report.kept_final = len(pool)

        split_by_video = self._split_by_video(pool, seed=seed)
        self._fill_report(pool, split_by_video, report)
        return FilterResult(kept=pool, split_by_video=split_by_video, report=report)

    def write(self, layout: ProcessedLayout, result: FilterResult) -> Path:
        """Write ``metadata/filtered_final.csv`` with the per-video ``split`` column."""
        rows = []
        for r in result.kept:
            row = r.index_row()
            row["split"] = result.split_by_video.get(r.video_id, "train")
            row["is_hard"] = int(_is_hard(r, self.fast_motion))
            row["complexity"] = _complexity(r, self.fast_motion)
            rows.append(row)
        fields = (list(rows[0].keys()) if rows
                  else list(OpenVidRecord.__annotations__.keys()) + ["split", "is_hard", "complexity"])
        return layout.write_csv(layout.filtered_final, rows, fields)

    # ------------------------------------------------------------------ #
    # L1 — basic hard filter (§2.1, Table 0)
    # ------------------------------------------------------------------ #

    def _level1_hard(self, pool: List[OpenVidRecord], rep: FilterReport) -> List[OpenVidRecord]:
        c = self.cfg
        kept = []
        for r in pool:
            if not r.caption.strip():
                rep.dropped["l1_no_caption"] = rep.dropped.get("l1_no_caption", 0) + 1
                continue
            # duration window (only when a duration is known); HD is always 1080p so
            # never dropped on resolution, the OpenVid-1M base set is ≥512² by spec.
            if r.seconds > 0 and (r.seconds < c.min_duration_s or r.seconds > c.max_duration_s):
                rep.dropped["l1_duration"] = rep.dropped.get("l1_duration", 0) + 1
                continue
            if not self._passes_blur(r):
                rep.dropped["l1_blur"] = rep.dropped.get("l1_blur", 0) + 1
                continue
            kept.append(r)
        return kept

    def _passes_blur(self, r: OpenVidRecord) -> bool:
        """Laplacian-variance gate — only active when a real perception hook + a
        decoded frame are available *and* the threshold is enabled (>0)."""
        if self.cfg.blur_laplacian_min <= 0 or self.perception is None:
            return True
        sharp = getattr(self.perception, "sharpness", None)
        if sharp is None:
            return True
        try:
            return float(sharp(r.path)) >= self.cfg.blur_laplacian_min
        except Exception:
            return True

    # ------------------------------------------------------------------ #
    # L2 — semantic filter (§2.2)
    # ------------------------------------------------------------------ #

    def _level2_semantic(self, pool: List[OpenVidRecord], rep: FilterReport) -> List[OpenVidRecord]:
        c = self.cfg
        # caption quality: drop gibberish / too-short captions
        kept = []
        for r in pool:
            if len(re.findall(r"[\w']+", r.caption)) < c.min_caption_words:
                rep.dropped["l2_caption_short"] = rep.dropped.get("l2_caption_short", 0) + 1
                continue
            kept.append(r)
        # aesthetic: drop the bottom fraction by aesthetic score (only when scores
        # are actually present — all-zero means the column was absent, skip the gate)
        if c.aesthetic_drop_frac > 0 and any(r.aesthetic for r in kept):
            ordered = sorted(kept, key=lambda r: r.aesthetic)
            cut = int(c.aesthetic_drop_frac * len(ordered))
            dropped_ids = {id(r) for r in ordered[:cut]}
            rep.dropped["l2_aesthetic"] = cut
            kept = [r for r in kept if id(r) not in dropped_ids]
        # content de-duplication by normalised caption (keep highest aesthetic)
        kept = self._dedup(kept, rep)
        return kept

    @staticmethod
    def _dedup(pool: List[OpenVidRecord], rep: FilterReport) -> List[OpenVidRecord]:
        best: Dict[str, OpenVidRecord] = {}
        dropped = 0
        for r in pool:
            key = re.sub(r"\s+", " ", r.caption.strip().lower())
            cur = best.get(key)
            if cur is None:
                best[key] = r
            else:
                dropped += 1
                if r.aesthetic > cur.aesthetic:
                    best[key] = r
        if dropped:
            rep.dropped["l2_dedup"] = dropped
        # preserve original order for determinism
        keep_ids = {id(r) for r in best.values()}
        return [r for r in pool if id(r) in keep_ids]

    # ------------------------------------------------------------------ #
    # L3 — task-fitness filter (§2.3)
    # ------------------------------------------------------------------ #

    def _level3_task_fitness(self, pool: List[OpenVidRecord], rep: FilterReport) -> List[OpenVidRecord]:
        c = self.cfg
        if not c.drop_static:
            return pool
        kept = []
        for r in pool:
            # drop pure-static / no-motion clips (no causal scheduling signal) — but
            # never drop a force-keep hard sample even if its motion reads low.
            if (r.scene_type == "static" and r.motion <= c.static_motion_max
                    and not _is_hard(r, self.fast_motion)):
                rep.dropped["l3_static"] = rep.dropped.get("l3_static", 0) + 1
                continue
            kept.append(r)
        return kept

    # ------------------------------------------------------------------ #
    # L4 — final sampling (§2.4): target count, HD floor, complexity floor
    # ------------------------------------------------------------------ #

    def _level4_final_sampling(self, pool: List[OpenVidRecord], rep: FilterReport) -> List[OpenVidRecord]:
        c = self.cfg
        target = min(c.target_samples, len(pool))
        if len(pool) <= target:
            return pool

        by_aes = sorted(pool, key=lambda r: r.aesthetic, reverse=True)
        selected: List[OpenVidRecord] = []
        chosen = set()

        def take(r: OpenVidRecord) -> None:
            if id(r) not in chosen and len(selected) < target:
                chosen.add(id(r))
                selected.append(r)

        # 1) force-keep hard samples (§2.3), highest-aesthetic first
        for r in by_aes:
            if _is_hard(r, self.fast_motion):
                take(r)
        # 2) meet the HD floor (§2.4 OpenVidHD ≥60%)
        hd_target = int(c.hd_min_frac * target)
        if sum(1 for r in selected if r.is_hd) < hd_target:
            for r in by_aes:
                if r.is_hd:
                    take(r)
                if sum(1 for s in selected if s.is_hd) >= hd_target:
                    break
        # 3) fill the rest by descending aesthetic
        for r in by_aes:
            take(r)
        rep.dropped["l4_oversampled"] = len(pool) - len(selected)
        return selected

    # ------------------------------------------------------------------ #
    # leakage-safe split by source video (§2.4 / §4.2)
    # ------------------------------------------------------------------ #

    def _split_by_video(self, pool: List[OpenVidRecord], *, seed: int) -> Dict[str, str]:
        c = self.cfg
        groups: Dict[str, List[OpenVidRecord]] = defaultdict(list)
        for r in pool:
            groups[base_video_id(r.video_id)].append(r)

        order = list(groups)
        random.Random(seed).shuffle(order)
        total = len(pool)
        val_cap = int(c.val_frac * total)
        test_cap = int(c.test_hard_frac * total)

        split_by_video: Dict[str, str] = {}
        n_val = n_test = 0
        for base in order:
            members = groups[base]
            group_is_hard = any(_is_hard(r, self.fast_motion) for r in members)
            if n_val < val_cap:
                tag, n_val = "val", n_val + len(members)
            elif group_is_hard and n_test < test_cap:
                tag, n_test = "test_hard", n_test + len(members)
            else:
                tag = "train"
            for r in members:
                split_by_video[r.video_id] = tag
        return split_by_video

    # ------------------------------------------------------------------ #
    # reporting
    # ------------------------------------------------------------------ #

    def _fill_report(
        self, pool: List[OpenVidRecord], split: Dict[str, str], rep: FilterReport
    ) -> None:
        n = max(1, len(pool))
        rep.scene_hist = scene_histogram(pool)
        rep.hd_frac = round(sum(1 for r in pool if r.is_hd) / n, 4)
        rep.complex_frac = round(
            sum(1 for r in pool if _complexity(r, self.fast_motion) == "complex") / n, 4
        )
        rep.mean_aesthetic = round(sum(r.aesthetic for r in pool) / n, 4)
        rep.n_train = sum(1 for v in split.values() if v == "train")
        rep.n_val = sum(1 for v in split.values() if v == "val")
        rep.n_test_hard = sum(1 for v in split.values() if v == "test_hard")
