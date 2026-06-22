"""Stage A: offline counterfactual teacher data generation (§1.1–§1.6).

Runs once with the frozen backbone to build the §3 ``LCOCF_OpenVid1M_Processed``
store that Stages B/C consume. It is independent of any training loop (no gradients,
no optimiser), embarrassingly parallel over clips, and — because the teacher
generates ``Y_full`` from the *caption* (text-to-video, §1.3), not by reconstructing
the source clip — it runs end-to-end with only the OpenVid metadata CSV present (no
mp4 files needed), which is what makes the whole pipeline CPU/mock-testable.

Pipeline (each step delegates to the already-built data layer — see the module map):

    §1.1  read_openvid_manifest → metadata/raw_dataset_index.csv
    §1.2/§2  QualityFilter (four levels) → metadata/filtered_final.csv (+ split),
             raw_filtered/captions.jsonl (Stage C source), metadata/filter_report.json
    §1.3–§1.4  TeacherForwardRunner → TeacherTrajectory per kept clip; persist the
             §3 level-3 baseline bucket and level-4 tube/causal-feature bucket
    §1.5  COCFDataGenerator.generate → single-hop counterfactual samples → LMDB
    §1.6  3σ damage-outlier cleaning, min-max norm stats, sample_index.csv, and the
             video_id-disjoint splits/*.txt lists (each sample inherits its video's
             split — no clip appears in two splits)

Only the *indexed* samples (``sample_index.csv`` + ``splits/*.txt``) are ever read by
Stage B, so 3σ outliers are simply omitted from the index rather than deleted from
the LMDB — a single streaming write pass, bounded memory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from cocf.backbones.base import BackboneAdapter
from cocf.common.config import Config
from cocf.common.logging import get_logger
from cocf.common.memory import free_memory
from cocf.data import (
    CounterfactualSampleWriter,
    OpenVidRecord,
    ProcessedLayout,
    QualityFilter,
    read_openvid_manifest,
    video_id_str,
    write_raw_dataset_index,
)
from cocf.lcocf.damage import MetricExtractor
from cocf.lcocf.data import (
    COCFDataGenerator,
    COCFTrainingSample,
    CounterfactualDamageComputer,
    StratifiedSamplingConfig,
    TeacherTrajectory,
)
from cocf.lcocf.strength import CausalStrengthFeatureBuilder
from cocf.training.teacher_forward import TeacherForwardConfig, TeacherForwardRunner

Tensor = torch.Tensor
_log = get_logger(__name__)

# norm-stat groups (§1.6 min-max over strength / tube-state / cost labels)
_NORM_GROUPS = ("strength_features", "tube_features", "cost_label")


@dataclass
class StageAConfig:
    """Stage-A inputs: where OpenVid lives and where the processed store goes.

    Generation knobs (steps, representative timesteps, seeds, caps) come from
    ``config.teacher`` / ``config.data``; filter thresholds from ``config.filter`` —
    all already defined in :class:`~cocf.common.config.Config`, so this only carries
    the paths and a few run-scoped overrides.
    """

    openvid_csvs: List[Path]                 # OpenVid-1M.csv [+ OpenVidHD.csv]
    processed_root: Path                     # LCOCF_OpenVid1M_Processed root (§3)
    data_root: str = ""                      # clips resolve to {data_root}/{video_subdir}/{video}
    config: Config = field(default_factory=Config)
    device: torch.device = field(default_factory=lambda: torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"))
    limit: Optional[int] = None              # cap rows per CSV (debug / smoke)
    samples_per_video: Optional[int] = None  # override config.teacher.samples_per_video
    persist_buckets: bool = True             # write §3 level-3/level-4 per-video buckets
    seed: int = 1234


class DataGenerationStage:
    """Stage A: build the six-level processed store from OpenVid (§1)."""

    def __init__(
        self,
        config: StageAConfig,
        backbone: BackboneAdapter,
        metric_extractor: MetricExtractor,
        accelerator,
    ) -> None:
        self.config = config
        self.backbone = backbone
        self.metric_extractor = metric_extractor
        self.accelerator = accelerator
        self.layout = ProcessedLayout(config.processed_root)

        cfg = config.config
        self.teacher_runner = TeacherForwardRunner(
            accelerator, TeacherForwardConfig.from_config(cfg), device=config.device
        )
        self.damage_computer = CounterfactualDamageComputer(metric_extractor)
        sampling_cfg = StratifiedSamplingConfig(
            interpolation_interval=5, use_label_interpolation=cfg.teacher.interpolate_adjacent_steps
        )
        self.data_generator = COCFDataGenerator(
            metric_extractor=metric_extractor,
            strength_feature_builder=CausalStrengthFeatureBuilder(),
            damage_computer=self.damage_computer,
            sampling_config=sampling_cfg,
            device=config.device,
            perception=accelerator.perception,
            seeds_per_prompt=cfg.teacher.seeds_per_prompt,
        )
        self.transition = accelerator.transition

    # ------------------------------------------------------------------ #
    # entry point
    # ------------------------------------------------------------------ #

    def run(self) -> Path:
        """Execute Stage A and return the processed-store root."""
        cfg = self.config.config
        layout = self.layout.create()
        _log.info("=== Stage A: Counterfactual Teacher Data Generation (§1) ===")

        # --- §1.1 ingest OpenVid metadata → raw_dataset_index.csv --------- #
        records = read_openvid_manifest(
            [str(p) for p in self.config.openvid_csvs],
            self.config.data_root,
            video_subdir=cfg.data.video_subdir,
            static_motion_max=cfg.filter.static_motion_max,
            limit_per_csv=self.config.limit,
        )
        write_raw_dataset_index(records, layout)
        _log.info("§1.1 ingested %d OpenVid records", len(records))

        # --- §1.2/§2 four-level quality filter → filtered_final.csv ------- #
        qfilter = QualityFilter(cfg.filter, perception=self.accelerator.perception)
        result = qfilter.apply(records, seed=self.config.seed)
        qfilter.write(layout, result)
        layout.write_captions(self._caption_rows(result.kept))
        self._write_report(layout, result.report.as_dict())
        _log.info(
            "§2 filter kept %d/%d clips (hd=%.0f%%, complex=%.0f%%); split %d/%d/%d",
            result.report.kept_final, result.report.total_in,
            100 * result.report.hd_frac, 100 * result.report.complex_frac,
            result.report.n_train, result.report.n_val, result.report.n_test_hard,
        )

        # --- §1.3–§1.5 teacher forward + single-hop counterfactual labels - #
        max_samples = self.config.samples_per_video or cfg.teacher.samples_per_video
        index_rows: List[Dict[str, object]] = []
        split_of_sample: Dict[str, str] = {}
        damage_scalars: List[float] = []
        norm_acc: Dict[str, List[np.ndarray]] = {g: [] for g in _NORM_GROUPS}
        tube_meta_rows: List[Dict[str, object]] = []

        n_clips = 0
        with CounterfactualSampleWriter(layout.lmdb_dir, shard_size=cfg.teacher.shard_size) as writer:
            for rec in self._scene_interleaved(result.kept):
                traj = self.teacher_runner.run(rec.video_id, rec.caption, rec.scene_type)
                if traj is None:
                    continue
                if self.config.persist_buckets:
                    self._persist_baseline(traj)
                    self._persist_features(traj)
                tube_meta_rows.extend(self._tube_meta(traj))

                samples = self.data_generator.generate(
                    traj, self.backbone, self.transition,
                    max_tubes=cfg.teacher.max_tubes_per_prompt,
                    max_samples=max_samples,
                )
                split = result.split_by_video.get(rec.video_id, "train")
                for s in samples:
                    sid = self._sample_id(s)
                    writer.put(sid, s)
                    index_rows.append({
                        "sample_id": sid,
                        "video_id": s.video_id,
                        "timestep": int(s.timestep),
                        "action": int(s.action),
                        "scene_type": s.scene_type,
                    })
                    split_of_sample[sid] = split
                    damage_scalars.append(s.damage_scalar())
                    self._accumulate_norm(norm_acc, s)
                n_clips += 1
                _log.info("  [%d] %s → %d samples", n_clips, rec.video_id, len(samples))
                free_memory()

        # --- §1.6 cleaning, normalisation, index & leakage-safe splits ---- #
        keep_mask = self._outlier_mask(damage_scalars)
        kept_rows = [r for r, k in zip(index_rows, keep_mask) if k]
        dropped = len(index_rows) - len(kept_rows)
        layout.write_sample_index(kept_rows)
        layout.write_tube_meta(tube_meta_rows)
        layout.write_norm_stats(self._norm_stats(norm_acc, keep_mask))
        self._write_splits(layout, kept_rows, split_of_sample)
        _log.info(
            "§1.6 wrote %d training samples (dropped %d 3σ outliers) from %d clips → %s",
            len(kept_rows), dropped, n_clips, layout.root,
        )
        return layout.root

    # ------------------------------------------------------------------ #
    # §1.6 helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _outlier_mask(damage: Sequence[float], sigma: float = 3.0) -> List[bool]:
        """3σ damage-outlier mask (§1.6). All-kept when too few samples to estimate σ."""
        if len(damage) < 8:
            return [True] * len(damage)
        arr = np.asarray(damage, dtype="float64")
        mu, sd = float(arr.mean()), float(arr.std())
        if sd <= 1e-9:
            return [True] * len(damage)
        lo, hi = mu - sigma * sd, mu + sigma * sd
        return [(lo <= x <= hi) for x in damage]

    @staticmethod
    def _accumulate_norm(acc: Dict[str, List[np.ndarray]], sample: COCFTrainingSample) -> None:
        for group in _NORM_GROUPS:
            vec = getattr(sample, group, None)
            if isinstance(vec, Tensor) and vec.numel():
                acc[group].append(vec.detach().cpu().float().numpy())

    @staticmethod
    def _norm_stats(acc: Dict[str, List[np.ndarray]], keep: Sequence[bool]) -> Dict[str, object]:
        """Per-field min-max over the kept samples (§1.6 min-max 标准化 stats)."""
        stats: Dict[str, object] = {}
        for group, rows in acc.items():
            kept = [r for r, k in zip(rows, keep) if k] if len(keep) == len(rows) else rows
            if not kept:
                continue
            mat = np.stack(kept)
            stats[group] = {"min": mat.min(0).tolist(), "max": mat.max(0).tolist()}
        return stats

    def _write_splits(
        self, layout: ProcessedLayout, rows: Sequence[Dict[str, object]], split_of: Dict[str, str]
    ) -> None:
        buckets: Dict[str, List[str]] = {"train": [], "val": [], "test_hard": []}
        for r in rows:
            sid = str(r["sample_id"])
            buckets.get(split_of.get(sid, "train"), buckets["train"]).append(sid)
        layout.write_splits(buckets["train"], buckets["val"], buckets["test_hard"])

    # ------------------------------------------------------------------ #
    # persistence of §3 level-3 / level-4 buckets
    # ------------------------------------------------------------------ #

    def _persist_baseline(self, traj: TeacherTrajectory) -> None:
        """Write the §3 level-3 ``full_baseline/<video_id>/`` bucket (z_t keyed by t)."""
        z_by_t = {traj.num_total_steps - step_idx: z for step_idx, z in traj.z_by_step.items()}
        text_emb = traj.text_embed if traj.text_embed is not None else torch.zeros(1)
        self.layout.save_baseline(
            traj.video_id, text_emb=text_emb, z_t_by_step=z_by_t, y_full=traj.video_full,
        )

    def _persist_features(self, traj: TeacherTrajectory) -> None:
        """Write the §3 level-4 ``tube_causal_features/<video_id>/`` bucket."""
        tubes = traj.tubes
        if not tubes:
            return
        states = torch.stack([traj.tube_states[t.tube_id].as_tensor() for t in tubes])      # [K,7]
        strength = torch.stack([traj.strength_feats[t.tube_id].as_tensor() for t in tubes])  # [K,3]
        visual = torch.stack([traj.tube_visual_embed_full[t.tube_id] for t in tubes])        # [K,d_v]
        # compact per-tube (id, token-count) record in lieu of full masks — masks are
        # only needed at generation time and would bloat the store; Stages B/C read
        # the LMDB samples, not this bucket.
        ident = torch.tensor([[float(t.tube_id), float(t.size)] for t in tubes])             # [K,2]
        self.layout.save_features(
            traj.video_id, tube_features=ident, tube_states=states,
            causal_strength=strength, tube_visual_emb=visual,
        )

    @staticmethod
    def _tube_meta(traj: TeacherTrajectory) -> List[Dict[str, object]]:
        rows = []
        for t in traj.tubes:
            f = traj.strength_feats[t.tube_id]
            rows.append({
                "video_id": traj.video_id, "tube_id": t.tube_id, "scene_type": traj.scene_type,
                "num_frames": t.length, "size": t.size,
                "s_E": round(f.s_E, 4), "s_A": round(f.s_A, 4), "s_T": round(f.s_T, 4),
            })
        return rows

    # ------------------------------------------------------------------ #
    # misc helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _sample_id(s: COCFTrainingSample) -> str:
        """Stable unique key: (video, tube, timestep, action) is unique per clip (§1.5)."""
        return f"{video_id_str(s.video_id)}__t{int(s.timestep):03d}__k{int(s.tube_id):04d}__a{int(s.action)}"

    @staticmethod
    def _caption_rows(records: Sequence[OpenVidRecord]) -> List[Dict[str, object]]:
        """``raw_filtered/captions.jsonl`` rows; ``path`` lets Stage C read source clips
        in place without duplicating the (large) mp4s into the store."""
        return [
            {"video_id": r.video_id, "caption": r.caption, "scene_type": r.scene_type,
             "is_hd": int(r.is_hd), "path": r.path, "seconds": r.seconds}
            for r in records
        ]

    def _write_report(self, layout: ProcessedLayout, report: Dict[str, object]) -> None:
        path = layout.metadata_dir / "filter_report.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)

    @staticmethod
    def _scene_interleaved(records: Sequence[OpenVidRecord]) -> List[OpenVidRecord]:
        """Round-robin records across scene types so a ``--limit`` truncation still
        covers all six scene classes (§1.2 场景覆盖度)."""
        by_scene: Dict[str, List[OpenVidRecord]] = {}
        for r in records:
            by_scene.setdefault(r.scene_type, []).append(r)
        queues = list(by_scene.values())
        out: List[OpenVidRecord] = []
        i = 0
        while len(out) < len(records):
            q = queues[i % len(queues)]
            if q:
                out.append(q.pop(0))
            i += 1
            if all(not q for q in queues):
                break
        return out
