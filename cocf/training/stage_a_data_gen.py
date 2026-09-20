"""Stage A: offline counterfactual teacher data generation.

Runs once with the frozen backbone to build the processed store that Stages B/C
consume. It is independent of any training loop (no gradients, no optimiser) and
parallel over clips. Because the teacher generates ``Y_full`` from the caption
(text-to-video) rather than reconstructing the source clip, it runs end-to-end with
only the OpenVid metadata CSV present (no mp4 files), which keeps the pipeline
CPU/mock-testable.

Steps: read the OpenVid manifest, run the four-level quality filter (writing the
kept set, splits and Stage-C captions), generate the teacher trajectory and
counterfactual samples into the store, then finalize with 3σ damage-outlier
cleaning, min-max norm stats, the sample index and video-disjoint splits. Only
indexed samples are read by Stage B, so outliers are omitted from the index rather
than deleted — a single streaming write pass with bounded memory.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

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
    iter_lmdb_records,
    read_openvid_manifest,
    store_is_lmdb,
    video_id_str,
    write_raw_dataset_index,
)
from cocf.data.quality_filter import read_filtered_final
from cocf.lcocf.damage import DAMAGE_DIMENSIONS, DEFAULT_DAMAGE_WEIGHTS, MetricExtractor
from cocf.lcocf.data import (
    COCFDataGenerator,
    COCFTrainingSample,
    CounterfactualDamageComputer,
    StratifiedSamplingConfig,
    TeacherTrajectory,
)
from cocf.lcocf.strength import CausalStrengthFeatureBuilder
from cocf.training.teacher_forward import TeacherForwardConfig, TeacherForwardRunner
from cocf.data.video_dataset import (
    VideoReader,
    _resize_clip,
    sample_frame_indices,
)

Tensor = torch.Tensor
_log = get_logger(__name__)

# norm-stat groups: min-max over strength / tube-state / cost labels
_NORM_GROUPS = ("strength_features", "tube_features", "cost_label")

# Consecutive per-clip failures that make the generate loop give up (a run of failures
# signals an environment fault, not a bad clip).
_MAX_CONSECUTIVE_CLIP_FAILURES = 5

# How long a non-owning shard waits for shard 0 to publish filtered_final.csv before
# filtering locally.
_FILTER_HANDOFF_TIMEOUT_S = 1800.0
_FILTER_HANDOFF_POLL_S = 5.0


@dataclass
class StageAConfig:
    """Stage-A inputs: OpenVid paths, the processed-store root and run-scoped overrides."""

    openvid_csvs: List[Path]                 # OpenVid-1M.csv [+ OpenVidHD.csv]
    processed_root: Path                     # processed store root
    data_root: str = ""                      # clips resolve to {data_root}/{video_subdir}/{video}
    config: Config = field(default_factory=Config)
    device: torch.device = field(default_factory=lambda: torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"))
    limit: Optional[int] = None              # cap rows per CSV (debug / smoke)
    samples_per_video: Optional[int] = None  # override config.teacher.samples_per_video
    persist_buckets: bool = True             # master switch for the baseline/feature buckets
    persist_baseline: bool = True            # write the full_baseline bucket (~TB)
    persist_tube_features: bool = True       # write the tube_causal_features bucket (small)
    # Write representative-step latents into the baseline bucket (off by default; no
    # stage reads them).
    persist_step_latents: bool = False
    seed: int = 1234
    video_subdir: Optional[str] = None       # override config.data.video_subdir (e.g. "videos")
    require_file: bool = False               # keep only clips whose mp4 exists on disk
    use_real_video: bool = False             # anchor the teacher trajectory on real mp4 pixels
    # -- shard-parallel + resume -------- #
    num_shards: int = 1                      # total parallel workers over the clip set
    shard_index: int = 0                     # this worker's 0-based shard id
    finalize_only: bool = False              # skip generation; only build manifest/splits/index
    # Abort the shard on the first clip that raises instead of logging and moving on.
    fail_fast: bool = False


class DataGenerationStage:
    """Stage A: build the processed store from OpenVid."""

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
        # Keep the accelerator's plugins on the run device, matching the backbone.
        self.accelerator.to(config.device)
        self.layout = ProcessedLayout(config.processed_root)
        # Bucket persistence switches, resolved once so the generation loop stays branch-cheap.
        self._do_baseline = config.persist_buckets and config.persist_baseline
        self._do_features = config.persist_buckets and config.persist_tube_features

        cfg = config.config
        self.teacher_runner = TeacherForwardRunner(
            accelerator, TeacherForwardConfig.from_config(cfg), device=config.device
        )
        self.damage_computer = CounterfactualDamageComputer(metric_extractor)
        sampling_cfg = StratifiedSamplingConfig()
        self.data_generator = COCFDataGenerator(
            metric_extractor=metric_extractor,
            strength_feature_builder=CausalStrengthFeatureBuilder(),
            damage_computer=self.damage_computer,
            sampling_config=sampling_cfg,
            device=config.device,
            perception=accelerator.perception,
            seeds_per_prompt=cfg.teacher.seeds_per_prompt,
            free_memory_every=cfg.teacher.free_memory_every,
        )
        self.transition = accelerator.transition
        # Real-clip decoder, built once and only used when ``use_real_video`` is set.
        self._video_reader: Optional[VideoReader] = (
            self._build_video_reader() if config.use_real_video else None
        )

    @staticmethod
    def _build_video_reader() -> VideoReader:
        """Return a real-mp4 reader, preferring ``decord`` and falling back to
        ``torchvision`` — whichever is installed. Raises if neither is available."""
        from cocf.data.video_dataset import DecordVideoReader, TorchvisionVideoReader
        try:
            return DecordVideoReader()
        except Exception as exc:  # decord missing / unbuildable (common on macOS)
            _log.info("decord unavailable (%s); falling back to torchvision reader", exc)
            return TorchvisionVideoReader()

    # ------------------------------------------------------------------ #
    # entry point
    # ------------------------------------------------------------------ #

    def run(self) -> Path:
        """Execute Stage A (optionally one shard of it) and return the store root.

        The pipeline splits into a VRAM-bound *generate* phase and a pure-CPU *finalize*
        phase. This makes the stage shard-parallel, resumable (a ``_progress`` line per
        finished clip) and bounded-memory (finalize streams the statistics back off the
        shards). A single-process run finalizes inline; a multi-shard run generates only
        and needs one ``--finalize-only`` pass afterwards.
        """
        layout = self.layout.create()
        ns, si = max(1, self.config.num_shards), self.config.shard_index
        _log.info("=== Stage A: Counterfactual Teacher Data Generation (§1) — shard %d/%d ===", si, ns)

        # Every shard needs the kept set + split map, but only shard 0 writes the shared
        # global metadata (concurrent writers would corrupt the large CSVs).
        result = self._ingest_and_filter(
            layout, write_global=(si == 0 and not self.config.finalize_only)
        )

        if self.config.finalize_only:
            return finalize_processed_store(layout, result.split_by_video)

        # Record the store geometry/schedule before any sample is written; shard 0 owns it.
        if si == 0:
            self._write_stage_a_env(layout)

        # Generate this shard's counterfactual samples (resumable).
        self._generate(layout, result, ns, si)

        if ns > 1:
            _log.info(
                "shard %d/%d generation done. Once ALL shards finish, run one more pass "
                "with --finalize-only to build manifest/splits/sample_index.", si, ns,
            )
            return layout.root
        # single-process run: finalize inline.
        return finalize_processed_store(layout, result.split_by_video)

    # ------------------------------------------------------------------ #
    # ingest + filter
    # ------------------------------------------------------------------ #

    def _write_stage_a_env(self, layout: ProcessedLayout) -> None:
        """Persist the backbone geometry + schedule into ``metadata/stage_a_env.json``.

        Stages B/C read the token width back off the store, keeping their checkpoints
        compatible without threading identical CLI flags. ``ensure_loaded`` runs first so
        the recorded geometry reflects the loaded checkpoint, not a pre-load guess.
        """
        bb = self.backbone
        try:
            bb.ensure_loaded()
        except Exception as exc:
            if self.config.config.backbone.name == "mock":
                # Mock adapters have nothing to load; their geometry is declared.
                _log.debug("ensure_loaded() before env write: %s", exc)
            else:
                # A real backbone that failed to load would record a pre-load geometry guess.
                raise RuntimeError(
                    "ensure_loaded() failed before writing stage_a_env.json; "
                    "aborting rather than persisting a pre-load geometry guess"
                ) from exc
        d = self.config.config.data
        env = {
            "backbone": self.config.config.backbone.name,
            "model_path": self.config.config.backbone.model_path,
            "backbone_extra": dict(self.config.config.backbone.extra or {}),
            "dtype": self.config.config.backbone.dtype,
            "token_dim": int(bb.hidden_dim),
            "latent_channels": int(bb.latent_channels),
            "patch": list(getattr(bb, "patch", (1, 1, 1))),
            "vae_compress": list(getattr(bb, "vae_compress", (1, 1, 1))),
            "num_frames": int(d.num_frames),
            "height": int(d.height),
            "width": int(d.width),
            "teacher_steps": int(self.config.config.teacher.num_inference_steps),
            "representative_step_fracs": list(
                self.config.config.teacher.representative_step_fracs
            ),
            "use_real_video": bool(self.config.use_real_video),
        }
        layout.write_stage_a_env(env)
        _log.info(
            "Stage A env → %s (token_dim=%d, %dx%dx%d, %d teacher steps)",
            layout.stage_a_env, env["token_dim"], env["num_frames"],
            env["height"], env["width"], env["teacher_steps"],
        )
        if not env["use_real_video"]:
            return
        # --use-real-video never samples z_init, which Stage C's cached-baseline path needs.
        _log.warning(
            "Stage A is running with --use-real-video, so z_init is NOT persisted. "
            "Stage C will be unable to reuse Y_full and will recompute the full-compute "
            "baseline for every batch. Drop --use-real-video if this store feeds Stage C."
        )

    def _ingest_and_filter(self, layout: ProcessedLayout, *, write_global: bool):
        """Read OpenVid metadata and run the four-level quality filter.

        ``write_global`` gates the one-time global-metadata writes to a single shard so
        parallel workers never race on the large CSVs; the other shards inherit the kept
        set and split map from ``filtered_final.csv``.
        """
        if not write_global:
            # Only a concurrent generate shard has a publisher to wait for; a standalone
            # finalize pass reuses the CSV if present and otherwise filters immediately.
            reused = self._reuse_filter(
                layout,
                wait=(not self.config.finalize_only
                      and self.config.num_shards > 1
                      and self.config.shard_index > 0),
            )
            if reused is not None:
                return reused

        cfg = self.config.config
        video_subdir = self.config.video_subdir or cfg.data.video_subdir
        # Real-video mode needs the mp4 on disk, so it implies the existence filter.
        require_file = self.config.require_file or self.config.use_real_video
        records = read_openvid_manifest(
            [str(p) for p in self.config.openvid_csvs],
            self.config.data_root,
            video_subdir=video_subdir,
            static_motion_max=cfg.filter.static_motion_max,
            limit_per_csv=self.config.limit,
            require_file=require_file,
        )
        _log.info("§1.1 ingested %d OpenVid records", len(records))

        qfilter = QualityFilter(cfg.filter, perception=self.accelerator.perception)
        result = qfilter.apply(records, seed=self.config.seed)
        if write_global:
            write_raw_dataset_index(records, layout)
            qfilter.write(layout, result)
            layout.write_captions(self._caption_rows(result.kept))
            self._write_report(layout, result.report.as_dict())
        _log.info(
            "§2 filter kept %d/%d clips (hd=%.0f%%, complex=%.0f%%); split %d/%d/%d",
            result.report.kept_final, result.report.total_in,
            100 * result.report.hd_frac, 100 * result.report.complex_frac,
            result.report.n_train, result.report.n_val, result.report.n_test_hard,
        )
        return result

    def _reuse_filter(self, layout: ProcessedLayout, *, wait: bool):
        """The owning shard's ``filtered_final.csv``, or ``None`` to filter locally.

        ``wait`` polls for the atomically-written file; giving up falls through to
        running the filter here, so a lost race never changes the kept set.
        """
        deadline = time.monotonic() + (_FILTER_HANDOFF_TIMEOUT_S if wait else 0.0)
        announced = False
        while True:
            result = read_filtered_final(layout)
            if result is not None:
                _log.info(
                    "§2 reusing %s: %d clips; split %d/%d/%d",
                    layout.filtered_final.name, result.report.kept_final,
                    result.report.n_train, result.report.n_val,
                    result.report.n_test_hard,
                )
                return result
            if time.monotonic() >= deadline:
                if wait:
                    _log.warning(
                        "§2 %s did not appear within %ds; running the filter in this "
                        "shard instead. Shard 0 may have failed, or it is still "
                        "ingesting.", layout.filtered_final, _FILTER_HANDOFF_TIMEOUT_S,
                    )
                return None
            if not announced:
                _log.info("§2 waiting for shard 0 to publish %s", layout.filtered_final.name)
                announced = True
            time.sleep(_FILTER_HANDOFF_POLL_S)

    # ------------------------------------------------------------------ #
    # generation (per shard, resumable, O(1) memory)
    # ------------------------------------------------------------------ #

    def _generate(self, layout: ProcessedLayout, result, num_shards: int, shard_index: int) -> None:
        """Teacher forward + counterfactual generation for this shard's clips.

        Holds no per-sample state: samples stream to the writer, per-tube meta to a
        sidecar, and one ``_progress`` line per clip enables resume. A clip that raises is
        logged and skipped (``--fail-fast`` aborts instead), but too many consecutive
        failures still abort the shard.
        """
        cfg = self.config.config
        max_samples = self.config.samples_per_video or cfg.teacher.samples_per_video
        prog_path = layout.lmdb_dir / f"_progress.s{shard_index:02d}.jsonl"
        tube_path = layout.lmdb_dir / f"_tube_meta.s{shard_index:02d}.jsonl"
        done = self._read_progress(prog_path)
        if done:
            _log.info("resuming shard %d: %d clips already processed, skipping them",
                      shard_index, len(done))

        # Sharded runs force the .pt backend (LMDB is single-writer) and namespace shards
        # by id; finalize builds the merged manifest.
        sharded = num_shards > 1
        writer = CounterfactualSampleWriter(
            layout.lmdb_dir,
            shard_size=cfg.teacher.shard_size,
            map_size=int(cfg.teacher.lmdb_map_size_gib) * 1024 ** 3,
            shard_prefix=(f"shard_s{shard_index:02d}" if sharded else "shard"),
            manifest_name=f"manifest.s{shard_index:02d}.json",
            resume=True,
            write_manifest=False,   # finalize owns the merged manifest.json
            force_fallback=sharded,
        )

        n_new = n_failed = n_consecutive = 0
        fail_path = layout.lmdb_dir / f"_failed.s{shard_index:02d}.jsonl"
        with writer, open(prog_path, "a", encoding="utf-8") as pf, \
                open(tube_path, "a", encoding="utf-8") as tf, \
                open(fail_path, "a", encoding="utf-8") as ff:
            for rec in self._scene_interleaved(result.kept):
                if not _belongs_to_shard(rec.video_id, num_shards, shard_index):
                    continue
                if rec.video_id in done:
                    continue
                split = result.split_by_video.get(rec.video_id, "train")
                # A failed clip is logged to ``_failed.sNN.jsonl`` and skipped, with no
                # ``_progress`` line so it is retried next run; ``--fail-fast`` aborts instead.
                try:
                    n = self._process_clip(
                        rec, split, writer=writer, pf=pf, tf=tf,
                        shard_index=shard_index, max_samples=max_samples,
                        clip_no=n_new + 1,
                    )
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    if self.config.fail_fast:
                        raise
                    n_failed += 1
                    n_consecutive += 1
                    _log.exception("  [shard %d] %s FAILED; skipping (%d failed so far)",
                                   shard_index, rec.video_id, n_failed)
                    ff.write(json.dumps({
                        "video_id": rec.video_id,
                        "error": traceback.format_exc(limit=12),
                    }, ensure_ascii=False) + "\n")
                    ff.flush()
                    if n_consecutive >= _MAX_CONSECUTIVE_CLIP_FAILURES:
                        raise RuntimeError(
                            f"{n_consecutive} clips failed in a row in shard {shard_index}; "
                            "that is an environment failure (disk, VRAM, weights), not a "
                            "clip-specific one, so aborting rather than burning the rest of "
                            f"the shard. Tracebacks: {fail_path}"
                        ) from exc
                else:
                    n_consecutive = 0
                    if n is not None:
                        n_new += 1
                finally:
                    free_memory()
        _log.info("shard %d: generated samples for %d new clips → %s",
                  shard_index, n_new, layout.lmdb_dir)
        if n_failed:
            _log.warning(
                "shard %d: %d clip(s) failed and were skipped; ids + tracebacks in %s. "
                "They carry no _progress line, so re-running this shard retries them.",
                shard_index, n_failed, fail_path,
            )

    def _process_clip(
        self, rec: OpenVidRecord, split: str, *, writer, pf, tf,
        shard_index: int, max_samples: Optional[int], clip_no: int,
    ) -> Optional[int]:
        """Teacher forward + counterfactuals for one clip; returns the samples written.

        Returns ``None`` for a degenerate clip (no tube survived), recorded as done.
        """
        cfg = self.config.config
        t_clip = time.perf_counter()
        video_frames = self._decode_clip(rec) if self.config.use_real_video else None
        traj = self.teacher_runner.run(
            rec.video_id, rec.caption, rec.scene_type, video_frames=video_frames
        )
        if traj is None:
            # Degenerate clip (no tube). Record it done so a restart won't retry.
            self._log_progress(pf, rec.video_id, 0, split)
            return None
        # One progress line per phase (with the VRAM peak) so a slow run is
        # distinguishable from a hung one.
        _log.info(
            "  [shard %d] %s §1.3-1.4 done in %.1fs: %d tubes%s",
            shard_index, rec.video_id, time.perf_counter() - t_clip,
            len(traj.tubes), self._vram_note(),
        )
        if self._do_baseline:
            self._persist_baseline(traj)
        if self._do_features:
            self._persist_features(traj)
        for row in self._tube_meta(traj):
            tf.write(json.dumps(row, ensure_ascii=False) + "\n")
        tf.flush()
        self._persist_text_embed(traj)

        samples = self.data_generator.generate(
            traj, self.backbone, self.transition,
            max_tubes=cfg.teacher.max_tubes_per_prompt,
            max_samples=max_samples,
        )
        n = 0
        for s in samples:
            writer.put(self._sample_id(s), s)
            n += 1
        writer.flush()
        self._log_progress(pf, rec.video_id, n, split)
        _log.info(
            "  [shard %d | +%d] %s → %d samples in %.1fs%s",
            shard_index, clip_no, rec.video_id, n,
            time.perf_counter() - t_clip, self._vram_note(),
        )
        return n

    @staticmethod
    def _vram_note() -> str:
        """`` (vram 31.8/40.0 GiB peak)`` on CUDA, empty elsewhere; resets the peak counter."""
        if not torch.cuda.is_available():
            return ""
        peak = torch.cuda.max_memory_reserved() / 1024 ** 3
        # Report capacity from the current device, matching max_memory_reserved().
        total = (torch.cuda.get_device_properties(torch.cuda.current_device())
                 .total_memory / 1024 ** 3)
        torch.cuda.reset_peak_memory_stats()
        return f" (vram {peak:.1f}/{total:.1f} GiB peak)"

    @staticmethod
    def _log_progress(fh, video_id: str, n_samples: int, split: str) -> None:
        """Append one durable (flushed) progress line so a restart can skip this clip."""
        fh.write(json.dumps({"video_id": video_id, "n": int(n_samples), "split": split}) + "\n")
        fh.flush()
        os.fsync(fh.fileno())

    @staticmethod
    def _read_progress(path: Path) -> set:
        """Set of video_ids already processed in a prior (interrupted) run of this shard."""
        done: set = set()
        if not path.exists():
            return done
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["video_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
        return done

    def finalize(self, layout: Optional[ProcessedLayout] = None,
                 split_by_video: Optional[Dict[str, str]] = None) -> Path:
        """Build the shared index / splits / norm from whatever shards exist.

        Thin wrapper over :func:`finalize_processed_store`; recomputes the split map from
        the filter when not supplied.
        """
        layout = layout or self.layout
        if split_by_video is None:
            split_by_video = self._ingest_and_filter(layout, write_global=False).split_by_video
        return finalize_processed_store(layout, split_by_video)

    # ------------------------------------------------------------------ #
    # finalize lives at module scope (``finalize_processed_store``)
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # persistence of the baseline / tube-feature buckets
    # ------------------------------------------------------------------ #

    def _persist_baseline(self, traj: TeacherTrajectory) -> None:
        """Write the ``full_baseline/<video_id>/`` bucket.

        ``z_t`` (representative-step latents) is written only when ``persist_step_latents``
        is on.
        """
        z_by_t = (
            {traj.num_total_steps - step_idx: z for step_idx, z in traj.z_by_step.items()}
            if self.config.persist_step_latents else {}
        )
        self.layout.save_baseline(
            traj.video_id, z_t_by_step=z_by_t, y_full=traj.video_full,
            z_init=traj.z_init,
        )

    def _persist_text_embed(self, traj: TeacherTrajectory) -> None:
        """Write the clip's prompt embedding once, trimmed to the caption length in fp16.

        Stage B joins it back by ``video_id``.
        """
        emb = traj.text_embed
        if emb is None:
            return
        mask = getattr(traj.cond, "mask", None)
        if mask is not None and mask.numel():
            used = mask[0].bool() if mask.dim() > 1 else mask.bool()
            keep = int(used.nonzero().max()) + 1 if bool(used.any()) else emb.shape[0]
            emb = emb[:keep]
        path = self.layout.text_embed_path(traj.video_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(emb.detach().to("cpu", torch.float16), path)

    def _persist_features(self, traj: TeacherTrajectory) -> None:
        """Write the ``tube_causal_features/<video_id>/`` bucket."""
        tubes = traj.tubes
        if not tubes:
            return
        states = torch.stack([traj.tube_states[t.tube_id].as_tensor() for t in tubes])      # [K,7]
        strength = torch.stack([traj.strength_feats[t.tube_id].as_tensor() for t in tubes])  # [K,3]
        visual = torch.stack([traj.tube_visual_embed_full[t.tube_id] for t in tubes])        # [K,d_v]
        # Compact per-tube (id, token-count) record in lieu of full masks.
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
    # real-clip decode
    # ------------------------------------------------------------------ #

    def _decode_clip(self, rec: OpenVidRecord) -> Tensor:
        """Decode ``rec``'s mp4 to ``[F, 3, H, W]`` in ``[-1, 1]`` for the teacher.

        Uses the *same* sampler and normalisation as :class:`VideoTextDataset` so the
        pixels the teacher encodes are byte-identical to what the Stage-C training
        loader would read (no train/serve skew). Frame count and resolution are pinned
        to ``config.data.num_frames`` / ``height`` / ``width`` — the exact geometry the
        teacher's ``TokenGrid`` is built from — so ``encode_video`` → ``to_grid`` aligns
        (no aspect-ratio bucketing here, which could pick a mismatched resolution).
        """
        dcfg = self.config.config.data
        reader = self._video_reader
        assert reader is not None, "real-video decode requested but no reader built"
        # Stable per-clip seed: sha1, not hash() — a str hash is salted per process
        # (PYTHONHASHSEED), so retries and other shards would sample different
        # frames for the same clip (see lcocf.data._seeded_noise).
        clip_seed = int(hashlib.sha1(rec.video_id.encode("utf-8")).hexdigest()[:8], 16)
        g = torch.Generator().manual_seed(dcfg.seed + clip_seed % (2 ** 20))
        available = reader.num_frames(rec.path)
        idx = sample_frame_indices(available, dcfg.num_frames, dcfg.frame_interval, generator=g)
        frames = reader.read(rec.path, idx)                 # [F, 3, h, w] in [0, 1]
        frames = _resize_clip(frames, dcfg.height, dcfg.width)
        if dcfg.normalize_to_unit:
            frames = frames * 2.0 - 1.0                     # [0, 1] → [-1, 1] (VAE input)
        return frames

    # ------------------------------------------------------------------ #
    # misc helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _sample_id(s: COCFTrainingSample) -> str:
        """Stable unique key: (video, tube, timestep, action) is unique per clip."""
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
        covers all scene classes. Linear in the record count.
        """
        by_scene: Dict[str, Deque[OpenVidRecord]] = {}
        for r in records:
            by_scene.setdefault(r.scene_type, deque()).append(r)
        queues = list(by_scene.values())
        out: List[OpenVidRecord] = []
        while queues:
            live: List[Deque[OpenVidRecord]] = []
            for q in queues:
                out.append(q.popleft())
                if q:
                    live.append(q)
            queues = live
        return out


# --------------------------------------------------------------------------- #
# Module-level helpers: shard routing + finalize (backbone-free, streaming)
# --------------------------------------------------------------------------- #


def _belongs_to_shard(video_id: str, num_shards: int, shard_index: int) -> bool:
    """Stable, process-independent clip→shard routing.

    Uses md5 (not Python's salted ``hash``) so every worker agrees on the partition,
    giving a disjoint, reproducible split with no coordination.
    """
    if num_shards <= 1:
        return True
    h = int(hashlib.md5(str(video_id).encode("utf-8")).hexdigest(), 16)
    return h % num_shards == shard_index


def _damage_scalar_from_payload(payload: Dict[str, object]) -> float:
    """Recompute the weighted-sum damage scalar from a stored payload's ``damage_label``."""
    dl = payload.get("damage_label")
    if dl is None:
        return 0.0
    arr = np.asarray(dl, dtype="float64").reshape(-1)
    scalar = 0.0
    for i, axis in enumerate(DAMAGE_DIMENSIONS):
        if i < arr.shape[0]:
            scalar += float(arr[i]) * DEFAULT_DAMAGE_WEIGHTS.get(axis, 0.0)
    return min(1.0, scalar)


def _outlier_mask(damage: Sequence[float], sigma: float = 3.0) -> List[bool]:
    """3σ damage-outlier mask. All-kept when too few samples to estimate σ."""
    if len(damage) < 8:
        return [True] * len(damage)
    arr = np.asarray(damage, dtype="float64")
    mu, sd = float(arr.mean()), float(arr.std())
    if sd <= 1e-9:
        return [True] * len(damage)
    lo, hi = mu - sigma * sd, mu + sigma * sd
    return [(lo <= x <= hi) for x in damage]


def _iter_shard_records(shard_paths: Sequence[Path]):
    """Yield ``(shard_filename, pos, sample_id, payload)`` streaming over the shards.

    One shard is resident at a time, so finalize's peak RAM is a single shard.
    """
    for sp in shard_paths:
        recs = torch.load(sp, map_location="cpu", weights_only=False)
        for pos, r in enumerate(recs):
            yield sp.name, pos, r["sample_id"], r["payload"]
        del recs
        free_memory()


def _iter_store_records(layout: ProcessedLayout, shard_paths: Sequence[Path]):
    """Stream ``(ref, pos, sample_id, payload)`` over whichever backend is on disk.

    Reads the LMDB store when present, else the ``.pt`` shards. ``ref``/``pos`` are the
    ``.pt`` manifest coordinates and are ``("", -1)`` for LMDB.
    """
    if store_is_lmdb(layout.lmdb_dir):
        for sid, payload in iter_lmdb_records(layout.lmdb_dir):
            yield "", -1, sid, payload
        return
    yield from _iter_shard_records(shard_paths)


def _merge_tube_meta(layout: ProcessedLayout) -> List[Dict[str, object]]:
    """Merge every shard's ``_tube_meta.sNN.jsonl`` sidecar into the tube_meta rows."""
    rows: List[Dict[str, object]] = []
    for p in sorted(layout.lmdb_dir.glob("_tube_meta.s*.jsonl")):
        with open(p, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def finalize_processed_store(layout: ProcessedLayout, split_by_video: Dict[str, str]) -> Path:
    """Build the shared index / splits / norm by streaming over all shards.

    Backbone-free single pass: it retains only the scalar index fields per record, emits
    the merged ``manifest.json`` plus ``sample_index.csv`` and leakage-safe ``splits/*.txt``
    (3σ outliers are omitted from the index, never deleted from the store). Works for the
    ``.pt`` shards and the LMDB backend. Idempotent: safe to re-run.
    """
    is_lmdb = store_is_lmdb(layout.lmdb_dir)
    shard_paths = sorted(layout.lmdb_dir.glob("shard_*.pt"))
    if not is_lmdb and not shard_paths:
        _log.warning(
            "finalize: no LMDB store and no .pt shards under %s — nothing to index",
            layout.lmdb_dir,
        )
        return layout.root
    if is_lmdb and shard_paths:
        # The store changed backend mid-flight; index the LMDB only and warn.
        _log.warning(
            "finalize: %s holds BOTH an LMDB store and %d .pt shard(s). Indexing the "
            "LMDB only — the shards were written by a run with a different backend and "
            "will not appear in sample_index/splits. Re-run that shard's generation, "
            "or move the .pt files aside.",
            layout.lmdb_dir, len(shard_paths),
        )
    backend = "LMDB" if is_lmdb else f"{len(shard_paths)} .pt shard(s)"

    # --- single pass: online min-max norm + damage + index fields ---- #
    norm_min: Dict[str, np.ndarray] = {}
    norm_max: Dict[str, np.ndarray] = {}
    damage: List[float] = []
    keys: List[str] = []
    index: Dict[str, list] = {}
    sample_rows: List[Dict[str, object]] = []
    seen: set = set()
    duplicates = 0
    for name, pos, sid, payload in _iter_store_records(layout, shard_paths):
        # A sample id addresses one record; index a repeat once so no consumer double-counts.
        if sid in seen:
            duplicates += 1
            if pos >= 0:
                index[sid] = [name, pos]   # last write wins, matching the writer
            continue
        seen.add(sid)
        damage.append(_damage_scalar_from_payload(payload))
        keys.append(sid)
        if pos >= 0:
            index[sid] = [name, pos]
        sample_rows.append({
            "sample_id": sid,
            "video_id": str(payload.get("video_id", "")),
            "timestep": int(payload.get("timestep", 0)),
            "action": int(payload.get("action", 0)),
            "scene_type": payload.get("scene_type", ""),
        })
        for g in _NORM_GROUPS:
            v = payload.get(g)
            if v is None:
                continue
            a = np.asarray(v, dtype="float64").reshape(-1)
            if not a.size:
                continue
            if g in norm_min:
                np.minimum(norm_min[g], a, out=norm_min[g])
                np.maximum(norm_max[g], a, out=norm_max[g])
            else:
                norm_min[g], norm_max[g] = a.copy(), a.copy()
    if duplicates:
        _log.warning("finalize: %d duplicate sample id(s) collapsed to one entry each",
                     duplicates)

    # --- apply the 3σ mask to the retained index rows ----------------------- #
    keep_mask = _outlier_mask(damage)
    kept_rows = [row for row, keep in zip(sample_rows, keep_mask) if keep]
    buckets: Dict[str, List[str]] = {"train": [], "val": [], "test_hard": []}
    for row in kept_rows:
        vid = str(row["video_id"])
        buckets.get(split_by_video.get(vid, "train"), buckets["train"]).append(
            str(row["sample_id"])
        )

    if not is_lmdb:
        # manifest.json is the .pt backend's addressing index; LMDB addresses by key.
        (layout.lmdb_dir / "manifest.json").write_text(
            json.dumps({"keys": keys, "index": index}), encoding="utf-8"
        )
    layout.write_sample_index(kept_rows)
    layout.write_splits(buckets["train"], buckets["val"], buckets["test_hard"])
    layout.write_norm_stats({
        g: {"min": norm_min[g].tolist(), "max": norm_max[g].tolist()} for g in norm_min
    })
    layout.write_tube_meta(_merge_tube_meta(layout))

    dropped = len(keys) - len(kept_rows)
    _log.info(
        "§1.6 finalize: indexed %d samples from %s; sample_index kept %d "
        "(dropped %d 3σ outliers); splits %d/%d/%d → %s",
        len(keys), backend, len(kept_rows), dropped,
        len(buckets["train"]), len(buckets["val"]), len(buckets["test_hard"]), layout.root,
    )
    return layout.root
