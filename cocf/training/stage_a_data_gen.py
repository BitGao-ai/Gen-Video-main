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

import hashlib
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
    iter_lmdb_records,
    read_openvid_manifest,
    store_is_lmdb,
    video_id_str,
    write_raw_dataset_index,
)
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
    persist_buckets: bool = True             # master switch for §3 level-3/level-4 buckets
    persist_baseline: bool = True            # write §3 level-3 full_baseline (the ~1TB bucket)
    persist_tube_features: bool = True       # write §3 level-4 tube_causal_features (small)
    seed: int = 1234
    video_subdir: Optional[str] = None       # override config.data.video_subdir (e.g. "videos")
    require_file: bool = False               # keep only clips whose mp4 exists on disk
    use_real_video: bool = False             # anchor the teacher trajectory on real mp4 pixels
    # -- shard-parallel + resume (§1 embarrassingly-parallel over clips) -------- #
    num_shards: int = 1                      # total parallel workers over the clip set
    shard_index: int = 0                     # this worker's 0-based shard id
    finalize_only: bool = False              # skip generation; only build manifest/splits/index


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
        # Keep the accelerator's plugins on the run device, matching the backbone
        # (placed via config.backbone.device) and Stages B/C. The teacher path leaves
        # the learnable plugins dormant today, but this makes the whole stage single-
        # device so any future plugin call (or a real perception provider feeding the
        # GPU tube path) composes without a CPU×CUDA mismatch.
        self.accelerator.to(config.device)
        self.layout = ProcessedLayout(config.processed_root)
        # Bucket persistence switches (§3 level-3/level-4). ``persist_buckets`` is the
        # master off-switch; the two fine-grained flags let a full-scale run drop the
        # ~1TB full_baseline bucket (unread by Stages B/C) while keeping the tiny tube
        # features. Kept as attributes so the generation loop stays branch-cheap.
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
        )
        self.transition = accelerator.transition
        # Real-clip decoder (§1.3 real-video anchor). Built once and reused; only
        # touched when ``use_real_video`` is set, so the caption-only path stays
        # dependency-free (no decord/torchvision import).
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

        The pipeline is split into a *generate* phase (§1.3–§1.5 — the VRAM-bound,
        days-long teacher forward) and a *finalize* phase (§1.6 — a pure-CPU index /
        splits / norm build). Decoupling them is what makes the stage:

        * **shard-parallel** — ``--num-shards N --shard-index i`` runs N processes, each
          over a stable-hash slice of the clips, all appending into one store;
        * **resumable** — each shard appends a ``_progress`` line per finished clip and
          skips them on restart, so an interrupted run continues instead of redoing the
          teacher forward (an OOM / pre-empt no longer forfeits the shard);
        * **bounded-memory** — the generate loop keeps *no* per-sample Python state; the
          §1.6 statistics are streamed back off the shards in finalize (online min-max),
          so RAM stays O(1) in sample count throughout the expensive phase.

        A single-process run (``num_shards==1``) finalizes inline, preserving the old
        one-call behaviour. A multi-shard run generates only; invoke once more with
        ``--finalize-only`` after all shards finish to build the shared index.
        """
        layout = self.layout.create()
        ns, si = max(1, self.config.num_shards), self.config.shard_index
        _log.info("=== Stage A: Counterfactual Teacher Data Generation (§1) — shard %d/%d ===", si, ns)

        # §1.1/§1.2 — every shard needs the kept set + split map, but only shard 0 writes
        # the shared global metadata (concurrent writers would corrupt the large CSVs).
        result = self._ingest_and_filter(
            layout, write_global=(si == 0 and not self.config.finalize_only)
        )

        if self.config.finalize_only:
            return finalize_processed_store(layout, result.split_by_video)

        # §1.3–§1.5 — generate this shard's counterfactual samples (resumable).
        self._generate(layout, result, ns, si)

        if ns > 1:
            _log.info(
                "shard %d/%d generation done. Once ALL shards finish, run one more pass "
                "with --finalize-only to build manifest/splits/sample_index.", si, ns,
            )
            return layout.root
        # single-process run: finalize inline (the historical one-call behaviour).
        return finalize_processed_store(layout, result.split_by_video)

    # ------------------------------------------------------------------ #
    # §1.1/§1.2 ingest + filter
    # ------------------------------------------------------------------ #

    def _ingest_and_filter(self, layout: ProcessedLayout, *, write_global: bool):
        """Read OpenVid metadata and run the four-level quality filter (§1.1/§1.2).

        ``write_global`` gates the one-time global-metadata writes to a single shard so
        parallel workers never race on the (large) raw_dataset_index / filtered_final CSVs.
        """
        cfg = self.config.config
        video_subdir = self.config.video_subdir or cfg.data.video_subdir
        # Real-video mode is meaningless without the mp4 on disk, so it implies the
        # existence filter (a clip with no file is silently dropped upstream).
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

    # ------------------------------------------------------------------ #
    # §1.3–§1.5 generation (per shard, resumable, O(1) memory)
    # ------------------------------------------------------------------ #

    def _generate(self, layout: ProcessedLayout, result, num_shards: int, shard_index: int) -> None:
        """Teacher forward + counterfactual generation for this shard's clips.

        Holds no per-sample state: samples stream straight to the writer, per-tube meta
        streams to a sidecar, and one ``_progress`` line per clip records what is done so
        a restart resumes. All §1.6 statistics are recomputed from the shards in finalize.
        """
        cfg = self.config.config
        max_samples = self.config.samples_per_video or cfg.teacher.samples_per_video
        prog_path = layout.lmdb_dir / f"_progress.s{shard_index:02d}.jsonl"
        tube_path = layout.lmdb_dir / f"_tube_meta.s{shard_index:02d}.jsonl"
        done = self._read_progress(prog_path)
        if done:
            _log.info("resuming shard %d: %d clips already processed, skipping them",
                      shard_index, len(done))

        # Sharded runs force the .pt backend (LMDB is single-writer) and namespace their
        # shards/manifest by shard id; the merged manifest is built later by finalize.
        sharded = num_shards > 1
        writer = CounterfactualSampleWriter(
            layout.lmdb_dir,
            shard_size=cfg.teacher.shard_size,
            shard_prefix=(f"shard_s{shard_index:02d}" if sharded else "shard"),
            manifest_name=f"manifest.s{shard_index:02d}.json",
            resume=True,
            write_manifest=False,   # finalize owns the merged manifest.json
            force_fallback=sharded,
        )

        n_new = 0
        with writer, open(prog_path, "a", encoding="utf-8") as pf, \
                open(tube_path, "a", encoding="utf-8") as tf:
            for rec in self._scene_interleaved(result.kept):
                if not _belongs_to_shard(rec.video_id, num_shards, shard_index):
                    continue
                if rec.video_id in done:
                    continue
                split = result.split_by_video.get(rec.video_id, "train")
                video_frames = self._decode_clip(rec) if self.config.use_real_video else None
                traj = self.teacher_runner.run(
                    rec.video_id, rec.caption, rec.scene_type, video_frames=video_frames
                )
                if traj is None:
                    # Degenerate clip (no tube). Record it done so a restart won't retry.
                    self._log_progress(pf, rec.video_id, 0, split)
                    continue
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
                self._log_progress(pf, rec.video_id, n, split)
                n_new += 1
                _log.info("  [shard %d | +%d] %s → %d samples", shard_index, n_new, rec.video_id, n)
                free_memory()
        _log.info("shard %d: generated samples for %d new clips → %s",
                  shard_index, n_new, layout.lmdb_dir)

    @staticmethod
    def _log_progress(fh, video_id: str, n_samples: int, split: str) -> None:
        """Append one durable (flushed) progress line so a restart can skip this clip."""
        fh.write(json.dumps({"video_id": video_id, "n": int(n_samples), "split": split}) + "\n")
        fh.flush()

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
        """Build the §1.6 shared index / splits / norm from whatever shards exist.

        Thin instance wrapper over :func:`finalize_processed_store`; recomputes the split
        map from the filter when not supplied (e.g. a standalone ``--finalize-only`` pass).
        """
        layout = layout or self.layout
        if split_by_video is None:
            split_by_video = self._ingest_and_filter(layout, write_global=False).split_by_video
        return finalize_processed_store(layout, split_by_video)

    # ------------------------------------------------------------------ #
    # §1.6 finalize lives at module scope (``finalize_processed_store``) so a
    # standalone --finalize-only / rebuild pass can call it without a backbone
    # or accelerator; it streams the stats back off the shards (online min-max).
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # persistence of §3 level-3 / level-4 buckets
    # ------------------------------------------------------------------ #

    def _persist_baseline(self, traj: TeacherTrajectory) -> None:
        """Write the §3 level-3 ``full_baseline/<video_id>/`` bucket (z_t keyed by t)."""
        z_by_t = {traj.num_total_steps - step_idx: z for step_idx, z in traj.z_by_step.items()}
        text_emb = traj.text_embed if traj.text_embed is not None else torch.zeros(1)
        self.layout.save_baseline(
            traj.video_id, text_emb=text_emb, z_t_by_step=z_by_t, y_full=traj.video_full,
            z_init=traj.z_init,
        )

    def _persist_text_embed(self, traj: TeacherTrajectory) -> None:
        """Write the clip's prompt embedding once (§P2-3).

        Trimmed to the caption's real length and stored in fp16: the tokenizer pads
        to 512 and a caption uses a few dozen positions, so the pair of measures turns
        a ~8 MiB per-*sample* duplicate into a ~0.3 MiB per-*clip* file. Stage B joins
        it back by ``video_id``.
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
    # real-clip decode (§1.3 real-video anchor)
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
        g = torch.Generator().manual_seed(dcfg.seed + (abs(hash(rec.video_id)) % (2 ** 20)))
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


# --------------------------------------------------------------------------- #
# Module-level helpers: shard routing + §1.6 finalize (backbone-free, streaming)
# --------------------------------------------------------------------------- #


def _belongs_to_shard(video_id: str, num_shards: int, shard_index: int) -> bool:
    """Stable, process-independent clip→shard routing (§1 embarrassingly parallel).

    Uses md5 (not Python's salted ``hash``) so every worker — a separate process with
    its own PYTHONHASHSEED — agrees on which shard owns a clip, giving a disjoint,
    reproducible partition of the kept set with zero coordination between workers.
    """
    if num_shards <= 1:
        return True
    h = int(hashlib.md5(str(video_id).encode("utf-8")).hexdigest(), 16)
    return h % num_shards == shard_index


def _damage_scalar_from_payload(payload: Dict[str, object]) -> float:
    """Recompute :meth:`COCFTrainingSample.damage_scalar` from a stored payload dict.

    Finalize reads raw payloads off the shards, so it reproduces the weighted-sum damage
    scalar straight from ``damage_label`` (identical maths to the typed method) to drive
    the §1.6 3σ outlier mask — no need to rebuild a typed sample per record.
    """
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
    """3σ damage-outlier mask (§1.6). All-kept when too few samples to estimate σ."""
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

    One shard is resident at a time (loaded, consumed, dropped), so finalize's peak RAM
    is a single shard rather than the whole store — the property that lets §1.6 scale to
    millions of samples on a memory-constrained box.
    """
    for sp in shard_paths:
        recs = torch.load(sp, map_location="cpu", weights_only=False)
        for pos, r in enumerate(recs):
            yield sp.name, pos, r["sample_id"], r["payload"]
        del recs
        free_memory()


def _iter_store_records(layout: ProcessedLayout, shard_paths: Sequence[Path]):
    """Stream ``(ref, pos, sample_id, payload)`` over whichever backend is on disk.

    The sample writer picks LMDB automatically whenever the package is importable
    (the documented production path), so finalize must read it too. Globbing only
    ``shard_*.pt`` meant that a machine following the README's ``pip install lmdb``
    wrote its samples to ``data.mdb`` and then had finalize declare the store empty —
    no sample_index, no splits, no norm stats, and a hard Stage-B preflight failure.
    ``ref``/``pos`` are the manifest coordinates for the ``.pt`` backend and are unused
    (``("", -1)``) for LMDB, which addresses records by key.
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
    """Build the §1.6 shared index / splits / norm by streaming over all shards.

    Backbone-free and, apart from the index/keys it must materialise for ``manifest.json``
    (an intrinsic output), O(1) in sample count. Two streaming passes:

    1. accumulate the per-field **online min-max** (§1.6 norm stats) and the damage
       scalars needed for the 3σ mask — one shard resident at a time;
    2. emit the merged ``manifest.json`` (every sample, so Stage B can address any
       record) plus ``sample_index.csv`` + leakage-safe ``splits/*.txt`` (kept,
       non-outlier samples only — 3σ outliers are omitted from the index, never deleted
       from the store, matching the original §1.6 contract).

    Works for both the sharded layout (``shard_sNN_*.pt``) and a legacy single-writer
    store (``shard_NNNNN.pt``) — the glob matches both — **and** for the LMDB backend
    the writer selects whenever the ``lmdb`` package is installed (§3 "LMDB 训练主库"),
    which is read by key rather than by shard. Idempotent: safe to re-run.
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
        # A store that changed backend mid-flight (e.g. ``pip install lmdb`` between
        # two Stage-A runs). Indexing only one of them would quietly drop the other's
        # samples, so say so rather than let the count look right. Warned once here,
        # not inside the (twice-consumed) record iterator.
        _log.warning(
            "finalize: %s holds BOTH an LMDB store and %d .pt shard(s). Indexing the "
            "LMDB only — the shards were written by a run with a different backend and "
            "will not appear in sample_index/splits. Re-run that shard's generation, "
            "or move the .pt files aside.",
            layout.lmdb_dir, len(shard_paths),
        )
    backend = "LMDB" if is_lmdb else f"{len(shard_paths)} .pt shard(s)"

    # --- pass 1: online min-max (§1.6 norm) + damage scalars (streaming) ---- #
    norm_min: Dict[str, np.ndarray] = {}
    norm_max: Dict[str, np.ndarray] = {}
    damage: List[float] = []
    for _name, _pos, _sid, payload in _iter_store_records(layout, shard_paths):
        damage.append(_damage_scalar_from_payload(payload))
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
    keep_mask = _outlier_mask(damage)

    # --- pass 2: manifest (all records) + sample_index/splits (kept) -------- #
    keys: List[str] = []
    index: Dict[str, list] = {}
    sample_rows: List[Dict[str, object]] = []
    buckets: Dict[str, List[str]] = {"train": [], "val": [], "test_hard": []}
    gi = 0
    for name, pos, sid, payload in _iter_store_records(layout, shard_paths):
        keys.append(sid)
        if pos >= 0:
            index[sid] = [name, pos]
        if keep_mask[gi]:
            vid = str(payload.get("video_id", ""))
            sample_rows.append({
                "sample_id": sid,
                "video_id": vid,
                "timestep": int(payload.get("timestep", 0)),
                "action": int(payload.get("action", 0)),
                "scene_type": payload.get("scene_type", ""),
            })
            buckets.get(split_by_video.get(vid, "train"), buckets["train"]).append(sid)
        gi += 1

    if not is_lmdb:
        # manifest.json is the .pt backend's *only* addressing index. The LMDB store
        # addresses by key and carries its own ``__keys__``, so writing a manifest
        # there would shadow it with a stale, position-based view.
        (layout.lmdb_dir / "manifest.json").write_text(
            json.dumps({"keys": keys, "index": index}), encoding="utf-8"
        )
    layout.write_sample_index(sample_rows)
    layout.write_splits(buckets["train"], buckets["val"], buckets["test_hard"])
    layout.write_norm_stats({
        g: {"min": norm_min[g].tolist(), "max": norm_max[g].tolist()} for g in norm_min
    })
    layout.write_tube_meta(_merge_tube_meta(layout))

    dropped = len(keys) - len(sample_rows)
    _log.info(
        "§1.6 finalize: indexed %d samples from %s; sample_index kept %d "
        "(dropped %d 3σ outliers); splits %d/%d/%d → %s",
        len(keys), backend, len(sample_rows), dropped,
        len(buckets["train"]), len(buckets["val"]), len(buckets["test_hard"]), layout.root,
    )
    return layout.root
