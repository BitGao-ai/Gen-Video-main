"""Six-level processed-data layout — ``LCOCF_OpenVid1M_Processed`` (§3).

A single helper that owns *every* path plus the small index/stat readers-writers,
so Stage A (the writer) and Stages B/C (the readers) never hard-code a path. The
tree matches the design document's §3 diagram verbatim::

    LCOCF_OpenVid1M_Processed/
    ├── metadata/                  # 一级 global indices (lightweight, no big files)
    │     raw_dataset_index.csv     video_id ─ path ─ basic info        (§1.1)
    │     filtered_final.csv        final kept samples + scene/quality   (§2.4)
    │     sample_index.csv          sample_id, video_id, timestep, action, scene
    │     tube_meta.csv             per-tube metadata
    │     norm_stats.json           min-max feature normalisation stats  (§1.6)
    ├── raw_filtered/              # 二级 kept raw clips + captions       (Stage C)
    │     vid_000001.mp4 …          captions.jsonl
    ├── full_baseline/vid_000001/  # 三级 teacher forward, bucketed by video_id
    │     text_emb.npy              z_t_sampled/{t_05,t_25,t_45}.npy
    │     Y_full.npy                kv_cache/
    ├── tube_causal_features/vid_000001/   # 四级 tube + causal features
    │     tube_features.npy  tube_states.npy  causal_strength.npy  tube_visual_emb.npy
    ├── counterfactual_lmdb/       # 五级 counterfactual training samples (Stage B)
    │     data.mdb  lock.mdb
    └── splits/                    # 六级 video_id-disjoint dataset splits
          train_list.txt  val_list.txt  test_hard_list.txt

Heavy per-video arrays are stored as ``.npy`` (numpy); the small indices are CSV /
JSON so dataset statistics and sampling configuration never require reading a big
file (§3 "元数据独立管理"). The actual counterfactual samples live in the LMDB
store (:mod:`cocf.data.sample_store`), not here.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import torch

from cocf.common.logging import get_logger

Tensor = torch.Tensor
_log = get_logger(__name__)

# Canonical column order for ``sample_index.csv`` (the §4.1 read entry point).
SAMPLE_INDEX_FIELDS: Sequence[str] = (
    "sample_id",
    "video_id",
    "timestep",
    "action",
    "scene_type",
)


def video_id_str(video_id) -> str:
    """Normalise a video id to the on-disk bucket name ``vid_000001``.

    Accepts an int (``1`` → ``vid_000001``) or an already-formatted string
    (``"vid_000001"`` / ``"000001"``) so callers can pass either freely.
    """
    if isinstance(video_id, str):
        s = video_id
        if s.startswith("vid_"):
            return s
        if s.isdigit():
            return f"vid_{int(s):06d}"
        return s  # already an arbitrary stable id (e.g. an OpenVid file stem)
    return f"vid_{int(video_id):06d}"


@dataclass
class ProcessedLayout:
    """Path + light-IO helper for the six-level processed store rooted at ``root``."""

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    # -- level directories ---------------------------------------------- #

    @property
    def metadata_dir(self) -> Path:
        return self.root / "metadata"

    @property
    def raw_filtered_dir(self) -> Path:
        return self.root / "raw_filtered"

    @property
    def full_baseline_dir(self) -> Path:
        return self.root / "full_baseline"

    @property
    def tube_features_dir(self) -> Path:
        return self.root / "tube_causal_features"

    @property
    def lmdb_dir(self) -> Path:
        return self.root / "counterfactual_lmdb"

    @property
    def splits_dir(self) -> Path:
        return self.root / "splits"

    # -- metadata files ------------------------------------------------- #

    @property
    def raw_dataset_index(self) -> Path:
        return self.metadata_dir / "raw_dataset_index.csv"

    @property
    def filtered_final(self) -> Path:
        return self.metadata_dir / "filtered_final.csv"

    @property
    def sample_index(self) -> Path:
        return self.metadata_dir / "sample_index.csv"

    @property
    def tube_meta(self) -> Path:
        return self.metadata_dir / "tube_meta.csv"

    @property
    def norm_stats(self) -> Path:
        return self.metadata_dir / "norm_stats.json"

    @property
    def stage_a_env(self) -> Path:
        """Backbone geometry / schedule the store was generated with (see
        :meth:`write_stage_a_env`)."""
        return self.metadata_dir / "stage_a_env.json"

    @property
    def train_list(self) -> Path:
        return self.splits_dir / "train_list.txt"

    @property
    def val_list(self) -> Path:
        return self.splits_dir / "val_list.txt"

    @property
    def test_hard_list(self) -> Path:
        return self.splits_dir / "test_hard_list.txt"

    # -- creation ------------------------------------------------------- #

    def create(self) -> "ProcessedLayout":
        """Create every level directory (idempotent). Returns self for chaining."""
        for d in (
            self.metadata_dir, self.raw_filtered_dir, self.full_baseline_dir,
            self.tube_features_dir, self.lmdb_dir, self.splits_dir,
            self.text_embed_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)
        return self

    # -- per-video buckets ---------------------------------------------- #

    def baseline_bucket(self, video_id) -> Path:
        return self.full_baseline_dir / video_id_str(video_id)

    def feature_bucket(self, video_id) -> Path:
        return self.tube_features_dir / video_id_str(video_id)

    @property
    def text_embed_dir(self) -> Path:
        """One prompt embedding per *clip* — not per counterfactual sample.

        The umT5 sequence is a property of the caption, so storing it inside every
        (tube, step, action) record duplicated a ~8 MiB tensor 15× per clip: ~21 TiB
        at the §2.4 target of 180k clips, before any of it is read. Stage B joins it
        back by ``video_id`` at load time.
        """
        return self.root / "text_embeds"

    def text_embed_path(self, video_id) -> Path:
        return self.text_embed_dir / f"{video_id_str(video_id)}.pt"

    def save_baseline(
        self,
        video_id,
        *,
        z_t_by_step: Mapping[int, Tensor],
        y_full: Tensor,
        z_init: Optional[Tensor] = None,
        text_emb: Optional[Tensor] = None,
        kv_cache: Optional[Mapping[str, Tensor]] = None,
        y_full_dtype: Optional[np.dtype] = np.float16,
    ) -> Path:
        """Write the §3 level-3 ``full_baseline/vid_XXXXXX/`` bucket for one video.

        ``Y_full`` dominates the bucket — 49×384×640 is 144 MB in fp32 and this is the
        per-clip cost of the whole ~TB-scale level-3 store — so it is stored in fp16 by
        default. It is only ever read as a *reference video* (Stage C's L1 and §6.3.2
        feature distances, both of which upcast), never re-entered into a trajectory, so
        the half-precision round trip is below the noise floor of the comparison. The
        latents (``z_t``, ``z_init``) stay fp32: those *are* re-entered, and Stage C's
        cached-baseline path is only valid if the noise it replays is bit-comparable.

        ``text_emb`` is optional and off by default: the prompt embedding is per-clip
        data and lives in ``text_embeds/<video_id>.pt``, trimmed to its real length and
        in fp16 (see :attr:`text_embed_dir`). Writing the untrimmed fp32 sequence here
        as well duplicated ~8 MiB per clip that nothing ever read.
        """
        bucket = self.baseline_bucket(video_id)
        (bucket / "z_t_sampled").mkdir(parents=True, exist_ok=True)
        if text_emb is not None:
            _save_npy(bucket / "text_emb.npy", text_emb, dtype=np.float16)
        for step, z in z_t_by_step.items():
            _save_npy(bucket / "z_t_sampled" / f"t_{int(step):02d}.npy", z)
        _save_npy(bucket / "Y_full.npy", y_full, dtype=y_full_dtype)
        if z_init is not None:
            # Y_full is only a usable reference for a run that starts from the same
            # noise, so the two are stored together (§P2-4).
            _save_npy(bucket / "z_init.npy", z_init)
        if kv_cache:
            kv_dir = bucket / "kv_cache"
            kv_dir.mkdir(parents=True, exist_ok=True)
            np.savez(
                kv_dir / "kv.npz",
                **{k: _to_numpy(v) for k, v in kv_cache.items()},
            )
        return bucket

    def load_baseline_latent(self, video_id, step: int, device=None) -> Optional[Tensor]:
        """Load one cached ``z_t`` for Stage C (or None if the bucket is absent)."""
        path = self.baseline_bucket(video_id) / "z_t_sampled" / f"t_{int(step):02d}.npy"
        if not path.exists():
            return None
        return _load_npy(path, device)

    def load_z_init(self, video_id, device=None) -> Optional[Tensor]:
        """The ``z_T`` ``Y_full`` was generated from (``None`` when not persisted)."""
        path = self.baseline_bucket(video_id) / "z_init.npy"
        if not path.exists():
            return None
        return _load_npy(path, device)

    def has_y_full(self, video_id) -> bool:
        """Whether this video's ``Y_full`` was persisted — without reading it.

        Lets a caller validate a whole batch's baselines before committing to the
        render, then read only the frames it turns out to need (see ``frames`` below).
        """
        return (self.baseline_bucket(video_id) / "Y_full.npy").exists()

    def load_y_full(self, video_id, device=None, *, frames=None) -> Optional[Tensor]:
        """Load the §3 level-3 reference video ``Y_full`` [F,3,H,W] for Stage C.

        Stage C's main quality loss compares the accelerated render against this
        full-compute baseline (§4.2 主损失). Returns ``None`` when the video's
        baseline bucket was not persisted (the sample is then trained on the
        regularisers only).

        ``frames=(start, stop)`` reads only that half-open frame range, memory-mapped,
        so the ~72 MB fp16 clip never lands on the device in full. Stage C's
        differentiable decode covers a *window* of the clip (``decode_grad_frames``),
        and the reference is cut to that same window before it is used — reading all 49
        frames to score 5 of them was pure transfer and residency.
        """
        path = self.baseline_bucket(video_id) / "Y_full.npy"
        if not path.exists():
            return None
        if frames is None:
            return _load_npy(path, device)
        start, stop = int(frames[0]), int(frames[1])
        arr = np.load(path, allow_pickle=False, mmap_mode="r")
        # ``np.array`` (a copy), not ``ascontiguousarray``: the slice of a memmap is
        # already contiguous, so the latter hands back a read-only view of the mapping
        # and ``torch.from_numpy`` warns about wrapping a non-writable buffer — and the
        # tensor would keep the whole file mapped for as long as it lives.
        window = np.array(arr[start:stop])
        t = torch.from_numpy(window)
        return t.to(device) if device is not None else t

    def save_features(
        self,
        video_id,
        *,
        tube_features: Tensor,
        tube_states: Tensor,
        causal_strength: Tensor,
        tube_visual_emb: Tensor,
    ) -> Path:
        """Write the §3 level-4 ``tube_causal_features/vid_XXXXXX/`` bucket."""
        bucket = self.feature_bucket(video_id)
        bucket.mkdir(parents=True, exist_ok=True)
        _save_npy(bucket / "tube_features.npy", tube_features)
        _save_npy(bucket / "tube_states.npy", tube_states)
        _save_npy(bucket / "causal_strength.npy", causal_strength)
        _save_npy(bucket / "tube_visual_emb.npy", tube_visual_emb)
        return bucket

    # -- raw_filtered (Stage C source) ---------------------------------- #

    def write_captions(self, captions: Iterable[Mapping[str, object]]) -> Path:
        """Write ``raw_filtered/captions.jsonl`` (one ``{video_id, caption}`` per line)."""
        self.raw_filtered_dir.mkdir(parents=True, exist_ok=True)
        path = self.raw_filtered_dir / "captions.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            for row in captions:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return path

    # -- CSV indices ---------------------------------------------------- #

    def write_csv(self, path: Path, rows: Sequence[Mapping[str, object]],
                  fieldnames: Optional[Sequence[str]] = None) -> Path:
        """Write a list-of-dicts to ``path`` as CSV (fieldnames inferred if absent).

        Published by an atomic rename, so a concurrent reader sees either the previous
        file or the complete new one — never a half-written prefix. The workers of a
        sharded Stage-A run poll ``filtered_final.csv`` to pick up shard 0's filter
        decision, and a large CSV spends a long time being partially on disk.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = list(rows)
        if fieldnames is None:
            fieldnames = list(rows[0].keys()) if rows else []
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            with open(tmp, "w", encoding="utf-8", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(fieldnames))
                w.writeheader()
                for r in rows:
                    w.writerow({k: r.get(k, "") for k in fieldnames})
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
        return path

    @staticmethod
    def read_csv(path: Path) -> List[Dict[str, str]]:
        if not Path(path).exists():
            return []
        with open(path, "r", encoding="utf-8", newline="") as fh:
            return list(csv.DictReader(fh))

    def write_sample_index(self, rows: Sequence[Mapping[str, object]]) -> Path:
        return self.write_csv(self.sample_index, rows, SAMPLE_INDEX_FIELDS)

    def read_sample_index(self) -> List[Dict[str, str]]:
        return self.read_csv(self.sample_index)

    def write_tube_meta(self, rows: Sequence[Mapping[str, object]]) -> Path:
        return self.write_csv(self.tube_meta, rows)

    # -- norm stats (§1.6) ---------------------------------------------- #

    def write_norm_stats(self, stats: Mapping[str, object]) -> Path:
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        with open(self.norm_stats, "w", encoding="utf-8") as fh:
            json.dump(stats, fh, indent=2, ensure_ascii=False)
        return self.norm_stats

    def read_norm_stats(self) -> Dict[str, object]:
        if not self.norm_stats.exists():
            return {}
        with open(self.norm_stats, "r", encoding="utf-8") as fh:
            return json.load(fh)

    # -- Stage-A generation environment (cross-stage geometry contract) --- #

    def write_stage_a_env(self, env: Mapping[str, object]) -> Path:
        """Record the backbone geometry and schedule this store was generated with.

        Stages B and C build their plugins from the *backbone's* token width, but
        neither loads the backbone: Stage B runs on cached labels and defaults to the
        mock adapter (``token_dim=32``), while Stage A's real Wan2.2-A14B gives 64. The
        residual-repair net is sized from that number, so the mismatch produced a Stage
        B checkpoint that could not be loaded into a Stage C running the real backbone —
        and nothing detected it until the shapes collided.

        Persisting the geometry next to the data makes it a property of the *store*
        rather than of whichever command line ran last, which is what the two later
        stages actually need to agree with. Also carries the resolution and teacher
        step count, because Stage C's target is the ``Y_full`` rendered here and both
        must match for the comparison to mean anything.
        """
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        with open(self.stage_a_env, "w", encoding="utf-8") as fh:
            json.dump(dict(env), fh, indent=2, ensure_ascii=False)
        return self.stage_a_env

    def read_stage_a_env(self) -> Dict[str, object]:
        """The recorded generation environment, or ``{}`` for a pre-existing store."""
        if not self.stage_a_env.exists():
            return {}
        with open(self.stage_a_env, "r", encoding="utf-8") as fh:
            return json.load(fh)

    # -- splits (§3 level-6, video_id-disjoint) ------------------------- #

    def write_splits(
        self,
        train: Sequence[str],
        val: Sequence[str],
        test_hard: Sequence[str] = (),
    ) -> None:
        self.splits_dir.mkdir(parents=True, exist_ok=True)
        _write_lines(self.train_list, train)
        _write_lines(self.val_list, val)
        _write_lines(self.test_hard_list, test_hard)

    def read_split(self, name: str) -> List[str]:
        """Read a split list by name: ``"train"`` | ``"val"`` | ``"test_hard"``."""
        path = {"train": self.train_list, "val": self.val_list,
                "test_hard": self.test_hard_list}[name]
        if not path.exists():
            return []
        return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


# --------------------------------------------------------------------------- #
# small IO helpers
# --------------------------------------------------------------------------- #


def _to_numpy(x: Tensor, dtype: Optional[np.dtype] = None) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        # ``.float()`` first unconditionally: numpy has no bfloat16, which is the dtype
        # every real backbone hands us, so a direct ``.numpy()`` raises.
        arr = x.detach().to("cpu").float().numpy()
    else:
        arr = np.asarray(x)
    return arr.astype(dtype, copy=False) if dtype is not None else arr


def _save_npy(path: Path, x: Tensor, dtype: Optional[np.dtype] = None) -> None:
    """Atomic ``.npy`` save (temp + replace) so a crashed run leaves no half files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    np.save(tmp, _to_numpy(x, dtype))
    # np.save appends .npy to the temp stem; normalise then atomically replace.
    written = tmp if tmp.exists() else tmp.with_suffix(tmp.suffix + ".npy")
    os.replace(written, path)


def _load_npy(path: Path, device=None) -> Tensor:
    arr = np.load(path, allow_pickle=False)
    return torch.from_numpy(arr).to(device) if device is not None else torch.from_numpy(arr)


def _write_lines(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(x) for x in lines) + ("\n" if lines else ""), encoding="utf-8")
