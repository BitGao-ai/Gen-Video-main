"""OpenVid-1M manifest ingestion & scene stratification (§1.1, §1.2).

OpenVid-1M ships its metadata as ``data/train/OpenVid-1M.csv`` (≈930k clips) and
``data/train/OpenVidHD.csv`` (≈433k 1080p clips). The column names **contain
spaces** (``aesthetic score``, ``motion score``, ``temporal consistency score``,
``camera motion``), so a reader must map them explicitly — that is exactly what
:data:`DEFAULT_OPENVID_COLUMNS` does. All 202 ``OpenVid_part*.zip`` parts extract
into one flat ``video/`` folder, so a clip resolves to
``{data_root}/{video_subdir}/{video}`` (§1.1).

This module turns those CSVs into typed :class:`OpenVidRecord`s with:

    * a stable ``video_id`` (the mp4 stem, which encodes the clip's unique id),
    * the resolved on-disk path,
    * the carried-through quality metadata (aesthetic / motion / temporal scores),
    * an HD flag (rows from ``OpenVidHD.csv``), and
    * a coarse **scene type** (§1.2) drawn from the six classes the rest of the
      framework balances over — ``static / dynamic / multi / text / face /
      occlusion`` — inferred cheaply from the caption + motion/camera cues.

It then writes the §3 level-1 ``metadata/raw_dataset_index.csv`` via
:class:`~cocf.data.processed_layout.ProcessedLayout`. The scene lexicons are reused
from the existing perception/parser code so the inference here matches what the
strength field and CMSC see downstream (no skew).
"""

from __future__ import annotations

import csv
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from cocf.common.logging import get_logger
from cocf.data.metrics import _TEXT_CUES
from cocf.data.processed_layout import ProcessedLayout, video_id_str
from cocf.data.video_dataset import VideoMeta
from cocf.lcocf.triplets import _CRITICAL_HINTS

_log = get_logger(__name__)

# Canonical field -> OpenVid CSV column name. The spaces are load-bearing: the HF
# dataset uses them verbatim, so they must be mapped, never assumed split-free.
DEFAULT_OPENVID_COLUMNS: Dict[str, str] = {
    "video": "video",
    "caption": "caption",
    "aesthetic": "aesthetic score",
    "motion": "motion score",
    "temporal_consistency": "temporal consistency score",
    "camera_motion": "camera motion",
    "frame": "frame",
    "fps": "fps",
    "seconds": "seconds",
}

# The six scene classes the pipeline balances over (matches VideoSample.scene and
# StratifiedSamplingConfig.scene_weights). Pinned so callers can iterate them.
SCENE_TYPES: Sequence[str] = ("static", "dynamic", "multi", "text", "face", "occlusion")

_FACE_HINTS = _CRITICAL_HINTS["face"]
_HANDS_HINTS = _CRITICAL_HINTS["hands"]
_MULTI_CUES = ("group", "crowd", "people", "two", "three", "several", "many",
               "interact", "together", "and", "多", "群", "两", "三", "互动")
_OCCLUSION_CUES = ("behind", "occlud", "overlap", "hidden", "cover", "in front of",
                   "block", "遮挡", "重叠", "前面", "后面")


@dataclass
class OpenVidRecord:
    """One parsed OpenVid clip: path + carried metadata + derived scene type."""

    video_id: str
    video: str          # mp4 filename only
    path: str           # resolved {data_root}/{video_subdir}/{video}
    caption: str
    is_hd: bool = False
    aesthetic: float = 0.0
    motion: float = 0.0
    temporal_consistency: float = 0.0
    camera_motion: str = ""
    frame: int = 0
    fps: float = 0.0
    seconds: float = 0.0
    scene_type: str = "dynamic"

    def to_video_meta(self) -> VideoMeta:
        """Adapt to the :class:`VideoMeta` the video dataset reads (Stage A/C)."""
        return VideoMeta(path=self.path, caption=self.caption, scene=self.scene_type)

    def index_row(self) -> Dict[str, object]:
        """Flat dict for ``raw_dataset_index.csv`` (§3 level-1)."""
        row = asdict(self)
        row["is_hd"] = int(self.is_hd)
        return row


def _to_float(x: object, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _to_int(x: object, default: int = 0) -> int:
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return default


def infer_scene_type(caption: str, motion: float, camera_motion: str = "",
                     static_motion_max: float = 0.02) -> str:
    """Classify a clip into one of :data:`SCENE_TYPES` (§1.2).

    Priority — occlusion ▸ text ▸ face ▸ multi ▸ (static | dynamic) — so the harder,
    rarer, quality-critical categories (which §2.3 force-keeps) win when several
    cues co-occur; the static/dynamic split is decided last from the motion score.
    """
    cap = (caption or "").lower()
    cam = (camera_motion or "").lower()
    if any(c in cap for c in _OCCLUSION_CUES):
        return "occlusion"
    if any(c in cap for c in _TEXT_CUES):
        return "text"
    if any(c in cap for c in _FACE_HINTS) or any(c in cap for c in _HANDS_HINTS):
        return "face"
    if any(c in cap for c in _MULTI_CUES):
        return "multi"
    # otherwise distinguish static vs single-subject dynamic by motion magnitude;
    # a non-"static"/"none" camera motion also implies a dynamic scene.
    moving_cam = bool(cam) and cam not in ("static", "none", "fixed", "")
    if motion <= static_motion_max and not moving_cam:
        return "static"
    return "dynamic"


def read_openvid_csv(
    csv_path: str,
    data_root: str,
    *,
    video_subdir: str = "video",
    columns: Mapping[str, str] = DEFAULT_OPENVID_COLUMNS,
    is_hd: Optional[bool] = None,
    static_motion_max: float = 0.02,
    limit: Optional[int] = None,
) -> List[OpenVidRecord]:
    """Parse one OpenVid CSV into :class:`OpenVidRecord`s.

    ``is_hd`` defaults to detecting ``OpenVidHD`` in the filename; pass it explicitly
    to override. Missing optional columns degrade gracefully to defaults so a
    trimmed/sample CSV still reads.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"OpenVid manifest not found: {csv_path}")
    hd = ("openvidhd" in path.name.lower()) if is_hd is None else bool(is_hd)
    col = dict(columns)
    records: List[OpenVidRecord] = []
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader):
            if limit is not None and i >= limit:
                break
            video = (row.get(col["video"]) or "").strip()
            if not video:
                continue
            caption = (row.get(col["caption"]) or "").strip()
            motion = _to_float(row.get(col.get("motion", "")))
            camera = (row.get(col.get("camera_motion", "")) or "").strip()
            records.append(
                OpenVidRecord(
                    video_id=Path(video).stem,
                    video=video,
                    path=os.path.join(data_root, video_subdir, video),
                    caption=caption,
                    is_hd=hd,
                    aesthetic=_to_float(row.get(col.get("aesthetic", ""))),
                    motion=motion,
                    temporal_consistency=_to_float(row.get(col.get("temporal_consistency", ""))),
                    camera_motion=camera,
                    frame=_to_int(row.get(col.get("frame", ""))),
                    fps=_to_float(row.get(col.get("fps", ""))),
                    seconds=_to_float(row.get(col.get("seconds", ""))),
                    scene_type=infer_scene_type(caption, motion, camera, static_motion_max),
                )
            )
    _log.info("Parsed %d OpenVid records from %s (hd=%s)", len(records), path.name, hd)
    return records


def read_openvid_manifest(
    csv_paths: Sequence[str],
    data_root: str,
    *,
    video_subdir: str = "video",
    columns: Mapping[str, str] = DEFAULT_OPENVID_COLUMNS,
    static_motion_max: float = 0.02,
    limit_per_csv: Optional[int] = None,
) -> List[OpenVidRecord]:
    """Read & concatenate several OpenVid CSVs (e.g. the 1M subset + the HD subset)."""
    out: List[OpenVidRecord] = []
    for p in csv_paths:
        out.extend(read_openvid_csv(
            p, data_root, video_subdir=video_subdir, columns=columns,
            static_motion_max=static_motion_max, limit=limit_per_csv,
        ))
    return out


def write_raw_dataset_index(records: Sequence[OpenVidRecord], layout: ProcessedLayout) -> Path:
    """Write the §3 level-1 ``metadata/raw_dataset_index.csv`` (video_id ─ path ─ info)."""
    rows = [r.index_row() for r in records]
    fields = list(rows[0].keys()) if rows else list(OpenVidRecord.__annotations__.keys())
    return layout.write_csv(layout.raw_dataset_index, rows, fields)


def scene_histogram(records: Sequence[OpenVidRecord]) -> Dict[str, int]:
    """Count records per scene type (for the §2.4 statistics report)."""
    hist = {s: 0 for s in SCENE_TYPES}
    for r in records:
        hist[r.scene_type] = hist.get(r.scene_type, 0) + 1
    return hist
