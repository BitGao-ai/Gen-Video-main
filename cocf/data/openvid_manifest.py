"""OpenVid-1M manifest ingestion and scene stratification.

Parses the OpenVid CSVs into typed :class:`OpenVidRecord`s (stable video_id, resolved
path, carried quality metadata, HD flag, derived scene type) and writes the level-1
``metadata/raw_dataset_index.csv``.
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

# Canonical field -> OpenVid CSV column name (spaces are part of the real header).
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

# The six scene classes the pipeline balances over.
SCENE_TYPES: Sequence[str] = ("static", "dynamic", "multi", "text", "face", "occlusion")

_FACE_HINTS = _CRITICAL_HINTS["face"]
_HANDS_HINTS = _CRITICAL_HINTS["hands"]
_MULTI_CUES = ("group", "crowd", "people", "two", "three", "several", "many",
               "interact", "together", "and", "多", "群", "两", "三", "互动")
_OCCLUSION_CUES = ("behind", "occlud", "overlap", "hidden", "cover", "in front of",
                   "block", "遮挡", "重叠", "前面", "后面")


def _cue_re(cues: Sequence[str]) -> Optional[re.Pattern]:
    """Word-start-anchored regex over the ASCII cues (``None`` when there are none)."""
    words = [c for c in cues if c.isascii()]
    if not words:
        return None
    return re.compile(r"\b(?:" + "|".join(re.escape(w) for w in words) + r")")


def _cjk_cues(cues: Sequence[str]) -> List[str]:
    return [c for c in cues if not c.isascii()]


def _has_cue(cap: str, rex: Optional[re.Pattern], cjk: Sequence[str]) -> bool:
    return bool(rex is not None and rex.search(cap)) or any(c in cap for c in cjk)


_FACE_RE = _cue_re(_FACE_HINTS)
_HANDS_RE = _cue_re(_HANDS_HINTS)
_FACE_CJK = _cjk_cues(_FACE_HINTS) + _cjk_cues(_HANDS_HINTS)
_MULTI_RE = _cue_re(_MULTI_CUES)
_MULTI_CJK = _cjk_cues(_MULTI_CUES)
_OCCLUSION_RE = _cue_re(_OCCLUSION_CUES)
_OCCLUSION_CJK = _cjk_cues(_OCCLUSION_CUES)
_TEXT_RE = _cue_re(_TEXT_CUES)
_TEXT_CJK = _cjk_cues(_TEXT_CUES)


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
        """Flat dict for ``raw_dataset_index.csv``."""
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
    """Classify a clip into one of :data:`SCENE_TYPES`.

    Priority — occlusion ▸ text ▸ face ▸ multi ▸ (static | dynamic); the
    static/dynamic split is decided last from the motion score.
    """
    cap = (caption or "").lower()
    cam = (camera_motion or "").lower()
    if _has_cue(cap, _OCCLUSION_RE, _OCCLUSION_CJK):
        return "occlusion"
    if _has_cue(cap, _TEXT_RE, _TEXT_CJK):
        return "text"
    if _has_cue(cap, _FACE_RE, _FACE_CJK) or _has_cue(cap, _HANDS_RE, ()):
        return "face"
    if _has_cue(cap, _MULTI_RE, _MULTI_CJK):
        return "multi"
    # otherwise distinguish static vs single-subject dynamic by motion magnitude.
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
    require_file: bool = False,
) -> List[OpenVidRecord]:
    """Parse one OpenVid CSV into :class:`OpenVidRecord`s.

    ``is_hd`` defaults to detecting ``OpenVidHD`` in the filename. ``require_file``
    keeps only clips whose resolved ``path`` exists, and then ``limit`` caps the number
    of *kept* rows rather than the number scanned.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"OpenVid manifest not found: {csv_path}")
    hd = ("openvidhd" in path.name.lower()) if is_hd is None else bool(is_hd)
    col = dict(columns)
    records: List[OpenVidRecord] = []
    n_missing = 0
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader):
            # ``limit`` caps rows scanned without ``require_file``, else rows kept.
            if limit is not None and not require_file and i >= limit:
                break
            video = (row.get(col["video"]) or "").strip()
            if not video:
                continue
            resolved = os.path.join(data_root, video_subdir, video)
            if require_file and not os.path.exists(resolved):
                n_missing += 1
                continue
            caption = (row.get(col["caption"]) or "").strip()
            motion = _to_float(row.get(col.get("motion", "")))
            camera = (row.get(col.get("camera_motion", "")) or "").strip()
            records.append(
                OpenVidRecord(
                    video_id=Path(video).stem,
                    video=video,
                    path=resolved,
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
            if limit is not None and require_file and len(records) >= limit:
                break
    if require_file:
        _log.info(
            "Parsed %d on-disk OpenVid records from %s (hd=%s; skipped %d rows with no mp4 under %s)",
            len(records), path.name, hd, n_missing, os.path.join(data_root, video_subdir),
        )
    else:
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
    require_file: bool = False,
) -> List[OpenVidRecord]:
    """Read & concatenate several OpenVid CSVs (e.g. the 1M subset + the HD subset)."""
    out: List[OpenVidRecord] = []
    for p in csv_paths:
        out.extend(read_openvid_csv(
            p, data_root, video_subdir=video_subdir, columns=columns,
            static_motion_max=static_motion_max, limit=limit_per_csv,
            require_file=require_file,
        ))
    return out


def write_raw_dataset_index(records: Sequence[OpenVidRecord], layout: ProcessedLayout) -> Path:
    """Write the level-1 ``metadata/raw_dataset_index.csv``."""
    rows = [r.index_row() for r in records]
    fields = list(rows[0].keys()) if rows else list(OpenVidRecord.__annotations__.keys())
    return layout.write_csv(layout.raw_dataset_index, rows, fields)


def scene_histogram(records: Sequence[OpenVidRecord]) -> Dict[str, int]:
    """Count records per scene type."""
    hist = {s: 0 for s in SCENE_TYPES}
    for r in records:
        hist[r.scene_type] = hist.get(r.scene_type, 0) + 1
    return hist
