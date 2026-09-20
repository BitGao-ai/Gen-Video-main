"""Stage-C raw-clip source: reads ``raw_filtered/`` and provides hard-sample sampling.

Reads the processed store's ``raw_filtered/captions.jsonl`` (or a plain video/caption
CSV/JSONL manifest as fallback) and up-weights hard scenes (multi / occlusion / text /
face). Batch planning touches only this lightweight metadata, never a video payload.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

import torch
from torch.utils.data import Dataset, Sampler

from cocf.common.logging import get_logger
from cocf.data.openvid_manifest import infer_scene_type
from cocf.data.processed_layout import ProcessedLayout

_log = get_logger(__name__)

# Hard scene classes whose sampling is up-weighted in Stage C.
HARD_SCENE_TYPES = frozenset({"multi", "occlusion", "text", "face"})


@dataclass
class RawFilteredItem:
    """One Stage-C training item: a prompt + its scene metadata (no payload)."""

    video_id: str
    caption: str
    scene_type: str = "dynamic"
    is_hd: bool = False
    path: str = ""
    seconds: float = 0.0

    @property
    def is_hard(self) -> bool:
        return self.scene_type in HARD_SCENE_TYPES


class RawFilteredDataset(Dataset):
    """Reads Stage-C raw-clip items from the processed store (or a fallback manifest)."""

    def __init__(
        self,
        processed_root: Optional[Path] = None,
        manifest_path: Optional[Path] = None,
    ) -> None:
        if processed_root is not None:
            self.items = self._from_processed(Path(processed_root))
            self.source = "raw_filtered"
        elif manifest_path is not None:
            self.items = self._from_manifest(Path(manifest_path))
            self.source = "manifest"
        else:
            raise ValueError("RawFilteredDataset needs processed_root or manifest_path")
        if not self.items:
            _log.warning("RawFilteredDataset: no items read from %s", self.source)

    # -- readers -------------------------------------------------------- #

    @staticmethod
    def _from_processed(root: Path) -> List[RawFilteredItem]:
        layout = ProcessedLayout(root)
        path = layout.raw_filtered_dir / "captions.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"Stage C expected {path} (run Stage A first, or pass --manifest)."
            )
        items: List[RawFilteredItem] = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                items.append(
                    RawFilteredItem(
                        video_id=str(row.get("video_id", "")),
                        caption=str(row.get("caption", "")),
                        scene_type=str(row.get("scene_type", "dynamic")),
                        is_hd=bool(row.get("is_hd", 0)),
                        path=str(row.get("path", "")),
                        seconds=float(row.get("seconds", 0.0) or 0.0),
                    )
                )
        return items

    @staticmethod
    def _from_manifest(path: Path) -> List[RawFilteredItem]:
        if not path.exists():
            raise FileNotFoundError(f"Stage C manifest not found: {path}")
        rows: List[Dict[str, object]] = []
        if path.suffix.lower() in (".jsonl", ".json"):
            with open(path, "r", encoding="utf-8") as fh:
                rows = [json.loads(ln) for ln in fh if ln.strip()]
        else:  # CSV
            with open(path, "r", encoding="utf-8", newline="") as fh:
                rows = list(csv.DictReader(fh))
        items: List[RawFilteredItem] = []
        for i, row in enumerate(rows):
            caption = str(row.get("caption", "") or "")
            video = str(row.get("video", row.get("path", "")) or "")
            scene = str(row.get("scene", row.get("scene_type", "")) or "") \
                or infer_scene_type(caption, motion=0.0)
            items.append(
                RawFilteredItem(
                    video_id=str(row.get("video_id", "") or Path(video).stem or f"clip_{i:06d}"),
                    caption=caption,
                    scene_type=scene,
                    is_hd=bool(int(row.get("is_hd", 0) or 0)) if str(row.get("is_hd", "")).strip() else False,
                    path=video,
                    seconds=float(row.get("seconds", 0.0) or 0.0),
                )
            )
        return items

    # -- Dataset API ---------------------------------------------------- #

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> RawFilteredItem:
        return self.items[idx]

    @property
    def scene_types(self) -> List[str]:
        return [it.scene_type for it in self.items]


def collate_raw_filtered(batch: Sequence[RawFilteredItem]) -> List[RawFilteredItem]:
    """Identity collate: Stage C runs the engine per clip (Y_full is per-video)."""
    return list(batch)


class HardSamplePrioritySampler(Sampler[int]):
    """Weighted index sampler that up-weights hard scenes, with replacement."""

    def __init__(
        self,
        scene_types: Sequence[str],
        *,
        hard_boost: float = 2.0,
        num_samples: Optional[int] = None,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.n = len(scene_types)
        self.weights = torch.tensor(
            [hard_boost if s in HARD_SCENE_TYPES else 1.0 for s in scene_types],
            dtype=torch.float64,
        )
        self.world_size = max(1, int(world_size))
        self.rank = int(rank) % self.world_size
        total = int(num_samples) if num_samples else self.n
        self.num_samples = max(1, total // self.world_size)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        if self.n == 0:
            return iter(())
        g = torch.Generator().manual_seed(
            self.seed + self.epoch * self.world_size + self.rank
        )
        idx = torch.multinomial(self.weights, self.num_samples, replacement=True, generator=g)
        return iter(idx.tolist())
