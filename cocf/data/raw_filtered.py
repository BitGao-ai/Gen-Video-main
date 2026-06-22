"""Stage-C raw-clip source — ``raw_filtered/`` reading + hard-sample sampling (§4.2).

§4.2 switches the data source away from the offline counterfactual LMDB: Stage C
"直接读取 raw_filtered/ 下的原始视频 - 文本对，以生成式任务做端到端训练". This module is that
reader plus the §4.2 采样策略:

    * **hard-sample priority** — multi / occlusion / text / face clips are up-weighted
      so the end-to-end fine-tune spends more steps on the model's short-board scenes
      (:class:`HardSamplePrioritySampler`);
    * **length / budget dynamic** — each item carries its scene type and (when known)
      duration so the stage can size a dynamic per-step budget ``B_t`` from scene
      complexity (the budget itself is computed by the shared
      :class:`~cocf.scheduler.budget.BudgetScheduler` in the stage).

Primary source is the processed store's ``raw_filtered/captions.jsonl``
(written by Stage A via :meth:`ProcessedLayout.write_captions`). When no processed
store is given, a plain video/caption manifest (CSV or JSONL) is read as a fallback,
so Stage C also runs against an ad-hoc prompt list. Like the Stage-B sampler, batch
planning only ever touches this lightweight metadata — never a video payload.
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

# The §2.3 / §4.2 hard scene classes whose sampling is up-weighted in Stage C.
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
    """Reads Stage-C raw-clip items from the processed store (or a fallback manifest).

    Parameters
    ----------
    processed_root
        Root of the six-level store; reads ``raw_filtered/captions.jsonl`` (§4.2). Takes
        precedence over ``manifest_path`` when both are given.
    manifest_path
        Fallback CSV/JSONL with ``{video|path, caption[, scene]}`` rows, used when no
        processed store is available.
    """

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
    """Weighted index sampler that up-weights hard scenes (§4.2 硬样本优先).

    Hard scene types (:data:`HARD_SCENE_TYPES`) are drawn ``hard_boost`` times more
    often than easy ones. Sampling is **with replacement** so the boost is exact and
    the epoch length stays fixed; reshuffled deterministically per epoch via
    :meth:`set_epoch`.
    """

    def __init__(
        self,
        scene_types: Sequence[str],
        *,
        hard_boost: float = 2.0,
        num_samples: Optional[int] = None,
        seed: int = 0,
    ) -> None:
        self.n = len(scene_types)
        self.weights = torch.tensor(
            [hard_boost if s in HARD_SCENE_TYPES else 1.0 for s in scene_types],
            dtype=torch.float64,
        )
        self.num_samples = int(num_samples) if num_samples else self.n
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        if self.n == 0:
            return iter(())
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        idx = torch.multinomial(self.weights, self.num_samples, replacement=True, generator=g)
        return iter(idx.tolist())
