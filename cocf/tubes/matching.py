"""Tube assembly by optimal cross-frame matching."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch

from cocf.common.config import TubeConfig
from cocf.common.types import Region, SemanticTube, TokenGrid

Tensor = torch.Tensor


def solve_assignment(affinity: Tensor, threshold: float) -> List[tuple]:
    """Match region pairs maximizing total affinity above threshold."""
    ra, rb = affinity.shape
    if ra == 0 or rb == 0:
        return []
    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(affinity.detach().float().cpu().numpy(),
                                           maximize=True)
        pairs = [(int(i), int(j)) for i, j in zip(rows, cols)]
    except Exception:
        pairs = []
        used_i, used_j = set(), set()
        flat = [
            (float(affinity[i, j]), i, j)
            for i in range(ra)
            for j in range(rb)
        ]
        for score, i, j in sorted(flat, reverse=True):
            if i in used_i or j in used_j:
                continue
            pairs.append((i, j))
            used_i.add(i)
            used_j.add(j)
    return [(i, j) for (i, j) in pairs if float(affinity[i, j]) >= threshold]


@dataclass
class _Track:
    """Growing tube under construction."""

    track_id: int
    regions: Dict[int, Region] = field(default_factory=dict)
    last_frame: int = -1
    gap: int = 0

    @property
    def length(self) -> int:
        """Number of regions in track."""
        return len(self.regions)


class TubeMatcher:
    """Links per-frame regions into semantic tubes."""

    def __init__(self, config: TubeConfig) -> None:
        """Store config and init tube id counter."""
        self.cfg = config
        self.max_gap = 2
        self._next_tube_id = 0

    def _allocate_id(self) -> int:
        """Allocate next tube id."""
        tid = self._next_tube_id
        self._next_tube_id += 1
        return tid

    def build_tubes(
        self,
        regions_by_frame: Dict[int, List[Region]],
        affinity_by_pair: Dict[tuple, Tensor],
        grid: TokenGrid,
        affinity_fn: Optional[Callable[[int, int], Optional[Tensor]]] = None,
    ) -> List[SemanticTube]:
        """Assemble tubes from regions and affinity matrices."""
        frames = sorted(regions_by_frame)
        if not frames:
            return []
        tracks: List[_Track] = []
        for r in regions_by_frame[frames[0]]:
            tracks.append(
                _Track(track_id=self._allocate_id(), regions={frames[0]: r},
                       last_frame=frames[0])
            )

        for a, b in zip(frames[:-1], frames[1:]):
            new_regions = regions_by_frame[b]
            live = [tr for tr in tracks if tr.gap <= self.max_gap]
            aff_full = affinity_by_pair.get((a, b))

            by_last: Dict[int, List[_Track]] = {}
            for tr in live:
                by_last.setdefault(tr.last_frame, []).append(tr)

            matched_new: set = set()
            matched_tracks: set = set()
            for fa in sorted(by_last, reverse=True):
                if fa == b or (b - fa) > self.max_gap + 1:
                    continue
                aff = aff_full if fa == a else (
                    affinity_fn(fa, b) if affinity_fn is not None else None
                )
                if aff is None:
                    continue
                avail = [j for j in range(len(new_regions)) if j not in matched_new]
                if not avail:
                    break
                trs = by_last[fa]
                sub_regions = [new_regions[j] for j in avail]
                aff_sub = aff[:, avail] if aff.numel() else aff
                for ti, rj_local in self._match(trs, sub_regions, fa, aff_sub):
                    tr = trs[ti]
                    rj = avail[rj_local]
                    tr.regions[b] = new_regions[rj]
                    tr.last_frame = b
                    tr.gap = 0
                    matched_new.add(rj)
                    matched_tracks.add(tr.track_id)

            for tr in live:
                if tr.track_id not in matched_tracks and tr.last_frame != b:
                    tr.gap += 1
            for rj, r in enumerate(new_regions):
                if rj not in matched_new:
                    tracks.append(
                        _Track(track_id=self._allocate_id(), regions={b: r}, last_frame=b)
                    )

        tubes = [self._to_tube(tr, grid) for tr in tracks if tr.length > 0]
        return self._split_long(tubes, grid)

    def _match(
        self, row_tracks: List[_Track], new_regions: List[Region], frame_a: int,
        aff_full: Optional[Tensor],
    ) -> List[tuple]:
        """Match tracks to new regions via affinity."""
        if not row_tracks or not new_regions or aff_full is None:
            return []
        rows = []
        for tr in row_tracks:
            rid = tr.regions[frame_a].region_id
            rows.append(aff_full[rid] if rid < aff_full.shape[0] else aff_full.new_zeros(aff_full.shape[1]))
        sub = torch.stack(rows)
        return solve_assignment(sub, self.cfg.affinity_match_threshold)

    def _to_tube(self, track: _Track, grid: TokenGrid) -> SemanticTube:
        """Convert track to semantic tube."""
        tube = SemanticTube(tube_id=track.track_id)
        feats = []
        for frame, r in sorted(track.regions.items()):
            tube.tokens_by_frame[frame] = r.token_indices
            tube.masks_by_frame[frame] = r.mask
            if r.identity_feat is not None:
                feats.append(r.identity_feat)
                tube.identity_feat_by_frame[frame] = r.identity_feat
        if feats:
            tube.identity_feat = torch.stack(feats).mean(0)
        return tube

    def _split_long(self, tubes: List[SemanticTube], grid: TokenGrid) -> List[SemanticTube]:
        """Split tubes longer than max length."""
        out: List[SemanticTube] = []
        for tube in tubes:
            frames = tube.frames
            if len(frames) <= self.cfg.max_tube_len:
                out.append(tube)
                continue
            for start in range(0, len(frames), self.cfg.max_tube_len):
                chunk = frames[start : start + self.cfg.max_tube_len]
                sub = SemanticTube(
                    tube_id=tube.tube_id if start == 0 else self._allocate_id(),
                    tokens_by_frame={f: tube.tokens_by_frame[f] for f in chunk},
                    masks_by_frame={f: tube.masks_by_frame[f] for f in chunk},
                    identity_feat=tube.identity_feat,
                    identity_feat_by_frame={
                        f: tube.identity_feat_by_frame[f]
                        for f in chunk
                        if f in tube.identity_feat_by_frame
                    },
                )
                out.append(sub)
        return out
