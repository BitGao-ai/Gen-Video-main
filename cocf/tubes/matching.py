"""Tube assembly by optimal cross-frame matching (§4.3.1).

Consumes per-frame :class:`Region` lists plus the pairwise affinity matrices and
links regions into :class:`SemanticTube` tracks:

    1. for each consecutive frame pair, solve the **optimal assignment** on the
       affinity matrix (Hungarian / ``scipy.linear_sum_assignment``; greedy
       fallback when SciPy is absent) gated by ``affinity_match_threshold``;
    2. matched region → extend the track; unmatched new region → open a track;
       unmatched track → keep alive across a short gap (broken-tube completion);
    3. split any track longer than ``max_tube_len`` (default 16) into sub-tubes.

The matcher is pure bookkeeping over affinities, so it has no perception
dependency and is deterministic/testable on synthetic affinity matrices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from cocf.common.config import TubeConfig
from cocf.common.types import Region, SemanticTube, TokenGrid

Tensor = torch.Tensor


def solve_assignment(affinity: Tensor, threshold: float) -> List[tuple]:
    """Return ``[(i, j), …]`` maximising total affinity, dropping pairs < ``threshold``.

    Uses the Hungarian algorithm when SciPy is available; otherwise a greedy
    descending-affinity matcher (optimal for the common near-diagonal case).
    """
    ra, rb = affinity.shape
    if ra == 0 or rb == 0:
        return []
    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(affinity.cpu().numpy(), maximize=True)
        pairs = [(int(i), int(j)) for i, j in zip(rows, cols)]
    except Exception:  # greedy fallback (no SciPy)
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
    """A growing tube under construction."""

    track_id: int
    regions: Dict[int, Region] = field(default_factory=dict)  # frame -> region
    last_frame: int = -1
    gap: int = 0  # consecutive frames this track failed to match (broken-tube)

    @property
    def length(self) -> int:
        return len(self.regions)


class TubeMatcher:
    """Links per-frame regions into semantic tubes via optimal matching."""

    def __init__(self, config: TubeConfig) -> None:
        self.cfg = config
        self.max_gap = 2  # frames a track may survive unmatched before closing

    def build_tubes(
        self,
        regions_by_frame: Dict[int, List[Region]],
        affinity_by_pair: Dict[tuple, Tensor],
        grid: TokenGrid,
    ) -> List[SemanticTube]:
        """Assemble tubes from regions and precomputed consecutive-frame affinities.

        ``affinity_by_pair[(a, b)]`` is the ``[R_a, R_b]`` matrix from frame ``a`` to
        ``b`` (as produced by :class:`AffinityComputer`).
        """
        frames = sorted(regions_by_frame)
        if not frames:
            return []
        tracks: List[_Track] = []
        next_id = 0
        # seed tracks from the first frame
        for r in regions_by_frame[frames[0]]:
            tracks.append(_Track(track_id=next_id, regions={frames[0]: r}, last_frame=frames[0]))
            next_id += 1

        for a, b in zip(frames[:-1], frames[1:]):
            new_regions = regions_by_frame[b]
            live = [tr for tr in tracks if tr.gap <= self.max_gap]
            aff_full = affinity_by_pair.get((a, b))
            # restrict affinity rows to *live tracks whose last region is on frame a*
            row_tracks = [tr for tr in live if tr.last_frame == a]
            pairs = self._match(row_tracks, new_regions, a, aff_full)

            matched_new = set()
            for ti, rj in pairs:
                tr = row_tracks[ti]
                tr.regions[b] = new_regions[rj]
                tr.last_frame = b
                tr.gap = 0
                matched_new.add(rj)
            # unmatched tracks age (broken-tube completion)
            matched_tracks = {row_tracks[ti].track_id for ti, _ in pairs}
            for tr in row_tracks:
                if tr.track_id not in matched_tracks:
                    tr.gap += 1
            # unmatched new regions open fresh tracks
            for rj, r in enumerate(new_regions):
                if rj not in matched_new:
                    tracks.append(_Track(track_id=next_id, regions={b: r}, last_frame=b))
                    next_id += 1

        tubes = [self._to_tube(tr, grid) for tr in tracks if tr.length > 0]
        return self._split_long(tubes, grid)

    # -- helpers --------------------------------------------------------- #

    def _match(
        self, row_tracks: List[_Track], new_regions: List[Region], frame_a: int,
        aff_full: Optional[Tensor],
    ) -> List[tuple]:
        if not row_tracks or not new_regions or aff_full is None:
            return []
        # gather the affinity rows for each track's region index on frame_a
        rows = []
        for tr in row_tracks:
            rid = tr.regions[frame_a].region_id
            rows.append(aff_full[rid] if rid < aff_full.shape[0] else torch.zeros(aff_full.shape[1]))
        sub = torch.stack(rows)  # [num_tracks, R_b]
        return solve_assignment(sub, self.cfg.affinity_match_threshold)

    def _to_tube(self, track: _Track, grid: TokenGrid) -> SemanticTube:
        tube = SemanticTube(tube_id=track.track_id)
        feats = []
        for frame, r in sorted(track.regions.items()):
            tube.tokens_by_frame[frame] = r.token_indices
            tube.masks_by_frame[frame] = r.mask
            if r.identity_feat is not None:
                feats.append(r.identity_feat)
        if feats:
            tube.identity_feat = torch.stack(feats).mean(0)
        return tube

    def _split_long(self, tubes: List[SemanticTube], grid: TokenGrid) -> List[SemanticTube]:
        """Split tubes spanning more than ``max_tube_len`` frames into sub-tubes."""
        out: List[SemanticTube] = []
        nid = max((t.tube_id for t in tubes), default=-1) + 1
        for tube in tubes:
            frames = tube.frames
            if len(frames) <= self.cfg.max_tube_len:
                out.append(tube)
                continue
            for start in range(0, len(frames), self.cfg.max_tube_len):
                chunk = frames[start : start + self.cfg.max_tube_len]
                sub = SemanticTube(
                    tube_id=tube.tube_id if start == 0 else nid,
                    tokens_by_frame={f: tube.tokens_by_frame[f] for f in chunk},
                    masks_by_frame={f: tube.masks_by_frame[f] for f in chunk},
                    identity_feat=tube.identity_feat,
                )
                if start != 0:
                    nid += 1
                out.append(sub)
        return out
