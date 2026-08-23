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
from typing import Callable, Dict, List, Optional

import torch

from cocf.common.config import TubeConfig
from cocf.common.types import Region, SemanticTube, TokenGrid

Tensor = torch.Tensor


def solve_assignment(affinity: Tensor, threshold: float) -> List[tuple]:
    """Return ``[(i, j), …]`` maximising total affinity, dropping pairs < ``threshold``.

    Uses the Hungarian algorithm when SciPy is available; otherwise a greedy
    descending-affinity matcher (optimal for the common near-diagonal case).

    ``.float()`` before ``.numpy()``: numpy has no bfloat16, so a provider whose
    affinity arrives in half precision raises ``TypeError: Got unsupported ScalarType
    BFloat16`` — which the ``except`` below would absorb as "no SciPy" and silently
    downgrade *every* frame pair to the greedy matcher. A quality regression that
    reports itself nowhere is worse than the crash.
    """
    ra, rb = affinity.shape
    if ra == 0 or rb == 0:
        return []
    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(affinity.detach().float().cpu().numpy(),
                                           maximize=True)
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
        # Monotonic across *calls*, not just within one. Re-segmentation
        # (``tube_refresh_every > 0``) used to restart numbering at 0, so the new
        # tube 3 inherited the anchor the old tube 3 left in the store — and that
        # anchor carries the *old* token indices, so a rollback wrote a stale patch
        # into unrelated positions of the latent (§P1-10).
        self._next_tube_id = 0

    def _allocate_id(self) -> int:
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
        """Assemble tubes from regions and precomputed consecutive-frame affinities.

        ``affinity_by_pair[(a, b)]`` is the ``[R_a, R_b]`` matrix from frame ``a`` to
        ``b`` (as produced by :class:`AffinityComputer`). ``affinity_fn(fa, fb)``
        computes one on demand for a *non-consecutive* pair, which is what lets a track
        that missed a frame be picked up again (see below); without it the matcher
        degrades to consecutive-only matching.
        """
        frames = sorted(regions_by_frame)
        if not frames:
            return []
        tracks: List[_Track] = []
        # seed tracks from the first frame
        for r in regions_by_frame[frames[0]]:
            tracks.append(
                _Track(track_id=self._allocate_id(), regions={frames[0]: r},
                       last_frame=frames[0])
            )

        for a, b in zip(frames[:-1], frames[1:]):
            new_regions = regions_by_frame[b]
            live = [tr for tr in tracks if tr.gap <= self.max_gap]
            aff_full = affinity_by_pair.get((a, b))

            # Match freshest-first: tracks whose last region is on frame ``a``, then
            # tracks that missed a frame but are still inside the gap window. Restricting
            # to ``last_frame == a`` (as this did) meant a single missed frame retired a
            # track permanently — ``max_gap`` only postponed its closure and the
            # "broken-tube completion" it exists for never happened (§P1-10).
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

            # unmatched tracks age (broken-tube completion)
            for tr in live:
                if tr.track_id not in matched_tracks and tr.last_frame != b:
                    tr.gap += 1
            # unmatched new regions open fresh tracks
            for rj, r in enumerate(new_regions):
                if rj not in matched_new:
                    tracks.append(
                        _Track(track_id=self._allocate_id(), regions={b: r}, last_frame=b)
                    )

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
            # new_zeros follows aff_full's device so the fallback row stacks with the
            # real (possibly GPU) affinity rows without a device mismatch.
            rows.append(aff_full[rid] if rid < aff_full.shape[0] else aff_full.new_zeros(aff_full.shape[1]))
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
                # Retain the per-frame feature too: the pooled mean below cannot
                # express identity *drift*, which is exactly what the tube state's
                # identity confidence measures (§4.3.1).
                tube.identity_feat_by_frame[frame] = r.identity_feat
        if feats:
            tube.identity_feat = torch.stack(feats).mean(0)
        return tube

    def _split_long(self, tubes: List[SemanticTube], grid: TokenGrid) -> List[SemanticTube]:
        """Split tubes spanning more than ``max_tube_len`` frames into sub-tubes."""
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
                    # Carry only this chunk's frames, so each sub-tube's identity
                    # confidence is measured over the frames it actually spans.
                    identity_feat_by_frame={
                        f: tube.identity_feat_by_frame[f]
                        for f in chunk
                        if f in tube.identity_feat_by_frame
                    },
                )
                out.append(sub)
        return out
