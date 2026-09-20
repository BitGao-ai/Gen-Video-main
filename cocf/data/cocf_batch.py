"""Stage-B batch assembly: stratified sampling and collation.

:class:`StratifiedBatchSampler` yields batches balanced 1:1:1:1 across the four
actions and stratified by scene and timestep, planning only from ``sample_index.csv``
rows (never a payload). :func:`collate_cocf_samples` stacks stored payload dicts into
batched tensors, padding the variable-length text token sequence with a mask.
"""

from __future__ import annotations

import random
from typing import Dict, Iterator, List, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Sampler

from cocf.common.logging import get_logger
from cocf.common.types import Action

Tensor = torch.Tensor
_log = get_logger(__name__)

# Per-sample tensor fields that collate stacks into ``[B, ...]``.
_VECTOR_FIELDS = (
    "tube_features", "strength_features", "damage_label", "cost_label",
    "uncertainty", "tube_visual_embed_full", "tube_visual_embed_cf",
)
_SCALAR_LONG_FIELDS = ("action", "timestep", "tube_token_count", "tube_id", "strength_level")
_SCALAR_FLOAT_FIELDS = ("step_frac", "interaction_density", "tube_stability",
                        "skip_residual")
_STRING_FIELDS = ("prompt", "scene_type", "video_id")


def timestep_stratum(timestep: int, num_total_steps: int) -> str:
    """Map a denoising timestep to its early / mid / late phase."""
    sf = float(timestep) / max(1, num_total_steps)
    if sf >= 0.8:
        return "early"
    if sf >= 0.3:
        return "mid"
    return "late"


class StratifiedBatchSampler(Sampler[List[int]]):
    """Yields batches of dataset indices, action-balanced 1:1:1:1.

    ``actions``/``scenes``/``strata`` are per-index lists aligned to the dataset keys.
    ``rank``/``world_size`` stride each action bucket so the balance holds per rank and
    every rank runs the same number of steps.
    """

    def __init__(
        self,
        actions: Sequence[int],
        scenes: Optional[Sequence[str]] = None,
        strata: Optional[Sequence[str]] = None,
        *,
        batch_size: int = 32,
        seed: int = 0,
        drop_last: bool = True,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.n = len(actions)
        self.batch_size = max(1, int(batch_size))
        self.seed = int(seed)
        self.epoch = 0
        self.world_size = max(1, int(world_size))
        self.rank = int(rank) % self.world_size
        self.scenes = list(scenes) if scenes is not None else [""] * self.n
        self.strata = list(strata) if strata is not None else [""] * self.n
        # bucket dataset indices by action, then keep this rank's stride of each
        buckets: Dict[int, List[int]] = {}
        for i, a in enumerate(actions):
            buckets.setdefault(int(a), []).append(i)
        self.action_buckets = {}
        for a, idxs in sorted(buckets.items()):
            shard = idxs[self.rank::self.world_size]
            if shard:
                self.action_buckets[a] = shard
        per_rank = self.n // self.world_size
        self._num_batches = per_rank // self.batch_size if drop_last else \
            (per_rank + self.batch_size - 1) // self.batch_size

    def set_epoch(self, epoch: int) -> None:
        """Reshuffle deterministically per epoch (call before each epoch)."""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return max(0, self._num_batches)

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        # Each action pool is ordered by interleaving its (stratum, scene) groups so
        # every contiguous draw spans as many strata and scenes as the data allows.
        pools = {a: self._interleaved(idxs, rng) for a, idxs in self.action_buckets.items()}
        present = list(pools)
        if not present:
            return
        per_action = max(1, self.batch_size // len(present))
        cursors = {a: 0 for a in present}

        def draw(a: int) -> int:
            # Small pools repeat (cursor wraps): a 1:1:1:1 mix oversamples rare actions.
            pool = pools[a]
            i = pool[cursors[a] % len(pool)]
            cursors[a] += 1
            return i

        for _ in range(len(self)):
            batch: List[int] = [draw(a) for a in present for _k in range(per_action)]
            # Top up round-robin so a batch always has batch_size entries.
            while len(batch) < self.batch_size:
                batch.append(draw(present[len(batch) % len(present)]))
            rng.shuffle(batch)
            yield batch[: self.batch_size]

    def _interleaved(self, idxs: Sequence[int], rng: random.Random) -> List[int]:
        """Order ``idxs`` so consecutive entries vary in timestep stratum and scene.

        Groups by ``(stratum, scene)``, shuffles within and across groups, then draws
        round-robin; degrades to a plain shuffle when there is only one group.
        """
        groups: Dict[tuple, List[int]] = {}
        for i in idxs:
            groups.setdefault((self.strata[i], self.scenes[i]), []).append(i)
        keys = list(groups)
        rng.shuffle(keys)
        for k in keys:
            rng.shuffle(groups[k])
        out: List[int] = []
        for r in range(max((len(g) for g in groups.values()), default=0)):
            for k in keys:
                if r < len(groups[k]):
                    out.append(groups[k][r])
        return out


def collate_cocf_samples(batch: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    """Collate stored counterfactual payload dicts into a batch of stacked tensors.

    Missing keys default to zeros; the variable-length text token sequence is
    right-padded to the batch max with a companion ``text_mask``.
    """
    out: Dict[str, object] = {}
    n = len(batch)

    for key in _VECTOR_FIELDS:
        vals = [_as_tensor(b.get(key)) for b in batch]
        ref = next((v for v in vals if v is not None and v.numel()), None)
        if ref is None:
            continue
        filled = [v if (v is not None and v.numel()) else torch.zeros_like(ref) for v in vals]
        out[key] = torch.stack(filled).float()

    for key in _SCALAR_LONG_FIELDS:
        out[key] = torch.tensor([int(b.get(key, 0) or 0) for b in batch], dtype=torch.long)
    for key in _SCALAR_FLOAT_FIELDS:
        out[key] = torch.tensor([float(b.get(key, 0.0) or 0.0) for b in batch], dtype=torch.float32)
    for key in _STRING_FIELDS:
        out[key] = [str(b.get(key, "")) for b in batch]

    # text token sequence: pad [L_i, d_c] -> [B, L_max, d_c] with a [B, L_max] mask.
    # A sample missing its per-clip embedding gets an all-zero row and mask.
    text = [_as_tensor(b.get("text_embed")) for b in batch]
    text = [t if (t is not None and t.dim() == 2) else None for t in text]
    out["text_missing"] = sum(1 for t in text if t is None)
    ref = next((t for t in text if t is not None), None)
    if ref is not None and n > 0:
        d_c = ref.shape[-1]
        l_max = max(t.shape[0] for t in text if t is not None)
        padded = torch.zeros(n, l_max, d_c)
        mask = torch.zeros(n, l_max)
        for i, t in enumerate(text):
            if t is None:
                continue
            li = t.shape[0]
            padded[i, :li] = t.float()
            mask[i, :li] = 1.0
        out["text_embed"] = padded
        out["text_mask"] = mask
    return out


def _as_tensor(x: object) -> Optional[Tensor]:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    try:
        return torch.as_tensor(x)
    except Exception:
        return None
