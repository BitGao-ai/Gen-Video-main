"""Counterfactual sample store — the §3 level-5 LMDB training main library.

The design doc stores every counterfactual training sample in an LMDB keyed by
``sample_id`` (§3 "LMDB 训练主库") to dodge the海量小文件 IO bottleneck and give
the ~5–10× random-read throughput Stage B's batch-random sampling needs (§3, §4.1).

This module provides that store behind one interface, with two interchangeable
backends chosen automatically:

    * **LMDB** when the ``lmdb`` package is importable (the production path; writes
      the ``data.mdb`` / ``lock.mdb`` the §3 diagram names).
    * **sharded ``.pt``** fallback otherwise — records are batched into
      ``shard_XXXXX.pt`` files with a ``manifest.json`` index, preserving the same
      "few large files, random access by key" property so the framework still runs
      where LMDB is not installed (e.g. this CPU box) with **no interface change**.

The store is deliberately **schema-agnostic**: it serialises whatever ``to_dict``
payload it is handed (a :class:`~cocf.lcocf.data.COCFTrainingSample` duck-types via
``.to_dict()``) and returns plain dicts on read. Reconstruction into a typed sample
is the consumer's job (``COCFTrainingSample.from_dict``), which keeps this module
free of any dependency on the L-COCF package — no import cycle.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset

from cocf.common.logging import get_logger

_log = get_logger(__name__)

# 32 GiB virtual map by default — LMDB only commits pages actually written, so an
# over-estimate is free on disk and avoids MDB_MAP_FULL on large dataset builds.
_DEFAULT_MAP_SIZE = 32 * 1024 ** 3


def _have_lmdb() -> bool:
    try:
        import lmdb  # noqa: F401
        return True
    except Exception:
        return False


def _encode(payload: Any) -> bytes:
    buf = io.BytesIO()
    torch.save(payload, buf)
    return buf.getvalue()


def _decode(blob: bytes) -> Any:
    return torch.load(io.BytesIO(blob), map_location="cpu", weights_only=False)


def _as_payload(sample: Any) -> Dict[str, Any]:
    """Accept a typed sample (``.to_dict()``) or an already-plain dict."""
    return sample.to_dict() if hasattr(sample, "to_dict") else dict(sample)


# --------------------------------------------------------------------------- #
# Writer
# --------------------------------------------------------------------------- #


class CounterfactualSampleWriter:
    """Writes counterfactual samples into the §3 level-5 store, keyed by ``sample_id``.

    Use as a context manager so the backend is flushed/closed deterministically::

        with CounterfactualSampleWriter(layout.lmdb_dir, shard_size=256) as w:
            for sid, sample in ...:
                w.put(sid, sample)
    """

    def __init__(self, lmdb_dir, *, shard_size: int = 256, map_size: int = _DEFAULT_MAP_SIZE):
        self.dir = Path(lmdb_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = max(1, int(shard_size))
        self._use_lmdb = _have_lmdb()
        self._keys: List[str] = []
        if self._use_lmdb:
            import lmdb

            self._env = lmdb.open(str(self.dir), map_size=int(map_size), subdir=True)
            self._txn = self._env.begin(write=True)
            self._pending = 0
        else:
            # sharded fallback: accumulate in a buffer, flush every shard_size
            self._shard_idx = 0
            self._buffer: List[Dict[str, Any]] = []
            self._manifest: Dict[str, List[int]] = {}  # sample_id -> [shard_idx, pos]

    # -- writing -------------------------------------------------------- #

    def put(self, sample_id: str, sample: Any) -> None:
        sample_id = str(sample_id)
        payload = _as_payload(sample)
        self._keys.append(sample_id)
        if self._use_lmdb:
            self._txn.put(sample_id.encode("utf-8"), _encode(payload))
            self._pending += 1
            if self._pending >= 1000:  # commit periodically to bound txn memory
                self._txn.commit()
                self._txn = self._env.begin(write=True)
                self._pending = 0
        else:
            self._manifest[sample_id] = [self._shard_idx, len(self._buffer)]
            self._buffer.append({"sample_id": sample_id, "payload": payload})
            if len(self._buffer) >= self.shard_size:
                self._flush_shard()

    def __len__(self) -> int:
        return len(self._keys)

    # -- lifecycle ------------------------------------------------------ #

    def _flush_shard(self) -> None:
        if not self._buffer:
            return
        shard_path = self.dir / f"shard_{self._shard_idx:05d}.pt"
        torch.save(self._buffer, shard_path)
        self._shard_idx += 1
        self._buffer = []

    def close(self) -> None:
        if self._use_lmdb:
            self._txn.commit()
            # persist the ordered key list so the dataset need not enumerate the env
            with self._env.begin(write=True) as txn:
                txn.put(b"__keys__", json.dumps(self._keys).encode("utf-8"))
            self._env.sync()
            self._env.close()
        else:
            self._flush_shard()
            (self.dir / "manifest.json").write_text(
                json.dumps({"keys": self._keys, "index": self._manifest}),
                encoding="utf-8",
            )

    def __enter__(self) -> "CounterfactualSampleWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# --------------------------------------------------------------------------- #
# Dataset (reader)
# --------------------------------------------------------------------------- #


class CounterfactualLMDBDataset(Dataset):
    """Reads counterfactual samples back from the §3 level-5 store.

    Returns the stored **payload dict** (not a typed sample) so this reader carries
    no L-COCF dependency; the Stage-B collate reconstructs / tensorises it. The
    optional ``sample_ids`` restricts the dataset to a split (the §3 level-6
    ``splits/`` lists), which is how Stage B reads only the training samples.
    """

    def __init__(self, lmdb_dir, sample_ids: Optional[Sequence[str]] = None) -> None:
        self.dir = Path(lmdb_dir)
        self._use_lmdb = _have_lmdb() and (self.dir / "data.mdb").exists()
        if self._use_lmdb:
            self._open_lmdb()
        else:
            self._open_fallback()
        # restrict to a requested subset (e.g. a split), preserving its order
        if sample_ids is not None:
            wanted = [s for s in sample_ids if s in self._key_set]
            missing = len(sample_ids) - len(wanted)
            if missing:
                _log.warning("CounterfactualLMDBDataset: %d requested ids not in store", missing)
            self.keys = wanted
        else:
            self.keys = list(self._all_keys)
        if not self.keys:
            _log.warning("CounterfactualLMDBDataset is empty at %s", self.dir)

    # -- backends ------------------------------------------------------- #

    def _open_lmdb(self) -> None:
        import lmdb

        self._env = lmdb.open(str(self.dir), readonly=True, lock=False, subdir=True)
        with self._env.begin() as txn:
            raw = txn.get(b"__keys__")
            if raw is not None:
                self._all_keys = json.loads(raw.decode("utf-8"))
            else:  # no key list written → enumerate (skip the meta key)
                self._all_keys = [
                    k.decode("utf-8") for k, _ in txn.cursor() if k != b"__keys__"
                ]
        self._key_set = set(self._all_keys)
        self._shard_cache: Dict[int, List[Dict[str, Any]]] = {}

    def _open_fallback(self) -> None:
        manifest_path = self.dir / "manifest.json"
        if not manifest_path.exists():
            self._all_keys, self._index = [], {}
        else:
            m = json.loads(manifest_path.read_text(encoding="utf-8"))
            self._all_keys, self._index = m["keys"], m["index"]
        self._key_set = set(self._all_keys)
        self._shard_cache = {}  # tiny LRU(1) on the most-recently-read shard

    # -- protocol ------------------------------------------------------- #

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.get(self.keys[index])

    def get(self, sample_id: str) -> Dict[str, Any]:
        if self._use_lmdb:
            with self._env.begin() as txn:
                blob = txn.get(sample_id.encode("utf-8"))
            if blob is None:
                raise KeyError(sample_id)
            return _decode(blob)
        # fallback: load (and cache) the shard, return the record's payload
        shard_idx, pos = self._index[sample_id]
        shard = self._shard_cache.get(shard_idx)
        if shard is None:
            shard = torch.load(self.dir / f"shard_{shard_idx:05d}.pt", weights_only=False)
            self._shard_cache = {shard_idx: shard}  # keep only the last shard resident
        return shard[pos]["payload"]
