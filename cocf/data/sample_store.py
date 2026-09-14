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
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from cocf.common.logging import get_logger

_log = get_logger(__name__)

# Virtual map reserved for the LMDB store. LMDB only commits pages actually written,
# so an over-estimate costs nothing on disk while an under-estimate aborts the run
# with MDB_MAP_FULL part-way through. Sized for the §2.4 target (180k clips x ~12
# samples x ~7 KB ≈ 15 GiB) with room for a wider CLIP embed or more samples per
# clip; ``CounterfactualSampleWriter(map_size=…)`` overrides it, and the writer grows
# the map on demand anyway (see :meth:`CounterfactualSampleWriter.put`).
_DEFAULT_MAP_SIZE = 256 * 1024 ** 3

# Records held in one LMDB write transaction. Bounds the transaction's memory and,
# with it, how much has to be replayed if the map turns out to need growing.
_COMMIT_EVERY = 1000


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


def _dedup(keys: Sequence[str]) -> List[str]:
    """``keys`` with duplicates removed, first occurrence wins, order preserved."""
    return list(dict.fromkeys(keys))


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

    def __init__(self, lmdb_dir, *, shard_size: int = 256, map_size: int = _DEFAULT_MAP_SIZE,
                 shard_prefix: str = "shard", manifest_name: str = "manifest.json",
                 resume: bool = False, write_manifest: bool = True,
                 force_fallback: bool = False):
        self.dir = Path(lmdb_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = max(1, int(shard_size))
        # Sharded-fallback layout knobs. ``shard_prefix`` / ``manifest_name`` give each
        # parallel Stage-A shard its own on-disk namespace inside one lmdb_dir, so
        # ``--num-shards`` workers never clobber each other's shard files. The merged
        # top-level ``manifest.json`` is built later by the finalize pass, so writers in
        # the pipeline pass ``write_manifest=False`` and let finalize own the index.
        self.shard_prefix = shard_prefix
        self.manifest_name = manifest_name
        self.write_manifest = write_manifest
        # LMDB is single-writer; a sharded parallel run forces the .pt backend so N
        # workers can each append to the shared dir. Non-sharded runs keep auto-select.
        self._use_lmdb = _have_lmdb() and not force_fallback
        self._keys: List[str] = []
        if self._use_lmdb:
            import lmdb

            self._map_size = int(map_size)
            self._env = lmdb.open(str(self.dir), map_size=self._map_size, subdir=True)
            # Read the prior key list *before* opening the write transaction. Both
            # orders work in LMDB (read txns are independent of the writer), but
            # reading first keeps the write txn's lifetime tight and avoids relying on
            # that guarantee. Keys already committed by an earlier (interrupted) run
            # matter because ``close()`` rewrites ``__keys__`` wholesale: without
            # seeding from the store, a resumed run would publish only *this* session's
            # keys and orphan every record written before the interruption — the data
            # is still in data.mdb, but nothing can address it.
            self._prior_keys: List[str] = self._read_keys() if resume else []
            self._txn = self._env.begin(write=True)
            # Records written into the open transaction, retained so a map growth
            # (which has to abort it) can replay them — see :meth:`_put_lmdb`.
            self._pending: List[Tuple[bytes, bytes]] = []
        else:
            # sharded fallback: accumulate in a buffer, flush every shard_size. When
            # ``resume`` picks up an interrupted shard, continue numbering *after* the
            # highest existing shard so a restart appends rather than overwrites.
            self._shard_idx = self._next_shard_index() if resume else 0
            self._buffer: List[Dict[str, Any]] = []
            self._manifest: Dict[str, List] = {}  # sample_id -> [shard_filename, pos]

    def _shard_name(self, idx: int) -> str:
        return f"{self.shard_prefix}_{idx:05d}.pt"

    def _read_keys(self) -> List[str]:
        """Ordered key list already stored in the LMDB env (``[]`` when absent)."""
        try:
            with self._env.begin() as txn:
                raw = txn.get(b"__keys__")
                if raw is not None:
                    return list(json.loads(raw.decode("utf-8")))
                # No ``__keys__`` yet (crash before the first close): enumerate.
                return [k.decode("utf-8") for k, _ in txn.cursor() if k != b"__keys__"]
        except Exception:  # pragma: no cover — unreadable/fresh env
            return []

    def _next_shard_index(self) -> int:
        """Highest existing ``{prefix}_NNNNN.pt`` + 1 (0 when none) — the resume anchor."""
        n = 0
        for p in self.dir.glob(f"{self.shard_prefix}_*.pt"):
            try:
                n = max(n, int(p.stem.rsplit("_", 1)[1]) + 1)
            except (ValueError, IndexError):
                continue
        return n

    # -- writing -------------------------------------------------------- #

    def put(self, sample_id: str, sample: Any) -> None:
        sample_id = str(sample_id)
        payload = _as_payload(sample)
        self._keys.append(sample_id)
        if self._use_lmdb:
            self._put_lmdb(sample_id, _encode(payload))
        else:
            self._manifest[sample_id] = [self._shard_name(self._shard_idx), len(self._buffer)]
            self._buffer.append({"sample_id": sample_id, "payload": payload})
            if len(self._buffer) >= self.shard_size:
                self._flush_shard()

    def _put_lmdb(self, sample_id: str, blob: bytes) -> None:
        """Write one record, growing the map rather than dying on MDB_MAP_FULL.

        The map is a *virtual* reservation, so outgrowing it is a recoverable
        bookkeeping fact, not a full disk — but LMDB reports it as an exception from
        the middle of a days-long generation run, which the caller then counts as a
        failed clip.

        Growing means aborting the open transaction, which discards every record it
        held — so the uncommitted batch is kept and replayed. Without that, up to
        ``_COMMIT_EVERY`` samples would be listed in ``__keys__`` without existing in
        the store, which reads as a silent data loss rather than as an error.
        """
        import lmdb

        record = (sample_id.encode("utf-8"), blob)
        try:
            self._txn.put(*record)
            self._pending.append(record)
        except lmdb.MapFullError:
            self._txn.abort()
            self._map_size *= 2
            _log.warning("LMDB map full; growing the reservation to %.0f GiB and "
                         "replaying %d uncommitted record(s)",
                         self._map_size / 1024 ** 3, len(self._pending))
            self._env.set_mapsize(self._map_size)
            self._txn = self._env.begin(write=True)
            replay, self._pending = self._pending, []
            for r in replay + [record]:
                self._txn.put(*r)
                self._pending.append(r)
        if len(self._pending) >= _COMMIT_EVERY:  # bound the transaction's memory
            self._commit_lmdb()

    def _commit_lmdb(self) -> None:
        self._txn.commit()
        self._txn = self._env.begin(write=True)
        self._pending = []

    def __len__(self) -> int:
        return len(self._keys)

    # -- lifecycle ------------------------------------------------------ #

    def _flush_shard(self) -> None:
        if not self._buffer:
            return
        shard_path = self.dir / self._shard_name(self._shard_idx)
        temporary = shard_path.with_suffix(".pt.tmp")
        with temporary.open("wb") as stream:
            torch.save(self._buffer, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, shard_path)
        self._shard_idx += 1
        self._buffer = []

    def flush(self) -> None:
        """Commit all samples before the caller publishes durable clip progress."""
        if self._use_lmdb:
            merged = _dedup(list(self._prior_keys) + self._keys)
            self._txn.put(b"__keys__", json.dumps(merged).encode("utf-8"))
            self._commit_lmdb()
            self._env.sync()
        else:
            self._flush_shard()
            if os.name == "posix":
                fd = os.open(self.dir, os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)

    def close(self) -> None:
        if self._use_lmdb:
            self._txn.commit()
            # Persist the ordered key list so the dataset need not enumerate the env.
            # Merge with the keys a previous run committed (``resume``), de-duplicating
            # while preserving order — a plain overwrite silently orphans them.
            merged = _dedup(list(self._prior_keys) + self._keys)
            with self._env.begin(write=True) as txn:
                txn.put(b"__keys__", json.dumps(merged).encode("utf-8"))
            self._env.sync()
            self._env.close()
        else:
            self._flush_shard()
            if self.write_manifest:
                # Same de-duplication as the LMDB branch. ``_manifest`` is a dict and
                # already keeps only the last write per id, but ``_keys`` is a list —
                # publishing it raw makes a re-``put`` id appear twice in every index
                # built from it, and the sharded (multi-worker) layout is exactly the
                # one that takes this branch.
                (self.dir / self.manifest_name).write_text(
                    json.dumps({"keys": _dedup(self._keys), "index": self._manifest}),
                    encoding="utf-8",
                )

    def __enter__(self) -> "CounterfactualSampleWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# --------------------------------------------------------------------------- #
# Backend introspection + streaming read (used by the §1.6 finalize pass)
# --------------------------------------------------------------------------- #


def store_is_lmdb(lmdb_dir) -> bool:
    """True when ``lmdb_dir`` holds a *readable* LMDB store.

    Finalize and any other whole-store pass must branch on what is actually on disk,
    not on which backend they would pick themselves — the writer auto-selects LMDB
    whenever the package is importable, so a reader that only globs ``shard_*.pt``
    finds nothing and silently concludes the store is empty (which is exactly how
    installing ``lmdb`` used to break Stage A → Stage B).
    """
    return _have_lmdb() and (Path(lmdb_dir) / "data.mdb").exists()


def iter_lmdb_records(lmdb_dir):
    """Stream ``(sample_id, payload)`` over an LMDB store, one record resident.

    Honours the stored ``__keys__`` order when present, else enumerates the env.
    """
    import lmdb

    env = lmdb.open(str(lmdb_dir), readonly=True, lock=False, subdir=True)
    try:
        with env.begin() as txn:
            raw = txn.get(b"__keys__")
            keys = (
                [k.encode("utf-8") for k in json.loads(raw.decode("utf-8"))]
                if raw is not None
                else [k for k, _ in txn.cursor() if k != b"__keys__"]
            )
            for k in keys:
                blob = txn.get(k)
                if blob is None:
                    _log.warning("iter_lmdb_records: key %r listed but absent", k)
                    continue
                yield k.decode("utf-8"), _decode(blob)
    finally:
        env.close()


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

    def __init__(self, lmdb_dir, sample_ids: Optional[Sequence[str]] = None,
                 text_embed_dir=None, shard_cache_size: int = 8) -> None:
        self.dir = Path(lmdb_dir)
        # Per-clip prompt embeddings live outside the sample store (§P2-3); when a
        # directory is given, each record is joined with its video's embedding on
        # read so consumers still see a self-contained payload.
        self.text_embed_dir = Path(text_embed_dir) if text_embed_dir else None
        self._text_cache: "OrderedDict[str, Any]" = OrderedDict()
        # Bounded LRU over decoded .pt shards. A single slot thrashed badly: the
        # stratified sampler draws a batch from all over the store, so consecutive
        # reads almost always landed in different shards and each one reloaded a
        # whole shard (hundreds of samples) to serve one record.
        self._shard_cache_size = max(1, int(shard_cache_size))
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
        self._env = None
        self._env_pid = None
        with self._lmdb_env().begin() as txn:
            raw = txn.get(b"__keys__")
            if raw is not None:
                self._all_keys = json.loads(raw.decode("utf-8"))
            else:  # no key list written → enumerate (skip the meta key)
                self._all_keys = [
                    k.decode("utf-8") for k, _ in txn.cursor() if k != b"__keys__"
                ]
        self._key_set = set(self._all_keys)
        self._shard_cache: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()

    def _lmdb_env(self):
        """This process's read env, opened lazily and re-opened after a fork.

        A ``DataLoader`` with ``num_workers > 0`` forks after the dataset is built, and
        an LMDB environment must not be shared across that boundary — the workers would
        inherit the parent's reader slots. Keying the handle on the pid gives each
        worker its own, which is what makes ``--num_workers`` usable at all: Stage B is
        IO-bound on this store, so single-process reads are its actual bottleneck.
        """
        import lmdb

        pid = os.getpid()
        if self._env is None or self._env_pid != pid:
            self._env = lmdb.open(str(self.dir), readonly=True, lock=False,
                                  subdir=True, max_readers=512)
            self._env_pid = pid
        return self._env

    def _open_fallback(self) -> None:
        manifest_path = self.dir / "manifest.json"
        if not manifest_path.exists():
            self._all_keys, self._index = [], {}
        else:
            m = json.loads(manifest_path.read_text(encoding="utf-8"))
            self._all_keys, self._index = m["keys"], m["index"]
        self._key_set = set(self._all_keys)
        self._shard_cache = OrderedDict()

    # -- protocol ------------------------------------------------------- #

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.get(self.keys[index])

    def get(self, sample_id: str) -> Dict[str, Any]:
        if self._use_lmdb:
            with self._lmdb_env().begin() as txn:
                blob = txn.get(sample_id.encode("utf-8"))
            if blob is None:
                raise KeyError(sample_id)
            return self._with_text_embed(_decode(blob))
        # fallback: load (and cache) the shard, return the record's payload. The index
        # value is either a shard *filename* (new shard-parallel layout) or a legacy
        # integer shard index — accept both so pre-existing stores keep reading unchanged.
        shard_ref, pos = self._index[sample_id]
        shard_name = shard_ref if isinstance(shard_ref, str) else f"shard_{int(shard_ref):05d}.pt"
        shard = self._shard_cache.get(shard_name)
        if shard is None:
            shard = torch.load(self.dir / shard_name, weights_only=False)
            self._shard_cache[shard_name] = shard
            while len(self._shard_cache) > self._shard_cache_size:
                self._shard_cache.popitem(last=False)   # evict least-recently-used
        else:
            self._shard_cache.move_to_end(shard_name)
        return self._with_text_embed(shard[pos]["payload"])

    def _with_text_embed(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Attach the clip's prompt embedding, loaded once per video (§P2-3)."""
        if self.text_embed_dir is None or "text_embed" in payload:
            return payload
        vid = str(payload.get("video_id", ""))
        if not vid:
            return payload
        emb = self._text_cache.get(vid)
        if emb is None:
            # Same normalisation Stage A wrote with, so an int-like id resolves.
            from cocf.data.processed_layout import video_id_str

            path = self.text_embed_dir / f"{video_id_str(vid)}.pt"
            if not path.exists():
                return payload
            emb = torch.load(path, map_location="cpu", weights_only=False).float()
            self._text_cache[vid] = emb
            while len(self._text_cache) > self._shard_cache_size:
                self._text_cache.popitem(last=False)
        else:
            self._text_cache.move_to_end(vid)
        out = dict(payload)
        out["text_embed"] = emb
        return out
