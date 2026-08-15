#!/usr/bin/env python
"""Rebuild a Stage-A store's index from its shards (§1.6, no teacher forward).

An interrupted Stage-A run leaves the counterfactual shards on disk but WITHOUT the
merged ``manifest.json`` / ``splits`` / ``sample_index.csv`` that Stage B reads through:
the sharded writer only emits its manifest on a clean close, and §1.6 runs after the
generation loop. Stage B then sees an empty index and loads 0 samples even though the
shards physically hold data (exactly the state a killed run leaves behind).

This script scans ``counterfactual_lmdb/shard_*.pt`` and rebuilds those artifacts by
calling the *same* streaming finalize the full pipeline uses — cheaply, on CPU, with no
backbone and O(1) memory in sample count. The per-clip train/val/test split is recovered
from ``metadata/filtered_final.csv`` (each sample inherits its video's split, §1.6
leakage-safe); clips absent there fall back to ``train``.

It is idempotent — safe to re-run — and equally rebuilds a store written by a modern
sharded run (``shard_sNN_*.pt``) or a legacy single-writer run (``shard_NNNNN.pt``).

NOTE: this only un-breaks the *index* so Stage B can read the shards for link testing.
The recovered samples are only as meaningful as the run that produced them — a
mock-backbone run yields mock labels, not trainable data.

Usage:
    python scripts/data/rebuild_index_from_shards.py \
        --processed-root ./LCOCF_OpenVid1M_Processed
"""

import argparse
import logging
from pathlib import Path

from cocf.common.logging import get_logger, setup_logging
from cocf.data.processed_layout import ProcessedLayout
from cocf.data.sample_store import CounterfactualLMDBDataset
from cocf.training.stage_a_data_gen import finalize_processed_store

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_ROOT = REPO_ROOT / "LCOCF_OpenVid1M_Processed"


def _split_map_from_filtered(layout: ProcessedLayout) -> dict:
    """``video_id -> split`` from filtered_final.csv (empty when the CSV is absent)."""
    mapping = {}
    for r in layout.read_csv(layout.filtered_final):
        vid = r.get("video_id")
        if vid:
            mapping[vid] = r.get("split") or "train"
    return mapping


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild manifest/splits/sample_index from existing Stage-A shards (§1.6)"
    )
    parser.add_argument(
        "--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT,
        help=f"Six-level processed store root. Defaults to {DEFAULT_PROCESSED_ROOT}.",
    )
    args = parser.parse_args()

    setup_logging(level=logging.INFO)
    # setup_logging attaches the stdout handler to the "cocf" logger and sets
    # propagate=False, so a bare getLogger("__main__") would emit nothing at
    # INFO — this script's own progress lines included.
    log = get_logger("cocf.rebuild_index")

    layout = ProcessedLayout(args.processed_root)
    if not layout.lmdb_dir.exists():
        parser.error(f"no counterfactual_lmdb under {args.processed_root}")

    split_by_video = _split_map_from_filtered(layout)
    log.info(
        "recovered split for %d videos from %s",
        len(split_by_video),
        layout.filtered_final.name if split_by_video else "(none; defaulting to train)",
    )

    root = finalize_processed_store(layout, split_by_video)

    # Read back through the real dataset reader so the user sees it actually loads now.
    ds = CounterfactualLMDBDataset(layout.lmdb_dir)
    log.info(
        "rebuild complete → %s | manifest indexes %d samples; splits train/val/test = %d/%d/%d",
        root, len(ds.keys),
        len(layout.read_split("train")),
        len(layout.read_split("val")),
        len(layout.read_split("test_hard")),
    )


if __name__ == "__main__":
    main()
