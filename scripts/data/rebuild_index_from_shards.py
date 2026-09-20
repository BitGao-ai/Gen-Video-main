#!/usr/bin/env python
"""Rebuild Stage-A index from existing shards."""

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
    """Build video to split map from filtered CSV."""
    mapping = {}
    for r in layout.read_csv(layout.filtered_final):
        vid = r.get("video_id")
        if vid:
            mapping[vid] = r.get("split") or "train"
    return mapping


def main():
    """Rebuild index from shards."""
    parser = argparse.ArgumentParser(
        description="Rebuild manifest/splits/sample_index from existing Stage-A shards (§1.6)"
    )
    parser.add_argument(
        "--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT,
        help=f"Six-level processed store root. Defaults to {DEFAULT_PROCESSED_ROOT}.",
    )
    args = parser.parse_args()

    setup_logging(level=logging.INFO)
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
