"""Data subsystem: training-data pipeline and quality-metric extraction.

Provides the :class:`~cocf.lcocf.damage.MetricExtractor` perception backends
(DINOv2/CLIP/RAFT/OCR, plus a deterministic mock) and video+caption reading with
HunyuanVideo/Wan2.1-style frame sampling, bucketing and normalisation. The
counterfactual teacher generation lives in :mod:`cocf.lcocf.data`.
"""

from __future__ import annotations

from cocf.data.cocf_batch import (
    StratifiedBatchSampler,
    collate_cocf_samples,
    timestep_stratum,
)
from cocf.data.metrics import MockMetricExtractor, ModelMetricExtractor
from cocf.data.openvid_manifest import (
    DEFAULT_OPENVID_COLUMNS,
    SCENE_TYPES,
    OpenVidRecord,
    infer_scene_type,
    read_openvid_csv,
    read_openvid_manifest,
    scene_histogram,
    write_raw_dataset_index,
)
from cocf.data.processed_layout import ProcessedLayout, video_id_str
from cocf.data.quality_filter import (
    FilterReport,
    FilterResult,
    QualityFilter,
    base_video_id,
)
from cocf.data.raw_filtered import (
    HARD_SCENE_TYPES,
    HardSamplePrioritySampler,
    RawFilteredDataset,
    RawFilteredItem,
    collate_raw_filtered,
)
from cocf.data.sample_store import (
    CounterfactualLMDBDataset,
    CounterfactualSampleWriter,
    iter_lmdb_records,
    store_is_lmdb,
)
from cocf.data.video_dataset import (
    DecordVideoReader,
    SyntheticVideoReader,
    TorchvisionVideoReader,
    VideoMeta,
    VideoReader,
    VideoSample,
    VideoTextDataset,
    collate_video_samples,
)

__all__ = [
    # metrics (perception backends)
    "MockMetricExtractor",
    "ModelMetricExtractor",
    # video reading & sampling
    "VideoTextDataset",
    "VideoSample",
    "VideoMeta",
    "VideoReader",
    "DecordVideoReader",
    "SyntheticVideoReader",
    "TorchvisionVideoReader",
    "collate_video_samples",
    # OpenVid manifest ingestion & scene stratification
    "OpenVidRecord",
    "read_openvid_csv",
    "read_openvid_manifest",
    "write_raw_dataset_index",
    "infer_scene_type",
    "scene_histogram",
    "SCENE_TYPES",
    "DEFAULT_OPENVID_COLUMNS",
    # four-level quality filter
    "QualityFilter",
    "FilterResult",
    "FilterReport",
    "base_video_id",
    # Stage-C raw-clip source + hard-sample sampling
    "RawFilteredDataset",
    "RawFilteredItem",
    "HardSamplePrioritySampler",
    "collate_raw_filtered",
    "HARD_SCENE_TYPES",
    # six-level processed store
    "ProcessedLayout",
    "video_id_str",
    # counterfactual LMDB store
    "CounterfactualSampleWriter",
    "CounterfactualLMDBDataset",
    "store_is_lmdb",
    "iter_lmdb_records",
    # Stage-B stratified batch assembly
    "StratifiedBatchSampler",
    "collate_cocf_samples",
    "timestep_stratum",
]
