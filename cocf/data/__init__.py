"""Data subsystem — training-data pipeline & quality-metric extraction.

Two responsibilities, both deliberately decoupled from the algorithm code so the
four innovations never depend on a particular dataset format or perception model:

    metrics         the :class:`~cocf.lcocf.damage.MetricExtractor` backends
                    (DINOv2/CLIP/RAFT/OCR) that score video quality — feeding the
                    L-COCF damage labels (§7.1.1) and the CMSC loss (§6). A
                    deterministic mock makes the whole pipeline CPU-testable.
    video_dataset   video+caption reading with HunyuanVideo/Wan2.1-style frame
                    sampling, resolution bucketing and normalisation (§7.1).
    latent_cache    pre-encode videos+prompts → latents+embeds on disk, the
                    dominant training memory saving (user requirement #1): the VAE
                    and text encoder never occupy VRAM during plugin training.

The counterfactual *teacher* generation that turns these into L-COCF training
labels lives in :mod:`cocf.lcocf.data` (it is L-COCF-specific), and consumes a
:class:`MetricExtractor` from here.
"""

from __future__ import annotations

from cocf.data.cocf_batch import (
    StratifiedBatchSampler,
    collate_cocf_samples,
    timestep_stratum,
)
from cocf.data.latent_cache import (
    LatentCacheDataset,
    LatentCacheWriter,
    LatentRecord,
    cache_key,
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
)
from cocf.data.video_dataset import (
    DecordVideoReader,
    SyntheticVideoReader,
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
    "collate_video_samples",
    # latent/text cache (training memory saving)
    "LatentCacheWriter",
    "LatentCacheDataset",
    "LatentRecord",
    "cache_key",
    # OpenVid manifest ingestion & scene stratification (§1.1/§1.2)
    "OpenVidRecord",
    "read_openvid_csv",
    "read_openvid_manifest",
    "write_raw_dataset_index",
    "infer_scene_type",
    "scene_histogram",
    "SCENE_TYPES",
    "DEFAULT_OPENVID_COLUMNS",
    # four-level quality filter (§2)
    "QualityFilter",
    "FilterResult",
    "FilterReport",
    "base_video_id",
    # Stage-C raw-clip source + hard-sample sampling (§4.2)
    "RawFilteredDataset",
    "RawFilteredItem",
    "HardSamplePrioritySampler",
    "collate_raw_filtered",
    "HARD_SCENE_TYPES",
    # six-level processed store (§3)
    "ProcessedLayout",
    "video_id_str",
    # counterfactual LMDB store (§3 level-5)
    "CounterfactualSampleWriter",
    "CounterfactualLMDBDataset",
    # Stage-B stratified batch assembly (§4.1)
    "StratifiedBatchSampler",
    "collate_cocf_samples",
    "timestep_stratum",
]
