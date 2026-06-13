"""Stage A: Offline counterfactual teacher data generation (§7.1.1).

This stage runs once on the full backbone to create a high-quality counterfactual
training dataset. It is completely independent of the training loop (no gradients,
no optimizer state), so it can be run offline and cached for all future training runs.

The output is a :class:`LatentCacheDataset` (or equivalent HDF5/Parquet) with
:class:`COCFTrainingSample` records, indexed by (prompt, scene, timestep, tube_id).

This stage is embarrassingly parallel: different videos can be processed on
different devices simultaneously.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn

from cocf.backbones.base import BackboneAdapter
from cocf.common.config import Config, DataConfig
from cocf.common.logging import get_logger
from cocf.common.memory import free_memory, teacher_forward
from cocf.data import VideoTextDataset
from cocf.lcocf.damage import MetricExtractor
from cocf.lcocf.data import (
    COCFDataGenerator,
    CounterfactualDamageComputer,
    StratifiedSamplingConfig,
)
from cocf.lcocf.strength import CausalStrengthFeatureBuilder

Tensor = torch.Tensor
_log = get_logger(__name__)


@dataclass
class StageAConfig:
    """Hyperparameters for Stage A data generation."""

    # Dataset
    manifest_path: Path  # CSV with video paths & captions
    output_dir: Path  # Where to write counterfactual samples
    num_workers: int = 4  # Parallel video processing

    # Generation parameters
    batch_size: int = 1  # Per-video batch (increase for multi-GPU)
    max_prompts: Optional[int] = None  # Limit dataset size for debugging
    samples_per_video: int = 30  # Cap on labels per video

    # Data parameters (copy from sampling_config)
    interpolation_interval: int = 5
    use_label_interpolation: bool = True

    # Device/mixed precision
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype: torch.dtype = torch.float32


class DataGenerationStage:
    """Stage A: Generate counterfactual training data (§7.1.1).

    Invoked once per training pipeline initialization. Output can be cached and
    reused across multiple Stage B/C runs.

    Workflow:
        1. Load video dataset from manifest
        2. For each video:
           a. Run full-compute backbone to get Y_full, z_t trajectory
           b. Detect semantic tubes G_t at representative timesteps
           c. For each sampled (tube, timestep, action):
              - Apply action (counterfactual intervention)
              - Decode → Y_cf
              - Compute damage D(Y_full, Y_cf)
           d. Store samples with metadata
        3. Write to disk with caching optimizations
    """

    def __init__(
        self,
        config: StageAConfig,
        backbone: BackboneAdapter,
        metric_extractor: MetricExtractor,
        accelerator,  # For accessing STA tube builder
    ) -> None:
        self.config = config
        self.backbone = backbone
        self.metric_extractor = metric_extractor
        self.accelerator = accelerator

        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Build data generator
        self.damage_computer = CounterfactualDamageComputer(metric_extractor)
        # CausalStrengthFeatureBuilder is stateless and takes no constructor args.
        self.strength_builder = CausalStrengthFeatureBuilder()

        sampling_cfg = StratifiedSamplingConfig(
            interpolation_interval=config.interpolation_interval,
            use_label_interpolation=config.use_label_interpolation,
        )

        self.data_generator = COCFDataGenerator(
            metric_extractor=metric_extractor,
            strength_feature_builder=self.strength_builder,
            damage_computer=self.damage_computer,
            sampling_config=sampling_cfg,
            device=config.device,
        )

        # Stage A emits COCFTrainingSample records (damage labels), not VAE latents.
        # These MUST live in their own directory: the latent cache (output_dir/"cache")
        # is read by Stage B's LatentCacheDataset expecting LatentRecord files, so
        # writing damage-label dicts there would silently collide with two
        # incompatible schemas in one directory.
        self.samples_dir = config.output_dir / "cocf_samples"
        self.samples_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> Path:
        """Execute Stage A and return the output dataset path."""
        _log.info("=== Stage A: Counterfactual Teacher Data Generation ===")

        # Load dataset. VideoTextDataset takes a DataConfig (its ``meta_file`` is the
        # manifest path); ``manifest_path`` / ``num_workers`` were never valid
        # constructor kwargs, so the old call raised TypeError on entry.
        #
        # NOTE (ambiguous, intentionally not auto-resolved): frame-count, resolution
        # bucket and normalisation params here fall back to DataConfig defaults. If
        # Stage A must mirror the main run's sampling exactly, thread the full
        # ``Config.data`` into StageAConfig rather than rebuilding a bare DataConfig.
        data_cfg = DataConfig(
            meta_file=str(self.config.manifest_path),
            num_workers=self.config.num_workers,
        )
        dataset = VideoTextDataset(data_cfg)

        if self.config.max_prompts:
            # The dataset keeps its manifest rows on ``.metas`` (there is no
            # ``.samples`` attribute — that would raise AttributeError).
            dataset.metas = dataset.metas[: self.config.max_prompts]

        total_samples = 0

        # Process each video
        for idx in range(0, len(dataset), self.config.batch_size):
            batch_end = min(idx + self.config.batch_size, len(dataset))
            # Fetch each sample once. Indexing the dataset decodes and frame-samples
            # the clip, so the previous double-index (``dataset[i].video`` then
            # ``dataset[i].caption``) decoded every video twice — wasted I/O and 2×
            # the transient clip memory.
            batch_samples = [dataset[i] for i in range(idx, batch_end)]
            batch_videos = [s.video for s in batch_samples]
            batch_prompts = [s.caption for s in batch_samples]

            _log.info(f"Processing batch {idx//self.config.batch_size + 1}/{(len(dataset)-1)//self.config.batch_size + 1}")

            # Run full-compute backbone (collect z_t, tubes, features). This stage is
            # label-only: ``teacher_forward`` (no_grad + inference_mode) guarantees no
            # autograd graph is ever built, which is both correct and the cheapest
            # path on VRAM for the full backbone pass.
            with teacher_forward():
                z_trajectory, tubes_by_step, strength_features, video_full = (
                    self._run_full_backbone(batch_videos, batch_prompts)
                )

            # Generate counterfactual samples
            samples = self.data_generator.generate_batch(
                video_full=video_full,
                prompts=batch_prompts,
                tubes_by_step=tubes_by_step,
                z_trajectory=z_trajectory,
                strength_features_by_step=strength_features,
                backbone_adapter=self.backbone,
                num_total_steps=self.config.batch_size * 50,  # Placeholder
                max_samples_per_video=self.config.samples_per_video,
            )

            # Write samples to disk (one .pt per COCFTrainingSample)
            for sample in samples:
                key = (
                    f"{idx:06d}_{sample.tube_id:04d}"
                    f"_{sample.timestep:03d}_{sample.action:01d}"
                )
                torch.save(sample.to_dict(), self.samples_dir / f"{key}.pt")

            total_samples += len(samples)
            _log.info(f"  Generated {len(samples)} counterfactual samples")

            # Stage A processes one video batch independently of every other, so the
            # full-compute intermediates (the decoded clip, the cached z_t trajectory
            # and per-step features) need not outlive the batch. Drop them and the
            # cached allocator blocks so peak VRAM stays at a single batch's footprint
            # rather than growing with the dataset.
            del batch_samples, batch_videos, batch_prompts, samples
            del z_trajectory, tubes_by_step, strength_features, video_full
            free_memory()

        _log.info(f"Stage A complete: {total_samples} total training samples")
        return self.config.output_dir

    def _run_full_backbone(
        self, batch_videos: List[Tensor], prompts: List[str]
    ) -> tuple:
        """Run full-compute backbone and collect trajectory data.

        Returns:
            (z_trajectory, tubes_by_step, strength_features, video_full)
        """
        # This is a stub; real impl would:
        #   1. Encode videos to latent space
        #   2. Run full denoising loop, caching z_t at each step
        #   3. Extract tubes at representative timesteps
        #   4. Compute strength features
        #   5. Decode latent to video
        video_full = torch.zeros(1, 1, 3, 512, 512)  # Stub
        z_trajectory = {}
        tubes_by_step = {}
        strength_features = {}
        return z_trajectory, tubes_by_step, strength_features, video_full
