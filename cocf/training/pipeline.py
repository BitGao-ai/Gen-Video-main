"""Unified training pipeline that orchestrates all three stages.

:class:`TrainingPipeline` runs the full process from data generation through
fine-tuning. Each stage is optional (skipped if its data/checkpoint exists) and
intermediate results are cached.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch

from cocf.common.config import Config
from cocf.common.logging import get_logger
from cocf.core.accelerator import Accelerator
from cocf.engine import InferenceEngine
from cocf.training.checkpoint import load_checkpoint, build_checkpoint
from cocf.training.distributed import resolve_device
from cocf.training.stage_a_data_gen import DataGenerationStage, StageAConfig
from cocf.training.stage_b_joint import JointTrainingStage, StageBConfig
from cocf.training.stage_c_finetune import FinettuneStage, StageCConfig

_log = get_logger(__name__)


@dataclass
class PipelineConfig:
    """Top-level configuration for the three-stage training pipeline.

    One ``processed_root`` is threaded through all three stages so the store written by
    Stage A is exactly what Stages B/C consume.
    """

    # Shared
    experiment_dir: Path
    checkpoint_load_path: Optional[Path] = None  # Resume from checkpoint
    seed: int = 42

    # Stage-A data inputs: OpenVid metadata CSV(s) and where the clips live.
    openvid_csvs: List[Path] = field(default_factory=list)
    data_root: str = ""
    # Shared processed-store root; defaults under experiment_dir when unset.
    processed_root: Optional[Path] = None

    # Individual stage configs (optional pre-built overrides; built lazily otherwise)
    stage_a: Optional[StageAConfig] = None
    stage_b: Optional[StageBConfig] = None
    stage_c: Optional[StageCConfig] = None

    def to_dict(self):
        return {
            "experiment_dir": str(self.experiment_dir),
            "checkpoint_load_path": str(self.checkpoint_load_path) if self.checkpoint_load_path else None,
            "seed": self.seed,
            "openvid_csvs": [str(p) for p in self.openvid_csvs],
            "data_root": self.data_root,
            "processed_root": str(self.processed_root) if self.processed_root else None,
        }

    @classmethod
    def from_yaml(cls, path: Path) -> PipelineConfig:
        """Load config from YAML file."""
        # Imported lazily so ``Config.load``'s JSON fallback works without PyYAML.
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(
            experiment_dir=Path(data.get("experiment_dir", "./experiments")),
            checkpoint_load_path=Path(data.get("checkpoint_load_path")) if data.get("checkpoint_load_path") else None,
            seed=data.get("seed", 42),
            openvid_csvs=[Path(p) for p in data.get("openvid_csvs", [])],
            data_root=data.get("data_root", ""),
            processed_root=Path(data["processed_root"]) if data.get("processed_root") else None,
        )


class TrainingPipeline:
    """Orchestrates the three-stage training workflow."""

    def __init__(
        self,
        config: Config,  # Full COCF config (backbones, modules, etc.)
        pipeline_cfg: PipelineConfig,
    ) -> None:
        self.config = config
        self.pipeline_cfg = pipeline_cfg
        # Device lives on the backbone sub-config.
        selected = resolve_device(config.backbone.device)
        if selected.startswith("cuda") and not torch.cuda.is_available():
            _log.warning("CUDA unavailable; pipeline using CPU")
            selected = "cpu"
        config.backbone.device = selected
        self.device = torch.device(selected)

        # Create experiment directory
        self.pipeline_cfg.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Build accelerator
        self.accelerator = Accelerator.from_config(config)

        # Load checkpoint if specified; re-attaches any LoRA adapters it carries.
        if pipeline_cfg.checkpoint_load_path:
            _log.info(f"Loading checkpoint from {pipeline_cfg.checkpoint_load_path}")
            ckpt = torch.load(
                pipeline_cfg.checkpoint_load_path, map_location=self.device,
                weights_only=False,
            )
            load_checkpoint(self.accelerator, ckpt, training_config=config.training)

        # Build engine. The trigger config is a top-level node on Config.
        self.engine = InferenceEngine(
            self.accelerator,
            config.engine,
            config.trigger,
        )

        # Stage instances (lazy-created)
        self._stage_a: Optional[DataGenerationStage] = None
        self._stage_b: Optional[JointTrainingStage] = None
        self._stage_c: Optional[FinettuneStage] = None

    def run(self, stages: List[str] = ["A", "B", "C"]) -> Accelerator:
        """Run the training pipeline for the given stages, in order.

        Returns the trained accelerator, ready for inference.
        """
        _log.info("=== COCF-SS-DCA Training Pipeline ===")
        _log.info(f"Running stages: {', '.join(stages)}")

        for stage in stages:
            if stage.upper() == "A":
                self._run_stage_a()
            elif stage.upper() == "B":
                self._run_stage_b()
            elif stage.upper() == "C":
                self._run_stage_c()
            else:
                _log.warning(f"Unknown stage: {stage}")

            # Save checkpoint after each stage
            self._save_checkpoint(stage)

        _log.info("=== Training Pipeline Complete ===")
        return self.accelerator

    def _processed_root(self) -> Path:
        """The shared six-level store root: A writes it, B/C read it."""
        return (
            self.pipeline_cfg.processed_root
            or self.pipeline_cfg.experiment_dir / "LCOCF_OpenVid1M_Processed"
        )

    def _run_stage_a(self) -> None:
        """Run Stage A: counterfactual teacher data generation."""
        _log.info("\n--- Stage A: Data Generation ---")

        if self.pipeline_cfg.stage_a is None:
            if not self.pipeline_cfg.openvid_csvs:
                raise RuntimeError(
                    "Stage A needs OpenVid CSV(s): set PipelineConfig.openvid_csvs "
                    "(or pass a pre-built PipelineConfig.stage_a)."
                )
            self.pipeline_cfg.stage_a = StageAConfig(
                openvid_csvs=list(self.pipeline_cfg.openvid_csvs),
                processed_root=self._processed_root(),
                data_root=self.pipeline_cfg.data_root,
                config=self.config,
                device=self.device,
                seed=self.pipeline_cfg.seed,
            )

        # The metric extractor is owned by the accelerator (mock by default).
        self._stage_a = DataGenerationStage(
            config=self.pipeline_cfg.stage_a,
            backbone=self.accelerator.backbone,
            metric_extractor=self.accelerator.metric_extractor,
            accelerator=self.accelerator,
        )
        self._stage_a.run()

    def _run_stage_b(self) -> None:
        """Run Stage B: joint module training."""
        _log.info("\n--- Stage B: Joint Training ---")

        if self.pipeline_cfg.stage_b is None:
            self.pipeline_cfg.stage_b = StageBConfig(
                processed_root=self._processed_root(),
                config=self.config,
                device=self.device,
                checkpoint_dir=self.pipeline_cfg.experiment_dir / "stage_b",
            )

        self._stage_b = JointTrainingStage(
            accelerator=self.accelerator,
            config=self.pipeline_cfg.stage_b,
        )
        self.accelerator = self._stage_b.run()

    def _run_stage_c(self) -> None:
        """Run Stage C: end-to-end lightweight fine-tuning."""
        _log.info("\n--- Stage C: Fine-tuning ---")

        if self.pipeline_cfg.stage_c is None:
            self.pipeline_cfg.stage_c = StageCConfig(
                processed_root=self._processed_root(),
                config=self.config,
                device=self.device,
            )

        self._stage_c = FinettuneStage(
            accelerator=self.accelerator,
            engine=self.engine,
            config=self.pipeline_cfg.stage_c,
        )
        self.accelerator = self._stage_c.run()

    def _save_checkpoint(self, stage: str) -> None:
        """Save the checkpoint for a completed stage.

        After Stage C this goes through :meth:`FinettuneStage.checkpoint` so the LoRA
        adapters inside the frozen backbone are not discarded.
        """
        ckpt_path = self.pipeline_cfg.experiment_dir / f"checkpoint_after_stage_{stage}.pt"
        if stage.upper() == "C" and self._stage_c is not None:
            payload = self._stage_c.checkpoint()
            n_lora = len(payload.get("lora", {}))
        else:
            payload = build_checkpoint(self.accelerator)
            n_lora = 0
        torch.save(payload, ckpt_path)
        _log.info("Saved checkpoint to %s (%d LoRA tensors)", ckpt_path, n_lora)

    @classmethod
    def from_config(cls, config_path: Path, pipeline_cfg_path: Optional[Path] = None) -> TrainingPipeline:
        """Create pipeline from config files."""
        config = Config.load(config_path)
        pipeline_cfg = (
            PipelineConfig.from_yaml(pipeline_cfg_path)
            if pipeline_cfg_path
            else PipelineConfig(experiment_dir=Path("./experiments"))
        )
        return cls(config, pipeline_cfg)
