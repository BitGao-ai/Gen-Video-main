"""Integration tests for the full training pipeline."""

import pytest
import tempfile
import torch
from pathlib import Path

from cocf.common.config import Config
from cocf.core.accelerator import Accelerator
from cocf.engine import InferenceEngine


class TestTrainingPipeline:
    """Integration tests for the three-stage training pipeline."""

    @pytest.fixture
    def accelerator(self):
        """Create a mock accelerator for testing."""
        config = Config()
        config.device = torch.device("cpu")
        accelerator = Accelerator.from_config(config)
        return accelerator

    @pytest.fixture
    def engine(self, accelerator):
        """Create inference engine."""
        config = Config()
        engine = InferenceEngine(accelerator, config.engine, config.trigger)
        return engine

    def test_accelerator_checkpoint(self, accelerator):
        """Accelerator should be saveable/loadable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test.pt"

            # Save
            torch.save(accelerator.state_dict(), ckpt_path)
            assert ckpt_path.exists()

            # Load
            new_accelerator = Accelerator.from_config(Config())
            state_dict = torch.load(ckpt_path)
            new_accelerator.load_state_dict(state_dict)

    def test_engine_device_agnostic(self, engine):
        """Engine should work on CPU and GPU (if available)."""
        device = torch.device("cpu")
        engine.to(device)

        # Check that parameters are on device
        for param in engine.parameters():
            assert param.device == device or param.numel() == 0

    def test_backbone_frozen_in_training(self, accelerator):
        """Backbone should be frozen while plugins are trainable."""
        from cocf.training.stage_b_joint import JointTrainingStage, StageBConfig

        # The backbone is an adapter; its weights live on `.module`. The mock
        # freezes them at construction, so unfreeze first to prove the stage
        # actually performs the freezing.
        for p in accelerator.backbone.module.parameters():
            p.requires_grad_(True)
        bb_params_grad = [p for p in accelerator.backbone.module.parameters() if p.requires_grad]
        assert len(bb_params_grad) > 0  # Backbone has (now) trainable params

        # Freeze and create training stage. StageBConfig's data root is
        # ``processed_root`` (the §3 store); its ctor does no IO, so a dummy path is
        # enough to exercise the freezing the stage performs at construction.
        stage_config = StageBConfig(processed_root=Path("./dummy"), device=torch.device("cpu"))
        stage = JointTrainingStage(accelerator, stage_config)

        # Check frozen
        bb_params_grad = [p for p in accelerator.backbone.module.parameters() if p.requires_grad]
        assert len(bb_params_grad) == 0  # All frozen


class TestInferenceEngine:
    """Test the inference engine."""

    def test_engine_state_initialization(self):
        """Engine should initialize state correctly."""
        from cocf.engine.state import EngineState

        z = torch.randn(1, 1024, 64)
        state = EngineState(
            z=z,
            grid=None,
            cond=None,
            subgraph=None,
            anchor_store=None,
        )

        assert state.z.shape == z.shape
        assert len(state.tubes) == 0

    def test_generation_result_efficiency(self):
        """GenerationResult should compute efficiency metrics."""
        from cocf.engine.state import GenerationResult, StepTrace

        traces = [
            StepTrace(
                step=i,
                active_ratio=0.8,
                budget=0.5,
                predicted_cost=0.1,
                num_tubes=5,
                rollbacks=0,
                repairs=0,
            )
            for i in range(30)
        ]

        result = GenerationResult(
            video=torch.zeros(1, 3, 49, 512, 512),
            z0=torch.zeros(1, 1024, 64),
            traces=traces,
        )

        assert result.mean_active_ratio == pytest.approx(0.8)
        assert result.num_rollbacks == 0
        assert result.num_repairs == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
