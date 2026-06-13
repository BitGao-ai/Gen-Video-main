"""Unit tests for the inference engine (§7.2)."""

import pytest
import torch

from cocf.engine.state import EngineState, GenerationResult, StepTrace


class TestEngineState:
    """Test EngineState functionality."""

    def test_engine_state_creation(self):
        """EngineState should be creatable with minimal inputs."""
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
        assert len(state.traces) == 0

    def test_engine_state_accumulation(self):
        """EngineState should accumulate traces over steps."""
        state = EngineState(
            z=torch.randn(1, 1024, 64),
            grid=None,
            cond=None,
            subgraph=None,
            anchor_store=None,
        )

        # Add traces
        for i in range(5):
            trace = StepTrace(
                step=i,
                active_ratio=0.8,
                budget=0.5 + 0.1 * i,
                predicted_cost=0.1 * i,
                num_tubes=3,
            )
            state.traces.append(trace)

        assert len(state.traces) == 5
        assert state.traces[0].step == 0
        assert state.traces[4].step == 4


class TestGenerationResult:
    """Test GenerationResult efficiency reporting."""

    def test_result_summary(self):
        """Result.summary() should return expected metrics."""
        video = torch.randn(1, 3, 49, 512, 512)
        z0 = torch.randn(1, 1024, 64)

        traces = [
            StepTrace(
                step=i,
                active_ratio=0.8,
                budget=0.6,
                predicted_cost=0.05,
                num_tubes=5,
                rollbacks=0,
                repairs=1 if i % 2 == 0 else 0,
            )
            for i in range(10)
        ]

        result = GenerationResult(video=video, z0=z0, traces=traces)
        summary = result.summary()

        assert "steps" in summary
        assert "mean_active_ratio" in summary
        assert "rollbacks" in summary
        assert "repairs" in summary
        assert summary["steps"] == 10
        assert summary["repairs"] == 5  # 5 even steps

    def test_efficiency_properties(self):
        """Test efficiency computation properties."""
        traces = [
            StepTrace(
                step=i,
                active_ratio=0.5 + 0.1 * i,
                budget=0.4,
                predicted_cost=0.02 * i,
                num_tubes=4,
                rollbacks=1 if i < 2 else 0,
                repairs=1 if i % 3 == 0 else 0,
            )
            for i in range(6)
        ]

        result = GenerationResult(
            video=torch.zeros(1, 3, 49, 512, 512),
            z0=torch.zeros(1, 1024, 64),
            traces=traces,
        )

        assert result.mean_active_ratio > 0
        assert result.num_rollbacks == 2
        assert result.num_repairs == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
