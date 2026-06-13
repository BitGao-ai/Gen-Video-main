"""Unit tests for L-COCF counterfactual data generation (§7.1.1)."""

import pytest
import torch

from cocf.lcocf.data import (
    COCFTrainingSample,
    CounterfactualDamageComputer,
    DamageLabelInterpolator,
    StratifiedSampler,
    StratifiedSamplingConfig,
)
from cocf.lcocf.damage import NUM_DAMAGE_DIMS


class TestCOCFTrainingSample:
    """Test COCFTrainingSample data structure."""

    def test_sample_creation(self):
        """COCFTrainingSample should be creatable with required fields."""
        sample = COCFTrainingSample(
            tube_features=torch.randn(7),
            timestep=25,
            action=0,  # FULL
            damage_label=torch.rand(NUM_DAMAGE_DIMS),
            prompt="a cat",
            tube_id=0,
        )

        assert sample.timestep == 25
        assert sample.action == 0
        assert sample.tube_id == 0

    def test_sample_damage_scalar(self):
        """Sample.damage_scalar() should return a value in [0,1]."""
        sample = COCFTrainingSample(
            tube_features=torch.randn(7),
            timestep=10,
            action=1,
            damage_label=torch.rand(NUM_DAMAGE_DIMS),
        )

        scalar = sample.damage_scalar()
        assert 0 <= scalar <= 1

    def test_sample_serialization(self):
        """Sample should serialize to dict."""
        sample = COCFTrainingSample(
            tube_features=torch.randn(7),
            timestep=15,
            action=2,
            damage_label=torch.rand(NUM_DAMAGE_DIMS),
            prompt="test",
            tube_id=42,
        )

        d = sample.to_dict()
        assert "tube_features" in d
        assert "timestep" in d
        assert d["tube_id"] == 42


class TestStratifiedSampler:
    """Test stratified sampling strategy."""

    def test_timestep_sampling(self):
        """Sampler should distribute timesteps across strata."""
        config = StratifiedSamplingConfig()
        sampler = StratifiedSampler(config, torch.device("cpu"))

        total_steps = 50
        samples = sampler.sample_timesteps(total_steps)

        assert len(samples) > 0
        # All samples should be valid timesteps
        for t, stratum in samples:
            assert 0 <= t <= total_steps

    def test_action_sampling(self):
        """Sampler should sample diverse actions."""
        config = StratifiedSamplingConfig()
        sampler = StratifiedSampler(config, torch.device("cpu"))

        actions = sampler.sample_actions(num_samples=12)
        assert len(actions) <= 12
        # Should include multiple action types
        assert len(set(actions)) > 1


class TestDamageLabelInterpolator:
    """Test label interpolation."""

    def test_interpolation_caching(self):
        """Interpolator should cache computed damages."""
        interp = DamageLabelInterpolator(interval=5)

        damage = torch.rand(NUM_DAMAGE_DIMS)
        interp.cache_damage(step=0, tube_id=0, action=0, damage=damage)

        retrieved = interp.interpolate(step=0, tube_id=0, action=0, num_total_steps=50)
        assert retrieved is not None
        assert torch.allclose(retrieved, damage)

    def test_interpolation_neighbors(self):
        """Interpolator should interpolate between neighbors."""
        interp = DamageLabelInterpolator(interval=5)

        damage_0 = torch.ones(NUM_DAMAGE_DIMS) * 0.0
        damage_5 = torch.ones(NUM_DAMAGE_DIMS) * 1.0

        interp.cache_damage(step=0, tube_id=0, action=0, damage=damage_0)
        interp.cache_damage(step=5, tube_id=0, action=0, damage=damage_5)

        # Interpolate at step 2: alpha = 2/5 = 0.4, so result = 0.4 * ones.
        result = interp.interpolate(step=2, tube_id=0, action=0, num_total_steps=50)
        assert result is not None
        expected = torch.ones(NUM_DAMAGE_DIMS) * 0.4
        assert torch.allclose(result, expected)
        assert (result >= 0).all() and (result <= 1).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
