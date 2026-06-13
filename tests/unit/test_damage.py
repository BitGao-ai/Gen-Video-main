"""Unit tests for L-COCF damage computation."""

import pytest
import torch

from cocf.lcocf.damage import (
    DAMAGE_DIMENSIONS,
    DEFAULT_DAMAGE_WEIGHTS,
    MultiDimDamageComputer,
    VideoFeatures,
)


class MockMetricExtractor:
    """Deterministic mock for testing."""

    def extract(self, video, prompt):
        """Return synthetic VideoFeatures."""
        num_frames = video.shape[0]
        return VideoFeatures(
            dino_per_frame=torch.randn(num_frames, 64),
            clip_per_frame=torch.randn(num_frames, 128),
            clip_text_score=0.75,
            flow_mag_per_pair=torch.randn(num_frames - 1),
            ocr_accuracy=0.9,
        )


class TestMultiDimDamageComputer:
    """Test multi-dimensional damage computation."""

    def test_damage_shape(self):
        """Damage vector should have NUM_DAMAGE_DIMS elements."""
        from cocf.lcocf.damage import NUM_DAMAGE_DIMS

        computer = MultiDimDamageComputer()

        # Mock feature extractor
        class MockExtractor:
            def extract(self, video, prompt):
                num_frames = video.shape[0]
                return VideoFeatures(
                    dino_per_frame=torch.randn(num_frames, 64),
                    clip_per_frame=torch.randn(num_frames, 128),
                    clip_text_score=0.75,
                    flow_mag_per_pair=torch.randn(num_frames - 1),
                    ocr_accuracy=0.9,
                )

        extractor = MockExtractor()

        # Create mock videos
        video_full = torch.randn(8, 3, 512, 512)  # 8 frames
        video_cf = torch.randn(8, 3, 512, 512)

        # Compute damage. ``compute`` returns a {axis: value} dict; convert it to
        # the ordered [NUM_DAMAGE_DIMS] tensor via ``as_vector`` before shape checks.
        damage_dict = computer.compute(
            VideoFeatures(
                dino_per_frame=torch.randn(8, 64),
                clip_per_frame=torch.randn(8, 128),
                clip_text_score=0.75,
                flow_mag_per_pair=torch.randn(7),
                ocr_accuracy=0.9,
            ),
            VideoFeatures(
                dino_per_frame=torch.randn(8, 64),
                clip_per_frame=torch.randn(8, 128),
                clip_text_score=0.70,
                flow_mag_per_pair=torch.randn(7),
                ocr_accuracy=0.85,
            ),
        )
        damage = computer.as_vector(damage_dict)

        assert damage.shape == (NUM_DAMAGE_DIMS,)
        assert (damage >= 0).all()
        assert (damage <= 1).all()

    def test_damage_axes_coverage(self):
        """All damage axes should be present in DAMAGE_DIMENSIONS."""
        assert len(DAMAGE_DIMENSIONS) == 8
        assert "subject_consistency" in DAMAGE_DIMENSIONS
        assert "ocr_accuracy" in DAMAGE_DIMENSIONS

    def test_default_weights_sum(self):
        """Default damage weights should sum to ~1.0."""
        total = sum(DEFAULT_DAMAGE_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
