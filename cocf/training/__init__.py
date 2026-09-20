"""Three-stage training system, decoupled from the inference engine.

Stage A generates counterfactual teacher data with the full backbone; Stage B jointly
trains the four learnable modules on that data; Stage C fine-tunes on the full
accelerator (residual repair nets, CMSC boundary fusion, optional backbone LoRA). All
three are integrated into a unified :class:`TrainingPipeline` handling device and
checkpoint management, and can also run independently.
"""

from __future__ import annotations

from cocf.training.stage_a_data_gen import DataGenerationStage
from cocf.training.stage_b_joint import JointTrainingStage
from cocf.training.stage_c_finetune import FinettuneStage
from cocf.training.pipeline import TrainingPipeline

__all__ = [
    "DataGenerationStage",
    "JointTrainingStage",
    "FinettuneStage",
    "TrainingPipeline",
]
