"""Three-stage training system (§7.1).

Training is completely decoupled from the inference engine. The three stages
progress from data generation → joint module training → lightweight fine-tuning:

    Stage A (§7.1.1)
        Offline counterfactual teacher generation. Runs once with the full
        backbone to create a dataset of (tube, timestep, action, damage_label)
        triplets. Uses :class:`COCFDataGenerator` and is independent of any
        training loop.

    Stage B (§7.1.2)
        Joint training of the four learnable modules on the Stage A data.
        Minimizes the combined loss:
            L_total = L_cocf + λ·L_tube + λ·L_cert + λ·L_cmsc + λ·L_budget
        with the losses weighted per §7.1.2.

    Stage C (§7.1.3)
        Lightweight fine-tuning on the full accelerator (engine ↔ backbone
        interactions). Optional LoRA adapters on backbone; primarily trains
        the residual repair nets and CMSC boundary fusion.

All three stages are integrated into a unified :class:`TrainingPipeline` that
handles device/checkpoint management, but can also be invoked independently.
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
