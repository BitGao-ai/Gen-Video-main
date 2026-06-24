# COCF-SS-DCA

**Causal-Counterfactual Compute Field with Semantic-Tube Anchoring**

A plug-in acceleration layer for video diffusion transformers (DiT). COCF wraps a frozen backbone (HunyuanVideo, Wan2.1, etc.) and dynamically allocates compute per cross-frame *semantic tube* — perceptually unimportant regions are cheaply approximated while causally important regions keep full fidelity.

## Architecture

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                   Accelerator (core/)                    │
                    │  Wraps backbone, plugins, engine into a single state     │
                    └────┬──────────┬──────────┬──────────┬───────────────────┘
                         │          │          │          │
               ┌─────────┘  ┌───────┘   ┌──────┘  ┌──────┘
               ▼             ▼           ▼          ▼
         ┌──────────┐ ┌──────────┐ ┌────────┐ ┌──────────┐
         │  L-COCF  │ │   STA    │ │  RAEC  │ │   CMSC   │
         │  (§3)    │ │  (§4)    │ │  (§5)  │ │  (§6)    │
         └──────────┘ └──────────┘ └────────┘ └──────────┘
```

### Subsystems

| Package | Path | Description |
|---------|------|-------------|
| `cocf.common` | `cocf/common/` | Types, config, registry, memory helpers, logging |
| `cocf.backbones` | `cocf/backbones/` | Backbone adapters (HunyuanVideo, Wan2.1, Mock) |
| `cocf.tubes` | `cocf/tubes/` | STA — Semantic Tube Anchoring (§4) |
| `cocf.lcocf` | `cocf/lcocf/` | L-COCF — Lightweight Counterfactual Causal Compute Field (§3) |
| `cocf.raec` | `cocf/raec/` | RAEC — Revocable Anchoring & Error Certificates (§5) |
| `cocf.cmsc` | `cocf/cmsc/` | CMSC — Cross-Modal Semantic Conservation (§6) |
| `cocf.scheduler` | `cocf/scheduler/` | Dynamic compute-budget scheduling & action allocation (§7.3, §2.2) |
| `cocf.engine` | `cocf/engine/` | Accelerated inference loop (§7.2) |
| `cocf.core` | `cocf/core/` | Top-level accelerator wiring everything together |
| `cocf.data` | `cocf/data/` | Training data pipeline & counterfactual teacher generation (§7.1.1) |
| `cocf.training` | `cocf/training/` | Three-stage training system (§7.1) |

## Training Pipeline (Three Stages)

### Stage A — Counterfactual Teacher Data Generation
```
scripts/data/generate_counterfactual_data.py \
    --openvid-csv data/train/OpenVid-1M.csv \
    --data-root /datasets/OpenVid-1M \
    --processed-root ./LCOCF_OpenVid1M_Processed \
    --backbone hunyuanvideo
```
Runs the frozen backbone teacher forward pass, applies quality filtering, generates counterfactual labels, and writes the six-level processed LMDB store (§3).

### Stage B — Joint Plugin Training
```
scripts/train/train_stage_b.py \
    --processed-root ./LCOCF_OpenVid1M_Processed \
    --checkpoint_load ./checkpoints/after_stage_a.pt
```
Trains the four learnable plugins (L-COCF predictor + strength weights, STA smoothing, RAEC certificate, CMSC alignment) jointly on the Stage-A store. The backbone stays frozen.

### Stage C — End-to-End Fine-Tuning
```
scripts/train/train_stage_c.py \
    --processed-root ./LCOCF_OpenVid1M_Processed \
    --checkpoint_load ./checkpoints/stage_b_final.pt \
    --use_lora
```
Embeds trained plugins into the full accelerated pipeline and fine-tunes differentiable components against the full-compute baseline `Y_full`.

## Inference
```
scripts/inference/infer_single_video.py \
    --prompt "a cat jumping" \
    --checkpoint ./checkpoints/stage_c_final.pt \
    --output ./output.mp4 \
    --quality balanced
```

## Testing
```
python -m pytest tests/
```

## Dependencies

- Python ≥ 3.10
- PyTorch ≥ 2.0
- Diffusers (for HuggingFace backbone adapters)
- LMDB (for processed data store)
- einops, scipy, scikit-image

See individual imports for full dependency list.

## Project Status

Early-stage research code. Several components are stubs (mock backbone, CMSC alignment, memory tracking, metric computation). The inference script has hardcoded tensor shapes and TODOs for proper VAE decode and video saving. Tests provide minimal coverage.
