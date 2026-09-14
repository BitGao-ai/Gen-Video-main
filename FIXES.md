# Training and inference corrections

## Behavior changes

- BPTT truncates before a new computed transition, preserving the last segment
  through trailing skips. Detached saved anchors can still replace that segment;
  gradient logs do not guarantee every trainable parameter received a gradient.
- Semantic tube masks map latent slots to full-video frames by nearest
  representative frame. Windowed losses pass the full frame count and offset.
- Stage C tube CLIP features retain pixel gradients through tensor preprocessing.
- Inference with a real backbone automatically loads real perception and requires
  RAFT. Pass --sam-model, --dino-model, --clip-model and --raft-weights for local
  weights. Mock inference remains available with --backbone mock.
- The engine and Stage C require one video per call. Distributed training can
  still use one video per rank; shared multi-video tube state is rejected.
- Stage B and inference share neutral-centered CMSC risk, including padding rules.
- Learning-rate warmup includes the final step that reaches the configured LR.
- Checkpoint loading is strict by default. Missing files, missing/unexpected keys,
  incompatible shapes and incompatible recorded geometry fail instead of silently
  using random parameters. Explicit Python migration can use
  load_checkpoint(..., allow_shape_mismatch=True); inference has no permissive flag.
- New checkpoints include model geometry and a risk-definition version. A mock
  training adapter does not constrain the destination backbone identity, since
  Stage B uses it to train plugins from stored real-model features.

## Existing artifacts

Additional review corrections retain preview optical flow during inference and
sync predictor anchor ages from the anchor store. Only actually computed tubes
refresh anchors. Training/validation use allocator-resolved action costs and the
same certificate residual/CMSC inputs. Tube refresh intervals start at the build
step. `engine.log_every_steps` controls INFO cadence (default 1); other steps are
DEBUG. `predicted_cost` now reports allocation cost; `predicted_damage` is separate.

Stage A commits sample buffers before publishing progress, atomically replaces
shard files, and uses same-seed full continuations as perturbed counterfactual
references. This adds full continuations for nonzero seeds and can increase runtime
and memory. Generate into a new store rather than mixing old and new labels.

Stage B validation interval <= 0 disables validation. FP16 training checks finite
gradients collectively before and after averaging so all ranks skip overflow
updates together. LoRA metadata is inferred from live adapters; unrepresentable
mixed geometries fail explicitly. Nested tuple configuration and variant extra
merging preserve configuration structure. Pipeline device resolution and best
checkpoint paths now follow the selected device and experiment directory.

Regenerate Stage A samples and tube visual features affected by temporal mapping.
Changing code does not rewrite existing processed stores. Recalibrate/retrain
legacy certificate weights for the neutral-centered risk input and validate
quality/trigger rates before relying on old thresholds. Legacy checkpoints without
metadata remain readable when their state dictionaries match strictly; model
identity cannot be verified from absent metadata.

## Validation

Run CPU regressions with:

```bash
python -m unittest discover -s tests
```

Coverage includes trailing skips, continuous BPTT segments, the actual ANCHOR
executor without a saved anchor, temporal mask expansion, offset windows, injected
perception pixel gradients, risk parity and checkpoint/batch validation. Real
SAM/CLIP/Wan weights, GPU peak memory, LoRA and multi-GPU execution require separate
hardware validation. The tensor CLIP resize can differ numerically from legacy
PIL preprocessing, so old feature stores should not be assumed identical.
