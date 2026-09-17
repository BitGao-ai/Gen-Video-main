#!/usr/bin/env bash
# One-shot diagnostic: full-compute render with VAE tile 256 (Stage A's setting),
# controlling against the mud produced at tile 128. Standalone copy of the probe
# command so it can be uploaded and run without syncing the repo.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

WEIGHTS_ROOT="${WEIGHTS_ROOT:-/workdir/8e650203-d2d8-4032-afba-e3f6636858a0/LongVideo/model/weights}"
CHECKPOINT="${CHECKPOINT:-checkpoints/stage_b.20260916_114011.LVJjMI/stage_b_final.pt}"
OUTPUT="${OUTPUT:-outputs/stage_b_probe/full_tile256.mp4}"
GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0}}"

[[ -f "$CHECKPOINT" ]] || { echo "[error] checkpoint missing: $CHECKPOINT" >&2; exit 1; }
mkdir -p "$(dirname "$OUTPUT")"

CUDA_VISIBLE_DEVICES="$GPU_IDS" python3 -u scripts/inference/infer_single_video.py \
  --prompt "A person walks slowly across a park, with trees in the background, steady camera." \
  --checkpoint "$CHECKPOINT" \
  --backbone wan22 --wan-variant a14b-t2v \
  --model-path "$WEIGHTS_ROOT/Wan2.2-T2V-A14B-Diffusers" \
  --backbone-dtype bfloat16 --perception-dtype bfloat16 \
  --sam-model "$WEIGHTS_ROOT/SAM" \
  --dino-model "$WEIGHTS_ROOT/DINO" \
  --clip-model "$WEIGHTS_ROOT/Clip" \
  --raft-weights "$WEIGHTS_ROOT/raft" \
  --flow-shift 5 --steps 20 --num-frames 49 --height 384 --width 640 \
  --vae-tile 256 --quality quality --seed 1234 --device cuda \
  --full-compute --output "$OUTPUT"

echo "[done] $OUTPUT"
