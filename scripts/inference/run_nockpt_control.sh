#!/usr/bin/env bash
# No-checkpoint control run: same accelerated inference as run_stage_b_probe.sh
# but with cold-start (untrained) plugins — discriminates "engine base path
# corruption" from "Stage-B-checkpoint-driven corruption". Expectation per the
# allocator analysis: cold-start also stays at the prior action plan, so a
# mosaic here too would confirm the engine base path (splice/priors) as the
# corruptor rather than the checkpoint.
# Preview: DRY_RUN=1 bash scripts/inference/run_nockpt_control.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0}}"
WEIGHTS_ROOT="${WEIGHTS_ROOT:-/workdir/8e650203-d2d8-4032-afba-e3f6636858a0/LongVideo/model/weights}"
MODEL_PATH="${MODEL_PATH:-$WEIGHTS_ROOT/Wan2.2-T2V-A14B-Diffusers}"
SAM_MODEL="${SAM_MODEL:-$WEIGHTS_ROOT/SAM}"
DINO_MODEL="${DINO_MODEL:-$WEIGHTS_ROOT/DINO}"
CLIP_MODEL="${CLIP_MODEL:-$WEIGHTS_ROOT/Clip}"
RAFT_WEIGHTS="${RAFT_WEIGHTS:-$WEIGHTS_ROOT/raft}"
PROMPT="${PROMPT:-A person walks slowly across a park, with trees in the background, steady camera.}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/stage_b_probe}"
QUALITY="${QUALITY:-quality}"
SEED="${SEED:-1234}"
DRY_RUN="${DRY_RUN:-0}"
VAE_TILE="${VAE_TILE:-128}"

die() { echo "[error] $*" >&2; exit 1; }
command -v "$PYTHON_BIN" >/dev/null || die "Python not found: $PYTHON_BIN"
[[ "$GPU_IDS" =~ ^([0-9]+|GPU-[a-zA-Z0-9-]+|MIG-[a-zA-Z0-9/-]+)$ ]] || die 'GPU_IDS must select exactly one GPU'
if [[ ${CUDA_VISIBLE_DEVICES+x} ]]; then
  [[ ",$CUDA_VISIBLE_DEVICES," == *",$GPU_IDS,"* ]] || die 'GPU_IDS is outside CUDA_VISIBLE_DEVICES'
fi
[[ "$DRY_RUN" =~ ^[01]$ ]] || die 'DRY_RUN must be 0 or 1'
[[ "$SEED" =~ ^(0|[1-9][0-9]*)$ ]] || die 'SEED must be a nonnegative integer'
[[ "$QUALITY" == quality || "$QUALITY" == balanced || "$QUALITY" == fast ]] || die 'Invalid QUALITY'
[[ "$VAE_TILE" =~ ^[0-9]+$ ]] || die 'VAE_TILE must be a nonnegative integer'

# Identical to run_stage_b_probe.sh except: no --checkpoint (cold-start plugins).
ARGS=(
  scripts/inference/infer_single_video.py
  --prompt "$PROMPT"
  --backbone wan22 --wan-variant a14b-t2v --model-path "$MODEL_PATH"
  --backbone-dtype bfloat16 --perception-dtype bfloat16
  --sam-model "$SAM_MODEL" --dino-model "$DINO_MODEL" --clip-model "$CLIP_MODEL"
  --raft-weights "$RAFT_WEIGHTS" --flow-shift 5
  --steps 20 --num-frames 49 --height 384 --width 640 --vae-tile "$VAE_TILE"
  --quality "$QUALITY" --seed "$SEED" --device cuda
)
if [[ "$DRY_RUN" == 1 ]]; then
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU_IDS"
  printf '%q ' "$PYTHON_BIN" -u "${ARGS[@]}" --output "$OUTPUT_ROOT/<unique-run>/accelerated_nockpt.mp4"
  printf '\n[preview] No model loaded or files written.\n'
  exit 0
fi
for path in "$MODEL_PATH" "$SAM_MODEL" "$DINO_MODEL" "$CLIP_MODEL"; do
  [[ -d "$path" ]] || die "Model directory missing: $path"
done
[[ -e "$RAFT_WEIGHTS" ]] || die "RAFT weights missing: $RAFT_WEIGHTS"
mkdir -p "$OUTPUT_ROOT"
run_dir="$(mktemp -d "$OUTPUT_ROOT/run.$(date +%Y%m%d_%H%M%S).XXXXXX")"
log_file="$run_dir/inference.log"
ARGS+=(--output "$run_dir/accelerated_nockpt.mp4")
echo "[start] no-checkpoint control GPU=$GPU_IDS quality=$QUALITY seed=$SEED"
echo "[output] $run_dir"
printf '[monitor] tail -f %q\n' "$log_file"
{
  printf '[command] CUDA_VISIBLE_DEVICES=%q ' "$GPU_IDS"
  printf '%q ' "$PYTHON_BIN" -u "${ARGS[@]}"
  printf '\n'
} > "$log_file"
if CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PYTHON_BIN" -u "${ARGS[@]}" >> "$log_file" 2>&1; then
  echo "[done] Output and log: $run_dir"
else
  status=$?
  echo "[error] Inference exited with status $status. Log: $log_file" >&2
  tail -n 40 "$log_file" >&2 || true
  exit "$status"
fi
