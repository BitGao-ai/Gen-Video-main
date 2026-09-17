#!/usr/bin/env bash
# Trajectory tracer for the accelerated path (per-step latent stats by action
# group + mid-trajectory decodes + action map). Diagnostic, not a benchmark.
# Preview: DRY_RUN=1 bash scripts/diagnose/run_trace_accelerated.sh
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
CHECKPOINT="${CHECKPOINT:-checkpoints/stage_b.20260916_114011.LVJjMI/stage_b_final.pt}"
PROMPT="${PROMPT:-A person walks slowly across a park, with trees in the background, steady camera.}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/trace_accelerated}"
QUALITY="${QUALITY:-quality}"
SEED="${SEED:-1234}"
DECODE_STEPS="${DECODE_STEPS:-1,2,3,5,10,15,20}"
VAE_TILE="${VAE_TILE:-128}"
DRY_RUN="${DRY_RUN:-0}"

die() { echo "[error] $*" >&2; exit 1; }
command -v "$PYTHON_BIN" >/dev/null || die "Python not found: $PYTHON_BIN"
[[ "$GPU_IDS" =~ ^([0-9]+|GPU-[a-zA-Z0-9-]+|MIG-[a-zA-Z0-9/-]+)$ ]] || die 'GPU_IDS must select exactly one GPU'
if [[ ${CUDA_VISIBLE_DEVICES+x} ]]; then
  [[ ",$CUDA_VISIBLE_DEVICES," == *",$GPU_IDS,"* ]] || die 'GPU_IDS is outside CUDA_VISIBLE_DEVICES'
fi
[[ "$DRY_RUN" =~ ^[01]$ ]] || die 'DRY_RUN must be 0 or 1'
[[ "$SEED" =~ ^(0|[1-9][0-9]*)$ ]] || die 'SEED must be a nonnegative integer'
[[ "$QUALITY" == quality || "$QUALITY" == balanced || "$QUALITY" == fast ]] || die 'Invalid QUALITY'

# Keep geometry and flow shift consistent with the probe script and the Stage A store.
ARGS=(
  scripts/diagnose/trace_accelerated.py
  --prompt "$PROMPT" --checkpoint "$CHECKPOINT"
  --backbone wan22 --wan-variant a14b-t2v --model-path "$MODEL_PATH"
  --backbone-dtype bfloat16 --perception-dtype bfloat16
  --sam-model "$SAM_MODEL" --dino-model "$DINO_MODEL" --clip-model "$CLIP_MODEL"
  --raft-weights "$RAFT_WEIGHTS" --flow-shift 5
  --steps 20 --num-frames 49 --height 384 --width 640 --vae-tile "$VAE_TILE"
  --quality "$QUALITY" --seed "$SEED" --device cuda
  --decode-steps "$DECODE_STEPS"
)
if [[ "$DRY_RUN" == 1 ]]; then
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU_IDS"
  printf '%q ' "$PYTHON_BIN" -u "${ARGS[@]}" --output-dir "$OUTPUT_ROOT/<unique-run>"
  printf '\n[preview] No model loaded or files written.\n'
  exit 0
fi
[[ -f "$CHECKPOINT" ]] || die "Checkpoint missing: $CHECKPOINT"
for path in "$MODEL_PATH" "$SAM_MODEL" "$DINO_MODEL" "$CLIP_MODEL"; do
  [[ -d "$path" ]] || die "Model directory missing: $path"
done
[[ -e "$RAFT_WEIGHTS" ]] || die "RAFT weights missing: $RAFT_WEIGHTS"
mkdir -p "$OUTPUT_ROOT"
run_dir="$(mktemp -d "$OUTPUT_ROOT/run.$(date +%Y%m%d_%H%M%S).XXXXXX")"
log_file="$run_dir/trace.log"
ARGS+=(--output-dir "$run_dir")
echo "[start] GPU=$GPU_IDS quality=$QUALITY seed=$SEED decode_steps=$DECODE_STEPS"
echo "[checkpoint] $CHECKPOINT"
echo "[output] $run_dir"
printf '[monitor] tail -f %q\n' "$log_file"
{
  printf '[command] CUDA_VISIBLE_DEVICES=%q ' "$GPU_IDS"
  printf '%q ' "$PYTHON_BIN" -u "${ARGS[@]}"
  printf '\n'
} > "$log_file"
if CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PYTHON_BIN" -u "${ARGS[@]}" >> "$log_file" 2>&1; then
  echo "[done] Trace + decodes + action map: $run_dir"
else
  status=$?
  echo "[error] Trace exited with status $status. Log: $log_file" >&2
  tail -n 40 "$log_file" >&2 || true
  exit "$status"
fi
