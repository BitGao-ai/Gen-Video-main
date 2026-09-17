#!/usr/bin/env bash
# Multi-GPU (torchrun) Stage C end-to-end fine-tuning.
# Example: NPROC=8 bash scripts/train/run_stage_c.sh
# Single-card smoke: NPROC=1 NUM_EPOCHS=1 bash scripts/train/run_stage_c.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
NPROC="${NPROC:-8}"
PROCESSED_ROOT="${PROCESSED_ROOT:-./LCOCF_OpenVid1M_Processed}"
STAGE_B_CKPT="${STAGE_B_CKPT:-./checkpoints/stage_b.20260916_114011.LVJjMI/stage_b_final.pt}"
WEIGHTS_ROOT="${WEIGHTS_ROOT:-/workdir/8e650203-d2d8-4032-afba-e3f6636858a0/LongVideo/model/weights}"
MODEL_PATH="${MODEL_PATH:-$WEIGHTS_ROOT/Wan2.2-T2V-A14B-Diffusers}"
SAM_MODEL="${SAM_MODEL:-$WEIGHTS_ROOT/SAM}"
DINO_MODEL="${DINO_MODEL:-$WEIGHTS_ROOT/DINO}"
CLIP_MODEL="${CLIP_MODEL:-$WEIGHTS_ROOT/Clip}"
# A *directory* holding one checkpoint per variant ("large"/"small" in the file
# names) satisfies both RAFT consumers; leave empty to use the torch hub cache.
RAFT_WEIGHTS="${RAFT_WEIGHTS:-$WEIGHTS_ROOT/raft}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
VAE_TILE="${VAE_TILE:-128}"
GRAD_WINDOW_STEPS="${GRAD_WINDOW_STEPS:-1}"
DECODE_GRAD_FRAMES="${DECODE_GRAD_FRAMES:-4}"
METRIC_FRAME_CHUNK="${METRIC_FRAME_CHUNK:-2}"
SEED="${SEED:-1234}"
LOG_DIR="${LOG_DIR:-logs}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints}"
CHECK_RAFT="${CHECK_RAFT:-1}"
DRY_RUN="${DRY_RUN:-0}"

die() { echo "[error] $*" >&2; exit 1; }
command -v "$PYTHON_BIN" >/dev/null || die "Python not found: $PYTHON_BIN"
command -v "$TORCHRUN_BIN" >/dev/null || die "torchrun not found: $TORCHRUN_BIN"
[[ "$NPROC" =~ ^[1-9][0-9]*$ ]] || die "NPROC must be a positive integer: $NPROC"
for value in "$NUM_EPOCHS" "$VAE_TILE" "$GRAD_WINDOW_STEPS" "$DECODE_GRAD_FRAMES" "$METRIC_FRAME_CHUNK"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "Expected positive integer: $value"
done
[[ "$SEED" =~ ^(0|[1-9][0-9]*)$ ]] || die "Expected nonnegative integer: $SEED"
[[ "$DRY_RUN" =~ ^[01]$ ]] || die 'DRY_RUN must be 0 or 1'
[[ "$CHECK_RAFT" =~ ^[01]$ ]] || die 'CHECK_RAFT must be 0 or 1'

CMD=(
  scripts/train/train_stage_c.py
  --processed-root "$PROCESSED_ROOT"
  --checkpoint_load "$STAGE_B_CKPT"
  --backbone wan22 --wan-variant a14b-t2v
  --model-path "$MODEL_PATH"
  --vae-tile "$VAE_TILE"
  --grad-window-steps "$GRAD_WINDOW_STEPS"
  --decode-grad-frames "$DECODE_GRAD_FRAMES"
  --real-models --metric-frame-chunk "$METRIC_FRAME_CHUNK"
  --sam-model "$SAM_MODEL" --dino-model "$DINO_MODEL" --clip-model "$CLIP_MODEL"
  --batch_size 1 --num_epochs "$NUM_EPOCHS" --seed "$SEED"
)
if [[ -n "$RAFT_WEIGHTS" ]]; then CMD+=(--raft-weights "$RAFT_WEIGHTS"); fi
if [[ "$NPROC" == 1 ]]; then CMD+=(--device cuda); fi

if [[ "$DRY_RUN" == 1 ]]; then
  if [[ "$NPROC" == 1 ]]; then
    printf '%q ' "$PYTHON_BIN" -u "${CMD[@]}" --checkpoint_save "$CHECKPOINT_DIR/<unique-run>/stage_c_final.pt"
  else
    printf '%q ' "$TORCHRUN_BIN" --standalone --nproc_per_node="$NPROC" "${CMD[@]}" \
      --checkpoint_save "$CHECKPOINT_DIR/<unique-run>/stage_c_final.pt"
  fi
  printf '\n[preview] No training started or files written.\n'
  exit 0
fi

[[ -d "$PROCESSED_ROOT" ]] || die "Processed store missing: $PROCESSED_ROOT"
[[ -f "$STAGE_B_CKPT" ]] || die "Stage B checkpoint missing: $STAGE_B_CKPT"
[[ -d "$MODEL_PATH" ]] || die "Model path missing: $MODEL_PATH"
# Stage C trains against the Y_full Stage A persisted here; an empty bucket means
# the baseline is re-denoised every batch (several-fold slowdown).
n_baseline="$(ls "$PROCESSED_ROOT/full_baseline" 2>/dev/null | wc -l | tr -d ' ')"
[[ "$n_baseline" =~ ^[1-9][0-9]*$ ]] \
  || die "No full_baseline buckets in $PROCESSED_ROOT — Stage C would recompute Y_full every batch"

# Both RAFT variants (perception: large, damage metrics: small) must load; a
# missing one is a hard failure under --real-models.
if [[ "$CHECK_RAFT" == 1 ]]; then
  RAFT_WEIGHTS="$RAFT_WEIGHTS" "$PYTHON_BIN" - <<'PY' || die "RAFT self-check failed — see run.md"
import os
import torch
from cocf.common.raft import load_raft
w = os.environ.get("RAFT_WEIGHTS") or None
for v in ("large", "small"):
    ok = load_raft("cuda", variant=v, weights_path=w, required=True) is not None
    print("[preflight] raft", v, ok)
    assert ok, v
PY
fi

mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR"
run_dir="$(mktemp -d "$CHECKPOINT_DIR/stage_c.$(date +%Y%m%d_%H%M%S).XXXXXX")"
log_file="$LOG_DIR/$(basename "$run_dir").log"
echo "[start] nproc=$NPROC epochs=$NUM_EPOCHS processed=$PROCESSED_ROOT"
echo "[start] baseline_buckets=$n_baseline stage_b_ckpt=$STAGE_B_CKPT"
echo "[log] $log_file"
echo "[checkpoint] $run_dir/stage_c_final.pt"
echo "[monitor] tail -f '$log_file'"

if [[ "$NPROC" == 1 ]]; then
  "$PYTHON_BIN" -u "${CMD[@]}" --checkpoint_save "$run_dir/stage_c_final.pt" \
    > "$log_file" 2>&1
else
  "$TORCHRUN_BIN" --standalone --nproc_per_node="$NPROC" "${CMD[@]}" \
    --checkpoint_save "$run_dir/stage_c_final.pt" > "$log_file" 2>&1
fi && {
  echo "[done] Log: $log_file; checkpoints: $run_dir"
} || {
  status=$?
  echo "[error] Stage C exited with status $status. Last 40 log lines:" >&2
  tail -n 40 "$log_file" >&2 || true
  exit "$status"
}
