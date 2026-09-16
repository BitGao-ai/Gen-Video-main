#!/usr/bin/env bash
# Single-GPU Stage B training using the shared non-OCR damage policy.
# Example: GPU_IDS=0 NUM_EPOCHS=10 bash scripts/train/run_stage_b.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0}}"
PROCESSED_ROOT="${PROCESSED_ROOT:-./LCOCF_OpenVid1M_Processed}"
BATCH_SIZE="${BATCH_SIZE:-16}"
# Budget cap, not a promise: classic mode still early-stops on validation
# stagnation, and phased mode stops when its step budgets are spent.
NUM_EPOCHS="${NUM_EPOCHS:-50}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SEED="${SEED:-1234}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-}"
LOG_DIR="${LOG_DIR:-logs}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints}"
DRY_RUN="${DRY_RUN:-0}"
# Optional phased predictor training (mean → variance → joint). All unset = classic.
PHASE_MEAN_STEPS="${PHASE_MEAN_STEPS:-}"
PHASE_VAR_STEPS="${PHASE_VAR_STEPS:-}"
PHASE_JOINT_LR_SCALE="${PHASE_JOINT_LR_SCALE:-}"
PHASE_MEAN_OBJECTIVE="${PHASE_MEAN_OBJECTIVE:-}"
PHASE_TARGET_SCALE="${PHASE_TARGET_SCALE:-}"
PHASE_AUX_ISOLATION="${PHASE_AUX_ISOLATION:-}"

die() { echo "[error] $*" >&2; exit 1; }
command -v "$PYTHON_BIN" >/dev/null || die "Python not found: $PYTHON_BIN"
[[ "$GPU_IDS" =~ ^([0-9]+|GPU-[a-zA-Z0-9-]+|MIG-[a-zA-Z0-9/-]+)$ ]] || die 'GPU_IDS must select exactly one GPU'
if [[ ${CUDA_VISIBLE_DEVICES+x} ]]; then
  [[ ",$CUDA_VISIBLE_DEVICES," == *",$GPU_IDS,"* ]] || die 'GPU_IDS is outside CUDA_VISIBLE_DEVICES'
fi
for value in "$BATCH_SIZE" "$NUM_EPOCHS"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "Expected positive integer: $value"
done
for value in "$NUM_WORKERS" "$SEED"; do
  [[ "$value" =~ ^(0|[1-9][0-9]*)$ ]] || die "Expected nonnegative integer: $value"
done
[[ "$DRY_RUN" =~ ^[01]$ ]] || die 'DRY_RUN must be 0 or 1'
if [[ -n $EARLY_STOP_PATIENCE ]]; then
  [[ "$EARLY_STOP_PATIENCE" =~ ^[1-9][0-9]*$ ]] || die 'EARLY_STOP_PATIENCE must be a positive integer'
fi
for value in "$PHASE_MEAN_STEPS" "$PHASE_VAR_STEPS"; do
  if [[ -n $value ]]; then
    [[ "$value" =~ ^(0|[1-9][0-9]*)$ ]] || die "Expected nonnegative integer: $value"
  fi
done
if [[ -n $PHASE_MEAN_OBJECTIVE ]]; then
  [[ "$PHASE_MEAN_OBJECTIVE" == mse || "$PHASE_MEAN_OBJECTIVE" == huber ]] \
    || die 'PHASE_MEAN_OBJECTIVE must be mse or huber'
fi
if [[ -n $PHASE_AUX_ISOLATION ]]; then
  [[ "$PHASE_AUX_ISOLATION" =~ ^[01]$ ]] || die 'PHASE_AUX_ISOLATION must be 0 or 1'
fi

COMMON=(
  --processed-root "$PROCESSED_ROOT"
  --batch_size "$BATCH_SIZE" --num_epochs "$NUM_EPOCHS"
  --num_workers "$NUM_WORKERS" --device cuda --seed "$SEED"
)
if [[ -n ${LR:-} ]]; then COMMON+=(--lr "$LR"); fi
if [[ -n $EARLY_STOP_PATIENCE ]]; then COMMON+=(--early_stop_patience "$EARLY_STOP_PATIENCE"); fi
if [[ -n $PHASE_MEAN_STEPS ]]; then COMMON+=(--predictor_mean_steps "$PHASE_MEAN_STEPS"); fi
if [[ -n $PHASE_VAR_STEPS ]]; then COMMON+=(--predictor_var_steps "$PHASE_VAR_STEPS"); fi
if [[ -n $PHASE_JOINT_LR_SCALE ]]; then COMMON+=(--predictor_joint_lr_scale "$PHASE_JOINT_LR_SCALE"); fi
if [[ -n $PHASE_MEAN_OBJECTIVE ]]; then COMMON+=(--predictor_mean_objective "$PHASE_MEAN_OBJECTIVE"); fi
if [[ -n $PHASE_TARGET_SCALE ]]; then COMMON+=(--predictor_target_scale "$PHASE_TARGET_SCALE"); fi
if [[ -n $PHASE_AUX_ISOLATION ]]; then COMMON+=(--predictor_aux_isolation "$PHASE_AUX_ISOLATION"); fi
if [[ "$DRY_RUN" == 1 ]]; then
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU_IDS"
  printf '%q ' "$PYTHON_BIN" -u scripts/train/train_stage_b.py "${COMMON[@]}" \
    --checkpoint_save "$CHECKPOINT_DIR/<unique-run>/stage_b_final.pt"
  printf '\n[preview] No training started or files written.\n'
  exit 0
fi
[[ -d "$PROCESSED_ROOT" ]] || die "Processed store missing: $PROCESSED_ROOT"
mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR"
run_dir="$(mktemp -d "$CHECKPOINT_DIR/stage_b.$(date +%Y%m%d_%H%M%S).XXXXXX")"
log_file="$LOG_DIR/$(basename "$run_dir").log"
echo "[start] GPU=$GPU_IDS epochs=$NUM_EPOCHS batch=$BATCH_SIZE processed=$PROCESSED_ROOT"
echo "[log] $log_file"
echo "[checkpoint] $run_dir/stage_b_final.pt"
echo '[policy] OCR scoring disabled; remaining damage weights normalized. Start a fresh checkpoint.'
echo "[monitor] tail -f '$log_file'"

if CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PYTHON_BIN" -u scripts/train/train_stage_b.py \
  "${COMMON[@]}" --checkpoint_save "$run_dir/stage_b_final.pt" > "$log_file" 2>&1; then
  echo "[done] Log: $log_file; checkpoints: $run_dir"
else
  status=$?
  echo "[error] Stage B exited with status $status. Last 40 log lines:" >&2
  tail -n 40 "$log_file" >&2 || true
  exit "$status"
fi
