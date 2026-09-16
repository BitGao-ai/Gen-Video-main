#!/usr/bin/env bash
# Sequential MSE/NLL predictor-only experiments on the same subset and seed.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
PROCESSED_ROOT="${PROCESSED_ROOT:-./LCOCF_OpenVid1M_Processed}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./checkpoints}"
TRAIN_VIDEOS="${TRAIN_VIDEOS:-8}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SEED="${SEED:-1234}"
DEVICE="${DEVICE:-cpu}"
LR="${LR:-0.001}"
DRY_RUN="${DRY_RUN:-0}"

die() { echo "[error] $*" >&2; exit 1; }
command -v "$PYTHON_BIN" >/dev/null || die "Python not found: $PYTHON_BIN"
[[ "$DRY_RUN" =~ ^[01]$ ]] || die 'DRY_RUN must be 0 or 1'
[[ "$DEVICE" == cpu || "$DEVICE" == cuda ]] || die 'DEVICE must be cpu or cuda'
for value in "$TRAIN_VIDEOS" "$EPOCHS" "$BATCH_SIZE"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "Expected positive integer: $value"
done
[[ "$SEED" =~ ^(0|[1-9][0-9]*)$ ]] || die 'SEED must be a nonnegative integer'
COMMON=(--processed-root "$PROCESSED_ROOT" --train-videos "$TRAIN_VIDEOS"
  --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --seed "$SEED" --device "$DEVICE" --lr "$LR")
if [[ "$DRY_RUN" == 1 ]]; then
  for objective in mse nll; do
    printf '%q ' "$PYTHON_BIN" -u scripts/train/diagnose_stage_b_fit.py "${COMMON[@]}" \
      --objective "$objective" --output-dir "$OUTPUT_ROOT/<unique-run>/$objective"
    printf '\n'
  done
  exit 0
fi
[[ -d "$PROCESSED_ROOT" ]] || die "Processed store missing: $PROCESSED_ROOT"
mkdir -p "$OUTPUT_ROOT"
RUN_DIR="$(mktemp -d "$OUTPUT_ROOT/stage_b_fit_probe.$(date +%Y%m%d_%H%M%S).XXXXXX")"
echo "[start] device=$DEVICE videos=$TRAIN_VIDEOS epochs=$EPOCHS seed=$SEED"
echo "[output] $RUN_DIR"
for objective in mse nll; do
  log_file="$RUN_DIR/$objective.log"
  echo "[run] $objective; monitor: tail -f '$log_file'"
  if "$PYTHON_BIN" -u scripts/train/diagnose_stage_b_fit.py "${COMMON[@]}" \
    --objective "$objective" --output-dir "$RUN_DIR/$objective" > "$log_file" 2>&1; then
    echo "[done] $RUN_DIR/$objective/history.jsonl"
  else
    status=$?
    tail -n 40 "$log_file" >&2 || true
    echo "[error] $objective failed (exit=$status); see $log_file. Remaining experiments skipped." >&2
    exit "$status"
  fi
done
echo "[done] Both experiments completed: $RUN_DIR"
echo '[note] Diagnostic models only; not production checkpoints.'
