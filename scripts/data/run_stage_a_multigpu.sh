#!/usr/bin/env bash
# Run from a new project/data directory after changes to Stage-A labels.
# Smoke: NUM_GPUS=1 LIMIT=10 FAIL_FAST=1 PROCESSED_ROOT=./processed_smoke bash "$0"
# Preview commands: DRY_RUN=1 GPU_IDS=0,1 NUM_GPUS=2 bash "$0"
# Resume an eight-shard store on one GPU: NUM_SHARDS=8 NUM_GPUS=1 bash "$0"
# Retry selected shards without finalizing: NUM_SHARDS=8 NUM_GPUS=1 SHARD_IDS=0,1,2 bash "$0"
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

PYTHON_BIN="${PYTHON_BIN:-python3}"
NUM_GPUS="${NUM_GPUS:-}"
NUM_SHARDS="${NUM_SHARDS:-}"
MAX_CONCURRENT="${MAX_CONCURRENT:-}"
FINALIZE_PARTIAL="${FINALIZE_PARTIAL:-0}"
RAM_PER_WORKER_GIB="${RAM_PER_WORKER_GIB:-80}"
OPENVID_CSV="${OPENVID_CSV:-/xxx/pkg/data/train/OpenVidHD.csv}"
DATA_ROOT="${DATA_ROOT:-/xxx/pkg/OpenVidHD_part}"
VIDEO_SUBDIR="${VIDEO_SUBDIR:-OpenVidHD_part_1}"
PROCESSED_ROOT="${PROCESSED_ROOT:-./LCOCF_OpenVid1M_Processed}"
WEIGHTS_ROOT="${WEIGHTS_ROOT:-/xxx/LongVideo/model/weights}"
BACKBONE="${BACKBONE:-wan22}"
WAN_VARIANT="${WAN_VARIANT:-a14b-t2v}"
MODEL_PATH="${MODEL_PATH:-${WEIGHTS_ROOT}/Wan2.2-T2V-A14B-Diffusers}"
BACKBONE_DTYPE="${BACKBONE_DTYPE:-bfloat16}"
SAM_MODEL="${SAM_MODEL:-${WEIGHTS_ROOT}/SAM}"
DINO_MODEL="${DINO_MODEL:-${WEIGHTS_ROOT}/DINO}"
CLIP_MODEL="${CLIP_MODEL:-${WEIGHTS_ROOT}/Clip}"
RAFT_WEIGHTS="${RAFT_WEIGHTS:-${WEIGHTS_ROOT}/raft}"
NUM_FRAMES="${NUM_FRAMES:-49}"
HEIGHT="${HEIGHT:-384}"
WIDTH="${WIDTH:-640}"
VAE_TILE="${VAE_TILE:-256}"
METRIC_FRAME_CHUNK="${METRIC_FRAME_CHUNK:-4}"
SAM_POINTS_PER_CROP="${SAM_POINTS_PER_CROP:-16}"
PERCEPTION_DTYPE="${PERCEPTION_DTYPE:-bfloat16}"
LIMIT="${LIMIT:-200}"
SEED="${SEED:-1234}"
LOG_DIR="${LOG_DIR:-logs}"
FAIL_FAST="${FAIL_FAST:-1}"
DRY_RUN="${DRY_RUN:-0}"

die() { echo "[error] $*" >&2; exit 1; }
command -v "$PYTHON_BIN" >/dev/null || die "Python not found: $PYTHON_BIN"
[[ "$FAIL_FAST" =~ ^[01]$ && "$DRY_RUN" =~ ^[01]$ ]] || die "FAIL_FAST and DRY_RUN must be 0 or 1"
[[ "$FINALIZE_PARTIAL" =~ ^[01]$ ]] || die "FINALIZE_PARTIAL must be 0 or 1"
for value in "$LIMIT" "$NUM_FRAMES" "$HEIGHT" "$WIDTH" "$VAE_TILE" "$METRIC_FRAME_CHUNK" "$SAM_POINTS_PER_CROP"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "Expected a positive integer, got: $value"
done

# Honor scheduler/container visibility, including GPU UUIDs. GPU_IDS can select
# a subset but cannot escape an existing CUDA_VISIBLE_DEVICES restriction.
visible="${CUDA_VISIBLE_DEVICES-}"
if [[ ${GPU_IDS+x} ]]; then
  selected="$GPU_IDS"
elif [[ ${CUDA_VISIBLE_DEVICES+x} ]]; then
  selected="$visible"
else
  command -v nvidia-smi >/dev/null || die "No nvidia-smi; specify GPU_IDS for command preview"
  selected="$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)"
fi
[[ -n "$selected" && "$selected" != -1 ]] || die "No GPUs selected"
[[ "$selected" != ,* && "$selected" != *, && "$selected" != *,,* ]] || die "Invalid GPU list"
IFS=',' read -r -a gpu_ids <<< "$selected"
NUM_GPUS="${NUM_GPUS:-${#gpu_ids[@]}}"
[[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]] || die "NUM_GPUS must be positive"
(( NUM_GPUS <= ${#gpu_ids[@]} )) || die "NUM_GPUS exceeds selected GPU count"
for ((i=0; i<NUM_GPUS; i++)); do
  gpu="${gpu_ids[$i]}"
  [[ "$gpu" =~ ^([0-9]+|GPU-[a-zA-Z0-9-]+|MIG-[a-zA-Z0-9/-]+)$ ]] || die "Invalid GPU identifier: $gpu"
  if [[ ${CUDA_VISIBLE_DEVICES+x} ]]; then
    [[ ",$visible," == *",$gpu,"* ]] || die "GPU $gpu is outside CUDA_VISIBLE_DEVICES"
  fi
  for ((j=0; j<i; j++)); do
    [[ "$gpu" != "${gpu_ids[$j]}" ]] || die "Duplicate GPU: $gpu"
  done
done

NUM_SHARDS="${NUM_SHARDS:-$NUM_GPUS}"
MAX_CONCURRENT="${MAX_CONCURRENT:-$NUM_GPUS}"
for value in "$NUM_SHARDS" "$MAX_CONCURRENT" "$RAM_PER_WORKER_GIB"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "Expected a positive integer, got: $value"
done
(( MAX_CONCURRENT <= NUM_GPUS )) || die "MAX_CONCURRENT exceeds NUM_GPUS"
shard_ids=()
if [[ ${SHARD_IDS+x} ]]; then
  [[ "$SHARD_IDS" =~ ^(0|[1-9][0-9]*)(,(0|[1-9][0-9]*))*$ ]] || die "Invalid SHARD_IDS"
  IFS=',' read -r -a shard_ids <<< "$SHARD_IDS"
else
  for ((i=0; i<NUM_SHARDS; i++)); do shard_ids+=("$i"); done
fi
for ((i=0; i<${#shard_ids[@]}; i++)); do
  (( shard_ids[i] < NUM_SHARDS )) || die "Shard ${shard_ids[$i]} exceeds NUM_SHARDS"
  for ((j=0; j<i; j++)); do
    [[ "${shard_ids[$i]}" != "${shard_ids[$j]}" ]] || die "Duplicate shard: ${shard_ids[$i]}"
  done
done
(( MAX_CONCURRENT <= ${#shard_ids[@]} )) || MAX_CONCURRENT=${#shard_ids[@]}

COMMON=(
  --openvid-csv "$OPENVID_CSV" --data-root "$DATA_ROOT" --video-subdir "$VIDEO_SUBDIR"
  --only-existing-videos --processed-root "$PROCESSED_ROOT"
  --backbone "$BACKBONE" --wan-variant "$WAN_VARIANT"
  --model-path "$MODEL_PATH" --backbone-dtype "$BACKBONE_DTYPE"
  --num-frames "$NUM_FRAMES" --height "$HEIGHT" --width "$WIDTH"
  --vae-tile "$VAE_TILE" --metric-frame-chunk "$METRIC_FRAME_CHUNK"
  --real-models --sam-points-per-crop "$SAM_POINTS_PER_CROP"
  --perception-dtype "$PERCEPTION_DTYPE" --sam-model "$SAM_MODEL"
  --dino-model "$DINO_MODEL" --clip-model "$CLIP_MODEL" --raft-weights "$RAFT_WEIGHTS"
  --limit "$LIMIT" --device cuda --seed "$SEED"
)
[[ "$FAIL_FAST" == 0 ]] || COMMON+=(--fail-fast)
if [[ "$DRY_RUN" == 1 ]]; then
  echo "[preview] total_shards=$NUM_SHARDS concurrent=$MAX_CONCURRENT selected=${shard_ids[*]}"
  for ((i=0; i<${#shard_ids[@]}; i++)); do
    printf 'CUDA_VISIBLE_DEVICES=%q ' "${gpu_ids[$((i % MAX_CONCURRENT))]}"
    printf '%q ' "$PYTHON_BIN" -u scripts/data/generate_counterfactual_data.py "${COMMON[@]}" --num-shards "$NUM_SHARDS" --shard-index "${shard_ids[$i]}"
    printf '\n'
  done
  echo '[preview] No models loaded, paths checked, or files written.'
  exit 0
fi

[[ -f "$OPENVID_CSV" ]] || die "CSV missing: $OPENVID_CSV"
[[ -d "$DATA_ROOT/$VIDEO_SUBDIR" ]] || die "Video directory missing: $DATA_ROOT/$VIDEO_SUBDIR"
for path in "$MODEL_PATH" "$SAM_MODEL" "$DINO_MODEL" "$CLIP_MODEL" "$RAFT_WEIGHTS"; do
  [[ -d "$path" ]] || die "Local model directory missing: $path"
done
command -v flock >/dev/null || die "flock is required to prevent concurrent writers"
mkdir -p "$PROCESSED_ROOT" "$LOG_DIR"
exec 9>"$PROCESSED_ROOT/.stage_a.lock"
flock -n 9 || die "Another launch is writing to $PROCESSED_ROOT"

# The manifest prevents changing shard routing or label settings on a resumed run.
"$PYTHON_BIN" - "$PROCESSED_ROOT" "$NUM_SHARDS" "${COMMON[@]}" <<'PY'
import json
import difflib
import pathlib
import sys
root = pathlib.Path(sys.argv[1])
manifest = root / '.stage_a_launch.json'
config = {'num_shards': int(sys.argv[2]), 'args': [a for a in sys.argv[3:] if a != '--fail-fast']}
if manifest.exists():
    saved = json.loads(manifest.read_text())
    if saved != config:
        print(f'[error] Resume configuration differs from {manifest}', file=sys.stderr)
        print(''.join(difflib.unified_diff(
            json.dumps(saved, indent=2, sort_keys=True).splitlines(keepends=True),
            json.dumps(config, indent=2, sort_keys=True).splitlines(keepends=True),
            fromfile='saved configuration (-)', tofile='requested configuration (+)',
        )), file=sys.stderr)
        raise SystemExit('Restore the saved settings to resume. Do not delete/edit the manifest to bypass this check. For intentionally changed label settings, use a new PROCESSED_ROOT.')
else:
    existing = [p for p in root.iterdir() if p.name != '.stage_a.lock']
    if existing:
        raise SystemExit('Unversioned nonempty processed directory. Use a new PROCESSED_ROOT to avoid mixing old labels.')
    with manifest.open('x') as f:
        json.dump(config, f, indent=2)
PY

RUN_LOG_DIR="$(mktemp -d "$LOG_DIR/stage_a.$(date +%Y%m%d_%H%M%S).XXXXXX")"
echo "[start] shards=$NUM_SHARDS concurrent=$MAX_CONCURRENT selected=${shard_ids[*]} limit=$LIMIT processed=$PROCESSED_ROOT logs=$RUN_LOG_DIR"
"$PYTHON_BIN" --version
memory_snapshot() {
  echo "[memory] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  for root in /sys/fs/cgroup /sys/fs/cgroup/memory; do
    for name in memory.max memory.current memory.peak memory.events \
      memory.limit_in_bytes memory.usage_in_bytes memory.max_usage_in_bytes memory.oom_control; do
      if [[ -r "$root/$name" ]]; then
        echo "$root/$name"
        cat "$root/$name" || true
      fi
    done
  done
}
memory_snapshot >> "$RUN_LOG_DIR/memory.log"
"$PYTHON_BIN" - "$MAX_CONCURRENT" "$RAM_PER_WORKER_GIB" <<'PY'
import pathlib
import sys

gib = 1024 ** 3
available = []
info = pathlib.Path('/proc/meminfo')
if info.exists():
    for line in info.read_text().splitlines():
        if line.startswith('MemAvailable:'):
            available.append(int(line.split()[1]) * 1024)
            print(f'[memory] host_available={available[-1] / gib:.1f}GiB')
roots = {pathlib.Path('/sys/fs/cgroup'), pathlib.Path('/sys/fs/cgroup/memory')}
membership = pathlib.Path('/proc/self/cgroup')
if membership.exists():
    for line in membership.read_text().splitlines():
        hierarchy, controllers, path = line.split(':', 2)
        if hierarchy == '0' and not controllers:
            roots.add(pathlib.Path('/sys/fs/cgroup') / path.lstrip('/'))
        elif 'memory' in controllers.split(','):
            roots.add(pathlib.Path('/sys/fs/cgroup/memory') / path.lstrip('/'))
found = False
for root in roots:
    for limit_name, usage_name in [('memory.max', 'memory.current'),
                                   ('memory.limit_in_bytes', 'memory.usage_in_bytes')]:
        try:
            limit = int((root / limit_name).read_text().strip())
            usage = int((root / usage_name).read_text().strip())
        except (OSError, ValueError):
            continue
        if limit >= 2 ** 60:
            continue
        found = True
        available.append(max(0, limit - usage))
        print(f'[memory] {root}: limit={limit / gib:.1f}GiB used={usage / gib:.1f}GiB')
if not found:
    print('[warn] No finite cgroup limit detected; ancestor/platform limits may still apply')
need = int(sys.argv[1]) * int(sys.argv[2]) * gib
print(f'[memory] estimated_worker_demand={need / gib:.1f}GiB (estimate, not a guarantee)')
if available and need > min(available) * 0.9:
    raise SystemExit('[error] Estimated demand exceeds 90% of available memory; reduce MAX_CONCURRENT')
PY

pids=()
cleanup() {
  trap - EXIT INT TERM
  for pid in ${pids[@]+"${pids[@]}"}; do
    kill "$pid" 2>/dev/null || true
  done
  for pid in ${pids[@]+"${pids[@]}"}; do
    wait "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
failed=()
# Batch scheduling bounds both GPU usage and simultaneous model loads.
for ((start=0; start<${#shard_ids[@]}; start+=MAX_CONCURRENT)); do
  pids=()
  for ((slot=0; slot<MAX_CONCURRENT && start+slot<${#shard_ids[@]}; slot++)); do
    shard="${shard_ids[$((start+slot))]}"
    CUDA_VISIBLE_DEVICES="${gpu_ids[$slot]}" "$PYTHON_BIN" -u scripts/data/generate_counterfactual_data.py \
      "${COMMON[@]}" --num-shards "$NUM_SHARDS" --shard-index "$shard" \
      >"$RUN_LOG_DIR/stage_a.s$shard.log" 2>&1 &
    pids+=("$!")
    echo "[launch] shard=$shard GPU=${gpu_ids[$slot]} pid=$!"
  done
  batch_count=${#pids[@]}
  for ((slot=0; slot<batch_count; slot++)); do
    shard="${shard_ids[$((start+slot))]}"
    if wait "${pids[$slot]}"; then
      echo "[exit] shard=$shard status=0"
    else
      status=$?
      echo "[exit] shard=$shard status=$status log=$RUN_LOG_DIR/stage_a.s$shard.log" >&2
      failed+=("$shard")
    fi
    unset 'pids[slot]'
  done
  pids=()
  memory_snapshot >> "$RUN_LOG_DIR/memory.log"
  (( ${#failed[@]} == 0 )) || break
done
pids=()
(( ${#failed[@]} == 0 )) || die "Failed shards: ${failed[*]}; finalize skipped"

if (( ${#shard_ids[@]} < NUM_SHARDS )) && [[ "$FINALIZE_PARTIAL" == 0 ]]; then
  echo '[done] Selected shards finished; finalize skipped. Run all shards to resume/check completion before merging.'
  exit 0
fi
if (( NUM_SHARDS > 1 )); then
  echo '[finalize] Merging indexes; see finalize.log'
  "$PYTHON_BIN" -u scripts/data/generate_counterfactual_data.py "${COMMON[@]}" \
    --num-shards "$NUM_SHARDS" --shard-index 0 --finalize-only \
    >"$RUN_LOG_DIR/finalize.log" 2>&1 &
  pids=("$!")
  wait "${pids[0]}" || die "Finalize failed; see $RUN_LOG_DIR/finalize.log"
  pids=()
fi
if [[ "$FAIL_FAST" == 0 ]]; then
  echo '[warn] Tolerant mode: successful exit can include failed clips. Check shard summaries and _failed.sNN.jsonl against completed progress.'
fi
echo "[done] Processed store: $PROCESSED_ROOT; logs: $RUN_LOG_DIR"
