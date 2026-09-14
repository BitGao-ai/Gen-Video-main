#!/usr/bin/env bash
# Usage: bash scripts/data/diagnose_stage_a.sh [run_log_dir] [report_file]
# Read-only diagnostics, except for the newly created report file.
set -euo pipefail
cd "$(dirname "$0")/../.."

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  echo "Usage: bash $0 [run_log_dir] [report_file]"
  echo 'Defaults: newest logs/stage_a.* directory; a unique report under /tmp.'
  exit 0
fi
(( $# <= 2 )) || { echo 'Expected at most two arguments' >&2; exit 1; }

run_dir="${1:-}"
if [[ -z "$run_dir" ]]; then
  shopt -s nullglob
  for candidate in logs/stage_a.*; do
    [[ -d "$candidate" ]] || continue
    if [[ -z "$run_dir" || "$candidate" -nt "$run_dir" ]]; then
      run_dir="$candidate"
    fi
  done
fi
[[ -n "$run_dir" && -d "$run_dir" ]] || {
  echo 'Run log directory not found; pass its path as the first argument.' >&2
  exit 1
}

if [[ -n ${2:-} ]]; then
  report="$2"
  # Do not overwrite an existing report or source log.
  (set -o noclobber; : > "$report") || exit 1
else
  report="$(mktemp "${TMPDIR:-/tmp}/stage_a_diagnostic.XXXXXX")"
fi

section() { printf '\n========== %s ==========\n' "$*"; }
read_file() {
  if [[ -r "$1" ]]; then
    printf '\n--- %s ---\n' "$1"
    cat "$1" || true
  fi
}

collect() {
  section 'Context'
  date -Is 2>/dev/null || date
  printf 'Project: %s\nRun logs: %s\n' "$PWD" "$run_dir"
  echo 'Memory events are cumulative; these counters alone cannot date an OOM.'

  section 'Host memory (not necessarily the container allowance)'
  if command -v free >/dev/null 2>&1; then free -h || true; fi
  read_file /proc/meminfo

  section 'Cgroup membership and mounts'
  read_file /proc/self/cgroup
  if [[ -r /proc/self/mountinfo ]]; then
    grep -E ' - cgroup2? ' /proc/self/mountinfo || true
  fi

  section 'Cgroup memory limits and events'
  # Check namespace roots and membership paths. Mount details above allow
  # diagnosis when a platform uses a nonstandard cgroup mount layout.
  roots=(/sys/fs/cgroup /sys/fs/cgroup/memory)
  if [[ -r /proc/self/cgroup ]]; then
    while IFS=: read -r hierarchy controllers path; do
      if [[ "$hierarchy" == 0 && -z "$controllers" ]]; then
        roots+=("/sys/fs/cgroup${path}")
      elif [[ ",$controllers," == *,memory,* ]]; then
        roots+=("/sys/fs/cgroup/memory${path}")
      fi
    done < /proc/self/cgroup
  fi
  for root in "${roots[@]}"; do
    for name in memory.max memory.high memory.current memory.peak memory.events \
      memory.events.local memory.swap.max memory.swap.current \
      memory.limit_in_bytes memory.usage_in_bytes memory.max_usage_in_bytes \
      memory.failcnt memory.oom_control; do
      read_file "$root/$name"
    done
  done

  section 'GPU snapshot'
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
  else
    echo 'nvidia-smi unavailable'
  fi

  section 'Shard log tails (last 80 lines each)'
  shopt -s nullglob
  logs=("$run_dir"/stage_a.s*.log)
  if (( ${#logs[@]} == 0 )); then echo 'No shard logs found'; fi
  for log in "${logs[@]}"; do
    section "$log"
    tail -n 80 "$log" || true
  done
  if [[ -f "$run_dir/finalize.log" ]]; then
    section 'Finalize log'
    tail -n 80 "$run_dir/finalize.log" || true
  fi

  section 'Recent kernel messages (permission denial is normal in containers)'
  if command -v dmesg >/dev/null 2>&1; then
    dmesg -T 2>&1 | tail -n 100 || true
  fi
}

collect 2>&1 | tee "$report"
printf '\nReport: %s\n' "$report"
