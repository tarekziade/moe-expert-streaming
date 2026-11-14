#!/bin/bash
# run_guarded.sh LOG SCRIPT ARGS... : run a Python script under a memory watchdog, so a run that pushes macOS into
# swapping is killed instead of freezing the machine.
#
# Kills the run on critical memory pressure or when swap grows by more than 2 GB. Close other apps (browsers) before
# benchmarking: they cost ~0.7 tok/s on a 32 GB Mac and make the watchdog fire more often.
#
#   PYTHON=python3 ./scripts/run_guarded.sh story.log scripts/bench.py /path/to/converted --cache-gib 12 \
#       --dense-bits 8 --io-threads 8 --skip-prefill --output story.json
set -uo pipefail
log=$1; shift
PYTHON=${PYTHON:-python3}
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-0.9}
export PYTORCH_MPS_LOW_WATERMARK_RATIO=${PYTORCH_MPS_LOW_WATERMARK_RATIO:-0.8}
swap_mb() { sysctl -n vm.swapusage | awk '{gsub("M","",$6); print int($6)}'; }
"$PYTHON" "$@" > "$log" 2>&1 &
pid=$!; start=$(swap_mb)
while kill -0 "$pid" 2>/dev/null; do
  level=$(sysctl -n kern.memorystatus_vm_pressure_level 2>/dev/null || echo 1)
  if [ "$level" -ge 4 ] || [ $(( $(swap_mb) - start )) -gt 2048 ]; then
    echo "WATCHDOG: pressure $level, swap +$(( $(swap_mb) - start )) MB: killing $pid" | tee -a "$log"
    kill -9 "$pid"; exit 1
  fi
  sleep 1
done
wait "$pid"; status=$?; echo "exit $status" >> "$log"; exit $status
