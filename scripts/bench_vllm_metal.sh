#!/bin/bash
# bench_vllm_metal.sh OUT_DIR [MODEL] : the three bench rows on vllm-metal, as scripts/vllm_metal_parity.py measures them.
#
# Serves MODEL (default mlx-community/Qwen3.6-35B-A3B-8bit) with a 12 GB expert cache and 1,024-token prefill chunks
# under the same memory watchdog as run_guarded.sh, waits for it, runs the parity client, then stops the server.
# Needs ~/.venv-vllm-metal (vllm-metal's install.sh). Close browsers first.
set -uo pipefail
out=$1; model=${2:-mlx-community/Qwen3.6-35B-A3B-8bit}; port=${PORT:-8123}; cache_gb=${CACHE_GB:-12}
# The default KV pool is 0.92 of free GPU memory (7 GB here) for one 8K sequence (~0.2 GB); cap it like the other engines.
kv_bytes=${KV_BYTES:-2147483648}
mkdir -p "$out"
here=$(cd "$(dirname "$0")" && pwd)
venv=${VLLM_METAL_VENV:-$HOME/.venv-vllm-metal}
swap_mb() { sysctl -n vm.swapusage | awk '{gsub("M","",$6); print int($6)}'; }

VLLM_METAL_MULTIMODAL_MODE=text-only "$venv/bin/vllm" serve "$model" --port "$port" \
  --max-model-len 8192 --max-num-seqs 1 --max-num-batched-tokens 1024 --kv-cache-memory-bytes "$kv_bytes" \
  --additional-config "{\"moe_expert_offload\": {\"cache_gb\": $cache_gb}}" > "$out/server.log" 2>&1 &
server=$!; start=$(swap_mb)
guard() {
  while kill -0 "$server" 2>/dev/null; do
    level=$(sysctl -n kern.memorystatus_vm_pressure_level 2>/dev/null || echo 1)
    if [ "$level" -ge 4 ] || [ $(( $(swap_mb) - start )) -gt ${SWAP_LIMIT_MB:-2048} ]; then
      echo "WATCHDOG: pressure $level, swap +$(( $(swap_mb) - start )) MB: killing server" | tee -a "$out/server.log"
      pkill -9 -P "$server"; kill -9 "$server"; return
    fi
    sleep 1
  done
}
guard & watchdog=$!
trap 'pkill -P "$server" 2>/dev/null; kill "$server" "$watchdog" 2>/dev/null' EXIT

until curl -sf "http://localhost:$port/health" > /dev/null; do
  kill -0 "$server" 2>/dev/null || { echo "server exited"; tail -30 "$out/server.log"; exit 1; }
  sleep 2
done
echo "server ready"
"$venv/bin/python" -I "$here/vllm_metal_parity.py" "$model" "$port" > "$out/parity.json" 2> "$out/parity.err"
status=$?
tail -1 "$out/parity.json"; echo "swap growth: $(( $(swap_mb) - start )) MB" | tee -a "$out/parity.json"
exit $status
