#!/usr/bin/env bash
# Start one vLLM server per model in the config and wait until all of them answer.
#   bash deploy/serve.sh configs/phase1.yaml
# Each server gets GPU_FRACTION of the GPU (default 0.2: four 4B models on one A100 80GB).
set -euo pipefail

CONFIG=${1:-configs/phase1.yaml}
FRACTION=${GPU_FRACTION:-0.2}
MAX_LEN=${MAX_MODEL_LEN:-32768}
LOG_DIR=${LOG_DIR:-results/logs}
mkdir -p "$LOG_DIR"

# name, hf_id, revision, port, chat template (or "-")
mapfile -t MODELS < <(python3 - "$CONFIG" <<'EOF'
import sys, yaml
from urllib.parse import urlparse
cfg = yaml.safe_load(open(sys.argv[1]))
for name, m in cfg["models"].items():
    print(name, m["hf_id"], m["revision"], urlparse(m["url"]).port, m.get("chat_template", "-"), sep="\t")
EOF
)

for line in "${MODELS[@]}"; do
  IFS=$'\t' read -r name hf_id revision port template <<< "$line"
  extra=()
  if [[ "$template" != "-" ]]; then
    path=$(python3 -c "from huggingface_hub import hf_hub_download as d; print(d('$hf_id', '$template', revision='$revision'))")
    extra+=(--chat-template "$path")
  fi
  echo "starting $name ($hf_id@${revision:0:8}) on port $port"
  vllm serve "$hf_id" --revision "$revision" --served-model-name "$name" --port "$port" \
    --gpu-memory-utilization "$FRACTION" --max-model-len "$MAX_LEN" --enable-prefix-caching \
    "${extra[@]}" > "$LOG_DIR/vllm_$name.log" 2>&1 &
  # Servers start one after another so their memory profiling does not overlap.
  for _ in $(seq 1 180); do
    curl -sf "http://localhost:$port/v1/models" > /dev/null && break
    sleep 5
  done
  curl -sf "http://localhost:$port/v1/models" > /dev/null || { echo "$name did not start; see $LOG_DIR/vllm_$name.log"; exit 1; }
  echo "$name ready"
done
