#!/usr/bin/env bash
# Download a phase's results from the Azure Files share into results/.
#   bash tools/fetch_results.sh 1          # analysis only (phase1/report)
#   bash tools/fetch_results.sh 1 --logs   # analysis and episode logs
set -euo pipefail

PHASE=${1:-1}
LOGS=${2:-}
RG=${RG:-hijack-research}
STORAGE=${STORAGE:-$(az storage account list -g "$RG" --query "[0].name" -o tsv 2>/dev/null | tr -d '\r')}
KEY=$(az storage account keys list -g "$RG" -n "$STORAGE" --query "[0].value" -o tsv 2>/dev/null | tr -d '\r')

pattern="phase$PHASE/report/*"
[[ "$LOGS" == "--logs" ]] && pattern="phase$PHASE/*"

mkdir -p results
PYTHONIOENCODING=utf-8 az storage file download-batch --account-name "$STORAGE" --account-key "$KEY" \
  --source results --destination results --pattern "$pattern" -o none
echo "downloaded $pattern into results/"
