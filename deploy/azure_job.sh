#!/usr/bin/env bash
# Create (or update) the Phase 1 job from deploy/job.yaml and start it.
#   bash deploy/azure_job.sh             # create/update and start
#   START=0 bash deploy/azure_job.sh     # create/update only
# Values are filled into the template in bash itself: Git Bash on Windows rewrites
# environment values that look like paths (such as Azure resource IDs) for other programs.
set -euo pipefail

RG=${RG:-hijack-research}
LOCATION=${LOCATION:-uksouth}
ENV=${ENV:-hijack-env}
JOB=${JOB:-hijack-phase1}
START=${START:-1}

tsv() { az "$@" -o tsv 2>/dev/null | tr -d '\r'; }   # az output without Windows line endings

ACR=${ACR:-$(tsv acr list -g "$RG" --query "[0].name")}
ENV_ID=$(tsv containerapp env show -g "$RG" -n "$ENV" --query id)
ACR_USER=$(tsv acr credential show -n "$ACR" --query username)
ACR_PASSWORD=$(tsv acr credential show -n "$ACR" --query "passwords[0].value")
for name in ACR ENV_ID ACR_USER ACR_PASSWORD; do
  [[ -n "${!name}" ]] || { echo "could not read $name (is the setup complete and az logged in?)"; exit 1; }
done

yaml=$(<deploy/job.yaml)
for name in LOCATION ENV_ID ACR ACR_USER ACR_PASSWORD; do
  yaml=${yaml//"\${$name}"/"${!name}"}
done
JOB_YAML=$(mktemp)
trap 'rm -f "$JOB_YAML"' EXIT
printf '%s\n' "$yaml" > "$JOB_YAML"

if az containerapp job show -g "$RG" -n "$JOB" -o none 2>/dev/null; then
  az containerapp job update -g "$RG" -n "$JOB" --yaml "$JOB_YAML" -o none
else
  az containerapp job create -g "$RG" -n "$JOB" --yaml "$JOB_YAML" -o none
fi
echo "job $JOB is ready"

if [[ "$START" == 1 ]]; then
  az containerapp job start -g "$RG" -n "$JOB" -o none
  echo "started. Follow the logs with:"
  echo "  az containerapp job logs show -g $RG -n $JOB --container phase1 --follow"
fi
