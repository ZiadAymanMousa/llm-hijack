#!/usr/bin/env bash
# Create an isolated Azure workload for Phase 1 and start the job.
# Needs the Azure CLI (az login done) with the containerapp extension.
#   bash deploy/azure_setup.sh
# Set the names below first; ACR and STORAGE names must be globally unique.
set -euo pipefail

RG=${RG:-hijack-research}
LOCATION=${LOCATION:-uksouth}
ENV=${ENV:-hijack-env}
ACR=${ACR:?set ACR to a unique registry name, e.g. hijackacr123}
STORAGE=${STORAGE:?set STORAGE to a unique storage account name, e.g. hijackresults123}
JOB=${JOB:-hijack-phase1}

az extension add --name containerapp --upgrade --only-show-errors

echo "== resource group, registry, image"
az group create -n "$RG" -l "$LOCATION" -o none
az acr create -g "$RG" -n "$ACR" --sku Basic --admin-enabled true -o none
az acr build -r "$ACR" -t hijack:phase1 -f deploy/Dockerfile .

echo "== results share"
az storage account create -g "$RG" -n "$STORAGE" -l "$LOCATION" --sku Standard_LRS --kind StorageV2 -o none
az storage share-rm create -g "$RG" --storage-account "$STORAGE" -n results --quota 200 -o none
STORAGE_KEY=$(az storage account keys list -g "$RG" -n "$STORAGE" --query "[0].value" -o tsv)

echo "== Container Apps environment with a serverless A100 profile"
az containerapp env create -g "$RG" -n "$ENV" -l "$LOCATION" --enable-workload-profiles -o none
az containerapp env workload-profile add -g "$RG" -n "$ENV" \
  --workload-profile-name gpu-a100 --workload-profile-type Consumption-GPU-NC24-A100 -o none
az containerapp env storage set -g "$RG" -n "$ENV" --storage-name results \
  --azure-file-account-name "$STORAGE" --azure-file-account-key "$STORAGE_KEY" \
  --azure-file-share-name results --access-mode ReadWrite -o none

echo "== job"
export LOCATION ACR
export ENV_ID=$(az containerapp env show -g "$RG" -n "$ENV" --query id -o tsv)
export ACR_USER=$(az acr credential show -n "$ACR" --query username -o tsv)
export ACR_PASSWORD=$(az acr credential show -n "$ACR" --query "passwords[0].value" -o tsv)
JOB_YAML=$(mktemp)
trap 'rm -f "$JOB_YAML"' EXIT
python3 -c "import os,sys; print(os.path.expandvars(open('deploy/job.yaml').read()))" > "$JOB_YAML"
az containerapp job create -g "$RG" -n "$JOB" --yaml "$JOB_YAML" -o none

az containerapp job start -g "$RG" -n "$JOB" -o none
echo "started. Follow logs with:"
echo "  az containerapp job logs show -g $RG -n $JOB --container phase1 --follow"
