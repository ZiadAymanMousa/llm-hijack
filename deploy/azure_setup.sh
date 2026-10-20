#!/usr/bin/env bash
# Create an isolated Azure workload for Phase 1, then create and start the job (deploy/azure_job.sh).
# Needs the Azure CLI (az login done) with the containerapp extension.
#   ACR=<unique-registry-name> STORAGE=<unique-storage-name> bash deploy/azure_setup.sh
# ACR and STORAGE names must be globally unique.
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
STORAGE_KEY=$(az storage account keys list -g "$RG" -n "$STORAGE" --query "[0].value" -o tsv | tr -d '\r')

echo "== Container Apps environment with a serverless A100 profile"
az containerapp env create -g "$RG" -n "$ENV" -l "$LOCATION" --enable-workload-profiles -o none
az containerapp env workload-profile add -g "$RG" -n "$ENV" \
  --workload-profile-name gpu-a100 --workload-profile-type Consumption-GPU-NC24-A100 -o none
az containerapp env storage set -g "$RG" -n "$ENV" --storage-name results \
  --azure-file-account-name "$STORAGE" --azure-file-account-key "$STORAGE_KEY" \
  --azure-file-share-name results --access-mode ReadWrite -o none

echo "== job"
RG=$RG LOCATION=$LOCATION ENV=$ENV ACR=$ACR JOB=$JOB bash deploy/azure_job.sh
