# Running Phase 1 on Azure (serverless A100)

The job runs one container on an A100 80GB (Azure Container Apps, workload profile `Consumption-GPU-NC24-A100`). The container serves four 4B models with vLLM (0.2 of GPU memory each) and runs `deploy/run_phase1.sh`. Results go to an Azure Files share mounted at `/app/results`.

## What gets created

| Resource | Purpose |
|---|---|
| Resource group (`hijack-research`) | Keeps the study separate from any other workload |
| Container registry | Holds the image built from `deploy/Dockerfile` |
| Storage account + file share `results` | Run logs, reports and the Hugging Face model cache |
| Container Apps environment with a `gpu-a100` profile | Serverless A100; billed only while the job runs |
| Container Apps job `hijack-phase1` (manual trigger) | Runs Phase 1 once per start |

## Steps

1. Check that the subscription has serverless GPU quota for `Consumption-GPU-NC24-A100` in the chosen region (default `uksouth`).
2. From the project folder, with the Azure CLI signed in:

   ```bash
   ACR=<unique-registry-name> STORAGE=<unique-storage-name> bash deploy/azure_setup.sh
   ```

3. Follow the logs:

   ```bash
   az containerapp job logs show -g hijack-research -n hijack-phase1 --container phase1 --follow
   ```

4. When the job finishes, download `phase1/report/` from the `results` share (Azure Portal → Storage account → File shares, or `az storage file download-batch`).

A stopped or failed job can be started again with `az containerapp job start -g hijack-research -n hijack-phase1`; every step resumes from the results already on the share.

## Running pieces by hand

Inside the container (or on any Linux machine with one 80GB GPU):

```bash
bash deploy/serve.sh configs/phase1.yaml
python3 -m hijack.run --config configs/phase1.yaml --arm A --benchmark agentdojo --suites banking --limit 1
```

## Notes

- All models are public; no Hugging Face token is needed.
- Arm P (Progent) uses the base model on port 8000 to write its least-privilege policies.
- `HF_HOME` points at the share so model weights are downloaded once. If loading from the share is slow, unset it in the job to use the container's local disk.
