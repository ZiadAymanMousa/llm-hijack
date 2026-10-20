# Hardening the Agent, the Guard, or Both?

Code for a study of two defenses against indirect prompt injection in tool-using LLM agents, alone and combined:

- **a hardened agent:** Qwen3-4B-Instruct-RETA, fine-tuned to ignore instructions in tool outputs;
- **a step-level guard:** StepGuard, which checks every tool call before it runs, either with its full input (full view) or with untrusted values masked by source (restricted view).

## Layout

```
configs/phase1.yaml     models (pinned revisions), agents, guards, arms, benchmarks
configs/task_buckets.json  task under-specification labels (from AutoDojo)
prompts/                RETA tool-calling prompt and trust boundary; StepGuard step-level prompt
src/hijack/
  agents.py             agent adapter for the <function=...> tool format (base, RETA, MetaSecAlign)
  guards.py             StepGuard / base-model guard (vLLM) and wolf-defender-jev guard
  views.py              what the guard sees: full and restricted views
  masker.py             provenance masking and mask-leak check for the restricted view
  executor.py           tool executor that logs every proposed call and applies the guard
  labels.py             which proposed calls carry out the attacker's goal
  record.py             episode recording (one JSON line per episode)
  factory.py            builds agents, guards and pipelines from the config
  run.py                live runs:   python -m hijack.run
  replay.py             score a guard on logged calls:   python -m hijack.replay
  analyze.py            metrics, contrasts, interaction model, figures:   python -m hijack.analyze
tools/                  download a finished phase's results from the Azure share
tests/                  unit tests and an end-to-end test on a real AgentDojo task (no GPU needed)
deploy/                 Docker image, model serving, Phase 1 runner, Azure Container Apps job
```

## Setup and tests

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

On Windows, set `PYTHONUTF8=1` before running on the AgentDyn suites (one of their data files is UTF-8).

## Running

The models are served by vLLM (`deploy/serve.sh`); the Phase 1 job runs everything in `deploy/run_phase1.sh`. Single steps:

```bash
python -m hijack.run --config configs/phase1.yaml --arm F --benchmark agentdojo
python -m hijack.run --config configs/phase1.yaml --arm A --benchmark agentdojo --suites banking --limit 1
python -m hijack.run --config configs/phase1.yaml --sep --agent reta
python -m hijack.replay --config configs/phase1.yaml --guard stepguard_restricted --sources A B
python -m hijack.analyze --config configs/phase1.yaml --out results/phase1/report
```

Results go to `results/phase1/<benchmark>/<arm>.jsonl` (one line per episode, with the full message history and every proposed tool call). Runs resume where they stopped. The report folder gets `summary.md`, CSV tables and two figures.

## Results from the cloud run

```bash
bash tools/fetch_results.sh 1 --logs      # download a finished phase from the Azure share into results/
```

Cloud setup on an Azure A100: see [deploy/README.md](deploy/README.md).
