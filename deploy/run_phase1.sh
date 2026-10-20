#!/usr/bin/env bash
# Phase 1 end to end: serve models, smoke test, live arms, held-out check, SEP, replay, analysis.
# Every step resumes from what is already in results/, so re-running the job continues a stopped run.
set -euo pipefail
cd /app

CONFIG=configs/phase1.yaml
RUN="python3 -m hijack.run --config $CONFIG"
REPLAY="python3 -m hijack.replay --config $CONFIG"

bash deploy/serve.sh "$CONFIG"

echo "== smoke test (first banking task, arms A and F)"
$RUN --arm A --benchmark agentdojo --suites banking --limit 1
$RUN --arm F --benchmark agentdojo --suites banking --limit 1

echo "== AgentDojo, fixed attack"
for arm in A B C D E F A0 M; do
  $RUN --arm "$arm" --benchmark agentdojo
done
$RUN --arm P --benchmark agentdojo || echo "arm P (Progent) failed; reported separately"

echo "== AgentDyn held-out check"
for arm in A B D F; do
  $RUN --arm "$arm" --benchmark agentdyn
done

echo "== SEP answer manipulation"
for agent in base reta; do
  $RUN --sep --agent "$agent"
done

echo "== guards scored on logged calls"
for benchmark in agentdojo agentdyn; do
  for guard in stepguard_full stepguard_restricted base_full base_restricted wolf_jev; do
    $REPLAY --guard "$guard" --sources A B --benchmark "$benchmark"
  done
done

echo "== analysis"
python3 -m hijack.analyze --config "$CONFIG" --out results/phase1/report
echo "done: results/phase1/report/summary.md"
