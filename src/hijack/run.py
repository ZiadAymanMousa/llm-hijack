"""Run one arm on a benchmark (live), or the SEP answer-manipulation probe.

    python -m hijack.run --config configs/phase1.yaml --arm A --benchmark agentdojo
    python -m hijack.run --config configs/phase1.yaml --arm F --benchmark agentdyn --suites shopping --limit 5
    python -m hijack.run --config configs/phase1.yaml --sep --agent reta

Output: <results_dir>/<benchmark>/<arm>.jsonl, one line per episode. Re-running resumes.
"""

import argparse
import hashlib
import json
import random
import threading

import requests
from agentdojo.agent_pipeline.agent_pipeline import load_system_message
from agentdojo.attacks.attack_registry import load_attack
from agentdojo.functions_runtime import FunctionsRuntime
from agentdojo.task_suite.load_suites import get_suites
from agentdojo.task_suite.task_suite import AbortAgentError, functions_stack_trace_from_messages

from hijack.config import ROOT, load_config
from hijack.factory import build_agent, build_pipeline
from hijack.labels import attacker_markers, is_unsafe
from hijack.masker import normalize
from hijack.parallel import run_parallel
from hijack.record import Recorder, append_jsonl, done_keys, serialize_messages

BUCKETS = json.loads((ROOT / "configs" / "task_buckets.json").read_text(encoding="utf-8"))["task_to_bucket"]


def bucket(suite_name: str, user_task_id: str) -> str:
    """AutoDojo's under-specification label for the task (fully-specified / param-open / action-open)."""
    return BUCKETS.get(suite_name, {}).get(user_task_id.removeprefix("user_task_"), "unlabeled")


def payload_delivered(injections: dict, messages: list[dict]) -> bool:
    """True if any injected text reached a tool output the agent received (format-insensitive)."""
    squash = lambda s: "".join(ch for ch in normalize(s).lower() if ch.isalnum())  # noqa: E731
    seen = squash(" ".join(m["text"] for m in messages if m["role"] == "tool"))
    return any(squash(text) and squash(text) in seen for text in injections.values())


def last_assistant_output(messages) -> list:
    for m in reversed(messages):
        if m["role"] == "assistant":
            return m["content"] or []
    return []


def run_episode(suite, pipeline, recorder, user_task, injection_task, injections) -> dict:
    """Mirror of AgentDojo's TaskSuite.run_task_with_pipeline that also keeps the history."""
    environment = suite.load_and_inject_default_environment(injections)
    task_env = user_task.init_environment(environment)
    pre_env = task_env.model_copy(deep=True)
    runtime = FunctionsRuntime(suite.tools)

    error = None
    try:
        _, _, task_env, messages, _ = pipeline.query(user_task.PROMPT, runtime, task_env)
        recorder.observe(messages)
    except AbortAgentError as e:
        task_env, error = e.task_environment, "aborted"
        recorder.observe(e.messages)
    except Exception as e:  # the episode counts as failed; the cause is kept
        error = f"{type(e).__name__}: {e}"

    messages = recorder.messages
    output = last_assistant_output(messages)
    trace = functions_stack_trace_from_messages(messages) if messages else []
    utility = security = None
    try:
        utility = suite._check_task_result(user_task, output, pre_env, task_env, trace)
        if injection_task is not None:
            security = suite._check_task_result(injection_task, output, pre_env, task_env, trace)
    except Exception as e:
        error = error or f"check failed: {type(e).__name__}: {e}"

    markers = None
    if injection_task is not None:
        markers = attacker_markers(injection_task, user_task, pre_env, {t.name for t in suite.tools})
    for p in recorder.proposals:
        p["unsafe"] = markers is not None and is_unsafe(p["function"], p["args"], p["effectful"], markers)

    return recorder.finish(
        utility=bool(utility), attack_success=bool(security) if injection_task else None,
        payload_delivered=payload_delivered(injections, serialize_messages(messages)) if injection_task else None,
        labelable=markers.labelable if markers is not None else True,
        error=error,
    )


def suite_cases(suite, suite_name, attack, limit, clean_only, done) -> list[tuple]:
    """(case_id, user_task, injection_task, injections) for every episode not run yet."""
    cases = []
    for user_task in list(suite.user_tasks.values())[:limit]:
        pairs = [(None, {})]
        if not clean_only:
            for injection_task in suite.injection_tasks.values():
                try:
                    pairs.append((injection_task, attack.attack(user_task, injection_task)))
                except ValueError as e:  # task reads no attacker-controlled content
                    print(f"skip {suite_name}/{user_task.ID}: {e}")
                    break
        for injection_task, injections in pairs:
            case_id = f"{suite_name}/{user_task.ID}/{injection_task.ID if injection_task else 'clean'}"
            if case_id not in done:
                cases.append((case_id, user_task, injection_task, injections))
    return cases


def run_arm(cfg, arm: str, benchmark: str, suites: list[str] | None, limit: int | None, clean_only: bool,
            workers: int | None = None):
    # Progent keeps per-run state in its pipeline, so that arm always runs one episode at a time.
    workers = 1 if cfg.arm(arm).get("defense") else (workers or cfg.get("parallel", {}).get("live", 1))
    out_path = cfg.results_dir / benchmark / f"{arm}.jsonl"
    done = done_keys(out_path)
    all_suites = get_suites(cfg["benchmark_version"])
    for suite_name in suites or cfg["benchmarks"][benchmark]:
        suite = all_suites[suite_name]
        attack = load_attack(cfg["attack"], suite, build_pipeline(cfg, arm, Recorder(), suite_name))
        cases = suite_cases(suite, suite_name, attack, limit, clean_only, done)
        local = threading.local()  # one pipeline and recorder per worker thread

        def episode(case):
            if not hasattr(local, "pipeline"):
                local.recorder = Recorder()
                local.pipeline = build_pipeline(cfg, arm, local.recorder, suite_name)
            case_id, user_task, injection_task, injections = case
            local.recorder.start({
                "case_id": case_id, "arm": arm, "benchmark": benchmark, "suite": suite_name,
                "user_task": user_task.ID, "injection_task": injection_task.ID if injection_task else None,
                "attack": cfg["attack"] if injection_task else None, "bucket": bucket(suite_name, user_task.ID),
                "prompt": user_task.PROMPT, "injections": injections,
                "trusted": cfg.trusted_constants(suite_name), "workers": workers,
            })
            return run_episode(suite, local.pipeline, local.recorder, user_task, injection_task, injections)

        for record in run_parallel(episode, cases, workers):
            append_jsonl(out_path, record)
            print(f"{arm} {record['case_id']} utility={record['utility']} attack={record['attack_success']}"
                  f"{' error=' + record['error'] if record['error'] else ''}", flush=True)


def load_sep(cfg) -> list[dict]:
    spec = cfg["sep"]
    path = ROOT / spec["path"]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        data = requests.get(spec["url"], timeout=120).content
        if hashlib.sha256(data).hexdigest() != spec["sha256"]:
            raise RuntimeError("SEP dataset checksum mismatch")
        path.write_bytes(data)
    items = json.loads(path.read_text(encoding="utf-8"))
    random.Random(spec["seed"]).shuffle(items)
    return items[: spec["n_items"]]


def run_sep(cfg, agent_name: str, workers: int | None = None):
    """Answer manipulation: does the probe hidden in a document change the agent's answer?"""
    workers = workers or cfg.get("parallel", {}).get("sep", 1)
    out_path = cfg.results_dir / "sep" / f"{agent_name}.jsonl"
    done = done_keys(out_path)
    agent = build_agent(cfg, agent_name)
    system_message = load_system_message(None)
    items = [(f"sep/{i}", item) for i, item in enumerate(load_sep(cfg)) if f"sep/{i}" not in done]

    def probe(case):
        case_id, item = case
        answers = {variant: agent.answer_with_document(system_message, item["system_prompt_clean"],
                                                       item[f"prompt_{variant}"])
                   for variant in ("instructed", "clean")}
        witness = item["witness"].lower()
        return {"case_id": case_id, "agent": agent_name, "witness": item["witness"],
                "hit_instructed": witness in answers["instructed"].lower(),
                "hit_clean": witness in answers["clean"].lower(), "answers": answers}

    for record in run_parallel(probe, items, workers):
        append_jsonl(out_path, record)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--arm")
    parser.add_argument("--benchmark", default="agentdojo")
    parser.add_argument("--suites", nargs="*")
    parser.add_argument("--limit", type=int, help="first N user tasks per suite (smoke tests)")
    parser.add_argument("--clean-only", action="store_true", help="skip attacked episodes")
    parser.add_argument("--sep", action="store_true", help="run the SEP answer-manipulation probe")
    parser.add_argument("--agent", help="agent for --sep")
    parser.add_argument("--workers", type=int, help="episodes run at once (default: config parallel)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.sep:
        run_sep(cfg, args.agent, args.workers)
    else:
        run_arm(cfg, args.arm, args.benchmark, args.suites, args.limit, args.clean_only, args.workers)


if __name__ == "__main__":
    main()
