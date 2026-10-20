"""Score a guard offline on the tool calls logged in unguarded runs (protocol §6.3).

    python -m hijack.replay --config configs/phase1.yaml --guard stepguard_restricted --sources A B
    python -m hijack.replay --config configs/phase1.yaml --guard wolf_jev --sources A B --benchmark agentdojo

The guard sees exactly the input it would have seen live (views are rebuilt from the
logged messages). Output: <results_dir>/<benchmark>/replay/<guard>.jsonl, one line per
proposal. Score-based guards (wolf_jev) store a score; thresholds are set in analyze.
"""

import argparse

from agentdojo.task_suite.load_suites import get_suites

from hijack import views
from hijack.config import load_config
from hijack.factory import build_guard
from hijack.parallel import run_parallel
from hijack.record import append_jsonl, done_keys, read_jsonl


def suite_tools(cfg, suite_name: str) -> list[dict]:
    suite = get_suites(cfg["benchmark_version"])[suite_name]
    return [{"name": f.name, "description": f.description} for f in suite.tools]


def guard_input(cfg, episode: dict, proposal: dict, tools: list[dict], restricted: bool):
    messages = episode["messages"]
    call_msg = messages[proposal["msg_index"]]
    return views.render(
        history=messages[: proposal["msg_index"]], function=proposal["function"], args=proposal["args"],
        thought=views.split_thought(call_msg["text"]), user_request=episode["prompt"], tools=tools,
        restricted=restricted, trusted=episode.get("trusted", []), min_len=cfg["masking"]["min_substring"])


def replay(cfg, guard_name: str, sources: list[str], benchmark: str, limit: int | None,
           workers: int | None = None):
    spec = cfg.guard(guard_name)
    guard = build_guard(cfg, guard_name)
    # The local wolf-defender-jev model is not shared across threads; LLM guards are served by vLLM.
    workers = 1 if spec["type"] == "wolf_jev" else (workers or cfg.get("parallel", {}).get("replay", 1))
    out_path = cfg.results_dir / benchmark / "replay" / f"{guard_name}.jsonl"
    done = done_keys(out_path, key="key")
    tools = {}
    jobs = []
    for source in sources:
        for ep in read_jsonl(cfg.results_dir / benchmark / f"{source}.jsonl")[:limit]:
            tools.setdefault(ep["suite"], suite_tools(cfg, ep["suite"]))
            jobs += [(source, ep, i, p) for i, p in enumerate(ep["proposals"])
                     if f"{source}|{ep['case_id']}|{i}" not in done]

    def score(job):
        source, ep, i, p = job
        suite_tool_list = tools[ep["suite"]]
        if spec["type"] == "wolf_jev":
            restricted = guard_input(cfg, ep, p, suite_tool_list, restricted=True)
            verdict = guard.check(views.jev_scene(ep["prompt"], suite_tool_list, restricted.action))
        else:
            inp = guard_input(cfg, ep, p, suite_tool_list, restricted=spec["view"] == "restricted")
            verdict = guard.check(inp.prompt)
        return {
            "key": f"{source}|{ep['case_id']}|{i}", "guard": guard_name, "source_arm": source,
            "case_id": ep["case_id"], "suite": ep["suite"], "user_task": ep["user_task"],
            "injection_task": ep["injection_task"], "bucket": ep["bucket"], "proposal": i,
            "function": p["function"], "unsafe": p["unsafe"], "labelable": ep.get("labelable", True),
            "episode_attack_success": ep["attack_success"], "blocked": verdict.blocked,
            "label": verdict.label, "score": verdict.score, "latency_s": verdict.latency_s,
        }

    for row in run_parallel(score, jobs, workers):
        append_jsonl(out_path, row)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--guard", required=True)
    parser.add_argument("--sources", nargs="+", default=["A", "B"], help="unguarded arms to replay")
    parser.add_argument("--benchmark", default="agentdojo")
    parser.add_argument("--limit", type=int, help="first N episodes per source (smoke tests)")
    parser.add_argument("--workers", type=int, help="guard calls at once (default: config parallel)")
    args = parser.parse_args()
    replay(load_config(args.config), args.guard, args.sources, args.benchmark, args.limit, args.workers)


if __name__ == "__main__":
    main()
