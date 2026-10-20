"""Episode recording: message serialization, per-episode context and JSONL output.

One JSON line per episode. Each line holds the case, the outcome, the full message
history and every proposed tool call (with the guard's verdict), which is enough to
re-render both guard views offline (protocol §13).
"""

import json
import os
import time
from pathlib import Path

from agentdojo.types import get_text_content_as_str


def serialize_messages(messages) -> list[dict]:
    """AgentDojo chat messages -> plain JSON-friendly dicts."""
    out = []
    for m in messages:
        role = m["role"]
        item = {"role": role, "text": get_text_content_as_str(m["content"]) if m.get("content") else ""}
        if role == "assistant":
            item["tool_calls"] = [{"function": c.function, "args": dict(c.args)} for c in (m.get("tool_calls") or [])]
        elif role == "tool":
            item["function"] = m["tool_call"].function
            item["args"] = dict(m["tool_call"].args)
            item["error"] = m.get("error")
        out.append(item)
    return out


class Recorder:
    """Collects everything that happens during one episode."""

    def __init__(self):
        self.case: dict = {}
        self.messages = []
        self.proposals: list[dict] = []
        self.usage: list[dict] = []

    def start(self, case: dict) -> None:
        self.case, self.messages, self.proposals, self.usage = case, [], [], []
        self.t0 = time.time()

    def observe(self, messages) -> None:
        """Keep the latest message list so a crashed episode still has its history."""
        self.messages = list(messages)

    def add_usage(self, role: str, response, latency: float) -> None:
        u = getattr(response, "usage", None)
        self.usage.append({
            "role": role,
            "latency_s": round(latency, 3),
            "prompt_tokens": getattr(u, "prompt_tokens", None),
            "completion_tokens": getattr(u, "completion_tokens", None),
        })

    def finish(self, **outcome) -> dict:
        return {
            **self.case,
            **outcome,
            "seconds": round(time.time() - self.t0, 2),
            "proposals": self.proposals,
            "usage": self.usage,
            "messages": serialize_messages(self.messages),
        }


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def done_keys(path: Path, key: str = "case_id") -> set:
    return {r[key] for r in read_jsonl(path)}
