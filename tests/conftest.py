"""Shared test helpers: a fake OpenAI-compatible client and the config."""

from types import SimpleNamespace

import pytest

from hijack.config import ROOT, load_config


class FakeClient:
    """Mimics openai.OpenAI().chat.completions.create with a scripted reply function."""

    def __init__(self, reply):
        self.reply = reply
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, model, messages, temperature, max_tokens):
        self.calls.append({"model": model, "messages": messages})
        content = self.reply(model, messages)
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))], usage=usage)


@pytest.fixture
def cfg():
    return load_config(ROOT / "configs" / "phase1.yaml")
