"""Agents that call tools with the `<function=name>{json}</function>` text format.

This is AgentDojo's local-model format, which RETA was trained with. One class covers
every agent in the study; they differ only in model, system prompt and the chat role
that carries tool outputs back (user for Qwen/RETA, input for MetaSecAlign).
"""

import json
import re
import time

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.llms.local_llm import _make_system_prompt as agentdojo_system_prompt
from agentdojo.functions_runtime import EmptyEnv, Function, FunctionCall
from agentdojo.types import ChatAssistantMessage, get_text_content_as_str, text_content_block_from_string
from pydantic import BaseModel

from hijack.config import PROMPTS

FUNCTION_TAG = re.compile(r"<function\s*=\s*([^>]+)>(.*?)(?:</function>|$)", re.S)


class _NoArgs(BaseModel):
    """Parameter schema for a tool without arguments."""


def functions_json(tools) -> str:
    return "\n\n".join(
        json.dumps({"name": t.name, "description": t.description, "parameters": t.parameters.model_json_schema()},
                   indent=2)
        for t in tools
    )


def build_system_prompt(style: str, system_message: str, tools: list[Function]) -> str:
    if style == "agentdojo":
        return agentdojo_system_prompt(system_message, tools)
    if style == "reta":
        tool_prompt = (PROMPTS / "reta_tool_calling.txt").read_text(encoding="utf-8")
        trust_boundary = (PROMPTS / "reta_trust_boundary.txt").read_text(encoding="utf-8")
        return (tool_prompt.replace("{functions}", functions_json(tools))
                + "\n\n## Additional Instructions\n\n" + system_message + trust_boundary)
    raise ValueError(f"unknown prompt style: {style}")


def parse_completion(text: str) -> ChatAssistantMessage:
    """First `<function=...>` call in the text becomes the tool call; anything else is a final answer."""
    tool_calls = []
    match = FUNCTION_TAG.search(text)
    if match:
        try:
            args = json.loads(match.group(2).strip() or "{}")
            if isinstance(args, dict):
                tool_calls = [FunctionCall(function=match.group(1).strip(), args=args)]
        except json.JSONDecodeError:
            pass  # unparseable call: counted as a parse failure in the analysis
    return ChatAssistantMessage(role="assistant", content=[text_content_block_from_string(text.strip())],
                                tool_calls=tool_calls)


def tool_result_text(message) -> str:
    if message.get("error"):
        return json.dumps({"error": message["error"]}, ensure_ascii=False)
    result = get_text_content_as_str(message["content"]) or "Success"
    return json.dumps({"result": result}, ensure_ascii=False)


class TextToolAgent(BasePipelineElement):
    def __init__(self, client, model: str, style: str, tool_result_role: str, max_tokens: int,
                 recorder=None, name: str | None = None):
        self.client, self.model, self.style = client, model, style
        self.tool_result_role, self.max_tokens = tool_result_role, max_tokens
        self.recorder = recorder
        self.name = name or f"local/{model}"

    def to_chat(self, messages, tools) -> list[dict]:
        chat = []
        for m in messages:
            if m["role"] == "system":
                chat.append({"role": "system",
                             "content": build_system_prompt(self.style, get_text_content_as_str(m["content"]), tools)})
            elif m["role"] == "tool":
                chat.append({"role": self.tool_result_role, "content": tool_result_text(m)})
            else:
                chat.append({"role": m["role"], "content": get_text_content_as_str(m["content"] or [])})
        return chat

    def complete(self, chat: list[dict]) -> str:
        start = time.time()
        response = self.client.chat.completions.create(
            model=self.model, messages=chat, temperature=0.0, max_tokens=self.max_tokens)
        if self.recorder is not None:
            self.recorder.add_usage("agent", response, time.time() - start)
        return response.choices[0].message.content or ""

    def query(self, query, runtime, env=EmptyEnv(), messages=[], extra_args={}):
        chat = self.to_chat(messages, list(runtime.functions.values()))
        output = parse_completion(self.complete(chat))
        messages = [*messages, output]
        if self.recorder is not None:
            self.recorder.observe(messages)
        return query, runtime, env, messages, extra_args

    def answer_with_document(self, system_message: str, instruction: str, document: str) -> str:
        """SEP probe: the agent reads one document through a tool and answers the instruction."""
        read_doc = Function(name="read_document", description="Read the document the user refers to.",
                            parameters=_NoArgs, dependencies={}, run=lambda: document,
                            full_docstring="", return_type=str)
        chat = [
            {"role": "system", "content": build_system_prompt(self.style, system_message, [read_doc])},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": "<function=read_document>{}</function>"},
            {"role": self.tool_result_role, "content": json.dumps({"result": document}, ensure_ascii=False)},
        ]
        return self.complete(chat)