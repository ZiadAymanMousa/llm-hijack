"""Build agents, guards and pipelines from the config."""

import os

import openai
from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, PipelineConfig, load_system_message
from agentdojo.agent_pipeline.basic_elements import InitQuery, SystemMessage
from agentdojo.agent_pipeline.tool_execution import ToolsExecutionLoop

from hijack.agents import TextToolAgent
from hijack.executor import GuardedToolsExecutor
from hijack.guards import LLMGuard, WolfJevGuard


def make_client(url: str) -> openai.OpenAI:
    return openai.OpenAI(base_url=url, api_key="EMPTY", timeout=600, max_retries=3)


def build_agent(cfg, name: str, recorder=None) -> TextToolAgent:
    spec = cfg.agent(name)
    model = cfg.model(spec["model"])
    return TextToolAgent(make_client(model["url"]), model=spec["model"], style=spec["prompt"],
                         tool_result_role=spec["tool_result_role"],
                         max_tokens=cfg["decoding"]["agent_max_tokens"], recorder=recorder, name=f"local/{name}")


def build_guard(cfg, name: str, recorder=None, threshold: float | None = None):
    spec = cfg.guard(name)
    if spec["type"] == "llm":
        model = cfg.model(spec["model"])
        return LLMGuard(make_client(model["url"]), model=spec["model"],
                        max_tokens=cfg["decoding"]["guard_max_tokens"], name=name, recorder=recorder)
    if spec["type"] == "wolf_jev":
        return WolfJevGuard(spec["hf_id"], spec["revision"], name=name, threshold=threshold)
    raise ValueError(f"unknown guard type: {spec['type']}")


def build_pipeline(cfg, arm_name: str, recorder, suite_name: str) -> AgentPipeline:
    """The agent pipeline for one arm: system prompt, user query, agent and the tool loop."""
    arm = cfg.arm(arm_name)
    agent = build_agent(cfg, arm["agent"], recorder)
    system_message = load_system_message(None)

    if arm.get("defense") == "progent":
        os.environ["SECAGENT_BASE_URL"] = cfg["progent"]["policy_url"]
        pipeline = AgentPipeline.from_config(PipelineConfig(
            llm=agent, model_id=None, defense="progent", system_message=system_message,
            system_message_name=None, progent_policy_model=cfg["progent"]["policy_model"],
            progent_suite=suite_name))
    else:
        guard_name = arm.get("guard")
        guard = build_guard(cfg, guard_name, recorder) if guard_name else None
        executor = GuardedToolsExecutor(
            recorder, guard=guard, view=cfg.guard(guard_name)["view"] if guard_name else "full",
            block_message=cfg["block_message"], min_substring=cfg["masking"]["min_substring"])
        loop = ToolsExecutionLoop([executor, agent], max_iters=cfg["decoding"]["max_tool_iterations"])
        pipeline = AgentPipeline([SystemMessage(system_message), InitQuery(), agent, loop])
    pipeline.name = f"{arm_name}/{arm['agent']}"
    return pipeline
