"""Tool executor that records every proposed call and, if a guard is set, asks it first.

Every proposal is logged in all arms (guarded or not), including whether it would change
the environment, the mask map and the mask-leak flag of the restricted view, so
unguarded runs can be replayed against any guard later. A blocked call is not executed;
the agent receives a fixed error message instead (the guard's rationale is never shown).
"""

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.tool_execution import ToolsExecutor
from agentdojo.functions_runtime import EmptyEnv
from agentdojo.types import ChatToolResultMessage, text_content_block_from_string

from hijack import views
from hijack.labels import changes_environment
from hijack.record import serialize_messages


def tool_list(runtime) -> list[dict]:
    return [{"name": f.name, "description": f.description} for f in runtime.functions.values()]


class GuardedToolsExecutor(BasePipelineElement):
    def __init__(self, recorder, guard=None, view: str = "full", block_message: str = "", min_substring: int = 12):
        self.recorder, self.guard, self.view = recorder, guard, view
        self.block_message, self.min_substring = block_message, min_substring
        self.executor = ToolsExecutor()

    def query(self, query, runtime, env=EmptyEnv(), messages=[], extra_args={}):
        if not messages or messages[-1]["role"] != "assistant" or not messages[-1].get("tool_calls"):
            return query, runtime, env, messages, extra_args

        case = self.recorder.case
        history = serialize_messages(messages[:-1])
        thought = views.split_thought(serialize_messages(messages[-1:])[0]["text"])
        tools, trusted = tool_list(runtime), case.get("trusted", [])

        results = []
        for index, call in enumerate(messages[-1]["tool_calls"]):
            args = dict(call.args)
            render = dict(history=history, function=call.function, args=args, thought=thought,
                          user_request=query, tools=tools, trusted=trusted, min_len=self.min_substring)
            restricted = views.render(**render, restricted=True)
            proposal = {
                "msg_index": len(messages) - 1,
                "call_index": index,
                "function": call.function,
                "args": args,
                "effectful": changes_environment(runtime, env, call.function, args),
                "mask_map": restricted.mask_map,
                "mask_leak": views.mask_leaked(restricted, list(case.get("injections", {}).values()),
                                               query, trusted, tools, self.min_substring),
                "guard": None,
                "blocked": False,
            }
            if self.guard is not None:
                guard_input = restricted if self.view == "restricted" else views.render(**render, restricted=False)
                verdict = self.guard.check(guard_input.prompt)
                proposal["guard"] = {"name": self.guard.name, "view": self.view, "label": verdict.label,
                                     "raw": verdict.raw, "latency_s": verdict.latency_s}
                proposal["blocked"] = verdict.blocked
            self.recorder.proposals.append(proposal)

            if proposal["blocked"]:
                results.append(ChatToolResultMessage(role="tool", content=[text_content_block_from_string("")],
                                                     tool_call_id=call.id, tool_call=call, error=self.block_message))
            else:
                single = {"role": "assistant", "content": None, "tool_calls": [call]}
                _, _, env, executed, _ = self.executor.query(query, runtime, env, [*messages, single], extra_args)
                results.append(executed[-1])

        messages = [*messages, *results]
        self.recorder.observe(messages)
        return query, runtime, env, messages, extra_args
