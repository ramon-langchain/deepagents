"""Tests for the local fork/yield/control tools."""

from __future__ import annotations

import re
import threading
import time
from typing import TYPE_CHECKING, Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

from deepagents.graph import create_deep_agent

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from langchain_core.callbacks import CallbackManagerForLLMRun
    from langchain_core.language_models import LanguageModelInput
    from langchain_core.runnables import Runnable
    from langchain_core.tools import BaseTool


class _ForkYieldCloneCtlModel(BaseChatModel):
    """Scripted model that exercises parent fork/clone_ctl and child yield_value."""

    scenario: str = "single"
    call_history: list[dict[str, Any]] = []  # noqa: RUF012  # pydantic field, per-instance
    tools: list[Any] = []  # noqa: RUF012  # pydantic field, per-instance
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    @property
    def _llm_type(self) -> str:
        return "fork-yield-control"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        self.tools = list(tools)
        return self

    def _tool_names(self) -> set[str]:
        names: set[str] = set()
        for tool in self.tools:
            name = tool.get("name") if isinstance(tool, dict) else getattr(tool, "name", None)
            if isinstance(name, str):
                names.add(name)
        return names

    def _generate(  # noqa: PLR0912  # scripted model branches mirror tool-call scenarios
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        with self._lock:
            tool_names = self._tool_names()
            self.call_history.append({"messages": list(messages), "tool_names": tool_names})

        human_texts = [str(m.content) for m in messages if isinstance(m, HumanMessage)]
        tool_texts = [str(m.content) for m in messages if isinstance(m, ToolMessage)]

        clone_prompts = [text for text in human_texts if text.startswith(("followup input", "Continue from the inherited conversation context."))]
        if clone_prompts:
            yield_count = sum(1 for text in tool_texts if text == "yielded")
            if yield_count < len(clone_prompts):
                if self.scenario == "child_blocked_tools" and not any("not available inside a forked child" in text for text in tool_texts):
                    message = AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "fork",
                                "args": {},
                                "id": "nested-fork-1",
                                "type": "tool_call",
                            },
                            {
                                "name": "clone_ctl",
                                "args": {"op": "wait", "token": "not-a-real-token"},
                                "id": "clone-ctl-1",
                                "type": "tool_call",
                            },
                            {
                                "name": "task",
                                "args": {"description": "nested task", "subagent_type": "general-purpose"},
                                "id": "child-task-1",
                                "type": "tool_call",
                            },
                        ],
                    )
                else:
                    value = clone_prompts[-1]
                    value = value.replace("followup input", "followup value")
                    value = value.replace("Continue from the inherited conversation context.", "default clone value")
                    message = AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "yield_value",
                                "args": {"value": value},
                                "id": f"yield-{value[-1] if value[-1].isdigit() else '1'}",
                                "type": "tool_call",
                            }
                        ],
                    )
            else:
                if self.scenario == "kill":
                    time.sleep(2)
                message = AIMessage(content="hidden child final output")
        elif fork_result := next((text for text in tool_texts if text.startswith("fork_id: ")), None):
            token = re.search(r"fork_id: ([a-f0-9]+)", fork_result)
            assert token is not None
            clone_ctl_count = sum(1 for text in tool_texts if text.startswith("fork_id: ") and "yielded values:" in text)
            if self.scenario == "input" and "default clone value" in "\n".join(tool_texts) and "followup value" not in "\n".join(tool_texts):
                message = AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "clone_ctl",
                            "args": {"op": "input", "token": token.group(1), "message": "followup input"},
                            "id": "clone-ctl-input",
                            "type": "tool_call",
                        }
                    ],
                )
            elif self.scenario == "double_wait" and clone_ctl_count == 1:
                message = AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "clone_ctl",
                            "args": {"op": "wait", "token": token.group(1)},
                            "id": "clone-ctl-2",
                            "type": "tool_call",
                        }
                    ],
                )
            elif self.scenario == "kill" and clone_ctl_count == 1:
                message = AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "clone_ctl",
                            "args": {"op": "kill", "token": token.group(1)},
                            "id": "clone-ctl-kill",
                            "type": "tool_call",
                        }
                    ],
                )
            elif not any("yielded values:" in text for text in tool_texts):
                message = AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "clone_ctl",
                            "args": {"op": "wait", "token": token.group(1)},
                            "id": "clone-ctl-1",
                            "type": "tool_call",
                        }
                    ],
                )
            else:
                message = AIMessage(content="parent done")
        else:
            message = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "fork",
                        "args": {},
                        "id": "fork-1",
                        "type": "tool_call",
                    },
                    *(
                        [
                            {
                                "name": "fork",
                                "args": {},
                                "id": "fork-2",
                                "type": "tool_call",
                            }
                        ]
                        if self.scenario == "two_forks"
                        else []
                    ),
                ],
            )
        return ChatResult(generations=[ChatGeneration(message=message)])


def test_fork_yield_clone_ctl_returns_yielded_values_without_child_final_output() -> None:
    model = _ForkYieldCloneCtlModel()
    agent = create_deep_agent(model=model, tools=[], enable_fork_tools=True)

    result = agent.invoke({"messages": [HumanMessage(content="start fork")]})

    tool_outputs = [m.content for m in result["messages"] if isinstance(m, ToolMessage)]
    assert any("default clone value" in output for output in tool_outputs)
    assert all("hidden child final output" not in output for output in tool_outputs)


def test_fork_child_state_excludes_current_parent_tool_call_turn() -> None:
    model = _ForkYieldCloneCtlModel()
    agent = create_deep_agent(model=model, tools=[], enable_fork_tools=True)

    agent.invoke({"messages": [HumanMessage(content="start fork")]})

    child_messages = next(
        call["messages"]
        for call in model.call_history
        if any(isinstance(m, HumanMessage) and m.content == "Continue from the inherited conversation context." for m in call["messages"])
    )
    assert not any(isinstance(m, AIMessage) and any(call["name"] == "fork" for call in (m.tool_calls or [])) for m in child_messages)
    assert not any(isinstance(m, ToolMessage) and str(m.content).startswith("fork_id: ") for m in child_messages)


def test_fork_child_tool_schema_matches_parent_tool_schema() -> None:
    model = _ForkYieldCloneCtlModel()
    agent = create_deep_agent(model=model, tools=[], enable_fork_tools=True)

    agent.invoke({"messages": [HumanMessage(content="start fork")]})

    parent_tools = model.call_history[0]["tool_names"]
    child_tools = next(
        call["tool_names"]
        for call in model.call_history
        if any(isinstance(m, HumanMessage) and m.content == "Continue from the inherited conversation context." for m in call["messages"])
    )
    assert parent_tools == child_tools
    assert {"fork", "clone_ctl", "yield_value", "task"}.issubset(parent_tools)


def test_fork_tool_schema_has_no_task_arguments() -> None:
    model = _ForkYieldCloneCtlModel()
    agent = create_deep_agent(model=model, tools=[], enable_fork_tools=True)

    agent.invoke({"messages": [HumanMessage(content="start fork")]})

    fork_tool = next(tool for tool in model.tools if (tool.get("name") if isinstance(tool, dict) else getattr(tool, "name", None)) == "fork")
    params = fork_tool.get("parameters", {}) if isinstance(fork_tool, dict) else {}
    properties = params.get("properties", {}) if isinstance(params, dict) else {}
    assert properties == {}


def test_fork_child_runtime_blocks_recursive_fanout_tools() -> None:
    model = _ForkYieldCloneCtlModel(scenario="child_blocked_tools")
    agent = create_deep_agent(model=model, tools=[], enable_fork_tools=True)

    result = agent.invoke({"messages": [HumanMessage(content="start fork")]})

    parent_tool_outputs = [str(m.content) for m in result["messages"] if isinstance(m, ToolMessage)]
    parent_joined = "\n".join(parent_tool_outputs)
    child_tool_outputs = [str(m.content) for call in model.call_history for m in call["messages"] if isinstance(m, ToolMessage)]
    child_joined = "\n".join(child_tool_outputs)
    assert "`fork` is not available inside a forked child." in child_joined
    assert "`clone_ctl` is not available inside a forked child." in child_joined
    assert "`task` is not available inside a forked child." in child_joined
    assert "default clone value" in parent_joined
    assert "`fork` is not available inside a forked child." not in parent_joined


def test_clone_ctl_wait_returns_yielded_values_once() -> None:
    model = _ForkYieldCloneCtlModel(scenario="double_wait")
    agent = create_deep_agent(model=model, tools=[], enable_fork_tools=True)

    result = agent.invoke({"messages": [HumanMessage(content="start fork")]})

    ctl_outputs = [
        str(m.content)
        for m in result["messages"]
        if isinstance(m, ToolMessage) and str(m.content).startswith("fork_id: ") and "yielded values:" in str(m.content)
    ]
    assert len(ctl_outputs) == 2
    assert "default clone value" in ctl_outputs[0]
    assert "yielded values: none" in ctl_outputs[1]


def test_clone_ctl_kill_marks_running_children_and_returns_yielded_values() -> None:
    model = _ForkYieldCloneCtlModel(scenario="kill")
    agent = create_deep_agent(model=model, tools=[], enable_fork_tools=True)

    result = agent.invoke({"messages": [HumanMessage(content="start fork")]})

    ctl_outputs = [
        str(m.content)
        for m in result["messages"]
        if isinstance(m, ToolMessage) and str(m.content).startswith("fork_id: ") and "yielded values:" in str(m.content)
    ]
    assert len(ctl_outputs) == 2
    assert "default clone value" in ctl_outputs[0]
    assert "yielded values: none" in ctl_outputs[1]
    assert "running" in ctl_outputs[1]
    assert "killed" in ctl_outputs[1]


def test_clone_ctl_input_sends_new_human_message_to_each_child() -> None:
    model = _ForkYieldCloneCtlModel(scenario="input")
    agent = create_deep_agent(model=model, tools=[], enable_fork_tools=True)

    result = agent.invoke({"messages": [HumanMessage(content="start fork")]})

    child_human_messages = [str(m.content) for call in model.call_history for m in call["messages"] if isinstance(m, HumanMessage)]
    tool_outputs = [str(m.content) for m in result["messages"] if isinstance(m, ToolMessage)]
    assert "followup input" in child_human_messages
    assert any("followup value" in output for output in tool_outputs)


def test_repeated_fork_calls_start_one_child_each() -> None:
    model = _ForkYieldCloneCtlModel(scenario="two_forks")
    agent = create_deep_agent(model=model, tools=[], enable_fork_tools=True)

    result = agent.invoke({"messages": [HumanMessage(content="start fork")]})

    child_starts = [
        str(m.content)
        for call in model.call_history
        for m in call["messages"]
        if isinstance(m, HumanMessage) and str(m.content).startswith("Continue from the inherited conversation context.")
    ]
    tool_outputs = [str(m.content) for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(child_starts) >= 2
    assert sum(1 for output in tool_outputs if "started: 1" in output) == 2


def test_fork_starts_one_child() -> None:
    model = _ForkYieldCloneCtlModel()
    agent = create_deep_agent(model=model, tools=[], enable_fork_tools=True)

    result = agent.invoke({"messages": [HumanMessage(content="start fork")]})

    child_starts = [
        str(m.content)
        for call in model.call_history
        for m in call["messages"]
        if isinstance(m, HumanMessage) and str(m.content).startswith("Continue from the inherited conversation context.")
    ]
    tool_outputs = [str(m.content) for m in result["messages"] if isinstance(m, ToolMessage)]
    assert child_starts
    assert any("started: 1" in output for output in tool_outputs)
