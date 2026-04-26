from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

from deepagents.backends import FilesystemBackend, LocalShellBackend
from deepagents.backends.utils import create_file_data
from deepagents.graph import create_deep_agent
from tests.unit_tests.chat_model import GenericFakeChatModel


def _smoke_model() -> GenericFakeChatModel:
    """Return a fake model with enough canned responses for smoke tests."""
    return GenericFakeChatModel(messages=iter([AIMessage(content="hello!") for _ in range(4)]))


def _system_message_as_text(message: SystemMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "\n".join(str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in content)


def _invoke_for_smoke_check(agent: object, payload: dict[str, Any]) -> None:
    """Invoke the agent and tolerate fake-model exhaustion after the first call."""
    try:
        if not hasattr(agent, "invoke"):
            msg = f"Expected compiled agent with invoke(), got {type(agent)!r}"
            raise TypeError(msg)
        agent.invoke(payload)
    except RuntimeError as exc:
        if "StopIteration" not in str(exc):
            raise


def _tool_names(tools: list[Any]) -> set[str]:
    return {convert_to_openai_tool(tool)["function"]["name"] for tool in tools}


def test_system_prompt_exposes_expected_tools_with_execute() -> None:
    model = _smoke_model()
    backend = LocalShellBackend(root_dir=Path.cwd(), virtual_mode=True)
    agent = create_deep_agent(model=model, backend=backend)

    _invoke_for_smoke_check(agent, {"messages": [HumanMessage(content="hi")]})

    history = model.call_history
    assert len(history) >= 1
    tool_names = _tool_names(history[0]["tools"])
    assert {
        "execute",
        "task",
        "write_todos",
    }.issubset(tool_names)
    assert not {"fork", "clone_ctl", "yield_value"} & tool_names

    messages = history[0]["messages"]
    system_messages = [m for m in messages if isinstance(m, SystemMessage)]
    assert len(system_messages) >= 1
    assert _system_message_as_text(system_messages[0])


def test_system_prompt_exposes_expected_tools_without_execute() -> None:
    model = _smoke_model()
    backend = FilesystemBackend(root_dir=str(Path.cwd()), virtual_mode=True)
    agent = create_deep_agent(model=model, backend=backend)

    _invoke_for_smoke_check(agent, {"messages": [HumanMessage(content="hi")]})

    history = model.call_history
    assert len(history) >= 1
    tool_names = _tool_names(history[0]["tools"])
    assert "execute" not in tool_names
    assert {"task", "read_file"}.issubset(tool_names)
    assert not {"fork", "clone_ctl", "yield_value"} & tool_names

    messages = history[0]["messages"]
    system_messages = [m for m in messages if isinstance(m, SystemMessage)]
    assert len(system_messages) >= 1
    assert _system_message_as_text(system_messages[0])


def test_system_prompt_exposes_fork_tools_when_enabled() -> None:
    model = _smoke_model()
    backend = FilesystemBackend(root_dir=str(Path.cwd()), virtual_mode=True)
    agent = create_deep_agent(model=model, backend=backend, enable_fork_tools=True)

    _invoke_for_smoke_check(agent, {"messages": [HumanMessage(content="hi")]})

    history = model.call_history
    assert len(history) >= 1
    tool_names = _tool_names(history[0]["tools"])
    assert {"fork", "clone_ctl", "yield_value"}.issubset(tool_names)


def test_custom_system_message_is_preserved() -> None:
    model = _smoke_model()
    backend = FilesystemBackend(root_dir=str(Path.cwd()), virtual_mode=True)

    agent = create_deep_agent(
        model=model,
        backend=backend,
        system_prompt="You are Bobby a virtual assistant for company X",
    )

    _invoke_for_smoke_check(agent, {"messages": [HumanMessage(content="hi")]})

    history = model.call_history
    assert len(history) >= 1
    assert "task" in _tool_names(history[0]["tools"])

    messages = history[0]["messages"]
    system_messages = [m for m in messages if isinstance(m, SystemMessage)]
    assert len(system_messages) >= 1
    system_text = _system_message_as_text(system_messages[0])
    assert "You are Bobby a virtual assistant for company X" in system_text


def test_system_prompt_includes_configured_sync_and_async_subagents() -> None:
    model = _smoke_model()
    backend = FilesystemBackend(root_dir=str(Path.cwd()), virtual_mode=True)

    agent = create_deep_agent(
        model=model,
        backend=backend,
        subagents=[
            {
                "name": "code-reviewer",
                "description": "Reviews code for quality and security issues",
                "system_prompt": "You are a code reviewer. Analyze code for bugs, security vulnerabilities, and style issues.",
            },
            {
                "name": "remote-researcher",
                "description": "Researches topics on a remote LangGraph server",
                "graph_id": "research_graph",
                "url": "http://localhost:8123",
            },
            {
                "name": "remote-analyst",
                "description": "Analyzes data on a remote LangGraph server",
                "graph_id": "analysis_graph",
                "url": "http://localhost:8123",
            },
        ],
    )

    _invoke_for_smoke_check(agent, {"messages": [HumanMessage(content="hi")]})

    history = model.call_history
    assert len(history) >= 1
    assert {
        "task",
        "start_async_task",
        "check_async_task",
        "update_async_task",
        "cancel_async_task",
        "list_async_tasks",
    }.issubset(_tool_names(history[0]["tools"]))

    messages = history[0]["messages"]
    system_messages = [m for m in messages if isinstance(m, SystemMessage)]
    assert len(system_messages) >= 1
    system_text = _system_message_as_text(system_messages[0])
    assert "code-reviewer" in system_text
    assert "remote-researcher" in system_text
    assert "remote-analyst" in system_text


def test_system_prompt_with_memory_and_skills() -> None:
    model = _smoke_model()

    agent = create_deep_agent(
        model=model,
        memory=["/memory/AGENTS.md", "/memory/user/AGENTS.md"],
        skills=["/skills/user/", "/skills/project/"],
    )

    user_skill_content = """\
---
name: web-research
description: Structured approach to conducting thorough web research on any topic
---

# Web Research Skill

## When to Use
- User asks you to research a topic
- You need to gather information from the web
"""

    project_skill_content = """\
---
name: code-review
description: Systematic code review process following best practices and style guides
---

# Code Review Skill

## When to Use
- User asks you to review code
- You need to provide feedback on a pull request
"""

    memory_content = """\
# Project Memory

- Always use Python type hints
- Prefer functional programming patterns
"""

    user_memory_content = """\
# User Memory

- Preferred language: Python
- Always add docstrings to public functions
"""

    files = {
        "/skills/user/web-research/SKILL.md": create_file_data(user_skill_content),
        "/skills/project/code-review/SKILL.md": create_file_data(project_skill_content),
        "/memory/AGENTS.md": create_file_data(memory_content),
        "/memory/user/AGENTS.md": create_file_data(user_memory_content),
    }

    _invoke_for_smoke_check(agent, {"messages": [HumanMessage(content="hi")], "files": files})

    history = model.call_history
    assert len(history) >= 1
    assert "task" in _tool_names(history[0]["tools"])

    messages = history[0]["messages"]
    system_messages = [m for m in messages if isinstance(m, SystemMessage)]
    assert len(system_messages) >= 1
    system_text = _system_message_as_text(system_messages[0])
    assert "Project Memory" in system_text
    assert "User Memory" in system_text
    assert "web-research" in system_text
    assert "code-review" in system_text
