"""Local fork/yield/control middleware."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentMiddleware, ContextT, ResponseT
from langchain.tools import ToolRuntime  # noqa: TC002  # needed by ToolNode get_type_hints
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig  # noqa: TC002  # needed by ToolNode get_type_hints
from langchain_core.tools import StructuredTool
from langgraph.types import Command

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from langchain_core.language_models import BaseChatModel
    from langchain_core.runnables import Runnable
    from langchain_core.tools import BaseTool


_FORK_MODE_CONFIG_KEY = "deepagents_fork_mode"
_FORK_TOKEN_CONFIG_KEY = "deepagents_fork_token"  # noqa: S105  # config key, not a secret
_FORK_CHILD_ID_CONFIG_KEY = "deepagents_fork_child_id"
_FORK_CHILD_MODE = "child"
_WAIT_TIMEOUT_SECONDS = 5.0
_DEFAULT_FORK_TASK = "Continue from the inherited conversation context."


def _is_fork_child_config(config: RunnableConfig) -> bool:
    """Return whether a runnable config is executing inside a local fork child."""
    configurable = config.get("configurable", {})
    return configurable.get(_FORK_MODE_CONFIG_KEY) == _FORK_CHILD_MODE


@dataclass
class _YieldedValue:
    child_id: str
    value: str


@dataclass
class _ForkRun:
    token: str
    child_ids: list[str]
    yielded: list[_YieldedValue] = field(default_factory=list)
    consumed: int = 0
    statuses: dict[str, str] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    child_messages: dict[str, list[Any]] = field(default_factory=dict)
    child_configs: dict[str, RunnableConfig] = field(default_factory=dict)
    pending_inputs: dict[str, list[str]] = field(default_factory=dict)
    condition: threading.Condition = field(default_factory=threading.Condition)


def _messages_before_current_tool_call(messages: Sequence[Any], tool_call_id: str | None) -> list[Any]:
    copied = list(messages)
    if not tool_call_id:
        return copied

    for idx in range(len(copied) - 1, -1, -1):
        msg = copied[idx]
        if isinstance(msg, AIMessage):
            tool_calls = getattr(msg, "tool_calls", None) or []
            if any(call.get("id") == tool_call_id for call in tool_calls):
                return copied[:idx]

    for idx in range(len(copied) - 1, -1, -1):
        msg = copied[idx]
        if isinstance(msg, ToolMessage) and msg.tool_call_id == tool_call_id:
            return copied[:idx]
    return copied


def _format_clone_ctl_result(run: _ForkRun, values: list[_YieldedValue]) -> str:
    running = sum(1 for status in run.statuses.values() if status == "running")
    done = sum(1 for status in run.statuses.values() if status == "done")
    failed = sum(1 for status in run.statuses.values() if status == "error")
    killed = sum(1 for status in run.statuses.values() if status == "killed")
    lines = [
        f"fork_id: {run.token}",
        f"status: {done} done, {running} running, {failed} failed, {killed} killed",
    ]
    if values:
        lines.append("yielded values:")
        lines.extend(f"- {value.child_id}: {value.value}" for value in values)
    else:
        lines.append("yielded values: none")
    if run.errors:
        lines.append("errors:")
        lines.extend(f"- {child_id}: {error}" for child_id, error in run.errors.items())
    return "\n".join(lines)


class ForkMiddleware(AgentMiddleware[Any, ContextT, ResponseT]):
    """Expose `fork`, `clone_ctl`, and `yield_value` with stable tool schemas."""

    def __init__(
        self,
        *,
        model: BaseChatModel,
        system_prompt: str | SystemMessage,
        tools: Sequence[BaseTool | Callable | dict[str, Any]] | None,
    ) -> None:
        """Initialize local fork tools with the parent's model and tools."""
        super().__init__()
        self._model = model
        self._system_prompt = system_prompt
        self._tools = list(tools or [])
        self._middleware: list[AgentMiddleware[Any, Any, Any]] | None = None
        self._child_agent: Runnable | None = None
        self._runs: dict[str, _ForkRun] = {}
        self._runs_lock = threading.Lock()

        def fork(
            runtime: ToolRuntime,
        ) -> Command:
            """Fork yourself into one clone that inherits the current context."""
            return self._fork(runtime)

        async def afork(
            runtime: ToolRuntime,
        ) -> Command:
            """Fork yourself into one clone that inherits the current context."""
            return await self._afork(runtime)

        def clone_ctl(
            op: Literal["wait", "kill", "input"],
            token: str,
            runtime: ToolRuntime,
            message: str | None = None,
        ) -> Command:
            """Control forked children for a token."""
            return self._clone_ctl(op, token, runtime, message)

        async def aclone_ctl(
            op: Literal["wait", "kill", "input"],
            token: str,
            runtime: ToolRuntime,
            message: str | None = None,
        ) -> Command:
            """Control forked children for a token."""
            return await self._aclone_ctl(op, token, runtime, message)

        def yield_value(
            value: str,
            runtime: ToolRuntime,
        ) -> Command:
            """Yield a value from a forked child agent back to its parent."""
            return self._yield_value(value, runtime)

        async def ayield_value(
            value: str,
            runtime: ToolRuntime,
        ) -> Command:
            """Yield a value from a forked child agent back to its parent."""
            return await self._ayield_value(value, runtime)

        self.tools = [
            StructuredTool.from_function(
                name="fork",
                func=fork,
                coroutine=afork,
                description=(
                    "Fork a clone of yourself to complete work in parallel. "
                    "Returns a fork token that can be passed to `clone_ctl`. "
                    "Call `fork` multiple times to create multiple clones."
                ),
                infer_schema=True,
            ),
            StructuredTool.from_function(
                name="clone_ctl",
                func=clone_ctl,
                coroutine=aclone_ctl,
                description=(
                    "Control a clone created by `fork`. Use op='wait' to wait up to 5 seconds for yielded values, "
                    "op='input' with message='...' to send a new message to that clone, "
                    "or op='kill' to stop that clone and return any yielded values."
                ),
                infer_schema=True,
            ),
            StructuredTool.from_function(
                name="yield_value",
                func=yield_value,
                coroutine=ayield_value,
                description="Yield a value to your parent.",
                infer_schema=True,
            ),
        ]

    def configure_child_agent(self, middleware: Sequence[AgentMiddleware[Any, Any, Any]]) -> None:
        """Configure the child agent with the same middleware/tool schema as the parent."""
        self._middleware = list(middleware)

    def _get_child_agent(self) -> Runnable:
        if self._child_agent is None:
            if self._middleware is None:
                msg = "ForkMiddleware child agent was not configured."
                raise ValueError(msg)
            self._child_agent = create_agent(
                self._model,
                system_prompt=self._system_prompt,
                tools=self._tools,
                middleware=self._middleware,
                name="fork-child",
            ).with_config({"recursion_limit": 9_999})
        return self._child_agent

    @staticmethod
    def _is_child(config: RunnableConfig) -> bool:
        return _is_fork_child_config(config)

    @staticmethod
    def _child_config(token: str, child_id: str, config: RunnableConfig) -> RunnableConfig:
        return {
            **config,
            "configurable": {
                **config.get("configurable", {}),
                _FORK_MODE_CONFIG_KEY: _FORK_CHILD_MODE,
                _FORK_TOKEN_CONFIG_KEY: token,
                _FORK_CHILD_ID_CONFIG_KEY: child_id,
                "ls_agent_type": "fork-child",
            },
            "metadata": {
                **config.get("metadata", {}),
                _FORK_MODE_CONFIG_KEY: _FORK_CHILD_MODE,
            },
        }

    def _start_child_thread(self, token: str, child_id: str) -> None:
        thread = threading.Thread(
            target=self._run_child,
            args=(token, child_id),
            daemon=True,
        )
        thread.start()

    def _run_child(self, token: str, child_id: str) -> None:
        run = self._runs[token]
        while True:
            with run.condition:
                if run.statuses.get(child_id) == "killed":
                    run.condition.notify_all()
                    return
                queued = run.pending_inputs.get(child_id, [])
                if queued:
                    next_input = queued.pop(0)
                    run.child_messages[child_id] = [*run.child_messages.get(child_id, []), HumanMessage(content=next_input)]
                child_state = {"messages": list(run.child_messages[child_id])}
                child_config = run.child_configs[child_id]

            try:
                result = self._get_child_agent().invoke(child_state, child_config)
                with run.condition:
                    if run.statuses.get(child_id) == "killed":
                        run.condition.notify_all()
                        return
                    if isinstance(result, dict) and isinstance(result.get("messages"), list):
                        run.child_messages[child_id] = list(result["messages"])
                    if run.pending_inputs.get(child_id):
                        continue
                    run.statuses[child_id] = "done"
                    run.condition.notify_all()
                    return
            except Exception as e:  # noqa: BLE001
                with run.condition:
                    if run.statuses.get(child_id) != "killed":
                        run.statuses[child_id] = "error"
                        run.errors[child_id] = str(e)
                    run.condition.notify_all()
                return

    def _fork(self, runtime: ToolRuntime) -> Command:
        tool_call_id = runtime.tool_call_id
        if not tool_call_id:
            msg = "Tool call ID is required for fork invocation"
            raise ValueError(msg)
        if self._is_child(runtime.config):
            return Command(update={"messages": [ToolMessage("`fork` is not available inside a forked child.", tool_call_id=tool_call_id)]})
        token = uuid.uuid4().hex
        child_ids = ["child-0"]
        run = _ForkRun(token=token, child_ids=child_ids, statuses=dict.fromkeys(child_ids, "running"))
        parent_messages = _messages_before_current_tool_call(runtime.state.get("messages", []), tool_call_id)
        for child_id in child_ids:
            run.child_messages[child_id] = [*parent_messages, HumanMessage(content=_DEFAULT_FORK_TASK)]
            run.child_configs[child_id] = self._child_config(token, child_id, runtime.config)
        with self._runs_lock:
            self._runs[token] = run
        for child_id in child_ids:
            self._start_child_thread(token, child_id)
        msg = f"fork_id: {token}\nstarted: 1"
        return Command(
            update={
                "messages": [ToolMessage(msg, tool_call_id=tool_call_id)],
            }
        )

    async def _afork(self, runtime: ToolRuntime) -> Command:
        return self._fork(runtime)

    @staticmethod
    def _has_running_children(run: _ForkRun) -> bool:
        return any(status == "running" for status in run.statuses.values())

    @staticmethod
    def _kill_children(run: _ForkRun) -> None:
        for child_id, status in run.statuses.items():
            if status == "running":
                run.statuses[child_id] = "killed"

    @staticmethod
    def _queue_child_input(run: _ForkRun, message: str) -> list[str]:
        children_to_start: list[str] = []
        for child_id, status in run.statuses.items():
            if status in ("error", "killed"):
                continue
            run.pending_inputs.setdefault(child_id, []).append(message)
            if status != "running":
                run.statuses[child_id] = "running"
                children_to_start.append(child_id)
        return children_to_start

    @staticmethod
    def _collect_clone_ctl_result(run: _ForkRun) -> str:
        values = run.yielded[run.consumed :]
        run.consumed = len(run.yielded)
        return _format_clone_ctl_result(run, values)

    def _clone_ctl(self, op: Literal["wait", "kill", "input"], token: str, runtime: ToolRuntime, message: str | None = None) -> Command:
        tool_call_id = runtime.tool_call_id
        if not tool_call_id:
            msg = "Tool call ID is required for clone_ctl invocation"
            raise ValueError(msg)
        if self._is_child(runtime.config):
            return Command(update={"messages": [ToolMessage("`clone_ctl` is not available inside a forked child.", tool_call_id=tool_call_id)]})
        if op not in ("wait", "kill", "input"):
            msg = "Invalid clone_ctl op. Expected 'wait', 'input', or 'kill'."
            return Command(update={"messages": [ToolMessage(msg, tool_call_id=tool_call_id)]})
        if op == "input" and message is None:
            msg = "clone_ctl op='input' requires a message."
            return Command(update={"messages": [ToolMessage(msg, tool_call_id=tool_call_id)]})
        with self._runs_lock:
            run = self._runs.get(token)
        if run is None:
            return Command(update={"messages": [ToolMessage(f"Unknown fork token: {token}", tool_call_id=tool_call_id)]})
        children_to_start: list[str] = []
        with run.condition:
            if op == "kill":
                self._kill_children(run)
                run.condition.notify_all()
            elif op == "input":
                children_to_start = self._queue_child_input(run, message or "")
                run.condition.notify_all()
        for child_id in children_to_start:
            self._start_child_thread(token, child_id)
        with run.condition:
            if run.consumed >= len(run.yielded) and self._has_running_children(run):
                run.condition.wait(timeout=_WAIT_TIMEOUT_SECONDS)
            content = self._collect_clone_ctl_result(run)
        return Command(update={"messages": [ToolMessage(content, tool_call_id=tool_call_id)]})

    async def _aclone_ctl(self, op: Literal["wait", "kill", "input"], token: str, runtime: ToolRuntime, message: str | None = None) -> Command:
        return self._clone_ctl(op, token, runtime, message)

    def _yield_value(self, value: str, runtime: ToolRuntime) -> Command:
        tool_call_id = runtime.tool_call_id
        if not tool_call_id:
            msg = "Tool call ID is required for yield_value invocation"
            raise ValueError(msg)
        configurable = runtime.config.get("configurable", {})
        if configurable.get(_FORK_MODE_CONFIG_KEY) != _FORK_CHILD_MODE:
            msg = "`yield_value` is only available inside a forked child."
            return Command(update={"messages": [ToolMessage(msg, tool_call_id=tool_call_id)]})
        token = configurable.get(_FORK_TOKEN_CONFIG_KEY)
        child_id = configurable.get(_FORK_CHILD_ID_CONFIG_KEY)
        if not isinstance(token, str) or not isinstance(child_id, str):
            return Command(update={"messages": [ToolMessage("Missing fork child context.", tool_call_id=tool_call_id)]})
        with self._runs_lock:
            run = self._runs.get(token)
        if run is None:
            return Command(update={"messages": [ToolMessage(f"Unknown fork token: {token}", tool_call_id=tool_call_id)]})
        with run.condition:
            if run.statuses.get(child_id) == "killed":
                return Command(update={"messages": [ToolMessage("fork was killed; value ignored", tool_call_id=tool_call_id)]})
            run.yielded.append(_YieldedValue(child_id=child_id, value=value))
            run.condition.notify_all()
        return Command(update={"messages": [ToolMessage("yielded", tool_call_id=tool_call_id)]})

    async def _ayield_value(self, value: str, runtime: ToolRuntime) -> Command:
        return self._yield_value(value, runtime)
