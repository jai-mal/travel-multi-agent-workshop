"""Close tool calls that an interrupted chat run left without a result.

If a run is cut off while a tool is running, the saved history keeps the tool
request with no matching result. The model rejects that history on every later
message, so the chat can never recover. A placeholder result lets the next turn start.
"""

from __future__ import annotations

from typing import Iterable

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

INTERRUPTED_RESULT = (
    "This step was interrupted before it returned a result. "
    "Continue without it, and say so if the answer depends on it."
)


def interrupted_tool_results(messages: Iterable[BaseMessage]) -> list[ToolMessage]:
    """Placeholder results for tool calls in ``messages`` that never got a result."""
    history = list(messages)
    answered = {m.tool_call_id for m in history if isinstance(m, ToolMessage)}
    results: list[ToolMessage] = []
    for message in history:
        if not isinstance(message, AIMessage):
            continue
        for call in getattr(message, "tool_calls", None) or []:
            call_id = call.get("id")
            if call_id and call_id not in answered:
                answered.add(call_id)
                results.append(
                    ToolMessage(
                        content=INTERRUPTED_RESULT,
                        tool_call_id=call_id,
                        name=call.get("name"),
                    )
                )
    return results
