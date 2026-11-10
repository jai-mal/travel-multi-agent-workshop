"""Remove tool requests that an interrupted chat run left without a result.

If a run is cut off while a tool is running, the saved history keeps the tool
request with no matching result. The model rejects that history on every later
message, so the chat can never recover. Removing the unfinished request lets the
next turn start from a valid history.
"""

from __future__ import annotations

from typing import Iterable

from langchain_core.messages import AIMessage, BaseMessage, RemoveMessage, ToolMessage


def interrupted_tool_removals(messages: Iterable[BaseMessage]) -> list[RemoveMessage]:
    """Removals for assistant tool requests in ``messages`` that have a call with no result."""
    history = list(messages)
    answered = {m.tool_call_id for m in history if isinstance(m, ToolMessage)}
    removals: list[RemoveMessage] = []
    for message in history:
        if not isinstance(message, AIMessage) or not message.id:
            continue
        calls = getattr(message, "tool_calls", None) or []
        if any(call.get("id") and call["id"] not in answered for call in calls):
            removals.append(RemoveMessage(id=message.id))
    return removals
