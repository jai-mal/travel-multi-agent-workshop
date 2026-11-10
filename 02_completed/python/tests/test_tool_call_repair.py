from __future__ import annotations

import sys
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


PYTHON_ROOT = Path(__file__).resolve().parents[1]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from src.app.services.tool_call_repair import interrupted_tool_removals  # noqa: E402


class InterruptedToolRequestTests(unittest.TestCase):
    def test_completed_requests_are_kept(self):
        history = [
            HumanMessage(content="plan a trip", id="h1"),
            AIMessage(content="", id="ai1", tool_calls=[{"name": "find_places", "args": {}, "id": "call_1"}]),
            ToolMessage(content="ok", tool_call_id="call_1", id="t1"),
            AIMessage(content="Here is a plan.", id="ai2"),
        ]
        self.assertEqual(interrupted_tool_removals(history), [])

    def test_unanswered_request_at_end_is_removed(self):
        history = [
            HumanMessage(content="2", id="h1"),
            AIMessage(
                content="",
                id="ai1",
                tool_calls=[{"name": "create_or_update_itinerary", "args": {}, "id": "call_2"}],
            ),
        ]
        self.assertEqual([r.id for r in interrupted_tool_removals(history)], ["ai1"])

    def test_request_with_one_unanswered_call_is_removed(self):
        history = [
            AIMessage(
                content="",
                id="ai1",
                tool_calls=[
                    {"name": "find_places", "args": {}, "id": "call_a"},
                    {"name": "recall_memories", "args": {}, "id": "call_b"},
                ],
            ),
            ToolMessage(content="ok", tool_call_id="call_a", id="t1"),
        ]
        self.assertEqual([r.id for r in interrupted_tool_removals(history)], ["ai1"])

    def test_requests_without_ids_are_left_alone(self):
        history = [
            AIMessage(content="", id="ai1", tool_calls=[{"name": "find_places", "args": {}, "id": None}]),
            AIMessage(content="", tool_calls=[{"name": "find_places", "args": {}, "id": "call_c"}]),
        ]
        self.assertEqual(interrupted_tool_removals(history), [])

    def test_nothing_left_to_remove_once_removed(self):
        history = [AIMessage(content="", id="ai1", tool_calls=[{"name": "find_places", "args": {}, "id": "call_3"}])]
        removed_ids = {r.id for r in interrupted_tool_removals(history)}
        remaining = [m for m in history if m.id not in removed_ids]
        self.assertEqual(interrupted_tool_removals(remaining), [])


if __name__ == "__main__":
    unittest.main()
