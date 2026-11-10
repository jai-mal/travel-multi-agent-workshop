from __future__ import annotations

import sys
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


PYTHON_ROOT = Path(__file__).resolve().parents[1]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from src.app.services.tool_call_repair import INTERRUPTED_RESULT, interrupted_tool_results  # noqa: E402


class InterruptedToolCallTests(unittest.TestCase):
    def test_completed_calls_get_no_placeholder(self):
        history = [
            HumanMessage(content="plan a trip"),
            AIMessage(content="", tool_calls=[{"name": "find_places", "args": {}, "id": "call_1"}]),
            ToolMessage(content="ok", tool_call_id="call_1"),
            AIMessage(content="Here is a plan."),
        ]
        self.assertEqual(interrupted_tool_results(history), [])

    def test_unanswered_call_at_end_gets_placeholder(self):
        history = [
            HumanMessage(content="2"),
            AIMessage(
                content="",
                tool_calls=[{"name": "create_or_update_itinerary", "args": {}, "id": "call_2"}],
            ),
        ]
        results = interrupted_tool_results(history)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].tool_call_id, "call_2")
        self.assertEqual(results[0].name, "create_or_update_itinerary")
        self.assertEqual(results[0].content, INTERRUPTED_RESULT)

    def test_only_unanswered_calls_in_a_batch_are_closed(self):
        history = [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "find_places", "args": {}, "id": "call_a"},
                    {"name": "recall_memories", "args": {}, "id": "call_b"},
                ],
            ),
            ToolMessage(content="ok", tool_call_id="call_a"),
        ]
        self.assertEqual([r.tool_call_id for r in interrupted_tool_results(history)], ["call_b"])

    def test_calls_without_an_id_cannot_be_matched_and_are_skipped(self):
        history = [AIMessage(content="", tool_calls=[{"name": "find_places", "args": {}, "id": None}])]
        self.assertEqual(interrupted_tool_results(history), [])

    def test_no_second_placeholder_once_appended(self):
        history = [AIMessage(content="", tool_calls=[{"name": "find_places", "args": {}, "id": "call_3"}])]
        history += interrupted_tool_results(history)
        self.assertEqual(interrupted_tool_results(history), [])


if __name__ == "__main__":
    unittest.main()
