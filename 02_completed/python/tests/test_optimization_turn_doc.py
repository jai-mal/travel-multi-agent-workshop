from __future__ import annotations

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path


PYTHON_ROOT = Path(__file__).resolve().parents[1]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from src.app.services.optimization_turn_doc import build_optimization_turn_doc  # noqa: E402


NOW = datetime(2026, 11, 10, 21, 30, tzinfo=timezone.utc)


class BuildOptimizationTurnDocTests(unittest.TestCase):
    def test_live_turn_matches_the_traffic_record_shape(self):
        doc = build_optimization_turn_doc(
            turn_id="abc",
            tenant_id="analytics",
            user_id="sam_carter",
            session_id="session_1",
            complexity_tier="routine",
            model_deployment="gpt-5-mini",
            model_name="gpt-5-mini-2025-08-07",
            input_tokens=1200,
            output_tokens=300,
            cached_tokens=800,
            handoff_count=1,
            agent_path="supervisor,find_places",
            now=NOW,
        )
        self.assertEqual(doc["id"], "turn-abc")
        self.assertEqual(doc["type"], "optimization_turn")
        self.assertEqual(doc["source"], "live")
        self.assertEqual((doc["tenantId"], doc["userId"], doc["sessionId"]), ("analytics", "sam_carter", "session_1"))
        self.assertEqual(doc["complexity_tier"], "routine")
        self.assertEqual(doc["model_deployment"], "gpt-5-mini")
        self.assertEqual(doc["total_tokens"], 1500)
        self.assertEqual(doc["cached_tokens"], 800)
        self.assertEqual(doc["handoff_count"], 1)
        self.assertEqual(doc["agent_path"], "supervisor,find_places")
        self.assertEqual(doc["timeStamp"], NOW.isoformat())
        self.assertEqual(doc["turn_epoch"], int(NOW.timestamp()))

    def test_missing_values_fall_back_to_safe_defaults(self):
        doc = build_optimization_turn_doc(
            turn_id="x",
            tenant_id="marvel",
            user_id="tony",
            session_id="session_2",
            complexity_tier=None,
            model_deployment=None,
            model_name=None,
            input_tokens=None,
            output_tokens=None,
            cached_tokens=None,
            handoff_count=None,
            agent_path="",
            now=NOW,
        )
        self.assertEqual(doc["complexity_tier"], "default")
        self.assertEqual(doc["model_deployment"], "unknown")
        self.assertEqual(doc["total_tokens"], 0)
        self.assertEqual(doc["handoff_count"], 0)
        self.assertEqual(doc["agent_path"], "supervisor")


if __name__ == "__main__":
    unittest.main()
