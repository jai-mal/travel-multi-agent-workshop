"""Shape one captured chat turn as an OptimizationTurns record.

The Fabric mirror and the analytics notebook read this table. The fields match
what the traffic generator writes, so live turns and practice turns analyze the
same way. Live turns carry source "live" so they can be told apart.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def build_optimization_turn_doc(
    *,
    turn_id: str,
    tenant_id: str,
    user_id: str,
    session_id: str,
    complexity_tier: str | None,
    model_deployment: str | None,
    model_name: str | None,
    input_tokens: int | None,
    output_tokens: int | None,
    cached_tokens: int | None,
    handoff_count: int | None,
    agent_path: str | None,
    now: datetime | None = None,
) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    input_tokens = int(input_tokens or 0)
    output_tokens = int(output_tokens or 0)
    deployment = model_deployment or model_name or "unknown"
    return {
        "id": f"turn-{turn_id}",
        "type": "optimization_turn",
        "source": "live",
        "tenantId": tenant_id,
        "userId": user_id,
        "sessionId": session_id,
        "complexity_tier": complexity_tier or "default",
        "model_deployment": deployment,
        "model_name": model_name or deployment,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cached_tokens": int(cached_tokens or 0),
        "handoff_count": int(handoff_count or 0),
        "agent_path": agent_path or "supervisor",
        "timeStamp": now.isoformat(),
        "turn_epoch": int(now.timestamp()),
    }
