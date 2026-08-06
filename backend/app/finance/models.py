"""Small typed models used by the financial adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class C0Forecast:
    """A single independent forecast returned by one investor Agent."""

    scenario_id: str
    agent_id: int
    agent_role: str
    agent_role_category: str
    agent_role_label: str
    as_of: str
    horizon: str
    agent_knowledge_level: Optional[str] = None
    agent_analysis_style: Optional[str] = None
    agent_risk_attitude: Optional[str] = None
    agent_investment_horizon: Optional[str] = None
    profile_version: Optional[str] = None
    direction: Optional[str] = None
    up_probability: Optional[float] = None
    neutral_probability: Optional[float] = None
    down_probability: Optional[float] = None
    expected_return: Optional[float] = None
    expected_return_unit: str = "decimal"
    confidence: Optional[float] = None
    evidence_event_ids: List[str] = field(default_factory=list)
    reason: str = ""
    raw_response: str = ""
    status: str = "ok"
    error: Optional[str] = None
    attempt_count: int = 1
    finish_reason: Optional[str] = None
    response_content_length: int = 0
    reasoning_content_present: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable record for the result JSONL file."""
        return {
            "scenario_id": self.scenario_id,
            "agent_id": self.agent_id,
            "agent_role": self.agent_role,
            "agent_role_category": self.agent_role_category,
            "agent_role_label": self.agent_role_label,
            "agent_knowledge_level": self.agent_knowledge_level,
            "agent_analysis_style": self.agent_analysis_style,
            "agent_risk_attitude": self.agent_risk_attitude,
            "agent_investment_horizon": self.agent_investment_horizon,
            "profile_version": self.profile_version,
            "as_of": self.as_of,
            "horizon": self.horizon,
            "direction": self.direction,
            "up_probability": self.up_probability,
            "neutral_probability": self.neutral_probability,
            "down_probability": self.down_probability,
            "expected_return": self.expected_return,
            "expected_return_unit": self.expected_return_unit,
            "confidence": self.confidence,
            "evidence_event_ids": self.evidence_event_ids,
            "reason": self.reason,
            "raw_response": self.raw_response,
            "status": self.status,
            "error": self.error,
            "attempt_count": self.attempt_count,
            "finish_reason": self.finish_reason,
            "response_content_length": self.response_content_length,
            "reasoning_content_present": self.reasoning_content_present,
        }
