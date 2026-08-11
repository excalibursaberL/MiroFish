"""Provider-reported LLM token accounting for finance experiments."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Sequence


TOKEN_PHASES = (
    "independent_forecast",
    "pre_social_prediction",
    "social_interaction",
    "belief_snapshot",
    "post_social_prediction",
    "manual_interview",
    "other",
)

AGENT_TOKEN_USAGE_FIELDS = (
    "run_id",
    "scenario_id",
    "agent_id",
    "full_population_agent_id",
    "agent_key",
    "agent_role",
    "agent_role_category",
    "agent_role_label",
    "agent_skill_names",
    "agent_skill_bundle_hash",
    "model",
    "api_call_count",
    "usage_available_call_count",
    "usage_missing_call_count",
    "usage_complete",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    *(f"{phase}_total_tokens" for phase in TOKEN_PHASES),
    "token_measurement",
)


def _as_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    for method_name in ("model_dump", "dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                payload = method()
            except Exception:
                continue
            if isinstance(payload, dict):
                return payload
    try:
        return {
            key: item
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    except TypeError:
        return {}


def normalize_token_usage(value: Any) -> Dict[str, Any]:
    """Normalize OpenAI/CAMEL usage without estimating missing tokens."""
    payload = _as_dict(value)
    if "usage" in payload and not any(
        key in payload
        for key in ("prompt_tokens", "input_tokens", "completion_tokens", "output_tokens")
    ):
        payload = _as_dict(payload.get("usage"))

    prompt = payload.get("prompt_tokens", payload.get("input_tokens"))
    completion = payload.get("completion_tokens", payload.get("output_tokens"))
    total = payload.get("total_tokens")
    available = any(item is not None for item in (prompt, completion, total))

    def integer(item: Any) -> int:
        try:
            return max(0, int(item or 0))
        except (TypeError, ValueError):
            return 0

    prompt_tokens = integer(prompt)
    completion_tokens = integer(completion)
    total_tokens = integer(total)
    if available and total is None:
        total_tokens = prompt_tokens + completion_tokens
    return {
        "usage_available": available,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "provider_usage": payload,
    }


def summarize_agent_token_usage(
    records: Iterable[Dict[str, Any]],
    profiles: Sequence[Dict[str, Any]],
    *,
    run_id: str,
    scenario_id: str = "",
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Aggregate every provider call into one auditable row per investor."""
    by_agent: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        try:
            by_agent[int(record.get("agent_id"))].append(record)
        except (TypeError, ValueError):
            continue

    rows: List[Dict[str, Any]] = []
    for profile in profiles:
        agent_id = int(profile["user_id"])
        agent_records = by_agent.get(agent_id, [])
        phase_totals = {phase: 0 for phase in TOKEN_PHASES}
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        available_count = 0
        models = set()
        for record in agent_records:
            if record.get("model"):
                models.add(str(record["model"]))
            available = bool(record.get("usage_available"))
            if not available:
                continue
            available_count += 1
            prompt_tokens += int(record.get("prompt_tokens", 0) or 0)
            completion_tokens += int(record.get("completion_tokens", 0) or 0)
            call_total = int(record.get("total_tokens", 0) or 0)
            total_tokens += call_total
            phase = str(record.get("phase") or "other")
            if phase not in phase_totals:
                phase = "other"
            phase_totals[phase] += call_total

        call_count = len(agent_records)
        missing_count = call_count - available_count
        rows.append(
            {
                "run_id": run_id,
                "scenario_id": scenario_id,
                "agent_id": agent_id,
                "full_population_agent_id": profile.get(
                    "full_population_agent_id"
                ),
                "agent_key": profile.get("agent_key", ""),
                "agent_role": profile.get("role_id", "investor"),
                "agent_role_category": profile.get("role_category", ""),
                "agent_role_label": profile.get("role_label", ""),
                "agent_skill_names": "|".join(
                    str(name) for name in profile.get("finance_skill_names", [])
                ),
                "agent_skill_bundle_hash": profile.get(
                    "finance_skill_bundle_hash", ""
                ),
                "model": "|".join(sorted(models)),
                "api_call_count": call_count,
                "usage_available_call_count": available_count,
                "usage_missing_call_count": missing_count,
                "usage_complete": call_count > 0 and missing_count == 0,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                **{
                    f"{phase}_total_tokens": phase_totals[phase]
                    for phase in TOKEN_PHASES
                },
                "token_measurement": "provider_reported_no_estimation",
            }
        )

    summary = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "agent_count": len(rows),
        "agents_with_usage": sum(row["usage_available_call_count"] > 0 for row in rows),
        "api_call_count": sum(row["api_call_count"] for row in rows),
        "usage_available_call_count": sum(
            row["usage_available_call_count"] for row in rows
        ),
        "usage_missing_call_count": sum(
            row["usage_missing_call_count"] for row in rows
        ),
        "prompt_tokens": sum(row["prompt_tokens"] for row in rows),
        "completion_tokens": sum(row["completion_tokens"] for row in rows),
        "total_tokens": sum(row["total_tokens"] for row in rows),
        "phase_total_tokens": {
            phase: sum(row[f"{phase}_total_tokens"] for row in rows)
            for phase in TOKEN_PHASES
        },
        "token_measurement": "provider_reported_no_estimation",
        "note": (
            "Missing provider usage is reported explicitly and is never replaced "
            "with a text-length estimate."
        ),
    }
    return rows, summary
