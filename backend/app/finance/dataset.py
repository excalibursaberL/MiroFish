"""Loader and leakage checks for the anonymous financial benchmark."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "Dataset" / "seed5_small_blind" / "mirofish_inputs.jsonl"


class DatasetValidationError(ValueError):
    """Raised when a financial input snapshot is malformed or leaks answers."""


_FORBIDDEN_KEY_NAMES = {
    "label",
    "change",
    "evaluator_targets",
    "future_return",
    "future_price",
    "original_price",
    "trade_date",
    "first_day",
    "second_day",
    "manifest",
}
_FORBIDDEN_KEY_PATTERNS = (
    re.compile(r"^(?:open|close|day)\d+$", re.IGNORECASE),
    re.compile(r"^(?:future|evaluator|target)(?:_|$)", re.IGNORECASE),
)
_ANONYMOUS_ID_PATTERNS = {
    "scenario_id": re.compile(r"^SCN_[A-Za-z0-9_-]+$"),
    "symbol": re.compile(r"^ASSET_[A-Za-z0-9_-]+$"),
    "name": re.compile(r"^COMPANY_[A-Za-z0-9_-]+$"),
    "event_id": re.compile(r"^EVT_[A-Za-z0-9_-]+$"),
}
_RELATIVE_TIME_PATTERN = re.compile(r"^T[+-]\d+d$")
_ABSOLUTE_DATE_PATTERN = re.compile(
    r"(?<![A-Za-z])20\d{2}(?:[-/.]\d{1,2}(?:[-/.]\d{1,2})?|年)"
)


def _find_forbidden_key(value: Any, path: str = "scenario") -> Optional[str]:
    """Find evaluator/future fields at any nesting level."""
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            normalized = key_text.lower()
            if normalized in _FORBIDDEN_KEY_NAMES or any(
                pattern.match(key_text) for pattern in _FORBIDDEN_KEY_PATTERNS
            ):
                return f"{path}.{key_text}"
            found = _find_forbidden_key(child, f"{path}.{key_text}")
            if found:
                return found
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found = _find_forbidden_key(child, f"{path}[{index}]")
            if found:
                return found
    return None


def _require_anonymous_id(field: str, value: Any, path: str) -> str:
    """Validate the blind benchmark's ID and return it as text."""
    value_text = str(value)
    pattern = _ANONYMOUS_ID_PATTERNS[field]
    if not pattern.fullmatch(value_text):
        raise DatasetValidationError(
            f"{path} must use an anonymous {field} such as {pattern.pattern}"
        )
    return value_text


def _require_relative_time(value: Any, path: str) -> str:
    value_text = str(value)
    if not _RELATIVE_TIME_PATTERN.fullmatch(value_text):
        raise DatasetValidationError(f"{path} must use a relative time such as T-5d")
    return value_text


def _relative_day(value: str) -> int:
    """Convert a validated relative time to a comparable day offset."""
    sign = -1 if value[1] == "-" else 1
    return sign * int(value[2:-1])


@dataclass(frozen=True)
class FinancialScenario:
    """One point-in-time scenario safe to expose to an Agent."""

    scenario_id: str
    symbol: str
    name: str
    prediction_cutoff: str
    horizon: str
    seed_events: List[Dict[str, Any]]
    current_event: Dict[str, Any]

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "FinancialScenario":
        if not isinstance(payload, dict):
            raise DatasetValidationError("scenario must be a JSON object")

        forbidden_path = _find_forbidden_key(payload)
        if forbidden_path:
            raise DatasetValidationError(
                f"scenario contains evaluator/future field: {forbidden_path}"
            )

        required = {
            "scenario_id",
            "symbol",
            "name",
            "prediction_cutoff",
            "horizon",
            "seed_events",
            "current_event",
        }
        missing = sorted(required - payload.keys())
        if missing:
            raise DatasetValidationError(
                f"scenario missing required fields: {', '.join(missing)}"
            )

        seeds = payload["seed_events"]
        if not isinstance(seeds, list) or not seeds:
            raise DatasetValidationError("seed_events must be a non-empty list")
        scenario_id = _require_anonymous_id(
            "scenario_id", payload["scenario_id"], "scenario_id"
        )
        symbol = _require_anonymous_id("symbol", payload["symbol"], "symbol")
        name = _require_anonymous_id("name", payload["name"], "name")
        prediction_cutoff = _require_relative_time(
            payload["prediction_cutoff"], "prediction_cutoff"
        )
        cutoff_day = _relative_day(prediction_cutoff)
        seed_days = []
        for index, event in enumerate(seeds, start=1):
            cls._validate_event(event, f"seed_events[{index}]")
            cls._validate_event_identity(event, symbol, name, f"seed_events[{index}]")
            seed_days.append(_relative_day(str(event["event_time"])))
        cls._validate_event(payload["current_event"], "current_event")
        cls._validate_event_identity(payload["current_event"], symbol, name, "current_event")
        current_day = _relative_day(str(payload["current_event"]["event_time"]))
        if current_day != cutoff_day:
            raise DatasetValidationError(
                "current_event.event_time must equal prediction_cutoff"
            )
        if any(day >= cutoff_day for day in seed_days):
            raise DatasetValidationError("all seed events must precede the prediction cutoff")
        if seed_days != sorted(seed_days):
            raise DatasetValidationError("seed_events must be in chronological order")

        return cls(
            scenario_id=scenario_id,
            symbol=symbol,
            name=name,
            prediction_cutoff=prediction_cutoff,
            horizon=str(payload["horizon"]),
            seed_events=seeds,
            current_event=payload["current_event"],
        )

    @staticmethod
    def _validate_event(event: Any, label: str) -> None:
        if not isinstance(event, dict):
            raise DatasetValidationError(f"{label} must be an object")
        required = {"event_id", "event_time", "text"}
        missing = sorted(required - event.keys())
        if missing:
            raise DatasetValidationError(
                f"{label} missing required fields: {', '.join(missing)}"
            )
        forbidden_path = _find_forbidden_key(event, label)
        if forbidden_path:
            raise DatasetValidationError(
                f"{label} contains evaluator/future field: {forbidden_path}"
            )
        _require_anonymous_id("event_id", event["event_id"], f"{label}.event_id")
        _require_relative_time(event["event_time"], f"{label}.event_time")
        if "created_time" in event:
            _require_relative_time(event["created_time"], f"{label}.created_time")
        text = str(event["text"])
        if _ABSOLUTE_DATE_PATTERN.search(text):
            raise DatasetValidationError(
                f"{label}.text contains an absolute date; use relative dates"
            )

    @staticmethod
    def _validate_event_identity(
        event: Dict[str, Any], symbol: str, name: str, label: str
    ) -> None:
        for field, expected in (("symbol", symbol), ("name", name)):
            if field in event and str(event[field]) != expected:
                raise DatasetValidationError(
                    f"{label}.{field} must match the scenario anonymous {field}"
                )

    def to_safe_dict(self) -> Dict[str, Any]:
        """Return only the fields permitted in a C0 prompt or snapshot."""
        return {
            "scenario_id": self.scenario_id,
            "symbol": self.symbol,
            "name": self.name,
            "prediction_cutoff": self.prediction_cutoff,
            "horizon": self.horizon,
            "seed_events": self.seed_events,
            "current_event": self.current_event,
        }


class FinancialDatasetLoader:
    """Read anonymous JSONL scenarios without opening evaluator files."""

    def __init__(self, dataset_path: Optional[str | Path] = None):
        self.dataset_path = self._resolve_path(dataset_path)

    @staticmethod
    def _resolve_path(dataset_path: Optional[str | Path]) -> Path:
        path = Path(dataset_path) if dataset_path else DEFAULT_DATASET_PATH
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path = path.resolve()
        if path.name in {
            "evaluator_targets.tsv",
            "manifest.tsv",
            "seed5_small_blind_mapping_private.tsv",
        } or "mapping_private" in path.name:
            raise DatasetValidationError(
                "financial Agent input cannot point to evaluator or private mapping files"
            )
        if path.suffix.lower() != ".jsonl":
            raise DatasetValidationError("financial Agent input must be a JSONL snapshot")
        if not path.exists():
            raise FileNotFoundError(f"financial dataset not found: {path}")
        return path

    def load(
        self,
        *,
        scenario_ids: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
    ) -> List[FinancialScenario]:
        selected_values = list(scenario_ids or [])
        if any(not isinstance(value, str) or not value.strip() for value in selected_values):
            raise DatasetValidationError("scenario_ids must contain non-empty strings")
        selected = set(selected_values)
        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0
        ):
            raise DatasetValidationError("limit must be a positive integer")

        scenarios: List[FinancialScenario] = []
        seen_ids = set()
        with self.dataset_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                    scenario = FinancialScenario.from_dict(payload)
                except (json.JSONDecodeError, DatasetValidationError) as error:
                    raise DatasetValidationError(
                        f"invalid financial JSONL at line {line_number}: {error}"
                    ) from error
                if scenario.scenario_id in seen_ids:
                    raise DatasetValidationError(
                        f"duplicate scenario_id: {scenario.scenario_id}"
                    )
                seen_ids.add(scenario.scenario_id)
                if selected and scenario.scenario_id not in selected:
                    continue
                scenarios.append(scenario)
                if limit is not None and len(scenarios) >= limit:
                    break

        if selected:
            loaded_ids = {scenario.scenario_id for scenario in scenarios}
            missing_ids = sorted(selected - loaded_ids)
            if missing_ids:
                raise DatasetValidationError(
                    f"scenario IDs not found: {', '.join(missing_ids)}"
                )
        if not scenarios:
            raise DatasetValidationError("no financial scenarios selected")
        return scenarios

    def load_safe_payloads(self, **kwargs: Any) -> List[Dict[str, Any]]:
        return [scenario.to_safe_dict() for scenario in self.load(**kwargs)]
