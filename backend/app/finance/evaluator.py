"""Researcher-only outcome reader for completed financial runs.

This module is deliberately separate from the Agent dataset loader. Evaluator
targets must never be passed to prompts, profiles, Zep, or social simulation.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_TARGET_PATH = (
    PROJECT_ROOT / "Dataset" / "seed5_small_blind" / "evaluator_targets.tsv"
)
ASTOCK_LABEL_DIRECTIONS = {
    "0": "down",
    "1": "neutral",
    "2": "up",
}

# Fixed before OOD evaluation.  The rounded 1.7% band is calibrated from the
# 20th percentile of absolute five-day returns in the Astock training split.
# Keep this value in one place so prompts and evaluator labels cannot drift.
FIVE_DAY_NEUTRAL_THRESHOLD = 0.017
_FIVE_DAY_NEUTRAL_PERCENT = FIVE_DAY_NEUTRAL_THRESHOLD * 100
FIVE_DAY_DIRECTION_DEFINITION = (
    f"down: R5 < -{_FIVE_DAY_NEUTRAL_PERCENT:.1f}%; "
    f"neutral: -{_FIVE_DAY_NEUTRAL_PERCENT:.1f}% <= R5 "
    f"<= {_FIVE_DAY_NEUTRAL_PERCENT:.1f}%; "
    f"up: R5 > {_FIVE_DAY_NEUTRAL_PERCENT:.1f}%; "
    "R5 = close5 / original_price - 1"
)


class FinancialOutcomeEvaluator:
    """Read hidden outcomes after an experiment has completed."""

    def __init__(self, target_path: str | Path = DEFAULT_TARGET_PATH):
        self.target_path = Path(target_path).resolve()

    def get_outcome(self, scenario_id: str) -> Dict[str, Any]:
        with self.target_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                if row.get("scenario_id") == scenario_id:
                    return self._to_outcome(row)
        raise ValueError(f"evaluator target not found for scenario: {scenario_id}")

    @staticmethod
    def _to_outcome(row: Dict[str, str]) -> Dict[str, Any]:
        label = row["label"]
        change_return = float(row["CHANGE"])
        original_price = float(row["original_price"])
        close5 = float(row["close5"])
        five_day_return = close5 / original_price - 1.0
        return {
            "scenario_id": row["scenario_id"],
            "as_of": row["trade_date"],
            "horizon": "next_5_trading_days",
            "astock_label": label,
            "astock_direction": ASTOCK_LABEL_DIRECTIONS.get(label, "unknown"),
            "astock_change_return": change_return,
            "five_day_close_return": five_day_return,
            "five_day_close_direction": FinancialOutcomeEvaluator._sign_direction(
                five_day_return
            ),
            "five_day_neutral_threshold": FIVE_DAY_NEUTRAL_THRESHOLD,
            "five_day_direction_definition": FIVE_DAY_DIRECTION_DEFINITION,
            "normalized_original_price": original_price,
            "normalized_close5": close5,
            "daily_closes": [float(row[f"close{day}"]) for day in range(1, 6)],
            "disclosure": (
                "Astock label/CHANGE and the five-day end-to-end close return are "
                "different evaluation definitions. This record is evaluator-only."
            ),
        }

    @staticmethod
    def _sign_direction(
        value: float,
        neutral_threshold: float = FIVE_DAY_NEUTRAL_THRESHOLD,
    ) -> str:
        """Map a five-day return to the fixed three-class benchmark label."""
        if neutral_threshold < 0:
            raise ValueError("neutral threshold must be non-negative")
        if value > neutral_threshold:
            return "up"
        if value < -neutral_threshold:
            return "down"
        return "neutral"
