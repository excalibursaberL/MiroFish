"""Analyze candidate stopping rounds from one completed S1 batch."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


DIRECTIONS = ("up", "neutral", "down")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def probability_vector(row: dict[str, Any]) -> tuple[float, float, float]:
    values = tuple(float(row[f"{name}_probability"]) for name in DIRECTIONS)
    total = sum(values)
    if total <= 0:
        raise ValueError("probability sum must be positive")
    return tuple(value / total for value in values)  # type: ignore[return-value]


def js_divergence(left: Iterable[float], right: Iterable[float]) -> float:
    p = tuple(left)
    q = tuple(right)
    middle = tuple((a + b) / 2 for a, b in zip(p, q))

    def kl(values: tuple[float, ...], target: tuple[float, ...]) -> float:
        return sum(
            value * math.log2(value / reference)
            for value, reference in zip(values, target)
            if value > 0 and reference > 0
        )

    return 0.5 * kl(p, middle) + 0.5 * kl(q, middle)


def entropy_bits(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if not total:
        return 0.0
    return -sum(
        (count / total) * math.log2(count / total)
        for count in counts.values()
        if count
    )


def average(values: Iterable[float]) -> float | None:
    valid = [float(value) for value in values if value is not None]
    return mean(valid) if valid else None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(batch_dir: Path) -> dict[str, Any]:
    manifest = json.loads((batch_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError("batch must be completed before round-selection analysis")
    finance_dir = batch_dir.parent
    max_round = int(manifest["social_rounds"])
    snapshots: dict[tuple[str, int, int], dict[str, Any]] = {}
    actual: dict[str, str] = {}
    actions_by_round: dict[int, list[dict[str, Any]]] = defaultdict(list)
    tokens_by_round: Counter[int] = Counter()
    scenario_ids: list[str] = []

    for run in manifest["runs"]:
        scenario_id = str(run["scenario_id"])
        scenario_ids.append(scenario_id)
        run_dir = finance_dir / str(run["run_id"])
        for row in read_jsonl(run_dir / "belief_snapshots.jsonl"):
            snapshots[(scenario_id, int(row["round"]), int(row["agent_id"]))] = row
        with (run_dir / "evaluation.csv").open(encoding="utf-8-sig", newline="") as handle:
            evaluation_rows = list(csv.DictReader(handle))
        actual[scenario_id] = str(evaluation_rows[0]["actual_five_day_close_direction"])
        for row in read_jsonl(run_dir / "social_actions.jsonl"):
            round_number = int(row.get("round", 0) or 0)
            if round_number > 0 and row.get("agent_class") == "investor":
                actions_by_round[round_number].append(row)
        for row in read_jsonl(run_dir / "llm_token_usage.jsonl"):
            round_number = row.get("round")
            if round_number is not None and row.get("total_tokens") is not None:
                tokens_by_round[int(round_number)] += int(row["total_tokens"])

    metrics: list[dict[str, Any]] = []
    cumulative_tokens = int(tokens_by_round.get(0, 0))
    for round_number in range(max_round + 1):
        rows = [
            row
            for (scenario_id, row_round, _), row in snapshots.items()
            if row_round == round_number and row.get("status") == "ok"
        ]
        expected = len(scenario_ids) * int(manifest["runs"][0].get("metrics", {}).get(
            "belief_snapshots", {}
        ).get("expected_count", 0) or 0) // (max_round + 1)
        if not expected:
            expected = len(scenario_ids) * 10

        transition_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
        cumulative_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
        if round_number > 0:
            for scenario_id in scenario_ids:
                for agent_id in range(10):
                    previous = snapshots.get((scenario_id, round_number - 1, agent_id))
                    current = snapshots.get((scenario_id, round_number, agent_id))
                    baseline = snapshots.get((scenario_id, 0, agent_id))
                    if previous and current and previous.get("status") == current.get("status") == "ok":
                        transition_pairs.append((previous, current))
                    if baseline and current and baseline.get("status") == current.get("status") == "ok":
                        cumulative_pairs.append((baseline, current))

        scenario_consensus: list[float] = []
        scenario_entropy: list[float] = []
        scenario_polarization: list[float] = []
        majority_correct = 0
        majority_defined = 0
        for scenario_id in scenario_ids:
            scenario_rows = [
                row
                for row in rows
                if row["scenario_id"] == scenario_id
            ]
            counts = Counter(str(row["direction"]) for row in scenario_rows)
            if counts:
                top_count = max(counts.values())
                leaders = [key for key, count in counts.items() if count == top_count]
                scenario_consensus.append(top_count / len(scenario_rows))
                scenario_entropy.append(entropy_bits(counts))
                if len(leaders) == 1:
                    majority_defined += 1
                    majority_correct += int(leaders[0] == actual[scenario_id])
            vectors = [probability_vector(row) for row in scenario_rows]
            pairwise = [js_divergence(a, b) for a, b in combinations(vectors, 2)]
            scenario_polarization.append(average(pairwise) or 0.0)

        round_actions = actions_by_round.get(round_number, [])
        content_actions = [
            row
            for row in round_actions
            if str(row.get("action_type", "")).lower() in {"create_post", "create_comment"}
        ]
        cumulative_tokens += int(tokens_by_round.get(round_number, 0)) if round_number > 0 else 0
        transition_js = [
            js_divergence(probability_vector(before), probability_vector(after))
            for before, after in transition_pairs
        ]
        cumulative_js = [
            js_divergence(probability_vector(before), probability_vector(after))
            for before, after in cumulative_pairs
        ]
        metrics.append(
            {
                "round": round_number,
                "valid_snapshot_count": len(rows),
                "expected_snapshot_count": expected,
                "valid_snapshot_rate": len(rows) / expected,
                "individual_direction_accuracy": average(
                    row["direction"] == actual[row["scenario_id"]] for row in rows
                ),
                "majority_correct_scenarios": majority_correct,
                "majority_defined_scenarios": majority_defined,
                "majority_direction_accuracy": (
                    majority_correct / majority_defined if majority_defined else None
                ),
                "mean_scenario_consensus_rate": average(scenario_consensus),
                "mean_scenario_direction_entropy_bits": average(scenario_entropy),
                "mean_scenario_pairwise_js": average(scenario_polarization),
                "mean_expected_return": average(row["expected_return"] for row in rows),
                "mean_confidence": average(row["confidence"] for row in rows),
                "transition_pair_count": len(transition_pairs),
                "transition_direction_flip_rate": average(
                    before["direction"] != after["direction"]
                    for before, after in transition_pairs
                ),
                "transition_mean_js": average(transition_js),
                "transition_mean_abs_return_delta": average(
                    abs(float(after["expected_return"]) - float(before["expected_return"]))
                    for before, after in transition_pairs
                ),
                "cumulative_pair_count": len(cumulative_pairs),
                "cumulative_direction_flip_rate": average(
                    before["direction"] != after["direction"]
                    for before, after in cumulative_pairs
                ),
                "cumulative_mean_js": average(cumulative_js),
                "investor_action_count": len(round_actions),
                "new_content_count": len(content_actions),
                "expressing_agent_scenario_count": len(
                    {(row["scenario_id"], int(row["agent_id"])) for row in content_actions}
                ),
                "round_total_tokens": int(tokens_by_round.get(round_number, 0)),
                "cumulative_total_tokens": cumulative_tokens,
            }
        )

    missing = [
        {
            "scenario_id": scenario_id,
            "round": round_number,
            "agent_id": agent_id,
            "status": row.get("status"),
            "error": row.get("error"),
        }
        for (scenario_id, round_number, agent_id), row in snapshots.items()
        if row.get("status") != "ok"
    ]
    summary = {
        "batch_id": manifest["batch_id"],
        "scenario_count": len(scenario_ids),
        "agent_count_per_scenario": 10,
        "max_round": max_round,
        "random_seed": manifest.get("random_seed"),
        "prompt_version": "finance_forecast_s1_v3_unified_belief",
        "snapshot_count": len(snapshots),
        "valid_snapshot_count": sum(row.get("status") == "ok" for row in snapshots.values()),
        "missing_snapshots": missing,
        "metrics": metrics,
    }
    write_csv(batch_dir / "round_selection_metrics.csv", metrics)
    (batch_dir / "round_selection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest.setdefault("files", {}).update(
        {
            "round_selection_metrics": "round_selection_metrics.csv",
            "round_selection_summary": "round_selection_summary.json",
            "round_selection_report": "round_selection_report.md",
        }
    )
    (batch_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_dir", type=Path)
    args = parser.parse_args()
    summary = analyze(args.batch_dir.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
