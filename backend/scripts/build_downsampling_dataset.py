#!/usr/bin/env python3
"""Build a leakage-safe S1 dataset for Agent downsampling research.

The source runs remain immutable.  This script joins the completed four-round
S1 artifacts at scenario/agent/round granularity and writes analysis tables
that deliberately exclude future market outcomes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DATASET_VERSION = "downsampling_s1_rounds4_v1"
STANCE_LABELS = ("positive", "mixed", "negative", "neutral", "uncertain")
ACTION_TYPES = (
    "create_post",
    "create_comment",
    "like_post",
    "like_comment",
    "dislike_post",
    "dislike_comment",
    "refresh",
    "trend",
    "search_posts",
    "search_user",
    "follow",
)
PROFILE_FIELDS = (
    "agent_role",
    "agent_role_category",
    "agent_role_label",
    "agent_knowledge_level",
    "agent_analysis_style",
    "agent_risk_attitude",
    "agent_investment_horizon",
    "agent_decision_source",
    "agent_social_role",
    "profile_version",
)
FORBIDDEN_ANALYSIS_COLUMNS = {
    "actual_direction",
    "actual_return",
    "actual_five_day_close_direction",
    "actual_five_day_close_return",
    "five_day_direction_correct",
    "five_day_return_error",
    "label",
    "change",
}


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    root = workspace_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=root
        / "MiroFish"
        / "backend"
        / "uploads"
        / "finance"
        / "s1_rounds4_all18_20260806",
        help="Directory containing scenario_run_index.csv",
    )
    parser.add_argument(
        "--finance-runs-dir",
        type=Path,
        default=root / "MiroFish" / "backend" / "uploads" / "finance",
        help="Directory containing individual s1_reddit_* runs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "Dataset" / DATASET_VERSION,
        help="Destination directory",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL in {path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"expected object in {path}:{line_number}")
            rows.append(value)
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def fieldnames_for(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                names.append(key)
    return names


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> list[str]:
    fields = fieldnames_for(rows)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: normalize_csv_value(row.get(key)) for key in fields})
    return fields


def normalize_csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def mean(values: Iterable[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None]
    return sum(valid) / len(valid) if valid else None


def population_std(values: Iterable[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None]
    return statistics.pstdev(valid) if valid else None


def probability_vector(row: Mapping[str, Any] | None) -> list[float] | None:
    if not row or row.get("status") != "ok":
        return None
    values = [
        safe_float(row.get("up_probability")),
        safe_float(row.get("neutral_probability")),
        safe_float(row.get("down_probability")),
    ]
    if any(value is None or value < 0 for value in values):
        return None
    total = sum(float(value) for value in values)
    if total <= 0:
        return None
    return [float(value) / total for value in values]


def js_divergence(left: Sequence[float] | None, right: Sequence[float] | None) -> float | None:
    if left is None or right is None or len(left) != len(right):
        return None
    midpoint = [(a + b) / 2 for a, b in zip(left, right)]

    def kl(values: Sequence[float], reference: Sequence[float]) -> float:
        return sum(
            value * math.log(value / ref, 2)
            for value, ref in zip(values, reference)
            if value > 0 and ref > 0
        )

    return (kl(left, midpoint) + kl(right, midpoint)) / 2


def normalized_stance(value: Any) -> str:
    label = str(value or "uncertain").strip().lower()
    return label if label in STANCE_LABELS else "uncertain"


def direction_entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if not total:
        return 0.0
    return -sum(
        (count / total) * math.log(count / total, 2)
        for count in counts.values()
        if count > 0
    )


def majority_direction(counts: Counter[str]) -> str:
    if not counts:
        return "missing"
    ranked = counts.most_common()
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return "tie"
    return ranked[0][0]


def group_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = [(row, probability_vector(row)) for row in rows]
    valid = [(row, vector) for row, vector in valid if vector is not None]
    vectors = [vector for _, vector in valid]
    directions = Counter(str(row.get("direction", "missing")) for row, _ in valid)
    count = len(valid)
    pairwise = [
        js_divergence(vectors[i], vectors[j])
        for i in range(count)
        for j in range(i + 1, count)
    ]
    return {
        "valid_agent_count": count,
        "mean_up_probability": mean(vector[0] for vector in vectors),
        "mean_neutral_probability": mean(vector[1] for vector in vectors),
        "mean_down_probability": mean(vector[2] for vector in vectors),
        "mean_expected_return": mean(safe_float(row.get("expected_return")) for row, _ in valid),
        "mean_confidence": mean(safe_float(row.get("confidence")) for row, _ in valid),
        "up_direction_count": directions.get("up", 0),
        "neutral_direction_count": directions.get("neutral", 0),
        "down_direction_count": directions.get("down", 0),
        "up_direction_proportion": directions.get("up", 0) / count if count else None,
        "neutral_direction_proportion": directions.get("neutral", 0) / count if count else None,
        "down_direction_proportion": directions.get("down", 0) / count if count else None,
        "majority_direction": majority_direction(directions),
        "consensus_rate": max(directions.values()) / count if count and directions else None,
        "direction_entropy_bits": direction_entropy(directions),
        "mean_pairwise_js_divergence": mean(pairwise),
    }


def prefixed(prefix: str, values: Mapping[str, Any]) -> dict[str, Any]:
    return {f"{prefix}{key}": value for key, value in values.items()}


def prediction_fields(prefix: str, row: Mapping[str, Any] | None) -> dict[str, Any]:
    row = row or {}
    vector = probability_vector(row)
    return {
        f"{prefix}status": row.get("status", "missing"),
        f"{prefix}direction": row.get("direction"),
        f"{prefix}up_probability": vector[0] if vector else None,
        f"{prefix}neutral_probability": vector[1] if vector else None,
        f"{prefix}down_probability": vector[2] if vector else None,
        f"{prefix}expected_return": safe_float(row.get("expected_return")),
        f"{prefix}confidence": safe_float(row.get("confidence")),
    }


def profile_fields(profile: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "agent_role": profile.get("role_id"),
        "agent_role_category": profile.get("role_category"),
        "agent_role_label": profile.get("role_label"),
        "agent_knowledge_level": profile.get("knowledge_level"),
        "agent_analysis_style": profile.get("analysis_style"),
        "agent_risk_attitude": profile.get("risk_attitude"),
        "agent_investment_horizon": profile.get("investment_horizon"),
        "agent_decision_source": profile.get("decision_source"),
        "agent_social_role": profile.get("social_role"),
        "profile_version": profile.get("profile_version"),
    }


def deduplicate_exposures(
    edges: Sequence[Mapping[str, Any]],
    scenario_id: str,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for edge in edges:
        viewer = safe_int(edge.get("viewer_agent_id"))
        round_number = safe_int(edge.get("round"))
        if viewer is None or not 0 <= viewer < 20 or round_number is None or round_number < 1:
            continue
        key = (
            viewer,
            round_number,
            str(edge.get("content_type")),
            str(edge.get("content_id")),
        )
        grouped[key].append(edge)

    rows: list[dict[str, Any]] = []
    for (viewer, round_number, content_type, content_id), items in sorted(grouped.items()):
        first = items[0]
        author_id = safe_int(first.get("author_agent_id"))
        first_seen = min(
            value
            for value in (safe_int(item.get("first_seen_round")) for item in items)
            if value is not None
        )
        stance = normalized_stance(first.get("content_stance"))
        scores = [safe_float(item.get("stance_score")) for item in items]
        confidences = [safe_float(item.get("stance_confidence")) for item in items]
        rows.append(
            {
                "scenario_id": scenario_id,
                "viewer_agent_id": viewer,
                "round": round_number,
                "content_type": content_type,
                "content_id": content_id,
                "author_agent_id": author_id,
                "author_class": "source" if author_id is not None and author_id >= 20 else "investor",
                "is_self_authored": author_id == viewer,
                "first_seen_round": first_seen,
                "is_new_this_round": first_seen == round_number,
                "raw_exposure_count": len(items),
                "feed_visible_count": sum(item.get("exposure_type") == "feed_visible" for item in items),
                "direct_action_count": sum(item.get("exposure_type") == "direct_action" for item in items),
                "interacted": any(bool(item.get("interacted")) for item in items),
                "content_stance": stance,
                "stance_score": mean(scores),
                "stance_confidence": mean(confidences),
                "event_valence": normalized_stance(first.get("event_valence")),
                "stance_target": first.get("stance_target"),
                "stance_source": first.get("stance_source"),
                "stance_annotation_status": first.get("stance_annotation_status"),
                "stance_annotation_id": first.get("stance_annotation_id"),
            }
        )
    return rows


def exposure_features(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    all_rows = list(rows)
    social_rows = [
        row
        for row in all_rows
        if row.get("author_class") == "investor" and not row.get("is_self_authored")
    ]
    source_rows = [row for row in all_rows if row.get("author_class") == "source"]

    def categorical_features(
        prefix: str,
        selected: Sequence[Mapping[str, Any]],
        field: str,
    ) -> dict[str, Any]:
        unique_counts = Counter(normalized_stance(row.get(field)) for row in selected)
        raw_counts: Counter[str] = Counter()
        for row in selected:
            raw_counts[normalized_stance(row.get(field))] += int(
                row.get("raw_exposure_count", 0) or 0
            )
        unique_total = len(selected)
        raw_total = sum(raw_counts.values())
        values: dict[str, Any] = {}
        for label in STANCE_LABELS:
            values[f"{prefix}{label}_unique_count"] = unique_counts.get(label, 0)
            values[f"{prefix}{label}_unique_proportion"] = (
                unique_counts.get(label, 0) / unique_total if unique_total else 0.0
            )
            values[f"{prefix}{label}_raw_count"] = raw_counts.get(label, 0)
            values[f"{prefix}{label}_raw_proportion"] = (
                raw_counts.get(label, 0) / raw_total if raw_total else 0.0
            )
        return values

    def stance_features(prefix: str, selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        unique_counts = Counter(normalized_stance(row.get("content_stance")) for row in selected)
        raw_counts: Counter[str] = Counter()
        raw_scores: list[float] = []
        unique_scores: list[float] = []
        for row in selected:
            label = normalized_stance(row.get("content_stance"))
            weight = int(row.get("raw_exposure_count", 0) or 0)
            raw_counts[label] += weight
            score = safe_float(row.get("stance_score"))
            if score is not None:
                unique_scores.append(score)
                raw_scores.extend([score] * weight)
        unique_total = len(selected)
        raw_total = sum(raw_counts.values())
        values: dict[str, Any] = {
            f"{prefix}unique_content_count": unique_total,
            f"{prefix}raw_exposure_count": raw_total,
            f"{prefix}mean_stance_score_unique": mean(unique_scores),
            f"{prefix}stance_score_std_unique": population_std(unique_scores),
            f"{prefix}mean_stance_score_raw": mean(raw_scores),
        }
        for label in STANCE_LABELS:
            values[f"{prefix}{label}_unique_count"] = unique_counts.get(label, 0)
            values[f"{prefix}{label}_unique_proportion"] = (
                unique_counts.get(label, 0) / unique_total if unique_total else 0.0
            )
            values[f"{prefix}{label}_raw_count"] = raw_counts.get(label, 0)
            values[f"{prefix}{label}_raw_proportion"] = (
                raw_counts.get(label, 0) / raw_total if raw_total else 0.0
            )
        return values

    author_ids = {row.get("author_agent_id") for row in all_rows if row.get("author_agent_id") is not None}
    return {
        **stance_features("exposure_all_", all_rows),
        **stance_features("exposure_social_", social_rows),
        **stance_features("exposure_source_", source_rows),
        **categorical_features(
            "exposure_source_event_valence_", source_rows, "event_valence"
        ),
        "exposure_unique_author_count": len(author_ids),
        "exposure_unique_post_count": len({row.get("content_id") for row in all_rows if row.get("content_type") == "post"}),
        "exposure_unique_comment_count": len({row.get("content_id") for row in all_rows if row.get("content_type") == "comment"}),
        "exposure_new_unique_content_count": sum(bool(row.get("is_new_this_round")) for row in all_rows),
        "exposure_repeated_from_prior_round_count": sum(not bool(row.get("is_new_this_round")) for row in all_rows),
        "exposure_interacted_unique_content_count": sum(bool(row.get("interacted")) for row in all_rows),
        "exposure_feed_visible_raw_count": sum(int(row.get("feed_visible_count", 0) or 0) for row in all_rows),
        "exposure_direct_action_raw_count": sum(int(row.get("direct_action_count", 0) or 0) for row in all_rows),
        "exposure_self_authored_unique_content_count": sum(bool(row.get("is_self_authored")) for row in all_rows),
    }


def action_features(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(row.get("action_type", "unknown")) for row in rows)
    known = sum(counts.get(action_type, 0) for action_type in ACTION_TYPES)
    result: dict[str, Any] = {"action_count": len(rows)}
    result.update({f"action_{action_type}_count": counts.get(action_type, 0) for action_type in ACTION_TYPES})
    result["action_other_count"] = len(rows) - known
    return result


def assert_unique(rows: Sequence[Mapping[str, Any]], keys: Sequence[str], table: str) -> None:
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        key = tuple(row.get(name) for name in keys)
        if key in seen:
            raise ValueError(f"duplicate key in {table}: {key}")
        seen.add(key)


def assert_no_future_columns(table_fields: Mapping[str, Sequence[str]]) -> None:
    for table, fields in table_fields.items():
        leaked = {
            field
            for field in fields
            if field.lower() in FORBIDDEN_ANALYSIS_COLUMNS
            or field.lower().startswith("actual_")
            or field.lower().endswith("_correct")
        }
        if leaked:
            raise ValueError(f"future/evaluator columns leaked into {table}: {sorted(leaked)}")


def build_dataset(summary_dir: Path, finance_runs_dir: Path, output_dir: Path) -> dict[str, Any]:
    index_path = summary_dir / "scenario_run_index.csv"
    index_rows = sorted(read_csv(index_path), key=lambda row: row["scenario_id"])
    if not index_rows:
        raise ValueError(f"no runs found in {index_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    source_files: list[Path] = [index_path]
    scenario_runs: list[dict[str, Any]] = []
    canonical_profiles: dict[int, dict[str, Any]] = {}
    content_catalog: list[dict[str, Any]] = []
    exposure_rows: list[dict[str, Any]] = []
    round_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    group_target_rows: list[dict[str, Any]] = []
    run_contexts: list[dict[str, Any]] = []

    for index_row in index_rows:
        scenario_id = index_row["scenario_id"]
        run_id = index_row["run_id"]
        run_dir = finance_runs_dir / run_id
        required = {
            "manifest": run_dir / "manifest.json",
            "profiles": run_dir / "profiles.json",
            "snapshots": run_dir / "belief_snapshots.jsonl",
            "pre": run_dir / "pre_social_predictions.jsonl",
            "post": run_dir / "post_social_predictions.jsonl",
            "actions": run_dir / "social_actions_annotated.jsonl",
            "exposures": run_dir / "exposure_edges_annotated.jsonl",
            "annotations": run_dir / "stance_annotations.jsonl",
        }
        missing = [str(path) for path in required.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"{run_id} is missing required artifacts: {missing}")
        source_files.extend(required.values())

        manifest = read_json(required["manifest"])
        profiles = read_json(required["profiles"])
        snapshots = read_jsonl(required["snapshots"])
        pre_predictions = read_jsonl(required["pre"])
        post_predictions = read_jsonl(required["post"])
        actions = read_jsonl(required["actions"])
        annotations = read_jsonl(required["annotations"])
        deduped = deduplicate_exposures(read_jsonl(required["exposures"]), scenario_id)

        profile_map: dict[int, dict[str, Any]] = {}
        for profile in profiles:
            agent_id = safe_int(profile.get("user_id"))
            if agent_id is None or not 0 <= agent_id < 20:
                continue
            current = {"agent_id": agent_id, "agent_key": profile.get("agent_key"), **profile_fields(profile)}
            profile_map[agent_id] = current
            if agent_id in canonical_profiles:
                comparable = {key: current.get(key) for key in PROFILE_FIELDS}
                baseline = {key: canonical_profiles[agent_id].get(key) for key in PROFILE_FIELDS}
                if comparable != baseline:
                    raise ValueError(f"profile drift for Agent {agent_id} in {scenario_id}")
            else:
                canonical_profiles[agent_id] = current

        if sorted(profile_map) != list(range(20)):
            raise ValueError(f"{scenario_id} does not contain investor profiles 0..19")

        snapshot_map: dict[tuple[int, int], dict[str, Any]] = {}
        for row in snapshots:
            key = (int(row["agent_id"]), int(row["round"]))
            if key in snapshot_map:
                raise ValueError(f"duplicate belief snapshot in {scenario_id}: {key}")
            snapshot_map[key] = row
        pre_map = {int(row["agent_id"]): row for row in pre_predictions}
        post_map = {int(row["agent_id"]): row for row in post_predictions}
        action_map: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for action in actions:
            agent_id = safe_int(action.get("agent_id"))
            round_number = safe_int(action.get("round"))
            if action.get("agent_class") == "investor" and agent_id is not None and round_number is not None and round_number >= 1:
                action_map[(agent_id, round_number)].append(action)
        exposure_map: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for row in deduped:
            exposure_map[(int(row["viewer_agent_id"]), int(row["round"]))].append(row)
            exposure_rows.append({"run_id": run_id, **row})

        annotation_map: dict[tuple[str, str], dict[str, Any]] = {}
        for item in annotations:
            key = (str(item.get("content_type")), str(item.get("content_id")))
            if key in annotation_map:
                raise ValueError(f"duplicate content annotation in {scenario_id}: {key}")
            annotation_map[key] = item
            content_catalog.append(
                {
                    "run_id": run_id,
                    "scenario_id": scenario_id,
                    "content_type": item.get("content_type"),
                    "content_id": item.get("content_id"),
                    "content_hash": item.get("content_hash"),
                    "author_agent_id": item.get("author_agent_id"),
                    "author_class": item.get("author_class"),
                    "creation_round": item.get("round"),
                    "parent_content_id": item.get("parent_content_id"),
                    "content_text": item.get("content_text"),
                    "content_stance": normalized_stance(item.get("stance")),
                    "stance_score": safe_float(item.get("stance_score")),
                    "stance_confidence": safe_float(item.get("confidence")),
                    "stance_target": item.get("target"),
                    "event_valence": normalized_stance(item.get("event_valence")),
                    "supports_content_id": item.get("supports_content_id"),
                    "challenges_content_id": item.get("challenges_content_id"),
                    "annotation_reason": item.get("reason"),
                    "annotation_status": item.get("status"),
                    "annotator_model": item.get("annotator_model"),
                    "annotation_prompt_version": item.get("prompt_version"),
                    "baseline_stance": item.get("baseline_stance"),
                    "baseline_stance_score": item.get("baseline_stance_score"),
                    "baseline_stance_source": item.get("baseline_stance_source"),
                }
            )

        metadata = {
            "run_id": run_id,
            "replicate_id": manifest.get("replicate_id"),
            "scenario_id": scenario_id,
            "random_seed": manifest.get("random_seed"),
            "agent_set_version": manifest.get("agent_set_version"),
            "sampling_method": manifest.get("sampling_method"),
            "data_split": manifest.get("data_split"),
            "input_snapshot_hash": manifest.get("input_snapshot_hash"),
            "prompt_version": manifest.get("prompt_version"),
            "prompt_hash": manifest.get("prompt_hash"),
            "social_rounds": manifest.get("social_rounds"),
            "source_mode": manifest.get("source_mode"),
            "graph_id": manifest.get("graph_id"),
        }
        scenario_runs.append(
            {
                **metadata,
                "investor_agent_count": manifest.get("investor_agent_count"),
                "source_agent_count": manifest.get("source_agent_count"),
                "belief_snapshot_count": len(snapshots),
                "raw_exposure_edge_count": sum(int(row.get("raw_exposure_count", 0)) for row in deduped),
                "deduplicated_exposure_count": len(deduped),
                "investor_action_count": sum(action.get("agent_class") == "investor" for action in actions),
                "annotated_content_count": len(annotations),
            }
        )

        # Group targets cover each private round interview and the separate
        # final post-social interview.  No market outcome enters these rows.
        round_group_maps: dict[int, dict[str, Any]] = {}
        for round_number in range(0, int(manifest.get("social_rounds", 4)) + 1):
            members = [snapshot_map.get((agent_id, round_number), {}) for agent_id in range(20)]
            metrics = group_metrics(members)
            round_group_maps[round_number] = metrics
            group_target_rows.append(
                {
                    **metadata,
                    "measurement_stage": f"round_{round_number}",
                    "round": round_number,
                    **metrics,
                }
            )
        final_group = group_metrics([post_map.get(agent_id, {}) for agent_id in range(20)])
        group_target_rows.append(
            {
                **metadata,
                "measurement_stage": "post_social_final",
                "round": None,
                **final_group,
            }
        )

        per_run_round_rows: dict[tuple[int, int], dict[str, Any]] = {}
        for agent_id in range(20):
            profile = profile_map[agent_id]
            for round_number in range(1, int(manifest.get("social_rounds", 4)) + 1):
                previous = snapshot_map.get((agent_id, round_number - 1))
                current = snapshot_map.get((agent_id, round_number))
                previous_vector = probability_vector(previous)
                current_vector = probability_vector(current)
                current_members = [snapshot_map.get((other, round_number), {}) for other in range(20)]
                loo_members = [snapshot_map.get((other, round_number), {}) for other in range(20) if other != agent_id]
                full_metrics = group_metrics(current_members)
                loo_metrics = group_metrics(loo_members)
                full_vector = [
                    full_metrics["mean_up_probability"],
                    full_metrics["mean_neutral_probability"],
                    full_metrics["mean_down_probability"],
                ]
                loo_vector = [
                    loo_metrics["mean_up_probability"],
                    loo_metrics["mean_neutral_probability"],
                    loo_metrics["mean_down_probability"],
                ]
                row = {
                    **metadata,
                    "agent_id": agent_id,
                    **{key: profile.get(key) for key in PROFILE_FIELDS},
                    "round": round_number,
                    **prediction_fields("previous_", previous),
                    **prediction_fields("current_", current),
                    "belief_js_divergence": js_divergence(previous_vector, current_vector),
                    "direction_changed": (
                        previous.get("direction") != current.get("direction")
                        if previous and current and probability_vector(previous) and probability_vector(current)
                        else None
                    ),
                    "up_probability_delta": (
                        current_vector[0] - previous_vector[0] if previous_vector and current_vector else None
                    ),
                    "neutral_probability_delta": (
                        current_vector[1] - previous_vector[1] if previous_vector and current_vector else None
                    ),
                    "down_probability_delta": (
                        current_vector[2] - previous_vector[2] if previous_vector and current_vector else None
                    ),
                    "expected_return_delta": (
                        safe_float(current.get("expected_return")) - safe_float(previous.get("expected_return"))
                        if current and previous and safe_float(current.get("expected_return")) is not None and safe_float(previous.get("expected_return")) is not None
                        else None
                    ),
                    "confidence_delta": (
                        safe_float(current.get("confidence")) - safe_float(previous.get("confidence"))
                        if current and previous and safe_float(current.get("confidence")) is not None and safe_float(previous.get("confidence")) is not None
                        else None
                    ),
                    **action_features(action_map.get((agent_id, round_number), [])),
                    **exposure_features(exposure_map.get((agent_id, round_number), [])),
                    **prefixed("full_group_", full_metrics),
                    **prefixed("loo_group_", loo_metrics),
                    "agent_vs_loo_group_js": js_divergence(current_vector, loo_vector),
                    "removed_agent_vs_full_group_js": js_divergence(loo_vector, full_vector),
                }
                per_run_round_rows[(agent_id, round_number)] = row
                round_rows.append(row)

        # Outbound exposure is an influence opportunity, not a causal effect.
        authored_content = Counter(
            safe_int(item.get("author_agent_id"))
            for item in annotations
            if safe_int(item.get("author_agent_id")) is not None and 0 <= safe_int(item.get("author_agent_id")) < 20
        )
        outbound_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
        audience_pairs: dict[int, set[tuple[int, int]]] = defaultdict(set)
        for edge in deduped:
            author_id = safe_int(edge.get("author_agent_id"))
            viewer_id = int(edge["viewer_agent_id"])
            if author_id is None or not 0 <= author_id < 20 or author_id == viewer_id:
                continue
            outbound_rows[author_id].append(edge)
            audience_pairs[author_id].add((viewer_id, int(edge["round"])))

        for agent_id in range(20):
            profile = profile_map[agent_id]
            pre = pre_map.get(agent_id)
            post = post_map.get(agent_id)
            pre_vector = probability_vector(pre)
            post_vector = probability_vector(post)
            loo_final = group_metrics([post_map.get(other, {}) for other in range(20) if other != agent_id])
            loo_final_vector = [
                loo_final["mean_up_probability"],
                loo_final["mean_neutral_probability"],
                loo_final["mean_down_probability"],
            ]
            full_final_vector = [
                final_group["mean_up_probability"],
                final_group["mean_neutral_probability"],
                final_group["mean_down_probability"],
            ]
            agent_rounds = [per_run_round_rows[(agent_id, round_number)] for round_number in range(1, int(manifest.get("social_rounds", 4)) + 1)]
            outbound = outbound_rows.get(agent_id, [])
            audience_changes = [
                per_run_round_rows[pair]
                for pair in audience_pairs.get(agent_id, set())
                if pair in per_run_round_rows
            ]
            all_authors = {
                exposure.get("author_agent_id")
                for round_row in agent_rounds
                for exposure in exposure_map.get((agent_id, int(round_row["round"])), [])
                if exposure.get("author_agent_id") is not None
            }
            scenario_rows.append(
                {
                    **metadata,
                    "agent_id": agent_id,
                    **{key: profile.get(key) for key in PROFILE_FIELDS},
                    **prediction_fields("pre_", pre),
                    **prediction_fields("post_", post),
                    "pre_post_js_divergence": js_divergence(pre_vector, post_vector),
                    "pre_post_direction_changed": (
                        pre.get("direction") != post.get("direction")
                        if pre and post and pre_vector and post_vector
                        else None
                    ),
                    "pre_post_expected_return_delta": (
                        safe_float(post.get("expected_return")) - safe_float(pre.get("expected_return"))
                        if pre and post and safe_float(post.get("expected_return")) is not None and safe_float(pre.get("expected_return")) is not None
                        else None
                    ),
                    "pre_post_confidence_delta": (
                        safe_float(post.get("confidence")) - safe_float(pre.get("confidence"))
                        if pre and post and safe_float(post.get("confidence")) is not None and safe_float(pre.get("confidence")) is not None
                        else None
                    ),
                    "round_transition_mean_js": mean(safe_float(row.get("belief_js_divergence")) for row in agent_rounds),
                    "round_transition_max_js": max(
                        (safe_float(row.get("belief_js_divergence"), 0.0) or 0.0 for row in agent_rounds),
                        default=0.0,
                    ),
                    "round_direction_flip_count": sum(row.get("direction_changed") is True for row in agent_rounds),
                    "total_action_count": sum(int(row.get("action_count", 0)) for row in agent_rounds),
                    "total_raw_exposure_count": sum(int(row.get("exposure_all_raw_exposure_count", 0)) for row in agent_rounds),
                    "total_agent_round_unique_content_count": sum(int(row.get("exposure_all_unique_content_count", 0)) for row in agent_rounds),
                    "total_social_unique_content_count": sum(int(row.get("exposure_social_unique_content_count", 0)) for row in agent_rounds),
                    "total_source_unique_content_count": sum(int(row.get("exposure_source_unique_content_count", 0)) for row in agent_rounds),
                    "inbound_unique_author_count": len(all_authors),
                    "authored_content_count": authored_content.get(agent_id, 0),
                    "outbound_deduplicated_exposure_count": len(outbound),
                    "outbound_raw_exposure_count": sum(int(row.get("raw_exposure_count", 0)) for row in outbound),
                    "outbound_unique_viewer_count": len({int(row["viewer_agent_id"]) for row in outbound}),
                    "outbound_unique_viewer_round_count": len(audience_pairs.get(agent_id, set())),
                    "audience_transition_mean_js": mean(safe_float(row.get("belief_js_divergence")) for row in audience_changes),
                    "audience_direction_flip_rate": (
                        sum(row.get("direction_changed") is True for row in audience_changes) / len(audience_changes)
                        if audience_changes
                        else None
                    ),
                    **prefixed("full_group_post_", final_group),
                    **prefixed("loo_group_post_", loo_final),
                    "agent_post_vs_loo_group_js": js_divergence(post_vector, loo_final_vector),
                    "removed_agent_vs_full_group_post_js": js_divergence(loo_final_vector, full_final_vector),
                }
            )

        run_contexts.append(
            {
                "scenario_id": scenario_id,
                "run_id": run_id,
                "snapshot_keys": set(snapshot_map),
                "post_valid": sum(probability_vector(post_map.get(agent_id)) is not None for agent_id in range(20)),
                "deduped_exposure_count": len(deduped),
                "raw_exposure_count": sum(int(row.get("raw_exposure_count", 0)) for row in deduped),
                "annotation_count": len(annotations),
            }
        )

    profile_rows = [canonical_profiles[agent_id] for agent_id in sorted(canonical_profiles)]
    mi_feature_fields = [
        "run_id",
        "replicate_id",
        "scenario_id",
        "random_seed",
        "agent_id",
        *PROFILE_FIELDS,
        "round",
        "previous_direction",
        "previous_up_probability",
        "previous_neutral_probability",
        "previous_down_probability",
        "previous_expected_return",
        "previous_confidence",
        "current_direction",
        "current_up_probability",
        "current_neutral_probability",
        "current_down_probability",
        "current_expected_return",
        "current_confidence",
        "belief_js_divergence",
        "direction_changed",
        "up_probability_delta",
        "neutral_probability_delta",
        "down_probability_delta",
        "expected_return_delta",
        "confidence_delta",
        "exposure_social_unique_content_count",
        "exposure_social_raw_exposure_count",
        "exposure_social_mean_stance_score_unique",
        "exposure_social_stance_score_std_unique",
        "exposure_social_positive_unique_proportion",
        "exposure_social_mixed_unique_proportion",
        "exposure_social_negative_unique_proportion",
        "exposure_social_neutral_unique_proportion",
        "exposure_source_unique_content_count",
        "exposure_source_raw_exposure_count",
        "exposure_source_mean_stance_score_unique",
        "exposure_source_event_valence_positive_unique_proportion",
        "exposure_source_event_valence_mixed_unique_proportion",
        "exposure_source_event_valence_negative_unique_proportion",
        "exposure_source_event_valence_neutral_unique_proportion",
        "exposure_unique_author_count",
        "exposure_new_unique_content_count",
        "exposure_repeated_from_prior_round_count",
        "exposure_interacted_unique_content_count",
        "action_count",
        "action_create_post_count",
        "action_create_comment_count",
        "action_like_post_count",
        "action_like_comment_count",
        "action_dislike_post_count",
        "action_dislike_comment_count",
        "action_refresh_count",
        "full_group_mean_up_probability",
        "full_group_mean_neutral_probability",
        "full_group_mean_down_probability",
        "full_group_mean_expected_return",
        "full_group_consensus_rate",
        "full_group_direction_entropy_bits",
        "loo_group_mean_up_probability",
        "loo_group_mean_neutral_probability",
        "loo_group_mean_down_probability",
        "loo_group_mean_expected_return",
        "agent_vs_loo_group_js",
        "removed_agent_vs_full_group_js",
    ]
    mi_rows = [
        {field: row.get(field) for field in mi_feature_fields}
        for row in round_rows
    ]
    tables: dict[str, list[dict[str, Any]]] = {
        "scenario_runs.csv": scenario_runs,
        "agent_profiles.csv": profile_rows,
        "content_stance_catalog.csv": content_catalog,
        "agent_round_content_exposures.csv": exposure_rows,
        "agent_round_mi_features.csv": mi_rows,
        "agent_round_features.csv": round_rows,
        "agent_scenario_features.csv": scenario_rows,
        "group_targets.csv": group_target_rows,
    }

    assert_unique(scenario_runs, ("scenario_id",), "scenario_runs")
    assert_unique(profile_rows, ("agent_id",), "agent_profiles")
    assert_unique(content_catalog, ("scenario_id", "content_type", "content_id"), "content_stance_catalog")
    assert_unique(
        exposure_rows,
        ("scenario_id", "viewer_agent_id", "round", "content_type", "content_id"),
        "agent_round_content_exposures",
    )
    assert_unique(round_rows, ("scenario_id", "agent_id", "round"), "agent_round_features")
    assert_unique(mi_rows, ("scenario_id", "agent_id", "round"), "agent_round_mi_features")
    assert_unique(scenario_rows, ("scenario_id", "agent_id"), "agent_scenario_features")
    assert_unique(group_target_rows, ("scenario_id", "measurement_stage"), "group_targets")

    table_fields: dict[str, list[str]] = {}
    for filename, rows in tables.items():
        table_fields[filename] = write_csv(output_dir / filename, rows)
    assert_no_future_columns(table_fields)

    probability_violations = 0
    for row in round_rows:
        for prefix in ("previous_", "current_", "full_group_", "loo_group_"):
            values = [safe_float(row.get(f"{prefix}{name}_probability")) for name in ("up", "neutral", "down")]
            if all(value is not None for value in values) and abs(sum(values) - 1.0) > 1e-6:
                probability_violations += 1

    stance_counts = Counter(row["content_stance"] for row in content_catalog)
    quality_report = {
        "dataset_version": DATASET_VERSION,
        "source_summary_dir": str(summary_dir.resolve()),
        "scenario_count": len(scenario_runs),
        "agent_count": len(profile_rows),
        "social_round_count": 4,
        "row_counts": {filename: len(rows) for filename, rows in tables.items()},
        "expected_row_counts": {
            "scenario_runs.csv": 18,
            "agent_profiles.csv": 20,
            "agent_round_features.csv": 18 * 20 * 4,
            "agent_round_mi_features.csv": 18 * 20 * 4,
            "agent_scenario_features.csv": 18 * 20,
            "group_targets.csv": 18 * 6,
        },
        "valid_belief_snapshot_count": sum(len(context["snapshot_keys"]) for context in run_contexts),
        "expected_belief_snapshot_count": 18 * 20 * 5,
        "valid_final_post_prediction_count": sum(context["post_valid"] for context in run_contexts),
        "expected_final_post_prediction_count": 18 * 20,
        "raw_exposure_count": sum(context["raw_exposure_count"] for context in run_contexts),
        "deduplicated_agent_round_content_exposure_count": len(exposure_rows),
        "exposure_deduplication_key": [
            "scenario_id",
            "viewer_agent_id",
            "round",
            "content_type",
            "content_id",
        ],
        "stance_distribution": dict(sorted(stance_counts.items())),
        "probability_sum_violation_count": probability_violations,
        "future_outcome_columns_included": False,
        "uniqueness_checks_passed": True,
        "cross_table_checks": {
            "all_round_rows_have_profiles": all(int(row["agent_id"]) in canonical_profiles for row in round_rows),
            "all_exposure_content_has_catalog_entry": all(
                any(
                    catalog["scenario_id"] == row["scenario_id"]
                    and str(catalog["content_type"]) == str(row["content_type"])
                    and str(catalog["content_id"]) == str(row["content_id"])
                    for catalog in content_catalog
                )
                for row in exposure_rows
            ),
        },
        "research_warnings": [
            "The 18 scenario runs share one random seed; rows within a scenario are not independent samples.",
            "The stance annotator uses the same model family as the forecasting Agents; manually audit or independently relabel a stratified sample before confirmatory inference.",
            "Offline row deletion cannot reproduce the changed social dynamics of a real reduced-Agent rerun.",
            "Use future market outcomes only after the Agent subset and hyperparameters have been frozen.",
        ],
    }
    write_json(output_dir / "quality_report.json", quality_report)

    schema = {
        "dataset_version": DATASET_VERSION,
        "future_outcomes_included": False,
        "tables": {
            filename: {
                "row_count": len(tables[filename]),
                "columns": table_fields[filename],
                "primary_key": {
                    "scenario_runs.csv": ["scenario_id"],
                    "agent_profiles.csv": ["agent_id"],
                    "content_stance_catalog.csv": ["scenario_id", "content_type", "content_id"],
                    "agent_round_content_exposures.csv": ["scenario_id", "viewer_agent_id", "round", "content_type", "content_id"],
                    "agent_round_mi_features.csv": ["scenario_id", "agent_id", "round"],
                    "agent_round_features.csv": ["scenario_id", "agent_id", "round"],
                    "agent_scenario_features.csv": ["scenario_id", "agent_id"],
                    "group_targets.csv": ["scenario_id", "measurement_stage"],
                }[filename],
            }
            for filename in tables
        },
        "semantic_groups": {
            "belief_transition": "previous_* is the snapshot before a round; current_* is the snapshot after that round.",
            "exposure_all": "All visible content, including source, self-authored, and other-investor content.",
            "exposure_social": "Only content authored by another investor Agent.",
            "exposure_source": "Only content authored by a graph-derived source account.",
            "raw_vs_unique": "raw counts preserve repeated visibility/direct-action records; unique counts use Agent-round-content deduplication.",
            "loo_group": "The full group aggregate after excluding the row's focal Agent.",
            "outbound": "How widely content authored by the focal Agent was exposed to other investors.",
        },
    }
    write_json(output_dir / "schema.json", schema)

    source_manifest = {
        "dataset_version": DATASET_VERSION,
        "source_files": [
            {"path": str(path.resolve()), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for path in sorted(set(source_files))
        ],
    }
    write_json(output_dir / "source_manifest.json", source_manifest)

    readme = f"""# S1 四轮 Agent 降采样分析数据集

## 用途

该数据集由 18 个匿名 A 股场景、20 个投资者 Agent、4 轮 Reddit 社会互动的完整 S1 运行日志构建。它用于回答：能否用更少的 Agent 保留完整 20 Agent 群体的预测和社会行为。

该目录不包含未来真实涨跌、未来收益、预测是否正确等 evaluator 字段。降采样名单、Agent 数量和算法参数冻结后，才能在外部 evaluator 中连接真实结果。

## 推荐入口

1. `agent_round_mi_features.csv`：推荐的互信息直接输入表，主键为 `scenario_id + agent_id + round`，共 {len(mi_rows)} 行，只保留最常用的曝光、信念变化、动作和群体字段。
2. `agent_round_features.csv`：逐轮完整宽表，共 {len(round_rows)} 行，适合扩展分析和字段审计。
3. `agent_scenario_features.csv`：每个场景每个 Agent 一行，共 {len(scenario_rows)} 行。用于 mRMR、Agent 排序和离线子集选择。
4. `group_targets.csv`：完整群体在 round 0-4 和最终 post-social 的聚合目标。降采样保真度应与这里比较。
5. `agent_round_content_exposures.csv`：按“场景-Agent-轮次-内容”去重的曝光关系。`raw_exposure_count` 保留重复刷到同一内容的次数。
6. `content_stance_catalog.csv`：{len(content_catalog)} 条唯一帖子/评论及离线 LLM 立场标注。
7. `agent_profiles.csv`：20 个固定 Agent 的精简画像，不包含会随场景变化的 persona 长文本。
8. `scenario_runs.csv`：场景、run_id、随机种子、Prompt 哈希和数据量。

## 最重要的字段口径

- `belief_js_divergence`：本轮前后 Agent 三分类概率的 JS divergence。
- `exposure_social_*`：只统计其他投资者发布的内容，是研究社会影响时的首选字段。
- `exposure_source_*`：只统计图谱信息主体发布的内容，应与其他投资者观点分开控制。
- `exposure_source_event_valence_*`：信息源所发布事件本身的利好/利空属性，不等同于信息源作者持有投资立场。
- `*_unique_*`：同一 Agent 在同一轮看到同一内容多次只算一次。
- `*_raw_*`：保留刷新和直接互动造成的重复曝光强度。
- `full_group_*`：完整 20 Agent 的群体结果。
- `loo_group_*`：排除当前 Agent 后的 19 Agent 群体结果，适合减少自包含偏差。
- `removed_agent_vs_full_group_js`：移除当前 Agent 后群体平均概率相对完整群体的变化。
- `audience_transition_mean_js`：看过该 Agent 内容的受众在相应轮次中的平均信念变化，只能解释为关联，不能直接称为因果影响。

## 建议用法

```text
曝光 E_t：读取 agent_round_features.csv 中第 t 轮 exposure_social_* 字段
信念变化 ΔB_t：读取同一行 belief_js_divergence、概率变化和 expected_return_delta
完整群体 G：读取 group_targets.csv 或 full_group_* 字段
候选 Agent A_i：读取 agent_scenario_features.csv，并在训练场景内聚合
```

正式选择时应按场景划分训练/验证，不能把同一场景的 20 Agent 或 4 轮随机拆到两边。不要直接使用全 18 场景的全局平均值选择 Agent 后，再把同一批场景称为独立测试集。

建议依次比较随机、角色分层、中心性、mRMR 和角色约束互信息方法，并在 `K=4/6/8/10/12/15` 下计算精简群体与完整群体的概率 JS、方向一致率、预期收益差、熵差和互动分布差。

## 研究限制

- 当前 18 个场景只有一个随机种子，1800 条快照并不等于 1800 个独立样本；统计推断必须按场景聚类。
- 内容标注模型与预测 Agent 属于同一模型家族，正式结论前需要人工抽查或独立模型复标。
- 离线删除日志中的 Agent 只能用于筛选候选方案；真正减少 Agent 后会改变帖子和曝光网络，因此候选方案必须重新运行 S1。
- `mixed` 是占比最高的内容立场，分析时应同时使用 `stance_score`，但该分数只有有限的离散取值，不应假装成高精度连续测量。

## 复现

从工作区根目录运行：

```powershell
MiroFish/backend/.venv/Scripts/python.exe MiroFish/backend/scripts/build_downsampling_dataset.py
```

`source_manifest.json` 保存全部输入文件的 SHA-256；`quality_report.json` 保存行数、唯一键、概率和与跨表连接检查；`schema.json` 供程序或 Agent 读取字段结构。
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    # Add generated-file hashes after every output has been finalized.
    generated = [path for path in output_dir.iterdir() if path.is_file()]
    quality_report["generated_files"] = [
        {"name": path.name, "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        for path in sorted(generated)
        if path.name != "quality_report.json"
    ]
    write_json(output_dir / "quality_report.json", quality_report)
    return quality_report


def main() -> None:
    args = parse_args()
    report = build_dataset(
        summary_dir=args.summary_dir.resolve(),
        finance_runs_dir=args.finance_runs_dir.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "output_dir": str(args.output_dir.resolve()),
                "row_counts": report["row_counts"],
                "future_outcome_columns_included": report["future_outcome_columns_included"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
