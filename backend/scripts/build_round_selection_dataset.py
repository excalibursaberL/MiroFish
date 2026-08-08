"""Build and analyze an 18-scenario, K=10 S1 round experiment dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sqlite3
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence


DIRECTIONS = ("up", "neutral", "down")
SELECTED_ROUND = 6


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    root = workspace_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-dir",
        type=Path,
        default=root
        / "MiroFish"
        / "backend"
        / "uploads"
        / "finance"
        / "s1_batch_b07b857111d1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="destination; defaults to Dataset/s1_round_selection_<rounds>rounds_k10_seed<seed>_v2",
    )
    return parser.parse_args()


def dataset_version(manifest: Mapping[str, Any]) -> str:
    rounds = int(manifest["social_rounds"])
    seed = int(manifest["random_seed"])
    return f"s1_round_selection_{rounds}rounds_k10_seed{seed}_v2"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected object at {path}:{number}")
        rows.append(value)
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def fieldnames(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                result.append(key)
    return result


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = fieldnames(rows)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        if not fields:
            return
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_value(row.get(key)) for key in fields})


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def average(values: Iterable[Any]) -> float | None:
    valid = [safe_float(value) for value in values]
    numbers = [value for value in valid if value is not None]
    return mean(numbers) if numbers else None


def probability_vector(row: Mapping[str, Any]) -> tuple[float, float, float] | None:
    values = tuple(safe_float(row.get(f"{name}_probability")) for name in DIRECTIONS)
    if any(value is None for value in values):
        return None
    vector = tuple(float(value) for value in values if value is not None)
    total = sum(vector)
    if total <= 0:
        return None
    return tuple(value / total for value in vector)  # type: ignore[return-value]


def js_divergence(left: Sequence[float], right: Sequence[float]) -> float:
    middle = tuple((a + b) / 2 for a, b in zip(left, right))

    def kl(values: Sequence[float], target: Sequence[float]) -> float:
        return sum(
            value * math.log2(value / reference)
            for value, reference in zip(values, target)
            if value > 0 and reference > 0
        )

    return 0.5 * kl(left, middle) + 0.5 * kl(right, middle)


def entropy_bits(values: Iterable[str]) -> float:
    counts = Counter(values)
    total = sum(counts.values())
    return -sum(
        (count / total) * math.log2(count / total)
        for count in counts.values()
        if count
    ) if total else 0.0


def brier_score(row: Mapping[str, Any], actual: str) -> float | None:
    vector = probability_vector(row)
    if vector is None or actual not in DIRECTIONS:
        return None
    return sum(
        (probability - float(direction == actual)) ** 2
        for direction, probability in zip(DIRECTIONS, vector)
    )


def log_loss(row: Mapping[str, Any], actual: str) -> float | None:
    vector = probability_vector(row)
    if vector is None or actual not in DIRECTIONS:
        return None
    return -math.log(max(vector[DIRECTIONS.index(actual)], 1e-12))


def balanced_accuracy(actual: Sequence[str], predicted: Sequence[str]) -> float | None:
    recalls: list[float] = []
    for label in DIRECTIONS:
        indices = [index for index, value in enumerate(actual) if value == label]
        if indices:
            recalls.append(sum(predicted[index] == label for index in indices) / len(indices))
    return average(recalls)


def macro_f1(actual: Sequence[str], predicted: Sequence[str]) -> float | None:
    scores: list[float] = []
    for label in DIRECTIONS:
        tp = sum(a == label and p == label for a, p in zip(actual, predicted))
        fp = sum(a != label and p == label for a, p in zip(actual, predicted))
        fn = sum(a == label and p != label for a, p in zip(actual, predicted))
        denominator = 2 * tp + fp + fn
        scores.append((2 * tp / denominator) if denominator else 0.0)
    return average(scores)


def rank_values(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda pair: pair[1])
    result = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        rank = (start + 1 + end) / 2
        for index in range(start, end):
            result[ordered[index][0]] = rank
        start = end
    return result


def pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 3 or len(left) != len(right):
        return None
    left_mean = mean(left)
    right_mean = mean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_var = sum((value - left_mean) ** 2 for value in left)
    right_var = sum((value - right_mean) ** 2 for value in right)
    denominator = math.sqrt(left_var * right_var)
    return numerator / denominator if denominator else None


def spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    return pearson(rank_values(left), rank_values(right))


def with_provenance(
    row: Mapping[str, Any], scenario_id: str, run_id: str
) -> dict[str, Any]:
    return {"scenario_id": scenario_id, "run_id": run_id, **dict(row)}


def majority_direction(rows: Sequence[Mapping[str, Any]]) -> str | None:
    counts = Counter(str(row.get("direction")) for row in rows)
    if not counts:
        return None
    top = max(counts.values())
    leaders = [label for label, count in counts.items() if count == top]
    return leaders[0] if len(leaders) == 1 else None


def categorical_score(label: Any) -> float | None:
    return {
        "positive": 1.0,
        "negative": -1.0,
        "mixed": 0.0,
        "neutral": 0.0,
    }.get(str(label))


def load_reddit_relations(
    db_path: Path,
) -> tuple[dict[int, int], dict[int, int], dict[int, int]]:
    """Load authoritative post/comment ownership and comment-parent links."""
    post_owners: dict[int, int] = {}
    comment_owners: dict[int, int] = {}
    comment_posts: dict[int, int] = {}
    if not db_path.exists():
        return post_owners, comment_owners, comment_posts
    with sqlite3.connect(db_path) as connection:
        table_names = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        user_to_agent: dict[int, int] = {}
        if "user" in table_names:
            for user_id, agent_id in connection.execute(
                "SELECT user_id, agent_id FROM user"
            ).fetchall():
                if user_id is not None:
                    user_to_agent[int(user_id)] = int(
                        agent_id if agent_id is not None else user_id
                    )
        if "post" in table_names:
            for post_id, user_id in connection.execute(
                "SELECT post_id, user_id FROM post"
            ).fetchall():
                if post_id is not None and user_id is not None:
                    post_owners[int(post_id)] = user_to_agent.get(
                        int(user_id), int(user_id)
                    )
        if "comment" in table_names:
            for comment_id, post_id, user_id in connection.execute(
                "SELECT comment_id, post_id, user_id FROM comment"
            ).fetchall():
                if comment_id is None:
                    continue
                normalized_comment_id = int(comment_id)
                if post_id is not None:
                    comment_posts[normalized_comment_id] = int(post_id)
                if user_id is not None:
                    comment_owners[normalized_comment_id] = user_to_agent.get(
                        int(user_id), int(user_id)
                    )
    return post_owners, comment_owners, comment_posts


def mapping_value(mapping: Mapping[int, int], value: Any) -> int | None:
    try:
        return mapping.get(int(value))
    except (TypeError, ValueError):
        return None


def enrich_action_targets(
    actions: Sequence[Mapping[str, Any]],
    post_owners: Mapping[int, int],
    comment_owners: Mapping[int, int],
    comment_posts: Mapping[int, int],
) -> list[dict[str, Any]]:
    """Recover targets omitted by OASIS trace rows from the Reddit tables."""
    enriched: list[dict[str, Any]] = []
    for source in actions:
        action = dict(source)
        action_type = str(action.get("action_type", "")).lower()
        if action_type == "create_comment" and action.get("target_post_id") is None:
            parent_post_id = mapping_value(comment_posts, action.get("comment_id"))
            if parent_post_id is not None:
                action["target_post_id"] = parent_post_id
        if action.get("target_agent_id") is None:
            target_agent_id = mapping_value(
                comment_owners, action.get("target_comment_id")
            )
            if target_agent_id is None:
                target_agent_id = mapping_value(
                    post_owners, action.get("target_post_id")
                )
            if target_agent_id is not None:
                action["target_agent_id"] = target_agent_id
        enriched.append(action)
    return enriched


def restore_direct_content_exposures(
    actions: Sequence[Mapping[str, Any]],
    exposures: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Add auditable direct rows that legacy trace-only exports could not form."""
    result = [dict(row) for row in exposures]
    existing = {
        (
            str(row.get("trace_id")),
            str(row.get("content_type")),
            str(row.get("content_id")),
        )
        for row in result
        if row.get("exposure_type") == "direct_action"
    }
    for action in actions:
        action_type = str(action.get("action_type", "")).lower()
        if action_type in {"like_post", "dislike_post"}:
            content_type, content_id = "post", action.get("target_post_id")
        elif action_type in {"like_comment", "dislike_comment"}:
            content_type, content_id = "comment", action.get("target_comment_id")
        elif action_type == "create_comment":
            if action.get("target_comment_id") is not None:
                content_type, content_id = "comment", action.get("target_comment_id")
            else:
                content_type, content_id = "post", action.get("target_post_id")
        else:
            continue
        if content_id is None or action.get("agent_class") != "investor":
            continue
        key = (str(action.get("trace_id")), content_type, str(content_id))
        if key in existing:
            continue
        try:
            viewer = int(action["agent_id"])
            round_number = int(action.get("round", 0) or 0)
        except (KeyError, TypeError, ValueError):
            continue
        comparable = [
            row
            for row in result
            if int(row.get("viewer_agent_id", -1)) == viewer
            and str(row.get("content_type")) == content_type
            and str(row.get("content_id")) == str(content_id)
        ]
        first_seen_round = min(
            [
                int(row.get("first_seen_round", row.get("round", round_number)))
                for row in comparable
            ]
            + [round_number]
        )
        target_agent_id = action.get("target_agent_id")
        try:
            normalized_target = int(target_agent_id)
        except (TypeError, ValueError):
            normalized_target = None
        result.append(
            {
                "scenario_id": action.get("scenario_id"),
                "run_id": action.get("run_id"),
                "exposure_id": (
                    f"{action.get('trace_id')}:{content_type}:{content_id}"
                ),
                "trace_id": action.get("trace_id"),
                "round": round_number,
                "timestamp": action.get("timestamp"),
                "viewer_agent_id": viewer,
                "content_type": content_type,
                "content_id": content_id,
                "author_agent_id": normalized_target,
                "content_text": "",
                "content_stance": "unknown",
                "stance_score": None,
                "stance_source": "database_relation_recovery",
                "edge_layer": "direct_interaction",
                "exposure_type": "direct_action",
                "action_type": action_type,
                "interacted": True,
                "interaction_sign": {
                    "like_post": 1,
                    "like_comment": 1,
                    "dislike_post": -1,
                    "dislike_comment": -1,
                    "create_comment": 0,
                }[action_type],
                "interaction_target_id": content_id,
                "first_seen_round": first_seen_round,
                "is_first_exposure": first_seen_round == round_number,
                "is_self_authored": normalized_target == viewer,
            }
        )
        existing.add(key)
    return result


def derive_interaction_edges(
    actions: Sequence[Mapping[str, Any]],
    exposures: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Recover typed direct relations from preserved S1 audit artifacts."""
    supported = {
        "like_post": ("reaction", 1, "post", "target_post_id"),
        "dislike_post": ("reaction", -1, "post", "target_post_id"),
        "like_comment": ("reaction", 1, "comment", "target_comment_id"),
        "dislike_comment": ("reaction", -1, "comment", "target_comment_id"),
        "create_comment": ("comment", 0, None, None),
        "follow": ("agent_relation", 1, "agent", None),
        "mute": ("agent_relation", -1, "agent", None),
    }
    target_by_trace: dict[int, int] = {}
    for edge in exposures:
        if edge.get("exposure_type") != "direct_action":
            continue
        try:
            target_by_trace[int(edge["trace_id"])] = int(edge["author_agent_id"])
        except (KeyError, TypeError, ValueError):
            continue

    result: list[dict[str, Any]] = []
    for action in actions:
        action_type = str(action.get("action_type", "")).lower()
        if action_type not in supported or action.get("agent_class") != "investor":
            continue
        try:
            actor_id = int(action["agent_id"])
            round_number = int(action.get("round", 0) or 0)
        except (KeyError, TypeError, ValueError):
            continue
        if not 0 <= actor_id < 10 or round_number < 1:
            continue
        kind, sign, content_type, content_key = supported[action_type]
        args = action.get("action_args") or {}
        target_id = action.get("target_agent_id")
        if target_id is None:
            try:
                target_id = target_by_trace.get(int(action.get("trace_id")))
            except (TypeError, ValueError):
                target_id = None
        if target_id is None and action_type == "follow":
            target_id = args.get("follow_id")
        if target_id is None and action_type == "mute":
            target_id = args.get("mute_id")

        content_id = action.get(content_key) if content_key else None
        if action_type == "create_comment":
            if action.get("target_comment_id") is not None:
                content_type = "comment"
                content_id = action.get("target_comment_id")
            elif action.get("target_post_id") is not None:
                content_type = "post"
                content_id = action.get("target_post_id")
            else:
                content_type = None
        try:
            normalized_target_id = int(target_id)
        except (TypeError, ValueError):
            continue
        trace_id = action.get("trace_id")
        result.append(
            {
                "scenario_id": action.get("scenario_id"),
                "run_id": action.get("run_id"),
                "interaction_id": (
                    f"{trace_id}:{action_type}:{normalized_target_id}:"
                    f"{content_type or 'none'}:{content_id or 'none'}"
                ),
                "trace_id": trace_id,
                "round": round_number,
                "timestamp": action.get("timestamp"),
                "actor_agent_id": actor_id,
                "actor_class": "investor",
                "target_agent_id": normalized_target_id,
                "target_class": (
                    "investor" if normalized_target_id < 10 else "source"
                ),
                "action_type": action_type,
                "interaction_kind": kind,
                "interaction_sign": sign,
                "content_type": content_type,
                "content_id": content_id,
            }
        )
    return result


def external_social_exposures(
    rows: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    return [
        row for row in rows
        if row.get("author_class") == "investor"
        and not bool(row.get("is_self_authored"))
    ]


def analyze_and_build(batch_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_json(batch_dir / "manifest.json")
    if manifest.get("status") != "completed":
        raise ValueError("batch must be completed")
    max_round = int(manifest.get("social_rounds", 0))
    if max_round < SELECTED_ROUND:
        raise ValueError(
            f"this dataset builder requires at least {SELECTED_ROUND} rounds"
        )
    version = dataset_version(manifest)
    if not manifest.get("stance_annotation"):
        raise ValueError("run batch stance annotation before building the dataset")

    finance_dir = batch_dir.parent
    all_snapshots: list[dict[str, Any]] = []
    all_actions: list[dict[str, Any]] = []
    all_exposures: list[dict[str, Any]] = []
    all_interactions: list[dict[str, Any]] = []
    all_states: list[dict[str, Any]] = []
    all_annotations: list[dict[str, Any]] = []
    all_tokens: list[dict[str, Any]] = []
    all_profiles: list[dict[str, Any]] = []
    all_source_profiles: list[dict[str, Any]] = []
    scenario_inputs: list[dict[str, Any]] = []
    scenario_runs: list[dict[str, Any]] = []
    evaluation_rows: list[dict[str, Any]] = []
    source_files: list[dict[str, Any]] = []
    actual_direction: dict[str, str] = {}
    actual_return: dict[str, float] = {}

    source_names = (
        "manifest.json",
        "profiles.json",
        "source_profiles.json",
        "scenarios.jsonl",
        "history_memory.jsonl",
        "current_event.json",
        "simulation_config.json",
        "random_seed_state.json",
        "belief_snapshots.jsonl",
        "social_actions.jsonl",
        "social_actions_annotated.jsonl",
        "exposure_edges.jsonl",
        "exposure_edges_annotated.jsonl",
        "interaction_edges.jsonl",
        "agent_round_states.jsonl",
        "stance_annotations.jsonl",
        "llm_token_usage.jsonl",
        "agent_token_usage.csv",
        "evaluation.csv",
        "round_metrics.csv",
    )
    for run in manifest["runs"]:
        if run.get("status") != "completed":
            continue
        scenario_id = str(run["scenario_id"])
        run_id = str(run["run_id"])
        run_dir = finance_dir / run_id
        run_manifest = read_json(run_dir / "manifest.json")
        annotations = read_jsonl(run_dir / "stance_annotations.jsonl")
        if not annotations or any(row.get("status") != "ok" for row in annotations):
            raise ValueError(f"incomplete stance annotations: {run_id}")

        snapshots = [
            with_provenance(row, scenario_id, run_id)
            for row in read_jsonl(run_dir / "belief_snapshots.jsonl")
        ]
        actions = [
            with_provenance(row, scenario_id, run_id)
            for row in read_jsonl(run_dir / "social_actions_annotated.jsonl")
        ]
        exposures = [
            with_provenance(row, scenario_id, run_id)
            for row in read_jsonl(run_dir / "exposure_edges_annotated.jsonl")
        ]
        simulation_id = str(run_manifest.get("simulation_id", ""))
        reddit_db = (
            workspace_root()
            / "MiroFish"
            / "backend"
            / "uploads"
            / "simulations"
            / simulation_id
            / "reddit_simulation.db"
        )
        post_owners, comment_owners, comment_posts = load_reddit_relations(
            reddit_db
        )
        actions = enrich_action_targets(
            actions, post_owners, comment_owners, comment_posts
        )
        exposures = restore_direct_content_exposures(actions, exposures)
        interactions = [
            with_provenance(row, scenario_id, run_id)
            for row in read_jsonl(run_dir / "interaction_edges.jsonl")
        ]
        if not interactions:
            interactions = derive_interaction_edges(actions, exposures)
        states = [
            with_provenance(row, scenario_id, run_id)
            for row in read_jsonl(run_dir / "agent_round_states.jsonl")
        ]
        tokens = [
            with_provenance(row, scenario_id, run_id)
            for row in read_jsonl(run_dir / "llm_token_usage.jsonl")
        ]
        profiles = read_json(run_dir / "profiles.json")
        source_profiles = read_json(run_dir / "source_profiles.json")
        all_snapshots.extend(snapshots)
        all_actions.extend(actions)
        all_exposures.extend(exposures)
        all_interactions.extend(interactions)
        all_states.extend(states)
        all_annotations.extend(
            with_provenance(row, scenario_id, run_id) for row in annotations
        )
        all_tokens.extend(tokens)
        all_profiles.extend(
            with_provenance(row, scenario_id, run_id) for row in profiles
        )
        all_source_profiles.extend(
            with_provenance(row, scenario_id, run_id) for row in source_profiles
        )

        current_event = read_json(run_dir / "current_event.json")
        history_memory = read_jsonl(run_dir / "history_memory.jsonl")
        scenario_inputs.append(
            {
                "scenario_id": scenario_id,
                "run_id": run_id,
                "current_event": current_event,
                "history_memory": history_memory,
            }
        )
        evaluations = read_csv(run_dir / "evaluation.csv")
        if not evaluations:
            raise ValueError(f"missing evaluation rows: {run_id}")
        first_evaluation = evaluations[0]
        actual_direction[scenario_id] = first_evaluation["actual_five_day_close_direction"]
        actual_return[scenario_id] = float(first_evaluation["actual_five_day_close_return"])
        evaluation_rows.extend(
            {"scenario_id": scenario_id, "run_id": run_id, **row}
            for row in evaluations
        )
        scenario_runs.append(
            {
                "scenario_id": scenario_id,
                "run_id": run_id,
                "project_id": run.get("project_id"),
                "graph_id": run.get("graph_id"),
                "random_seed": run_manifest.get("random_seed"),
                "social_rounds": run_manifest.get("social_rounds"),
                "agent_count": len(profiles),
                "snapshot_count": len(snapshots),
                "valid_snapshot_count": sum(row.get("status") == "ok" for row in snapshots),
                "content_count": len(annotations),
                "action_count": len(actions),
                "exposure_edge_count": len(exposures),
                "actual_five_day_close_direction": actual_direction[scenario_id],
                "actual_five_day_close_return": actual_return[scenario_id],
                "prompt_version": run_manifest.get("prompt_version"),
                "input_snapshot_hash": run_manifest.get("input_snapshot_hash"),
            }
        )
        for name in source_names:
            path = run_dir / name
            if path.exists():
                source_files.append(
                    {
                        "scenario_id": scenario_id,
                        "run_id": run_id,
                        "path": str(path.relative_to(workspace_root())).replace("\\", "/"),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
        if reddit_db.exists():
            source_files.append(
                {
                    "scenario_id": scenario_id,
                    "run_id": run_id,
                    "path": str(reddit_db.relative_to(workspace_root())).replace(
                        "\\", "/"
                    ),
                    "bytes": reddit_db.stat().st_size,
                    "sha256": sha256_file(reddit_db),
                    "purpose": "recover authoritative comment-to-post relations",
                }
            )

    snapshots_by_key = {
        (row["scenario_id"], int(row["round"]), int(row["agent_id"])): row
        for row in all_snapshots
    }
    annotation_by_key = {
        (
            row["scenario_id"],
            str(row.get("content_type")),
            str(row.get("content_id")),
        ): row
        for row in all_annotations
    }

    # Keep feed opportunities and explicit interactions as separate layers.
    # The compatibility raw artifact contains both record types.
    exposure_groups: dict[tuple[str, int, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in all_exposures:
        if row.get("content_type") not in {"post", "comment"}:
            continue
        key = (
            row["scenario_id"],
            int(row["round"]),
            int(row["viewer_agent_id"]),
            str(row["content_type"]),
            str(row["content_id"]),
        )
        exposure_groups[key].append(row)
    unique_exposures: list[dict[str, Any]] = []
    for key, rows in sorted(exposure_groups.items()):
        scenario_id, round_number, viewer, content_type, content_id = key
        first = rows[0]
        annotation = annotation_by_key.get((scenario_id, content_type, content_id), {})
        feed_rows = [row for row in rows if row.get("exposure_type") == "feed_visible"]
        direct_rows = [row for row in rows if row.get("exposure_type") == "direct_action"]
        author_agent_id = first.get("author_agent_id")
        try:
            normalized_author_id = int(author_agent_id)
        except (TypeError, ValueError):
            normalized_author_id = None
        author_class = annotation.get("author_class")
        if author_class not in {"investor", "source"}:
            author_class = (
                "source"
                if normalized_author_id is not None and normalized_author_id >= 10
                else "investor"
            )
        first_seen_round = min(
            int(row.get("first_seen_round", round_number) or round_number)
            for row in rows
        )
        interaction_types = sorted({
            str(row.get("action_type", "")).lower()
            for row in direct_rows
            if row.get("action_type")
        })
        unique_exposures.append(
            {
                "scenario_id": scenario_id,
                "run_id": first["run_id"],
                "round": round_number,
                "viewer_agent_id": viewer,
                "content_type": content_type,
                "content_id": content_id,
                "author_agent_id": normalized_author_id,
                "author_class": author_class,
                "is_self_authored": normalized_author_id == viewer,
                "content_hash": annotation.get("content_hash"),
                "content_stance": annotation.get("stance", first.get("content_stance")),
                "stance_score": annotation.get("stance_score", first.get("stance_score")),
                "stance_confidence": annotation.get("confidence"),
                "stance_target": annotation.get("target"),
                "event_valence": annotation.get("event_valence"),
                "event_valence_score": categorical_score(annotation.get("event_valence")),
                # Legacy total retained for audit only.  Analysis must use the
                # separated feed/direct fields below.
                "raw_exposure_count": len(rows),
                "feed_impression_count": len(feed_rows),
                "direct_interaction_count": len(direct_rows),
                "interacted_any": bool(direct_rows),
                "interaction_action_types": interaction_types,
                "positive_interaction_count": sum(
                    str(row.get("action_type", "")).lower()
                    in {"like_post", "like_comment", "follow"}
                    for row in direct_rows
                ),
                "negative_interaction_count": sum(
                    str(row.get("action_type", "")).lower()
                    in {"dislike_post", "dislike_comment", "mute"}
                    for row in direct_rows
                ),
                "comment_interaction_count": sum(
                    str(row.get("action_type", "")).lower() == "create_comment"
                    for row in direct_rows
                ),
                "first_seen_round": first_seen_round,
                "is_first_exposure": first_seen_round == round_number,
            }
        )

    actions_by_key: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in all_actions:
        try:
            agent_id = int(row["agent_id"])
        except (KeyError, TypeError, ValueError):
            continue
        if agent_id < 10 and int(row.get("round", 0) or 0) > 0:
            actions_by_key[(row["scenario_id"], int(row["round"]), agent_id)].append(row)
    exposures_by_key: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in unique_exposures:
        exposures_by_key[
            (row["scenario_id"], int(row["round"]), int(row["viewer_agent_id"]))
        ].append(row)

    panel_rows: list[dict[str, Any]] = []
    for scenario_id in sorted(actual_direction):
        for round_number in range(1, max_round + 1):
            for agent_id in range(10):
                before = snapshots_by_key.get((scenario_id, round_number - 1, agent_id), {})
                after = snapshots_by_key.get((scenario_id, round_number, agent_id), {})
                before_vector = probability_vector(before)
                after_vector = probability_vector(after)
                actions = actions_by_key.get((scenario_id, round_number, agent_id), [])
                exposures = exposures_by_key.get((scenario_id, round_number, agent_id), [])
                social = external_social_exposures(exposures)
                self_authored = [row for row in exposures if row.get("is_self_authored")]
                social_new = [row for row in social if row.get("is_first_exposure")]
                social_repeated = [row for row in social if not row.get("is_first_exposure")]
                social_interacted = [
                    row for row in social
                    if int(row.get("direct_interaction_count", 0) or 0) > 0
                ]
                sources = [row for row in exposures if row.get("author_class") == "source"]
                social_scores = [
                    safe_float(row.get("stance_score")) for row in social
                ]
                social_scores = [value for value in social_scores if value is not None]
                social_new_scores = [
                    safe_float(row.get("stance_score")) for row in social_new
                ]
                social_new_scores = [
                    value for value in social_new_scores if value is not None
                ]
                social_interacted_scores = [
                    safe_float(row.get("stance_score")) for row in social_interacted
                ]
                social_interacted_scores = [
                    value for value in social_interacted_scores if value is not None
                ]
                source_scores = [
                    safe_float(row.get("event_valence_score")) for row in sources
                ]
                source_scores = [value for value in source_scores if value is not None]
                panel_rows.append(
                    {
                        "scenario_id": scenario_id,
                        "run_id": str(after.get("run_id") or before.get("run_id") or ""),
                        "round": round_number,
                        "agent_id": agent_id,
                        "full_population_agent_id": after.get("full_population_agent_id", before.get("full_population_agent_id")),
                        "agent_role": after.get("agent_role", before.get("agent_role")),
                        "agent_role_category": after.get("agent_role_category", before.get("agent_role_category")),
                        "agent_role_label": after.get("agent_role_label", before.get("agent_role_label")),
                        "random_seed": manifest.get("random_seed"),
                        "prompt_version": after.get("prompt_version", before.get("prompt_version")),
                        "before_status": before.get("status"),
                        "after_status": after.get("status"),
                        "before_direction": before.get("direction"),
                        "after_direction": after.get("direction"),
                        "before_expected_return": safe_float(before.get("expected_return")),
                        "after_expected_return": safe_float(after.get("expected_return")),
                        "expected_return_delta": (
                            safe_float(after.get("expected_return")) - safe_float(before.get("expected_return"))
                            if safe_float(after.get("expected_return")) is not None
                            and safe_float(before.get("expected_return")) is not None
                            else None
                        ),
                        "direction_flip": (
                            before.get("direction") != after.get("direction")
                            if before.get("status") == after.get("status") == "ok"
                            else None
                        ),
                        "belief_js_divergence": (
                            js_divergence(before_vector, after_vector)
                            if before_vector and after_vector
                            else None
                        ),
                        "after_up_probability": after.get("up_probability"),
                        "after_neutral_probability": after.get("neutral_probability"),
                        "after_down_probability": after.get("down_probability"),
                        "after_confidence": after.get("confidence"),
                        "action_count": len(actions),
                        "authored_content_count": sum(
                            str(row.get("action_type")).lower() in {"create_post", "create_comment"}
                            for row in actions
                        ),
                        "like_count": sum("like" in str(row.get("action_type")).lower() for row in actions),
                        "refresh_count": sum(str(row.get("action_type")).lower() == "refresh" for row in actions),
                        "exposure_raw_count": sum(int(row["raw_exposure_count"]) for row in exposures),
                        "exposure_feed_impression_count": sum(
                            int(row["feed_impression_count"]) for row in exposures
                        ),
                        "exposure_direct_interaction_count": sum(
                            int(row["direct_interaction_count"]) for row in exposures
                        ),
                        "exposure_unique_count": len(exposures),
                        "exposure_self_authored_unique_count": len(self_authored),
                        "exposure_social_unique_count": len(social),
                        "exposure_social_new_unique_count": len(social_new),
                        "exposure_social_repeated_unique_count": len(social_repeated),
                        "exposure_social_interacted_unique_count": len(social_interacted),
                        "exposure_source_unique_count": len(sources),
                        "exposure_social_mean_stance_score": average(social_scores),
                        "exposure_social_new_mean_stance_score": average(social_new_scores),
                        "exposure_social_interacted_mean_stance_score": average(
                            social_interacted_scores
                        ),
                        "exposure_social_positive_share": average(row.get("content_stance") == "positive" for row in social),
                        "exposure_social_negative_share": average(row.get("content_stance") == "negative" for row in social),
                        "exposure_social_mixed_share": average(row.get("content_stance") == "mixed" for row in social),
                        "exposure_source_mean_event_valence_score": average(source_scores),
                        "actual_five_day_close_direction": actual_direction[scenario_id],
                        "actual_five_day_close_return": actual_return[scenario_id],
                        "after_direction_correct": after.get("direction") == actual_direction[scenario_id] if after.get("status") == "ok" else None,
                        "after_return_absolute_error": abs(float(after["expected_return"]) - actual_return[scenario_id]) if safe_float(after.get("expected_return")) is not None else None,
                    }
                )

    round_rows: list[dict[str, Any]] = []
    scenario_round_rows: list[dict[str, Any]] = []
    for round_number in range(max_round + 1):
        rows = [
            row for row in all_snapshots
            if int(row["round"]) == round_number and row.get("status") == "ok"
        ]
        actual = [actual_direction[row["scenario_id"]] for row in rows]
        predicted = [str(row["direction"]) for row in rows]
        majority_actual: list[str] = []
        majority_predicted: list[str] = []
        correct_scenarios = 0
        ties = 0
        for scenario_id in sorted(actual_direction):
            scenario_rows = [row for row in rows if row["scenario_id"] == scenario_id]
            vectors = [probability_vector(row) for row in scenario_rows]
            vectors = [value for value in vectors if value]
            direction = majority_direction(scenario_rows)
            if direction is None:
                ties += 1
            else:
                majority_actual.append(actual_direction[scenario_id])
                majority_predicted.append(direction)
                correct_scenarios += int(direction == actual_direction[scenario_id])
            counts = Counter(str(row["direction"]) for row in scenario_rows)
            scenario_round_rows.append(
                {
                    "scenario_id": scenario_id,
                    "round": round_number,
                    "valid_agent_count": len(scenario_rows),
                    "majority_direction": direction or "tie",
                    "majority_correct": direction == actual_direction[scenario_id] if direction else False,
                    "actual_direction": actual_direction[scenario_id],
                    "actual_return": actual_return[scenario_id],
                    "individual_accuracy": average(row["direction"] == actual_direction[scenario_id] for row in scenario_rows),
                    "consensus_rate": max(counts.values()) / len(scenario_rows) if scenario_rows else None,
                    "direction_entropy_bits": entropy_bits(predicted for predicted in counts.elements()),
                    "mean_pairwise_js": average(
                        js_divergence(left, right) for left, right in combinations(vectors, 2)
                    ),
                    "mean_expected_return": average(row.get("expected_return") for row in scenario_rows),
                    "mean_brier_score": average(brier_score(row, actual_direction[scenario_id]) for row in scenario_rows),
                    "mean_log_loss": average(log_loss(row, actual_direction[scenario_id]) for row in scenario_rows),
                    "mean_return_absolute_error": average(
                        abs(float(row["expected_return"]) - actual_return[scenario_id])
                        for row in scenario_rows
                        if safe_float(row.get("expected_return")) is not None
                    ),
                }
            )
        transition = [row for row in panel_rows if int(row["round"]) == round_number] if round_number else []
        round_rows.append(
            {
                "round": round_number,
                "valid_snapshot_count": len(rows),
                "individual_direction_accuracy": average(a == p for a, p in zip(actual, predicted)),
                "individual_balanced_accuracy": balanced_accuracy(actual, predicted),
                "individual_macro_f1": macro_f1(actual, predicted),
                "mean_brier_score": average(brier_score(row, actual_direction[row["scenario_id"]]) for row in rows),
                "mean_log_loss": average(log_loss(row, actual_direction[row["scenario_id"]]) for row in rows),
                "mean_return_absolute_error": average(
                    abs(float(row["expected_return"]) - actual_return[row["scenario_id"]])
                    for row in rows
                    if safe_float(row.get("expected_return")) is not None
                ),
                "majority_correct_scenarios": correct_scenarios,
                "majority_tie_scenarios": ties,
                "majority_accuracy_all_scenarios": correct_scenarios / len(actual_direction),
                "majority_balanced_accuracy_defined": balanced_accuracy(majority_actual, majority_predicted),
                "mean_scenario_consensus_rate": average(row["consensus_rate"] for row in scenario_round_rows if int(row["round"]) == round_number),
                "mean_scenario_direction_entropy_bits": average(row["direction_entropy_bits"] for row in scenario_round_rows if int(row["round"]) == round_number),
                "mean_scenario_pairwise_js": average(row["mean_pairwise_js"] for row in scenario_round_rows if int(row["round"]) == round_number),
                "transition_direction_flip_rate": average(row["direction_flip"] for row in transition),
                "transition_mean_js": average(row["belief_js_divergence"] for row in transition),
                "transition_mean_abs_return_delta": average(abs(float(row["expected_return_delta"])) for row in transition if row["expected_return_delta"] is not None),
            }
        )

    # Preserve provider-token and content-count fields computed by the original analyzer.
    original_rounds = {
        int(row["round"]): row for row in read_csv(batch_dir / "round_selection_metrics.csv")
    }
    for row in round_rows:
        source = original_rounds.get(int(row["round"]), {})
        for name in (
            "investor_action_count",
            "new_content_count",
            "expressing_agent_scenario_count",
            "round_total_tokens",
            "cumulative_total_tokens",
            "cumulative_direction_flip_rate",
            "cumulative_mean_js",
        ):
            row[name] = safe_float(source.get(name))

    stance_round_rows: list[dict[str, Any]] = []
    for round_number in range(max_round + 1):
        contents = [row for row in all_annotations if int(row.get("round", 0) or 0) == round_number]
        round_exposures = [row for row in unique_exposures if int(row["round"]) == round_number]
        social_exposures = external_social_exposures(round_exposures)
        self_exposures = [row for row in round_exposures if row.get("is_self_authored")]
        source_exposures = [row for row in round_exposures if row.get("author_class") == "source"]
        stance_round_rows.append(
            {
                "round": round_number,
                "new_unique_content_count": len(contents),
                "investor_content_count": sum(row.get("author_class") == "investor" for row in contents),
                "source_content_count": sum(row.get("author_class") == "source" for row in contents),
                "content_positive_count": sum(row.get("stance") == "positive" for row in contents),
                "content_mixed_count": sum(row.get("stance") == "mixed" for row in contents),
                "content_negative_count": sum(row.get("stance") == "negative" for row in contents),
                "content_neutral_count": sum(row.get("stance") == "neutral" for row in contents),
                "mean_content_stance_score": average(row.get("stance_score") for row in contents),
                "unique_exposure_count": len(round_exposures),
                "social_unique_exposure_count": len(social_exposures),
                "self_authored_unique_exposure_count": len(self_exposures),
                "source_unique_exposure_count": len(source_exposures),
                "raw_exposure_count": sum(int(row["raw_exposure_count"]) for row in round_exposures),
                "feed_impression_count": sum(
                    int(row["feed_impression_count"]) for row in round_exposures
                ),
                "direct_interaction_count": sum(
                    int(row["direct_interaction_count"]) for row in round_exposures
                ),
                "mean_social_exposure_stance_score": average(row.get("stance_score") for row in social_exposures),
                "mean_source_event_valence_score": average(row.get("event_valence_score") for row in source_exposures),
            }
        )

    profile_rows: list[dict[str, Any]] = []
    role_lookup: dict[int, dict[str, Any]] = {}
    for row in all_snapshots:
        if row.get("status") == "ok":
            role_lookup.setdefault(int(row["agent_id"]), row)
    for agent_id in range(10):
        agent_snapshots = [row for row in all_snapshots if int(row["agent_id"]) == agent_id and row.get("status") == "ok"]
        agent_panel = [row for row in panel_rows if int(row["agent_id"]) == agent_id]
        agent_actions = [row for row in all_actions if row.get("agent_class") == "investor" and int(row.get("agent_id", -1)) == agent_id]
        agent_tokens = [row for row in all_tokens if row.get("agent_class") == "investor" and int(row.get("agent_id", -1)) == agent_id]
        role = role_lookup.get(agent_id, {})
        result: dict[str, Any] = {
            "agent_id": agent_id,
            "full_population_agent_id": role.get("full_population_agent_id"),
            "agent_role": role.get("agent_role"),
            "agent_role_category": role.get("agent_role_category"),
            "agent_role_label": role.get("agent_role_label"),
            "authored_content_count": sum(str(row.get("action_type")).lower() in {"create_post", "create_comment"} for row in agent_actions),
            "total_action_count": len(agent_actions),
            "total_provider_tokens": sum(int(row.get("total_tokens", 0) or 0) for row in agent_tokens if row.get("usage_available")),
            "mean_transition_js": average(row["belief_js_divergence"] for row in agent_panel),
            "direction_flip_rate": average(row["direction_flip"] for row in agent_panel),
            "mean_unique_exposures_per_round": average(row["exposure_unique_count"] for row in agent_panel),
        }
        for round_number in sorted({0, SELECTED_ROUND, max_round}):
            rows = [row for row in agent_snapshots if int(row["round"]) == round_number]
            result[f"round_{round_number}_accuracy"] = average(row["direction"] == actual_direction[row["scenario_id"]] for row in rows)
            result[f"round_{round_number}_mean_brier"] = average(brier_score(row, actual_direction[row["scenario_id"]]) for row in rows)
        profile_rows.append(result)

    association_specs = (
        ("social_stance_vs_return_delta", "exposure_social_mean_stance_score", "expected_return_delta"),
        ("social_stance_vs_after_return", "exposure_social_mean_stance_score", "after_expected_return"),
        ("unique_exposure_count_vs_belief_js", "exposure_unique_count", "belief_js_divergence"),
        ("social_exposure_count_vs_belief_js", "exposure_social_unique_count", "belief_js_divergence"),
    )
    association_rows: list[dict[str, Any]] = []
    for name, left_name, right_name in association_specs:
        pairs = [
            (safe_float(row[left_name]), safe_float(row[right_name]))
            for row in panel_rows
        ]
        pairs = [(left, right) for left, right in pairs if left is not None and right is not None]
        left = [float(pair[0]) for pair in pairs]
        right = [float(pair[1]) for pair in pairs]
        association_rows.append(
            {
                "association": name,
                "x_field": left_name,
                "y_field": right_name,
                "sample_count": len(pairs),
                "pearson_correlation": pearson(left, right),
                "spearman_correlation": spearman(left, right),
                "interpretation": "descriptive association only; repeated observations are clustered within scenarios and agents",
            }
        )

    # Center stance and belief changes inside each scenario-round to reduce common-event confounding.
    centered_left: list[float] = []
    centered_right: list[float] = []
    panel_groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in panel_rows:
        panel_groups[(row["scenario_id"], int(row["round"]))].append(row)
    for rows in panel_groups.values():
        valid = [
            row for row in rows
            if safe_float(row["exposure_social_mean_stance_score"]) is not None
            and safe_float(row["expected_return_delta"]) is not None
        ]
        if len(valid) < 2:
            continue
        stance_mean = mean(float(row["exposure_social_mean_stance_score"]) for row in valid)
        delta_mean = mean(float(row["expected_return_delta"]) for row in valid)
        centered_left.extend(float(row["exposure_social_mean_stance_score"]) - stance_mean for row in valid)
        centered_right.extend(float(row["expected_return_delta"]) - delta_mean for row in valid)
    association_rows.append(
        {
            "association": "within_scenario_round_social_stance_vs_return_delta",
            "x_field": "centered exposure_social_mean_stance_score",
            "y_field": "centered expected_return_delta",
            "sample_count": len(centered_left),
            "pearson_correlation": pearson(centered_left, centered_right),
            "spearman_correlation": spearman(centered_left, centered_right),
            "interpretation": "descriptive within-scenario-round association; not causal attribution",
        }
    )

    selected = round_rows[SELECTED_ROUND]
    baseline = round_rows[0]
    final = round_rows[max_round]
    actual_counts = Counter(actual_direction.values())
    constant_majority_accuracy = max(actual_counts.values()) / len(actual_direction)
    stance_counts = Counter(str(row.get("stance")) for row in all_annotations)
    summary = {
        "dataset_version": version,
        "source_batch_id": manifest["batch_id"],
        "design": {
            "scenario_count": len(scenario_runs),
            "agent_count_per_scenario": 10,
            "rounds": max_round,
            "random_seed": manifest.get("random_seed"),
            "selected_provisional_round": SELECTED_ROUND,
        },
        "quality": {
            "snapshot_count": len(all_snapshots),
            "valid_snapshot_count": sum(row.get("status") == "ok" for row in all_snapshots),
            "stance_annotation_count": len(all_annotations),
            "successful_stance_annotation_count": sum(row.get("status") == "ok" for row in all_annotations),
            "unique_exposure_count": len(unique_exposures),
            "raw_exposure_count": len(all_exposures),
            "interaction_edge_count": len(all_interactions),
            "self_authored_unique_exposure_count": sum(
                bool(row.get("is_self_authored")) for row in unique_exposures
            ),
        },
        "actual_direction_counts": dict(actual_counts),
        "constant_actual_majority_accuracy": constant_majority_accuracy,
        "content_stance_counts": dict(stance_counts),
        "round_0": baseline,
        "round_6": selected,
        "round_final": final,
        "associations": association_rows,
        "interpretation": {
            "round_6": "provisional short-horizon interaction baseline, not a convergence claim",
            "causality": "exposure-belief statistics are associations because exposure is endogenous and only one seed was run",
            "forecasting": "accuracy is descriptive and must be compared with the constant-majority baseline",
        },
    }

    # Analysis-ready tables.
    write_csv(output_dir / "scenario_runs.csv", scenario_runs)
    write_csv(output_dir / "belief_snapshots.csv", all_snapshots)
    write_csv(output_dir / "evaluation_snapshots.csv", [
        {
            **row,
            "actual_five_day_close_direction": actual_direction[row["scenario_id"]],
            "actual_five_day_close_return": actual_return[row["scenario_id"]],
            "direction_correct": row.get("direction") == actual_direction[row["scenario_id"]] if row.get("status") == "ok" else None,
            "brier_score": brier_score(row, actual_direction[row["scenario_id"]]),
            "log_loss": log_loss(row, actual_direction[row["scenario_id"]]),
            "return_absolute_error": abs(float(row["expected_return"]) - actual_return[row["scenario_id"]]) if safe_float(row.get("expected_return")) is not None else None,
        }
        for row in all_snapshots
    ])
    write_csv(output_dir / "agent_round_panel.csv", panel_rows)
    write_csv(output_dir / "agent_round_content_exposures.csv", unique_exposures)
    write_csv(output_dir / "interaction_edges.csv", all_interactions)
    write_csv(output_dir / "content_stance_catalog.csv", all_annotations)
    write_csv(output_dir / "round_metrics.csv", round_rows)
    write_csv(output_dir / "scenario_round_metrics.csv", scenario_round_rows)
    write_csv(output_dir / "stance_round_metrics.csv", stance_round_rows)
    write_csv(output_dir / "agent_role_metrics.csv", profile_rows)
    write_csv(output_dir / "exposure_belief_associations.csv", association_rows)
    write_csv(output_dir / "agent_token_usage.csv", [
        {"scenario_id": run["scenario_id"], "run_id": run["run_id"], **row}
        for run in manifest["runs"] if run.get("status") == "completed"
        for row in read_csv(finance_dir / run["run_id"] / "agent_token_usage.csv")
    ])

    # Auditable JSONL keeps nested fields and raw model responses intact.
    write_jsonl(output_dir / "belief_snapshots.jsonl", all_snapshots)
    write_jsonl(output_dir / "social_actions_annotated.jsonl", all_actions)
    write_jsonl(output_dir / "exposure_edges_annotated.jsonl", all_exposures)
    write_jsonl(output_dir / "interaction_edges.jsonl", all_interactions)
    write_jsonl(output_dir / "agent_round_states.jsonl", all_states)
    write_jsonl(output_dir / "stance_annotations.jsonl", all_annotations)
    write_jsonl(output_dir / "llm_token_usage.jsonl", all_tokens)
    write_jsonl(output_dir / "profiles.jsonl", all_profiles)
    write_jsonl(output_dir / "source_profiles.jsonl", all_source_profiles)
    write_jsonl(output_dir / "scenario_inputs.jsonl", scenario_inputs)

    for name in (
        "round_selection_metrics.csv",
        "round_selection_summary.json",
        "round_selection_report.md",
        "stance_annotation_scenarios.csv",
        "stance_annotation_summary.json",
    ):
        source = batch_dir / name
        if source.exists():
            shutil.copy2(source, output_dir / name)

    write_json(output_dir / "analysis_summary.json", summary)
    write_json(output_dir / "source_manifest.json", {
        "dataset_version": version,
        "source_batch": str(batch_dir.relative_to(workspace_root())).replace("\\", "/"),
        "source_batch_manifest_sha256": sha256_file(batch_dir / "manifest.json"),
        "files": source_files,
    })

    probability_errors = [
        row for row in all_snapshots
        if row.get("status") == "ok" and probability_vector(row) is None
    ]
    snapshot_keys = [
        (row["scenario_id"], int(row["round"]), int(row["agent_id"]))
        for row in all_snapshots
    ]
    expected_snapshot_keys = {
        (scenario_id, round_number, agent_id)
        for scenario_id in actual_direction
        for round_number in range(max_round + 1)
        for agent_id in range(10)
    }
    post_matches = 0
    post_count = 0
    for run in manifest["runs"]:
        if run.get("status") != "completed":
            continue
        scenario_id = str(run["scenario_id"])
        posts = read_jsonl(finance_dir / run["run_id"] / "post_social_predictions.jsonl")
        for post in posts:
            post_count += 1
            snapshot = snapshots_by_key.get(
                (scenario_id, max_round, int(post["agent_id"]))
            )
            if snapshot and all(post.get(name) == snapshot.get(name) for name in ("direction", "expected_return")):
                post_matches += 1
    expected_annotation_count = int(
        manifest.get("stance_annotation", {}).get("content_count", 0)
    )
    quality = {
        "passed": True,
        "scenario_count": len(scenario_runs),
        "random_seeds": sorted({row["random_seed"] for row in scenario_runs}),
        "expected_snapshot_count": len(expected_snapshot_keys),
        "snapshot_count": len(all_snapshots),
        "unique_snapshot_key_count": len(set(snapshot_keys)),
        "missing_snapshot_keys": [list(key) for key in sorted(expected_snapshot_keys - set(snapshot_keys))],
        "valid_snapshot_count": sum(row.get("status") == "ok" for row in all_snapshots),
        "invalid_snapshot_records": [
            {"scenario_id": row["scenario_id"], "round": row["round"], "agent_id": row["agent_id"], "error": row.get("error")}
            for row in all_snapshots if row.get("status") != "ok"
        ],
        "valid_probability_error_count": len(probability_errors),
        "expected_stance_annotation_count": expected_annotation_count,
        "stance_annotation_count": len(all_annotations),
        "stance_annotation_failure_count": sum(row.get("status") != "ok" for row in all_annotations),
        "raw_exposure_count": len(all_exposures),
        "unique_agent_round_content_exposure_count": len(unique_exposures),
        "feed_impression_count": sum(
            int(row.get("feed_impression_count", 0) or 0)
            for row in unique_exposures
        ),
        "direct_content_interaction_count": sum(
            int(row.get("direct_interaction_count", 0) or 0)
            for row in unique_exposures
        ),
        "interaction_edge_count": len(all_interactions),
        "self_authored_unique_exposure_count": sum(
            bool(row.get("is_self_authored")) for row in unique_exposures
        ),
        "exposure_layer_count_mismatch": sum(
            int(row.get("raw_exposure_count", 0) or 0)
            != int(row.get("feed_impression_count", 0) or 0)
            + int(row.get("direct_interaction_count", 0) or 0)
            for row in unique_exposures
        ),
        "social_self_exposure_leak_count": sum(
            int(row.get("exposure_social_unique_count", 0) or 0)
            != sum(
                candidate.get("author_class") == "investor"
                and not candidate.get("is_self_authored")
                for candidate in exposures_by_key.get(
                    (
                        row["scenario_id"],
                        int(row["round"]),
                        int(row["agent_id"]),
                    ),
                    [],
                )
            )
            for row in panel_rows
        ),
        "annotated_exposure_count": sum(row.get("stance_source") == "offline_llm" for row in all_exposures),
        "post_social_prediction_count": post_count,
        "post_social_source_round": max_round,
        "post_social_matching_final_round_count": post_matches,
        "extra_post_social_llm_calls": sum(row.get("phase") == "post_social_prediction" for row in all_tokens),
    }
    quality["passed"] = all((
        quality["scenario_count"] == 18,
        quality["random_seeds"] == [int(manifest["random_seed"])],
        quality["snapshot_count"] == len(expected_snapshot_keys),
        quality["unique_snapshot_key_count"] == len(expected_snapshot_keys),
        quality["valid_snapshot_count"] >= len(expected_snapshot_keys) - 1,
        not quality["missing_snapshot_keys"],
        quality["valid_probability_error_count"] == 0,
        expected_annotation_count > 0,
        quality["stance_annotation_count"] == expected_annotation_count,
        quality["stance_annotation_failure_count"] == 0,
        quality["exposure_layer_count_mismatch"] == 0,
        quality["social_self_exposure_leak_count"] == 0,
        quality["post_social_prediction_count"] == 180,
        quality["post_social_matching_final_round_count"] == 180,
        quality["extra_post_social_llm_calls"] == 0,
    ))
    write_json(output_dir / "quality_report.json", quality)

    schema = {
        "dataset_version": version,
        "primary_keys": {
            "belief_snapshots.csv": ["scenario_id", "round", "agent_id"],
            "agent_round_panel.csv": ["scenario_id", "round", "agent_id"],
            "agent_round_content_exposures.csv": ["scenario_id", "round", "viewer_agent_id", "content_type", "content_id"],
            "interaction_edges.csv": ["scenario_id", "interaction_id"],
            "content_stance_catalog.csv": ["scenario_id", "content_type", "content_id"],
            "scenario_round_metrics.csv": ["scenario_id", "round"],
        },
        "important_fields": {
            "belief_js_divergence": "同一 Agent 相邻两轮三分类概率分布的 JS divergence（bits）",
            "expected_return_delta": "本轮预测收益率减去上一轮信念快照，单位为小数",
            "exposure_social_mean_stance_score": "本轮看到的其他投资者唯一内容的 LLM 平均立场分，[-1,1]",
            "raw_exposure_count": "兼容审计总数；不得作为纯曝光量使用",
            "feed_impression_count": "OASIS feed 中出现该内容的次数，表示曝光机会而非已阅读",
            "direct_interaction_count": "针对该内容的点赞、点踩或评论动作次数",
            "is_self_authored": "内容作者是否就是查看者；社会曝光统计会排除此类记录",
            "is_first_exposure": "该 Agent 是否在本轮第一次看到该内容",
            "unique_exposure_count": "按 Agent-轮次-内容去重后的曝光数",
            "brier_score": "三分类概率预测的 Brier 分数，越小越好",
            "majority_accuracy_all_scenarios": "场景多数方向正确数除以 18；平票按错误计，避免分母变化",
        },
        "leakage_note": "evaluation_snapshots.csv and scenario_runs.csv contain future T+5 labels; do not use them as Agent input.",
    }
    write_json(output_dir / "schema.json", schema)

    report = render_report(
        summary, round_rows, profile_rows, stance_round_rows, association_rows
    )
    (output_dir / "analysis_report.md").write_text(report, encoding="utf-8")
    (batch_dir / "comprehensive_analysis_report.md").write_text(report, encoding="utf-8")
    write_json(batch_dir / "comprehensive_analysis_summary.json", summary)

    readme = render_readme(summary, quality)
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    checksum_lines = []
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "CHECKSUMS.sha256":
            checksum_lines.append(f"{sha256_file(path)}  {path.name}")
    (output_dir / "CHECKSUMS.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="ascii")
    return {"output_dir": str(output_dir), "quality": quality, "summary": summary}


def pct(value: Any) -> str:
    number = safe_float(value)
    return "NA" if number is None else f"{number * 100:.1f}%"


def num(value: Any, digits: int = 4) -> str:
    number = safe_float(value)
    return "NA" if number is None else f"{number:.{digits}f}"


def render_report(
    summary: Mapping[str, Any],
    rounds: Sequence[Mapping[str, Any]],
    profiles: Sequence[Mapping[str, Any]],
    stance_rounds: Sequence[Mapping[str, Any]],
    associations: Sequence[Mapping[str, Any]],
) -> str:
    max_round = int(summary["design"]["rounds"])
    seed = summary["design"]["random_seed"]
    r0 = rounds[0]
    r6 = rounds[SELECTED_ROUND]
    final = rounds[max_round]
    final_tokens = float(final["cumulative_total_tokens"] or 0)
    best_accuracy = max(rounds, key=lambda row: float(row["individual_direction_accuracy"] or 0))
    best_majority = max(rounds, key=lambda row: int(row["majority_correct_scenarios"]))
    stance_counts = summary["content_stance_counts"]
    within = next(row for row in associations if row["association"].startswith("within_scenario"))
    level_association = next(
        row for row in associations
        if row["association"] == "social_stance_vs_after_return"
    )
    exposure_change = next(
        row for row in associations
        if row["association"] == "social_exposure_count_vs_belief_js"
    )
    role_by_change = sorted(profiles, key=lambda row: float(row["mean_transition_js"] or 0), reverse=True)
    return f"""# S1 {max_round} 轮固定基准实验综合分析

## 结论

本批实验固定使用随机种子 **{seed}**、10 个 Agent 和 {max_round} 轮互动。它的主要用途是复核已经暂定的 **6 轮基准**在新随机种子下是否保持相近性质，而不是利用 seed={seed} 的结果重新挑选最佳轮次。

从 pre-social 到第 6 轮，个体方向准确率由 **{pct(r0['individual_direction_accuracy'])}** 变为 **{pct(r6['individual_direction_accuracy'])}**，场景多数方向正确数由 **{int(r0['majority_correct_scenarios'])}/18** 变为 **{int(r6['majority_correct_scenarios'])}/18**。第 6 轮相邻轮信念 JS 为 **{num(r6['transition_mean_js'])} bits**，累计方向改判率为 **{pct(r6['cumulative_direction_flip_rate'])}**。这些结果用于衡量 seed 稳健性，不能单独证明第 6 轮收敛或最优。

## 预测表现

| 指标 | 第 0 轮 | 第 {max_round} 轮 |
| --- | ---: | ---: |
| 个体方向准确率 | {pct(r0['individual_direction_accuracy'])} | {pct(final['individual_direction_accuracy'])} |
| 个体平衡准确率 | {pct(r0['individual_balanced_accuracy'])} | {pct(final['individual_balanced_accuracy'])} |
| Brier 分数（越低越好） | {num(r0['mean_brier_score'])} | {num(final['mean_brier_score'])} |
| 场景多数正确数 | {int(r0['majority_correct_scenarios'])}/18 | {int(final['majority_correct_scenarios'])}/18 |
| 平均收益率绝对误差 | {pct(r0['mean_return_absolute_error'])} | {pct(final['mean_return_absolute_error'])} |

真实标签为上涨 {summary['actual_direction_counts'].get('up', 0)} 个、中性 {summary['actual_direction_counts'].get('neutral', 0)} 个、下跌 {summary['actual_direction_counts'].get('down', 0)} 个。恒定预测样本最多类别可得到 **{pct(summary['constant_actual_majority_accuracy'])}** 的场景准确率。第 {max_round} 轮群体多数准确率为 **{pct(final['majority_accuracy_all_scenarios'])}**，必须与这一朴素基线一起解释。当前结果可以检验“社会互动是否改变预测”，但不能仅凭本批数据宣称系统已有竞争力。

在本批可观察的 0-{max_round} 轮中，个体方向准确率最高的是第 **{int(best_accuracy['round'])}** 轮（{pct(best_accuracy['individual_direction_accuracy'])}），场景多数正确数最高的是第 **{int(best_majority['round'])}** 轮（{int(best_majority['majority_correct_scenarios'])}/18）。这两项是描述性结果，不用于改变预先固定的第 6 轮口径。

## 社会互动造成了什么变化

- 从第 0 轮到第 6 轮，累计方向改判率为 **{pct(r6['cumulative_direction_flip_rate'])}**，累计平均 JS 为 **{num(r6['cumulative_mean_js'])} bits**。互动确实改变了相当一部分 Agent 的判断。
- 平均场景共识率从 **{pct(r0['mean_scenario_consensus_rate'])}** 变为 **{pct(r6['mean_scenario_consensus_rate'])}**；方向熵从 **{num(r0['mean_scenario_direction_entropy_bits'], 3)}** 变为 **{num(r6['mean_scenario_direction_entropy_bits'], 3)}**。这说明互动没有简单地把所有 Agent 推向同一个方向。
- 概率层面的场景内两两 JS 从 **{num(r0['mean_scenario_pairwise_js'])}** 变为 **{num(r6['mean_scenario_pairwise_js'])}**。方向标签可以翻转，但完整概率分布的差异不一定同步放大，后续分析不能只看上涨/中性/下跌标签。
- 到第 {max_round} 轮累计 provider-reported token 为 **{final_tokens / 1_000_000:.1f}M**。该信息仅描述计算规模，不等同于人民币费用。

## LLM 内容立场

本次共标注 **{sum(stance_counts.values())}** 条唯一帖子/评论：positive {stance_counts.get('positive', 0)}、mixed {stance_counts.get('mixed', 0)}、negative {stance_counts.get('negative', 0)}、neutral {stance_counts.get('neutral', 0)}、uncertain {stance_counts.get('uncertain', 0)}。标签已回填到逐条行动和逐条曝光边，并额外提供按“场景-Agent-轮次-内容”去重的曝光表。

社会内容平均立场与 Agent 当轮预测收益率水平的 Spearman 相关为 **{num(level_association['spearman_correlation'])}**，但与同轮 expected_return 变化的场景内中心化相关只有 **{num(within['spearman_correlation'])}**（n={within['sample_count']}）。这更符合“同一场景的信息背景同时影响内容和判断”，尚不能证明看到某类立场直接造成改判。社会内容曝光数量与信念 JS 的 Spearman 相关也只有 **{num(exposure_change['spearman_correlation'])}**，说明“看得多”不等于“改得多”。

以上均只能视为描述性相关：Agent 选择看什么内容并非随机，而且同一场景和同一 Agent 有重复观测，不能把相关直接解释为某条内容的因果影响。

## 角色差异

相邻轮信念 JS 最大的三个运行时 Agent 为：{', '.join(f"Agent {row['agent_id']}（{row['agent_role_label']}，{num(row['mean_transition_js'])}）" for row in role_by_change[:3])}。这说明不同角色的观点敏感度存在差异，但每个角色只有 18 个场景且只有一个随机种子，当前只能用于提出后续假设，不能把角色差异当作稳定人口规律。

## 对第 6 轮基准的含义

本批实验只运行到第 6 轮，因此不能观察第 7 轮以后是否反弹，也不能再次完成轮数选择。它应与 seed=4004 的 10 轮探索实验并列比较：若多个新种子下，第 6 轮相对于 pre-social 的信念变化、预测表现和互动强度方向大体一致，才说明 6 轮口径具有一定稳健性。若差异很大，应增加随机种子重复，而不是从本批 0-6 轮中事后改选一个更好看的轮次。

## 研究限制与下一步

1. 本报告仍是单个随机种子的完整截面。应将 seed={seed} 与 seed=4004 及后续至少 1-2 个新种子按场景配对比较。
2. 18 个场景不是 180 个独立 Agent 样本；显著性检验和置信区间应按场景聚类。
3. LLM 标注器与预测模型属于同一模型家族。正式论文前应抽取约 10%-20% 内容做双人人工复核，并报告一致性。
4. 下一步优先用 `agent_round_panel.csv` 做互信息或条件互信息分析，再进行 K=6/8/10 的真实重跑；离线删除 Agent 只能筛选候选，不能替代重新模拟网络互动。
"""


def render_readme(
    summary: Mapping[str, Any], quality: Mapping[str, Any]
) -> str:
    design = summary["design"]
    source_batch_id = summary["source_batch_id"]
    expected_snapshots = quality["expected_snapshot_count"]
    stance_count = quality["stance_annotation_count"]
    expected_stance_count = quality["expected_stance_annotation_count"]
    final_round = quality["post_social_source_round"]
    dataset_name = summary["dataset_version"]
    return f"""# S1 固定轮次复现实验数据集

该目录归档 `{source_batch_id}`：{design['scenario_count']} 个匿名 A 股场景、每场景 {design['agent_count_per_scenario']} 个 Agent、每场景 {design['rounds']} 轮 Reddit 式社会互动、固定本地随机种子 {design['random_seed']}。用途是复核暂定的第 {design['selected_provisional_round']} 轮基准，并继续研究内容曝光、信念变化、角色差异与 Agent 降采样。

## 推荐入口

- `analysis_report.md`：给人阅读的综合结论。
- `analysis_summary.json`：给程序或 Agent 读取的结论。
- `agent_round_panel.csv`：互信息和信念变化分析的主表，主键为 `scenario_id + round + agent_id`。
- `agent_round_content_exposures.csv`：按“场景-Agent-轮次-内容”去重；分开记录 feed 曝光机会、直接互动、自身内容以及首次/重复曝光。
- `interaction_edges.csv`：点赞、点踩、回复、关注等显式有向互动，不与 feed 曝光混合。
- `content_stance_catalog.csv`：{stance_count} 条唯一内容的独立 LLM 离线立场标签。
- `evaluation_snapshots.csv`：每轮信念与真实 T+5 结果的评估表。
- `round_metrics.csv`、`scenario_round_metrics.csv`：轮次选择和场景差异分析。
- `agent_role_metrics.csv`：10 个角色的描述性表现和行为统计。
- `social_actions_annotated.jsonl`、`exposure_edges_annotated.jsonl`：保留完整嵌套字段的审计记录。

## 重要边界

`evaluation_snapshots.csv` 和 `scenario_runs.csv` 含未来 T+5 真实标签，只能用于评估，不能输入 Agent。内容立场是离线标注，不参与本次 S1 运行，因此不会反向改变互动过程。

构建投资者 Agent 图时，应使用 `interaction_edges.csv` 并筛选 `actor_class=investor`、`target_class=investor`。指向 `target_class=source` 的边表示外生信息接触，应单独建层；`feed_impression_count` 仅表示可见机会，不能作为已发生的影响边。

第 {design['selected_provisional_round']} 轮只是预先暂定的短程互动基准，不是已经证明收敛。本数据集用于随机种子稳健性复核，不应据此重新选择轮次。互信息分析必须按场景划分训练/验证，不能把同一场景的不同行或不同 Agent 随机拆到两侧。

## 数据质量

- 场景：{quality['scenario_count']}/18
- 信念快照：{quality['snapshot_count']}/{expected_snapshots}，其中有效 {quality['valid_snapshot_count']}/{expected_snapshots}
- LLM 内容立场：{stance_count}/{expected_stance_count}，失败 {quality['stance_annotation_failure_count']}
- 社会曝光已排除自身内容：泄漏检查 {quality['social_self_exposure_leak_count']}
- 最终预测由第 {final_round} 轮快照派生：{quality['post_social_matching_final_round_count']}/{quality['post_social_prediction_count']}
- 额外 post-social LLM 调用：{quality['extra_post_social_llm_calls']}
- 总体质量检查：{'通过' if quality['passed'] else '未通过'}

`source_manifest.json` 记录所有源文件路径、大小和 SHA-256；`CHECKSUMS.sha256` 校验本目录生成文件；`schema.json` 解释主键和关键字段。

## 复现

从工作区根目录运行：

```powershell
MiroFish/backend/.venv/Scripts/python.exe MiroFish/backend/scripts/build_round_selection_dataset.py --batch-dir MiroFish/backend/uploads/finance/{source_batch_id} --output-dir Dataset/{dataset_name}
```
"""


def main() -> None:
    args = parse_args()
    batch_dir = args.batch_dir.resolve()
    manifest = read_json(batch_dir / "manifest.json")
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else workspace_root() / "Dataset" / dataset_version(manifest)
    )
    result = analyze_and_build(batch_dir, output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
