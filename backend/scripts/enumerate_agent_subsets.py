#!/usr/bin/env python3
"""Enumerate fixed-size Agent subsets and rank their offline fidelity.

The script compares every feasible subset with the observed 20-Agent system.
It is an offline screening tool: removing Agents from existing logs does not
reproduce the network and content that a reduced simulation would generate.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional progress dependency
    def tqdm(iterable=None, **_kwargs):
        return iterable


try:
    import networkx as nx
except ImportError:  # pragma: no cover - optional fallback is tested indirectly
    nx = None

try:
    import torch
except ImportError:  # pragma: no cover - reported by choose_device
    torch = None


ANALYSIS_VERSION = "agent_subset_enumeration_v1"
DEFAULT_K_VALUES = (4, 6, 8, 10, 12, 15)
DEFAULT_BATCH_SIZE = 16_384
EPSILON = 1e-8

DIRECTIONS = ("up", "neutral", "down")
PROBABILITY_COLUMNS = (
    "up_probability",
    "neutral_probability",
    "down_probability",
)
ACTION_COLUMNS = (
    "action_create_post_count",
    "action_create_comment_count",
    "action_like_post_count",
    "action_like_comment_count",
    "action_dislike_post_count",
    "action_dislike_comment_count",
    "action_refresh_count",
    "action_trend_count",
    "action_search_posts_count",
    "action_search_user_count",
    "action_follow_count",
    "action_other_count",
)
STANCE_COUNT_COLUMNS = (
    "exposure_social_positive_unique_count",
    "exposure_social_mixed_unique_count",
    "exposure_social_negative_unique_count",
    "exposure_social_neutral_unique_count",
    "exposure_social_uncertain_unique_count",
)
PROFILE_FIELDS = (
    ("agent_role_category", 0.30),
    ("agent_analysis_style", 0.20),
    ("agent_risk_attitude", 0.20),
    ("agent_investment_horizon", 0.15),
    ("agent_decision_source", 0.10),
    ("agent_social_role", 0.05),
)
REQUIRED_ROLE_CATEGORIES = (
    "institution",
    "retail_basic",
    "retail_mature",
    "retail_novice",
)
REQUIRED_ANALYSIS_STYLES = ("fundamental", "technical")


@dataclass(frozen=True)
class PreparedData:
    agent_ids: np.ndarray
    scenarios: tuple[str, ...]
    stage_scenario_indices: np.ndarray
    round_scenario_indices: np.ndarray
    stage_probabilities: np.ndarray
    stage_expected_return: np.ndarray
    stage_confidence: np.ndarray
    stage_directions: np.ndarray
    stage_valid: np.ndarray
    target_probabilities: np.ndarray
    target_expected_return: np.ndarray
    target_confidence: np.ndarray
    target_directions: np.ndarray
    target_majority_direction: np.ndarray
    target_direction_entropy: np.ndarray
    target_pairwise_js: np.ndarray
    pairwise_js: np.ndarray
    action_counts: np.ndarray
    stance_counts: np.ndarray
    stance_score_numerator: np.ndarray
    stance_score_denominator: np.ndarray
    profile_matrices: dict[str, np.ndarray]
    profile_categories: dict[str, tuple[str, ...]]
    graph_node_features: np.ndarray
    graph_source_reach: np.ndarray
    graph_adjacency: np.ndarray
    graph_reciprocal_adjacency: np.ndarray
    graph_community_membership: np.ndarray
    graph_communities: tuple[tuple[int, ...], ...]
    input_paths: dict[str, Path]


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[3]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    dataset = workspace_root() / "Dataset" / "downsampling_s1_rounds4_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=dataset)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=dataset / ANALYSIS_VERSION,
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=list(DEFAULT_K_VALUES),
    )
    parser.add_argument(
        "--all-k",
        action="store_true",
        help="Enumerate every non-empty proper subset size (K=1..N-1)",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float32",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--top-per-k", type=int, default=100)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument(
        "--min-majority-agreement",
        type=float,
        default=0.95,
        help="Report this direction-agreement quality gate without filtering",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show tqdm progress bars (disable for log files)",
    )
    parser.add_argument("--min-valid-fraction", type=float, default=0.75)
    parser.add_argument(
        "--profile-constraints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require role/style coverage and profile diversity",
    )
    parser.add_argument("--min-risk-categories", type=int, default=2)
    parser.add_argument("--min-horizon-categories", type=int, default=2)
    parser.add_argument(
        "--write-all-scores",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--max-combinations-per-k",
        type=int,
        default=None,
        help="Debug only: evaluate the first N feasible combinations per K",
    )
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def as_numeric(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    return frame.loc[:, columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)


def direction_one_hot(values: pd.Series) -> np.ndarray:
    normalized = values.astype(str).str.strip().str.lower()
    result = np.zeros((len(values), len(DIRECTIONS)), dtype=float)
    for index, label in enumerate(DIRECTIONS):
        result[:, index] = normalized.eq(label).to_numpy(float)
    return result


def numpy_js_divergence(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    midpoint = 0.5 * (left + right)
    left_term = np.where(
        left > 0.0,
        left * np.log2(np.maximum(left, EPSILON) / np.maximum(midpoint, EPSILON)),
        0.0,
    )
    right_term = np.where(
        right > 0.0,
        right * np.log2(np.maximum(right, EPSILON) / np.maximum(midpoint, EPSILON)),
        0.0,
    )
    return 0.5 * np.sum(left_term + right_term, axis=-1)


def pairwise_js_cube(probabilities: np.ndarray, valid: np.ndarray) -> np.ndarray:
    group_count, agent_count, _ = probabilities.shape
    result = np.zeros((group_count, agent_count, agent_count), dtype=np.float32)
    for group_index in range(group_count):
        values = np.nan_to_num(probabilities[group_index], nan=0.0)
        left = values[:, None, :]
        right = values[None, :, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            distances = numpy_js_divergence(left, right)
        pair_valid = valid[group_index, :, None] & valid[group_index, None, :]
        result[group_index] = np.where(pair_valid, distances, 0.0)
    return result


def _ordered_agent_rows(
    frame: pd.DataFrame,
    agent_ids: np.ndarray,
    *,
    name: str,
) -> pd.DataFrame:
    indexed = frame.copy()
    indexed["agent_id"] = pd.to_numeric(indexed["agent_id"], errors="raise").astype(int)
    if indexed["agent_id"].duplicated().any():
        raise ValueError(f"{name} contains duplicate Agent rows")
    ordered = indexed.set_index("agent_id").reindex(agent_ids)
    if ordered.index.has_duplicates or len(ordered) != len(agent_ids):
        raise ValueError(f"{name} could not be aligned to all Agents")
    return ordered


def _stage_values(
    rows: pd.DataFrame,
    prefix: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    probability_columns = [f"{prefix}_{name}" for name in PROBABILITY_COLUMNS]
    required = [
        f"{prefix}_status",
        f"{prefix}_direction",
        f"{prefix}_expected_return",
        f"{prefix}_confidence",
        *probability_columns,
    ]
    require_columns(rows, required, "stage rows")
    probabilities = as_numeric(rows, probability_columns)
    expected_return = pd.to_numeric(
        rows[f"{prefix}_expected_return"], errors="coerce"
    ).to_numpy(float)
    confidence = pd.to_numeric(
        rows[f"{prefix}_confidence"], errors="coerce"
    ).to_numpy(float)
    directions = direction_one_hot(rows[f"{prefix}_direction"])
    status_valid = rows[f"{prefix}_status"].astype(str).str.lower().eq("ok").to_numpy()
    valid = (
        status_valid
        & np.isfinite(probabilities).all(axis=1)
        & np.isfinite(expected_return)
        & np.isfinite(confidence)
        & np.isclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
        & np.isclose(directions.sum(axis=1), 1.0, atol=1e-6)
    )
    return probabilities, expected_return, confidence, directions, valid


def _target_stage_name(round_number: int | None) -> str:
    if round_number is None:
        return "post_social_final"
    return f"round_{round_number}"


def load_stage_data(
    round_frame: pd.DataFrame,
    scenario_frame: pd.DataFrame,
    target_frame: pd.DataFrame,
    agent_ids: np.ndarray,
    scenarios: tuple[str, ...],
) -> dict[str, np.ndarray]:
    round_required = {
        "scenario_id",
        "round",
        "agent_id",
        "current_status",
        "current_direction",
        "current_expected_return",
        "current_confidence",
        *(f"current_{name}" for name in PROBABILITY_COLUMNS),
    }
    scenario_required = {
        "scenario_id",
        "agent_id",
        "pre_status",
        "pre_direction",
        "pre_expected_return",
        "pre_confidence",
        "post_status",
        "post_direction",
        "post_expected_return",
        "post_confidence",
        *(f"pre_{name}" for name in PROBABILITY_COLUMNS),
        *(f"post_{name}" for name in PROBABILITY_COLUMNS),
    }
    target_required = {
        "scenario_id",
        "measurement_stage",
        "valid_agent_count",
        "mean_up_probability",
        "mean_neutral_probability",
        "mean_down_probability",
        "mean_expected_return",
        "mean_confidence",
        "up_direction_proportion",
        "neutral_direction_proportion",
        "down_direction_proportion",
        "majority_direction",
        "direction_entropy_bits",
        "mean_pairwise_js_divergence",
    }
    require_columns(round_frame, round_required, "agent_round_features.csv")
    require_columns(scenario_frame, scenario_required, "agent_scenario_features.csv")
    require_columns(target_frame, target_required, "group_targets.csv")

    round_frame = round_frame.copy()
    round_frame["round"] = pd.to_numeric(round_frame["round"], errors="raise").astype(int)
    target_lookup = target_frame.set_index(["scenario_id", "measurement_stage"])

    probabilities: list[np.ndarray] = []
    expected_returns: list[np.ndarray] = []
    confidences: list[np.ndarray] = []
    directions: list[np.ndarray] = []
    valid_rows: list[np.ndarray] = []
    target_probabilities: list[np.ndarray] = []
    target_expected_returns: list[float] = []
    target_confidences: list[float] = []
    target_directions: list[np.ndarray] = []
    target_majority_directions: list[int] = []
    target_entropies: list[float] = []
    target_pairwise: list[float] = []
    stage_scenario_indices: list[int] = []

    for scenario_index, scenario_id in enumerate(scenarios):
        scenario_rows = _ordered_agent_rows(
            scenario_frame[scenario_frame["scenario_id"].eq(scenario_id)],
            agent_ids,
            name=f"{scenario_id} agent_scenario_features",
        )
        stages: list[tuple[int | None, str, pd.DataFrame, str]] = [
            (0, "round_0", scenario_rows, "pre")
        ]
        for round_number in (1, 2, 3, 4):
            rows = _ordered_agent_rows(
                round_frame[
                    round_frame["scenario_id"].eq(scenario_id)
                    & round_frame["round"].eq(round_number)
                ],
                agent_ids,
                name=f"{scenario_id} round {round_number}",
            )
            stages.append((round_number, f"round_{round_number}", rows, "current"))
        stages.append((None, "post_social_final", scenario_rows, "post"))

        for _, target_stage, rows, prefix in stages:
            stage = _stage_values(rows, prefix)
            probabilities.append(stage[0])
            expected_returns.append(stage[1])
            confidences.append(stage[2])
            directions.append(stage[3])
            valid_rows.append(stage[4])
            stage_scenario_indices.append(scenario_index)

            try:
                target = target_lookup.loc[(scenario_id, target_stage)]
            except KeyError as exc:
                raise ValueError(
                    f"group_targets.csv is missing {(scenario_id, target_stage)}"
                ) from exc
            target_probabilities.append(
                target[
                    [
                        "mean_up_probability",
                        "mean_neutral_probability",
                        "mean_down_probability",
                    ]
                ].to_numpy(float)
            )
            target_expected_returns.append(float(target["mean_expected_return"]))
            target_confidences.append(float(target["mean_confidence"]))
            target_directions.append(
                target[
                    [
                        "up_direction_proportion",
                        "neutral_direction_proportion",
                        "down_direction_proportion",
                    ]
                ].to_numpy(float)
            )
            majority = str(target["majority_direction"]).strip().lower()
            target_majority_directions.append(
                DIRECTIONS.index(majority)
                if majority in DIRECTIONS
                else int(np.argmax(target_directions[-1]))
            )
            target_entropies.append(float(target["direction_entropy_bits"]))
            target_pairwise.append(float(target["mean_pairwise_js_divergence"]))
            if int(target["valid_agent_count"]) != int(stage[4].sum()):
                raise ValueError(
                    f"valid Agent count mismatch for {(scenario_id, target_stage)}"
                )

    result = {
        "probabilities": np.asarray(probabilities, dtype=np.float32),
        "expected_return": np.asarray(expected_returns, dtype=np.float32),
        "confidence": np.asarray(confidences, dtype=np.float32),
        "directions": np.asarray(directions, dtype=np.float32),
        "valid": np.asarray(valid_rows, dtype=bool),
        "target_probabilities": np.asarray(target_probabilities, dtype=np.float32),
        "target_expected_return": np.asarray(target_expected_returns, dtype=np.float32),
        "target_confidence": np.asarray(target_confidences, dtype=np.float32),
        "target_directions": np.asarray(target_directions, dtype=np.float32),
        "target_majority_direction": np.asarray(
            target_majority_directions, dtype=np.int64
        ),
        "target_direction_entropy": np.asarray(target_entropies, dtype=np.float32),
        "target_pairwise_js": np.asarray(target_pairwise, dtype=np.float32),
        "stage_scenario_indices": np.asarray(stage_scenario_indices, dtype=np.int16),
    }

    valid = result["valid"]
    full_probabilities = np.nansum(
        np.where(valid[..., None], result["probabilities"], 0.0), axis=1
    ) / valid.sum(axis=1, keepdims=True)
    if not np.allclose(
        full_probabilities, result["target_probabilities"], atol=2e-6
    ):
        difference = float(
            np.max(np.abs(full_probabilities - result["target_probabilities"]))
        )
        raise ValueError(f"full-group probability target mismatch: max diff={difference}")
    return result


def load_round_context(
    round_frame: pd.DataFrame,
    agent_ids: np.ndarray,
    scenarios: tuple[str, ...],
) -> dict[str, np.ndarray]:
    required = {
        "scenario_id",
        "round",
        "agent_id",
        *ACTION_COLUMNS,
        *STANCE_COUNT_COLUMNS,
        "exposure_social_unique_content_count",
        "exposure_social_mean_stance_score_unique",
    }
    require_columns(round_frame, required, "agent_round_features.csv")
    frame = round_frame.copy()
    frame["round"] = pd.to_numeric(frame["round"], errors="raise").astype(int)
    actions: list[np.ndarray] = []
    stances: list[np.ndarray] = []
    stance_numerator: list[np.ndarray] = []
    stance_denominator: list[np.ndarray] = []
    scenario_indices: list[int] = []
    for scenario_index, scenario_id in enumerate(scenarios):
        for round_number in (1, 2, 3, 4):
            rows = _ordered_agent_rows(
                frame[
                    frame["scenario_id"].eq(scenario_id)
                    & frame["round"].eq(round_number)
                ],
                agent_ids,
                name=f"{scenario_id} round {round_number} context",
            )
            action_values = np.nan_to_num(as_numeric(rows, ACTION_COLUMNS), nan=0.0)
            stance_values = np.nan_to_num(
                as_numeric(rows, STANCE_COUNT_COLUMNS), nan=0.0
            )
            unique_count = pd.to_numeric(
                rows["exposure_social_unique_content_count"], errors="coerce"
            ).fillna(0.0).to_numpy(float)
            mean_score = pd.to_numeric(
                rows["exposure_social_mean_stance_score_unique"], errors="coerce"
            ).fillna(0.0).to_numpy(float)
            if (action_values < 0).any() or (stance_values < 0).any():
                raise ValueError("action and stance counts must be non-negative")
            actions.append(action_values)
            stances.append(stance_values)
            stance_numerator.append(unique_count * mean_score)
            stance_denominator.append(unique_count)
            scenario_indices.append(scenario_index)
    return {
        "actions": np.asarray(actions, dtype=np.float32),
        "stances": np.asarray(stances, dtype=np.float32),
        "stance_numerator": np.asarray(stance_numerator, dtype=np.float32),
        "stance_denominator": np.asarray(stance_denominator, dtype=np.float32),
        "scenario_indices": np.asarray(scenario_indices, dtype=np.int16),
    }


def load_profiles(
    profile_frame: pd.DataFrame,
    agent_ids: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, tuple[str, ...]]]:
    required = {"agent_id", *(field for field, _ in PROFILE_FIELDS)}
    require_columns(profile_frame, required, "agent_profiles.csv")
    rows = _ordered_agent_rows(profile_frame, agent_ids, name="agent_profiles.csv")
    matrices: dict[str, np.ndarray] = {}
    categories: dict[str, tuple[str, ...]] = {}
    for field, _ in PROFILE_FIELDS:
        values = rows[field].astype(str).str.strip()
        field_categories = tuple(sorted(values.unique().tolist()))
        matrix = np.column_stack(
            [values.eq(category).to_numpy(float) for category in field_categories]
        )
        if not np.isclose(matrix.sum(axis=1), 1.0).all():
            raise ValueError(f"invalid profile categories in {field}")
        matrices[field] = matrix.astype(np.float32)
        categories[field] = field_categories
    return matrices, categories


def _community_membership(adjacency: np.ndarray) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    agent_count = adjacency.shape[0]
    if nx is None:
        communities = (tuple(range(agent_count)),)
    else:
        graph = nx.Graph()
        graph.add_nodes_from(range(agent_count))
        symmetric = adjacency + adjacency.T
        for left in range(agent_count):
            for right in range(left + 1, agent_count):
                weight = float(symmetric[left, right])
                if weight > 0.0:
                    graph.add_edge(left, right, weight=weight)
        detected = nx.algorithms.community.greedy_modularity_communities(
            graph, weight="weight"
        )
        communities = tuple(tuple(sorted(group)) for group in detected)
        if not communities:
            communities = (tuple(range(agent_count)),)
    membership = np.zeros((agent_count, len(communities)), dtype=np.float32)
    for community_index, members in enumerate(communities):
        membership[list(members), community_index] = 1.0
    return membership, communities


def load_graph_data(
    exposure_frame: pd.DataFrame,
    agent_ids: np.ndarray,
    scenarios: tuple[str, ...],
) -> dict[str, Any]:
    required = {
        "scenario_id",
        "viewer_agent_id",
        "round",
        "author_agent_id",
        "author_class",
        "is_self_authored",
    }
    require_columns(exposure_frame, required, "agent_round_content_exposures.csv")
    agent_to_index = {int(agent_id): index for index, agent_id in enumerate(agent_ids)}
    agent_count = len(agent_ids)
    graph_keys = [(scenario, round_number) for scenario in scenarios for round_number in (1, 2, 3, 4)]
    graph_key_to_index = {key: index for index, key in enumerate(graph_keys)}
    adjacency_cube = np.zeros((len(graph_keys), agent_count, agent_count), dtype=np.float64)
    source_seen = np.zeros((len(graph_keys), agent_count), dtype=bool)

    for row in exposure_frame.itertuples(index=False):
        scenario_id = str(row.scenario_id)
        try:
            round_number = int(row.round)
        except (TypeError, ValueError):
            continue
        graph_index = graph_key_to_index.get((scenario_id, round_number))
        if graph_index is None:
            continue
        viewer = agent_to_index.get(int(row.viewer_agent_id))
        if viewer is None:
            continue
        if str(row.author_class).strip().lower() == "source":
            source_seen[graph_index, viewer] = True
            continue
        if str(row.author_class).strip().lower() != "investor":
            continue
        author = agent_to_index.get(int(row.author_agent_id))
        if author is None or author == viewer:
            continue
        adjacency_cube[graph_index, author, viewer] += 1.0

    normalized = np.zeros_like(adjacency_cube)
    totals = adjacency_cube.sum(axis=(1, 2))
    nonzero = totals > 0.0
    normalized[nonzero] = adjacency_cube[nonzero] / totals[nonzero, None, None]
    adjacency = normalized[nonzero].mean(axis=0)
    reciprocal = np.minimum(adjacency, adjacency.T)

    in_strength = normalized.sum(axis=1).mean(axis=0)
    out_strength = normalized.sum(axis=2).mean(axis=0)
    in_neighbors = (adjacency_cube > 0.0).sum(axis=1).mean(axis=0) / max(agent_count - 1, 1)
    out_neighbors = (adjacency_cube > 0.0).sum(axis=2).mean(axis=0) / max(agent_count - 1, 1)
    node_features = np.column_stack(
        [in_strength, out_strength, in_neighbors, out_neighbors]
    ).astype(np.float32)
    source_reach = source_seen.mean(axis=0).astype(np.float32)
    membership, communities = _community_membership(adjacency)
    return {
        "node_features": node_features,
        "source_reach": source_reach,
        "adjacency": adjacency.astype(np.float32),
        "reciprocal": reciprocal.astype(np.float32),
        "community_membership": membership,
        "communities": communities,
    }


def load_data(dataset_dir: Path) -> PreparedData:
    paths = {
        "round": dataset_dir / "agent_round_features.csv",
        "scenario": dataset_dir / "agent_scenario_features.csv",
        "targets": dataset_dir / "group_targets.csv",
        "profiles": dataset_dir / "agent_profiles.csv",
        "exposures": dataset_dir / "agent_round_content_exposures.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing input files: {missing}")

    profile_frame = pd.read_csv(paths["profiles"])
    require_columns(profile_frame, ["agent_id"], "agent_profiles.csv")
    agent_ids = np.asarray(
        sorted(pd.to_numeric(profile_frame["agent_id"], errors="raise").astype(int).unique()),
        dtype=np.int16,
    )
    if len(agent_ids) < 2:
        raise ValueError("at least two Agents are required")

    round_frame = pd.read_csv(paths["round"])
    scenario_frame = pd.read_csv(paths["scenario"])
    target_frame = pd.read_csv(paths["targets"])
    scenarios = tuple(sorted(round_frame["scenario_id"].astype(str).unique().tolist()))
    if len(scenarios) < 2:
        raise ValueError("at least two scenarios are required")

    stage = load_stage_data(
        round_frame, scenario_frame, target_frame, agent_ids, scenarios
    )
    context = load_round_context(round_frame, agent_ids, scenarios)
    profile_matrices, profile_categories = load_profiles(profile_frame, agent_ids)
    exposure_frame = pd.read_csv(paths["exposures"])
    graph = load_graph_data(exposure_frame, agent_ids, scenarios)

    return PreparedData(
        agent_ids=agent_ids,
        scenarios=scenarios,
        stage_scenario_indices=stage["stage_scenario_indices"],
        round_scenario_indices=context["scenario_indices"],
        stage_probabilities=stage["probabilities"],
        stage_expected_return=stage["expected_return"],
        stage_confidence=stage["confidence"],
        stage_directions=stage["directions"],
        stage_valid=stage["valid"],
        target_probabilities=stage["target_probabilities"],
        target_expected_return=stage["target_expected_return"],
        target_confidence=stage["target_confidence"],
        target_directions=stage["target_directions"],
        target_majority_direction=stage["target_majority_direction"],
        target_direction_entropy=stage["target_direction_entropy"],
        target_pairwise_js=stage["target_pairwise_js"],
        pairwise_js=pairwise_js_cube(stage["probabilities"], stage["valid"]),
        action_counts=context["actions"],
        stance_counts=context["stances"],
        stance_score_numerator=context["stance_numerator"],
        stance_score_denominator=context["stance_denominator"],
        profile_matrices=profile_matrices,
        profile_categories=profile_categories,
        graph_node_features=graph["node_features"],
        graph_source_reach=graph["source_reach"],
        graph_adjacency=graph["adjacency"],
        graph_reciprocal_adjacency=graph["reciprocal"],
        graph_community_membership=graph["community_membership"],
        graph_communities=graph["communities"],
        input_paths=paths,
    )


def choose_device(requested: str) -> "torch.device":
    if torch is None:
        raise RuntimeError(
            "PyTorch is required. Run this script with the cifar10 environment."
        )
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is not available")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def torch_dtype(name: str) -> "torch.dtype":
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required")
    return torch.float32 if name == "float32" else torch.float64


def _tensor(array: np.ndarray, device: "torch.device", dtype: "torch.dtype") -> "torch.Tensor":
    return torch.as_tensor(array, dtype=dtype, device=device)


def prepare_tensors(
    data: PreparedData,
    device: "torch.device",
    dtype: "torch.dtype",
) -> dict[str, Any]:
    stage_probabilities = np.nan_to_num(data.stage_probabilities, nan=0.0)
    expected_return = np.nan_to_num(data.stage_expected_return, nan=0.0)
    confidence = np.nan_to_num(data.stage_confidence, nan=0.0)
    valid = data.stage_valid.astype(np.float32)
    action_full = data.action_counts.sum(axis=1)
    stance_full = data.stance_counts.sum(axis=1)
    stance_numerator_full = data.stance_score_numerator.sum(axis=1)
    stance_denominator_full = data.stance_score_denominator.sum(axis=1)

    node_features = data.graph_node_features.astype(np.float64)
    node_mean = node_features.mean(axis=0)
    node_std = node_features.std(axis=0)
    node_scale = np.maximum.reduce(
        [node_std, np.abs(node_mean) * 0.1, np.full_like(node_std, EPSILON)]
    )
    source_mean = float(data.graph_source_reach.mean())
    source_scale = max(
        float(data.graph_source_reach.std()), abs(source_mean) * 0.1, EPSILON
    )

    return {
        "probability_flat": _tensor(
            stage_probabilities.transpose(1, 0, 2).reshape(len(data.agent_ids), -1),
            device,
            dtype,
        ),
        "expected_flat": _tensor(expected_return.T, device, dtype),
        "confidence_flat": _tensor(confidence.T, device, dtype),
        "direction_flat": _tensor(
            data.stage_directions.transpose(1, 0, 2).reshape(len(data.agent_ids), -1),
            device,
            dtype,
        ),
        "valid_flat": _tensor(valid.T, device, dtype),
        "target_probabilities": _tensor(data.target_probabilities, device, dtype),
        "target_expected": _tensor(data.target_expected_return, device, dtype),
        "target_confidence": _tensor(data.target_confidence, device, dtype),
        "target_directions": _tensor(data.target_directions, device, dtype),
        "target_majority_direction": torch.as_tensor(
            data.target_majority_direction, dtype=torch.long, device=device
        ),
        "target_entropy": _tensor(data.target_direction_entropy, device, dtype),
        "target_pairwise": _tensor(data.target_pairwise_js, device, dtype),
        "pairwise": _tensor(data.pairwise_js, device, dtype),
        "action_flat": _tensor(
            data.action_counts.transpose(1, 0, 2).reshape(len(data.agent_ids), -1),
            device,
            dtype,
        ),
        "action_full": _tensor(action_full, device, dtype),
        "stance_flat": _tensor(
            data.stance_counts.transpose(1, 0, 2).reshape(len(data.agent_ids), -1),
            device,
            dtype,
        ),
        "stance_full": _tensor(stance_full, device, dtype),
        "stance_numerator_flat": _tensor(
            data.stance_score_numerator.T, device, dtype
        ),
        "stance_denominator_flat": _tensor(
            data.stance_score_denominator.T, device, dtype
        ),
        "stance_numerator_full": _tensor(stance_numerator_full, device, dtype),
        "stance_denominator_full": _tensor(stance_denominator_full, device, dtype),
        "profile": {
            field: _tensor(matrix, device, dtype)
            for field, matrix in data.profile_matrices.items()
        },
        "graph_node_features": _tensor(data.graph_node_features, device, dtype),
        "graph_node_mean": _tensor(node_mean, device, dtype),
        "graph_node_std": _tensor(node_std, device, dtype),
        "graph_node_scale": _tensor(node_scale, device, dtype),
        "graph_source_reach": _tensor(data.graph_source_reach, device, dtype),
        "graph_source_mean": torch.tensor(source_mean, device=device, dtype=dtype),
        "graph_source_scale": torch.tensor(source_scale, device=device, dtype=dtype),
        "graph_adjacency": _tensor(data.graph_adjacency, device, dtype),
        "graph_reciprocal": _tensor(
            data.graph_reciprocal_adjacency, device, dtype
        ),
        "graph_community": _tensor(
            data.graph_community_membership, device, dtype
        ),
        "stage_scenario_indices": data.stage_scenario_indices,
        "round_scenario_indices": data.round_scenario_indices,
        "stage_count": data.stage_probabilities.shape[0],
        "round_group_count": data.action_counts.shape[0],
        "agent_count": len(data.agent_ids),
    }


def torch_js_divergence(left: "torch.Tensor", right: "torch.Tensor") -> "torch.Tensor":
    midpoint = 0.5 * (left + right)
    left_term = torch.where(
        left > 0.0,
        left * torch.log2(left.clamp_min(EPSILON) / midpoint.clamp_min(EPSILON)),
        torch.zeros_like(left),
    )
    right_term = torch.where(
        right > 0.0,
        right * torch.log2(right.clamp_min(EPSILON) / midpoint.clamp_min(EPSILON)),
        torch.zeros_like(right),
    )
    return 0.5 * torch.sum(left_term + right_term, dim=-1)


def counts_to_distribution(counts: "torch.Tensor") -> "torch.Tensor":
    totals = counts.sum(dim=-1, keepdim=True)
    normalized = counts / totals.clamp_min(EPSILON)
    no_activity = (totals <= EPSILON).to(counts.dtype)
    return torch.cat([normalized, no_activity], dim=-1)


def normalized_weighted_mean_error(
    numerator: "torch.Tensor",
    denominator: "torch.Tensor",
    target_numerator: "torch.Tensor",
    target_denominator: "torch.Tensor",
) -> "torch.Tensor":
    value = numerator / denominator.clamp_min(EPSILON)
    target = target_numerator / target_denominator.clamp_min(EPSILON)
    value_present = denominator > EPSILON
    target_present = target_denominator > EPSILON
    return torch.where(
        value_present & target_present,
        torch.abs(value - target.unsqueeze(0)) / 2.0,
        torch.where(
            value_present == target_present,
            torch.zeros_like(value),
            torch.ones_like(value),
        ),
    )


def robust_summary(values: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    return (
        values.mean(dim=1),
        torch.quantile(values, 0.9, dim=1),
        values.max(dim=1).values,
    )


def _to_numpy(value: "torch.Tensor") -> np.ndarray:
    return value.detach().cpu().numpy()


def _add_error_summary(
    result: dict[str, np.ndarray],
    name: str,
    values: "torch.Tensor",
) -> None:
    mean, p90, worst = robust_summary(values)
    result[f"{name}_mean"] = _to_numpy(mean)
    result[f"{name}_p90"] = _to_numpy(p90)
    result[f"{name}_worst"] = _to_numpy(worst)


def _fold_means(
    values: "torch.Tensor",
    scenario_indices: np.ndarray,
    folds: int,
) -> list[np.ndarray]:
    output: list[np.ndarray] = []
    for fold in range(folds):
        indices = np.flatnonzero(scenario_indices % folds == fold)
        if len(indices) == 0:
            output.append(np.full(values.shape[0], np.nan))
        else:
            output.append(_to_numpy(values[:, indices].mean(dim=1)))
    return output


def calculate_batch(
    selection: "torch.Tensor",
    tensors: dict[str, Any],
    *,
    k: int,
    folds: int,
    min_valid_fraction: float,
) -> dict[str, np.ndarray]:
    batch_size = selection.shape[0]
    stage_count = tensors["stage_count"]
    round_group_count = tensors["round_group_count"]
    valid_count = selection @ tensors["valid_flat"]
    denominator = valid_count.clamp_min(1.0)
    minimum_valid = math.ceil(k * min_valid_fraction - EPSILON)
    valid_constraint = (valid_count >= minimum_valid).all(dim=1)

    probabilities = (
        selection @ tensors["probability_flat"]
    ).reshape(batch_size, stage_count, 3) / denominator.unsqueeze(-1)
    expected_return = (selection @ tensors["expected_flat"]) / denominator
    confidence = (selection @ tensors["confidence_flat"]) / denominator
    directions = (
        selection @ tensors["direction_flat"]
    ).reshape(batch_size, stage_count, 3) / denominator.unsqueeze(-1)

    belief_js = torch_js_divergence(
        probabilities, tensors["target_probabilities"].unsqueeze(0)
    )
    direction_js = torch_js_divergence(
        directions, tensors["target_directions"].unsqueeze(0)
    )
    expected_error = torch.abs(
        expected_return - tensors["target_expected"].unsqueeze(0)
    )
    confidence_error = torch.abs(
        confidence - tensors["target_confidence"].unsqueeze(0)
    )
    direction_entropy = -torch.sum(
        torch.where(
            directions > 0.0,
            directions * torch.log2(directions.clamp_min(EPSILON)),
            torch.zeros_like(directions),
        ),
        dim=-1,
    )
    entropy_error = torch.abs(
        direction_entropy - tensors["target_entropy"].unsqueeze(0)
    )
    pairwise_sum = torch.einsum(
        "bi,gij,bj->bg", selection, tensors["pairwise"], selection
    )
    pairwise_denominator = (valid_count * (valid_count - 1.0)).clamp_min(1.0)
    pairwise_mean = pairwise_sum / pairwise_denominator
    pairwise_error = torch.abs(
        pairwise_mean - tensors["target_pairwise"].unsqueeze(0)
    )
    majority_disagreement = (
        directions.argmax(dim=-1)
        != tensors["target_majority_direction"].unsqueeze(0)
    ).to(selection.dtype)

    action_counts = (
        selection @ tensors["action_flat"]
    ).reshape(batch_size, round_group_count, len(ACTION_COLUMNS))
    action_js = torch_js_divergence(
        counts_to_distribution(action_counts),
        counts_to_distribution(tensors["action_full"]).unsqueeze(0),
    )
    stance_counts = (
        selection @ tensors["stance_flat"]
    ).reshape(batch_size, round_group_count, len(STANCE_COUNT_COLUMNS))
    stance_js = torch_js_divergence(
        counts_to_distribution(stance_counts),
        counts_to_distribution(tensors["stance_full"]).unsqueeze(0),
    )
    stance_denominator = selection @ tensors["stance_denominator_flat"]
    stance_score_error = normalized_weighted_mean_error(
        selection @ tensors["stance_numerator_flat"],
        stance_denominator,
        tensors["stance_numerator_full"],
        tensors["stance_denominator_full"],
    )

    result: dict[str, np.ndarray] = {
        "min_valid_agent_count": _to_numpy(valid_count.min(dim=1).values),
        "valid_data_constraint": _to_numpy(valid_constraint),
        "majority_direction_agreement": _to_numpy(
            1.0 - majority_disagreement.mean(dim=1)
        ),
        "majority_direction_agreement_worst": _to_numpy(
            1.0 - majority_disagreement.max(dim=1).values
        ),
    }
    _add_error_summary(result, "belief_js", belief_js)
    _add_error_summary(result, "direction_js", direction_js)
    _add_error_summary(result, "expected_return_mae", expected_error)
    _add_error_summary(result, "confidence_mae", confidence_error)
    _add_error_summary(result, "direction_entropy_abs_error", entropy_error)
    _add_error_summary(result, "pairwise_js_abs_error", pairwise_error)
    _add_error_summary(result, "action_mix_js", action_js)
    _add_error_summary(result, "stance_distribution_js", stance_js)
    _add_error_summary(result, "stance_score_normalized_abs_error", stance_score_error)

    fold_sources = {
        "belief_js": (belief_js, tensors["stage_scenario_indices"]),
        "majority_direction_disagreement": (
            majority_disagreement,
            tensors["stage_scenario_indices"],
        ),
        "direction_js": (direction_js, tensors["stage_scenario_indices"]),
        "expected_return_mae": (expected_error, tensors["stage_scenario_indices"]),
        "confidence_mae": (confidence_error, tensors["stage_scenario_indices"]),
        "direction_entropy_abs_error": (
            entropy_error,
            tensors["stage_scenario_indices"],
        ),
        "pairwise_js_abs_error": (
            pairwise_error,
            tensors["stage_scenario_indices"],
        ),
        "action_mix_js": (action_js, tensors["round_scenario_indices"]),
        "stance_distribution_js": (
            stance_js,
            tensors["round_scenario_indices"],
        ),
        "stance_score_normalized_abs_error": (
            stance_score_error,
            tensors["round_scenario_indices"],
        ),
    }
    for metric_name, (values, scenario_indices) in fold_sources.items():
        for fold, fold_values in enumerate(
            _fold_means(values, scenario_indices, folds)
        ):
            result[f"{metric_name}_fold_{fold}"] = fold_values

    profile_error = torch.zeros(batch_size, device=selection.device, dtype=selection.dtype)
    profile_weights = dict(PROFILE_FIELDS)
    for field, matrix in tensors["profile"].items():
        distribution = (selection @ matrix) / float(k)
        target = matrix.mean(dim=0).unsqueeze(0)
        field_error = torch_js_divergence(distribution, target)
        result[f"profile_{field}_js"] = _to_numpy(field_error)
        profile_error = profile_error + profile_weights[field] * field_error
    result["profile_error"] = _to_numpy(profile_error)

    graph_features = tensors["graph_node_features"]
    graph_mean = (selection @ graph_features) / float(k)
    graph_second = (selection @ (graph_features * graph_features)) / float(k)
    graph_std = torch.sqrt((graph_second - graph_mean * graph_mean).clamp_min(0.0))
    graph_strength_error = 0.5 * torch.mean(
        torch.abs(graph_mean - tensors["graph_node_mean"])
        / tensors["graph_node_scale"],
        dim=1,
    ) + 0.5 * torch.mean(
        torch.abs(graph_std - tensors["graph_node_std"])
        / tensors["graph_node_scale"],
        dim=1,
    )

    internal_weight = torch.sum(
        (selection @ tensors["graph_adjacency"]) * selection, dim=1
    )
    internal_reciprocal = torch.sum(
        (selection @ tensors["graph_reciprocal"]) * selection, dim=1
    )
    if k == 1:
        density_error = torch.ones_like(internal_weight)
        reciprocity_error = torch.ones_like(internal_weight)
    else:
        full_density = tensors["graph_adjacency"].sum() / (
            tensors["agent_count"] * (tensors["agent_count"] - 1)
        )
        subset_density = internal_weight / float(k * (k - 1))
        density_error = torch.abs(
            subset_density - full_density
        ) / full_density.clamp_min(EPSILON)
        full_reciprocity = tensors["graph_reciprocal"].sum() / tensors[
            "graph_adjacency"
        ].sum().clamp_min(EPSILON)
        subset_reciprocity = internal_reciprocal / internal_weight.clamp_min(EPSILON)
        reciprocity_error = torch.where(
            internal_weight > EPSILON,
            torch.abs(subset_reciprocity - full_reciprocity),
            torch.ones_like(internal_weight),
        )
    topology_error = 0.5 * density_error.clamp(max=1.0) + 0.5 * reciprocity_error

    community_hits = (selection @ tensors["graph_community"] > 0.0).to(selection.dtype)
    community_error = 1.0 - community_hits.mean(dim=1)
    source_mean = (selection @ tensors["graph_source_reach"]) / float(k)
    source_error = torch.abs(source_mean - tensors["graph_source_mean"]) / tensors[
        "graph_source_scale"
    ]
    graph_error = (
        0.40 * torch.tanh(graph_strength_error)
        + 0.25 * community_error
        + 0.20 * torch.tanh(source_error)
        + 0.15 * topology_error
    )
    result.update(
        {
            "graph_strength_error": _to_numpy(graph_strength_error),
            "graph_density_relative_error": _to_numpy(density_error),
            "graph_reciprocity_abs_error": _to_numpy(reciprocity_error),
            "graph_community_error": _to_numpy(community_error),
            "graph_source_reach_error": _to_numpy(source_error),
        "graph_error": _to_numpy(graph_error),
        }
    )
    return result


def combination_array(agent_count: int, k: int) -> np.ndarray:
    if not 1 <= k <= agent_count:
        raise ValueError(f"K={k} is outside [1, {agent_count}]")
    count = math.comb(agent_count, k)
    values = np.fromiter(
        (index for subset in itertools.combinations(range(agent_count), k) for index in subset),
        dtype=np.int16,
        count=count * k,
    )
    return values.reshape(count, k)


def profile_constraint_mask(
    combinations: np.ndarray,
    data: PreparedData,
    *,
    enabled: bool,
    min_risk_categories: int,
    min_horizon_categories: int,
) -> tuple[np.ndarray, dict[str, int]]:
    feasible = np.ones(len(combinations), dtype=bool)
    failures: dict[str, int] = {}
    if not enabled:
        return feasible, failures

    checks: list[tuple[str, np.ndarray]] = []
    role_matrix = data.profile_matrices["agent_role_category"]
    role_categories = data.profile_categories["agent_role_category"]
    for category in REQUIRED_ROLE_CATEGORIES:
        if category not in role_categories:
            raise ValueError(f"required role category is absent: {category}")
        category_index = role_categories.index(category)
        checks.append(
            (
                f"role_{category}",
                role_matrix[combinations, category_index].sum(axis=1) >= 1.0,
            )
        )

    style_matrix = data.profile_matrices["agent_analysis_style"]
    style_categories = data.profile_categories["agent_analysis_style"]
    for category in REQUIRED_ANALYSIS_STYLES:
        if category not in style_categories:
            raise ValueError(f"required analysis style is absent: {category}")
        category_index = style_categories.index(category)
        checks.append(
            (
                f"style_{category}",
                style_matrix[combinations, category_index].sum(axis=1) >= 1.0,
            )
        )

    risk_matrix = data.profile_matrices["agent_risk_attitude"]
    risk_coverage = (risk_matrix[combinations].sum(axis=1) > 0.0).sum(axis=1)
    checks.append(("risk_diversity", risk_coverage >= min_risk_categories))
    horizon_matrix = data.profile_matrices["agent_investment_horizon"]
    horizon_coverage = (horizon_matrix[combinations].sum(axis=1) > 0.0).sum(axis=1)
    checks.append(("horizon_diversity", horizon_coverage >= min_horizon_categories))

    for name, check in checks:
        failures[name] = int((~check).sum())
        feasible &= check
    return feasible, failures


def rank_percentile(series: pd.Series) -> pd.Series:
    return series.rank(method="average", pct=True, ascending=True)


def pareto_front_3d(values: np.ndarray) -> np.ndarray:
    """Return the exact minimization frontier for three objectives."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("pareto_front_3d expects an N x 3 array")
    if len(values) == 0:
        return np.zeros(0, dtype=bool)
    order = np.lexsort((values[:, 2], values[:, 1], values[:, 0]))
    front = np.ones(len(values), dtype=bool)
    frontier_indices: list[int] = []
    for index in order:
        point = values[index]
        if frontier_indices:
            existing = values[np.asarray(frontier_indices)]
            dominates = np.all(existing <= point, axis=1) & np.any(
                existing < point, axis=1
            )
            if bool(dominates.any()):
                front[index] = False
                continue
            dominated = np.all(point <= existing, axis=1) & np.any(
                point < existing, axis=1
            )
            if bool(dominated.any()):
                frontier_indices = [
                    candidate
                    for candidate, remove in zip(frontier_indices, dominated, strict=True)
                    if not remove
                ]
        frontier_indices.append(int(index))
    return front


def add_rankings(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["belief_error"] = (
        0.7 * result["belief_js_mean"] + 0.3 * result["belief_js_p90"]
    )
    result["majority_direction_disagreement"] = 1.0 - result[
        "majority_direction_agreement"
    ]

    outcome_columns = (
        "direction_js_mean",
        "majority_direction_disagreement",
        "expected_return_mae_mean",
        "confidence_mae_mean",
    )
    diversity_columns = (
        "direction_entropy_abs_error_mean",
        "pairwise_js_abs_error_mean",
    )
    social_columns = (
        "stance_distribution_js_mean",
        "stance_score_normalized_abs_error_mean",
        "action_mix_js_mean",
    )
    result["outcome_error"] = sum(
        rank_percentile(result[column]) for column in outcome_columns
    ) / len(outcome_columns)
    result["diversity_error"] = sum(
        rank_percentile(result[column]) for column in diversity_columns
    ) / len(diversity_columns)
    result["social_error"] = (
        0.60 * rank_percentile(result[social_columns[0]])
        + 0.25 * rank_percentile(result[social_columns[1]])
        + 0.15 * rank_percentile(result[social_columns[2]])
    )

    belief_rank = rank_percentile(result["belief_error"])
    profile_rank = rank_percentile(result["profile_error"])
    graph_rank = rank_percentile(result["graph_error"])
    result["overall_score"] = (
        0.35 * belief_rank
        + 0.15 * result["outcome_error"]
        + 0.15 * result["diversity_error"]
        + 0.15 * result["social_error"]
        + 0.10 * profile_rank
        + 0.10 * graph_rank
    )
    result["overall_rank"] = result["overall_score"].rank(
        method="min", ascending=True
    ).astype(int)
    result["pareto_fidelity_objective"] = (
        0.55 * belief_rank
        + 0.25 * result["outcome_error"]
        + 0.20 * result["diversity_error"]
    )
    result["pareto_social_objective"] = result["social_error"]
    result["pareto_representation_objective"] = (
        0.50 * profile_rank + 0.50 * graph_rank
    )
    result["pareto_front"] = pareto_front_3d(
        result[
            [
                "pareto_fidelity_objective",
                "pareto_social_objective",
                "pareto_representation_objective",
            ]
        ].to_numpy()
    )
    return result


def fold_selection_rows(frame: pd.DataFrame, folds: int, k: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    component_names = (
        "belief_js",
        "majority_direction_disagreement",
        "direction_js",
        "expected_return_mae",
        "confidence_mae",
        "direction_entropy_abs_error",
        "pairwise_js_abs_error",
        "action_mix_js",
        "stance_distribution_js",
        "stance_score_normalized_abs_error",
    )
    for holdout_fold in range(folds):
        train_folds = [fold for fold in range(folds) if fold != holdout_fold]
        temporary = pd.DataFrame(index=frame.index)
        for name in component_names:
            temporary[name] = frame[
                [f"{name}_fold_{fold}" for fold in train_folds]
            ].mean(axis=1)
        temporary["profile_error"] = frame["profile_error"]
        temporary["graph_error"] = frame["graph_error"]
        temporary["outcome"] = sum(
            rank_percentile(temporary[column])
            for column in (
                "direction_js",
                "expected_return_mae",
                "confidence_mae",
                "majority_direction_disagreement",
            )
        ) / 4.0
        temporary["diversity"] = 0.5 * (
            rank_percentile(temporary["direction_entropy_abs_error"])
            + rank_percentile(temporary["pairwise_js_abs_error"])
        )
        temporary["social"] = (
            0.60 * rank_percentile(temporary["stance_distribution_js"])
            + 0.25
            * rank_percentile(temporary["stance_score_normalized_abs_error"])
            + 0.15 * rank_percentile(temporary["action_mix_js"])
        )
        temporary["score"] = (
            0.35 * rank_percentile(temporary["belief_js"])
            + 0.15 * temporary["outcome"]
            + 0.15 * temporary["diversity"]
            + 0.15 * temporary["social"]
            + 0.10 * rank_percentile(temporary["profile_error"])
            + 0.10 * rank_percentile(temporary["graph_error"])
        )
        best_index = temporary["score"].idxmin()
        selected = frame.loc[best_index]
        rows.append(
            {
                "k": k,
                "holdout_fold": holdout_fold,
                "subset_mask": int(selected["subset_mask"]),
                "agent_ids": selected["agent_ids"],
                "train_score": float(temporary.loc[best_index, "score"]),
                "holdout_belief_js": float(
                    selected[f"belief_js_fold_{holdout_fold}"]
                ),
                "holdout_direction_js": float(
                    selected[f"direction_js_fold_{holdout_fold}"]
                ),
                "holdout_majority_direction_agreement": float(
                    1.0
                    - selected[
                        f"majority_direction_disagreement_fold_{holdout_fold}"
                    ]
                ),
                "holdout_expected_return_mae": float(
                    selected[f"expected_return_mae_fold_{holdout_fold}"]
                ),
                "holdout_stance_distribution_js": float(
                    selected[f"stance_distribution_js_fold_{holdout_fold}"]
                ),
                "holdout_action_mix_js": float(
                    selected[f"action_mix_js_fold_{holdout_fold}"]
                ),
            }
        )
    return rows


def subset_masks(combinations: np.ndarray, agent_ids: np.ndarray) -> tuple[np.ndarray, list[str]]:
    actual_ids = agent_ids[combinations].astype(np.int64)
    bit_masks = np.bitwise_or.reduce(np.left_shift(1, actual_ids), axis=1)
    labels = ["|".join(str(int(agent_id)) for agent_id in row) for row in actual_ids]
    return bit_masks, labels


def enumerate_k(
    data: PreparedData,
    tensors: dict[str, Any],
    args: argparse.Namespace,
    k: int,
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    started = time.perf_counter()
    all_combinations = combination_array(len(data.agent_ids), k)
    feasible, failures = profile_constraint_mask(
        all_combinations,
        data,
        enabled=args.profile_constraints,
        min_risk_categories=args.min_risk_categories,
        min_horizon_categories=args.min_horizon_categories,
    )
    combinations = all_combinations[feasible]
    del all_combinations
    feasible_count = len(combinations)
    truncated = False
    if args.max_combinations_per_k is not None:
        combinations = combinations[: args.max_combinations_per_k]
        truncated = len(combinations) < feasible_count
    if len(combinations) == 0:
        raise ValueError(f"no feasible combinations for K={k}")

    metric_batches: list[pd.DataFrame] = []
    batch_starts = range(0, len(combinations), args.batch_size)
    batch_iterator = tqdm(
        batch_starts,
        total=math.ceil(len(combinations) / args.batch_size),
        desc=f"K={k}",
        unit="batch",
        disable=not args.progress,
        leave=False,
    )
    for start in batch_iterator:
        batch_combinations = combinations[start : start + args.batch_size]
        selection = torch.zeros(
            (len(batch_combinations), len(data.agent_ids)),
            device=tensors["probability_flat"].device,
            dtype=tensors["probability_flat"].dtype,
        )
        row_indices = torch.arange(len(batch_combinations), device=selection.device)
        for column in range(k):
            indices = torch.as_tensor(
                batch_combinations[:, column], dtype=torch.long, device=selection.device
            )
            selection[row_indices, indices] = 1.0
        with torch.inference_mode():
            metrics = calculate_batch(
                selection,
                tensors,
                k=k,
                folds=args.folds,
                min_valid_fraction=args.min_valid_fraction,
            )
        metric_batches.append(pd.DataFrame(metrics))

    scores = pd.concat(metric_batches, ignore_index=True)
    valid_data = scores["valid_data_constraint"].astype(bool).to_numpy()
    invalid_data_count = int((~valid_data).sum())
    scores = scores.loc[valid_data].reset_index(drop=True)
    combinations = combinations[valid_data]
    masks, labels = subset_masks(combinations, data.agent_ids)
    scores.insert(0, "agent_ids", labels)
    scores.insert(0, "subset_mask", masks)
    scores.insert(0, "k", k)
    scores = add_rankings(scores)
    scores["meets_majority_agreement_gate"] = (
        scores["majority_direction_agreement"] >= args.min_majority_agreement
    )
    fold_rows = fold_selection_rows(scores, args.folds, k)
    best = scores.loc[scores["overall_rank"].idxmin()]
    summary = {
        "k": k,
        "combination_count": math.comb(len(data.agent_ids), k),
        "profile_feasible_count": feasible_count,
        "evaluated_count": len(combinations),
        "valid_data_count": len(scores),
        "invalid_data_count": invalid_data_count,
        "truncated": truncated,
        "constraint_failure_counts": failures,
        "pareto_count": int(scores["pareto_front"].sum()),
        "majority_agreement_gate": args.min_majority_agreement,
        "majority_agreement_gate_count": int(
            scores["meets_majority_agreement_gate"].sum()
        ),
        "best_subset_mask": int(best["subset_mask"]),
        "best_agent_ids": str(best["agent_ids"]),
        "best_overall_score": float(best["overall_score"]),
        "best_belief_error": float(best["belief_error"]),
        "best_profile_error": float(best["profile_error"]),
        "best_graph_error": float(best["graph_error"]),
        "elapsed_seconds": time.perf_counter() - started,
    }
    return scores, summary, fold_rows


def write_readme(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Agent subset enumeration",
        "",
        "This directory contains offline exact-enumeration results. Lower error scores are better.",
        "The logs come from the full 20-Agent simulation; shortlisted subsets must still be rerun.",
        "",
        "## Outputs",
        "",
        "- `scores_kXX.csv.gz`: every feasible evaluated subset for K.",
        "- `top_candidates.csv`: best ranked subsets for each K.",
        "- `pareto_candidates.csv`: three-objective Pareto front for each K.",
        "- `fold_results.csv`: subset selected on two scenario folds and evaluated on the third.",
        "- `k_summary.csv`: counts, runtime, and best subset per K.",
        "- `summary.json`: configuration, hashes, environment, and metric definitions.",
        "- `run_status.json`: `running` while interrupted and `complete` after all requested K values finish.",
        "",
        "## Ranking",
        "",
        "The overall score uses within-K percentile ranks: belief 35%, outcome 15%, diversity 15%, social context 15%, profile 10%, and graph 10%.",
        "The Pareto front minimizes fidelity, social-context, and representation objectives.",
        "",
        f"Device: `{summary['environment']['device']}`",
        f"Dtype: `{summary['configuration']['dtype']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_args(args: argparse.Namespace, agent_count: int) -> None:
    if args.batch_size <= 0 or args.top_per_k <= 0:
        raise ValueError("batch-size and top-per-k must be positive")
    if args.folds < 2:
        raise ValueError("folds must be at least 2")
    if not 0.0 <= args.min_majority_agreement <= 1.0:
        raise ValueError("min-majority-agreement must be in [0, 1]")
    if not 0.0 < args.min_valid_fraction <= 1.0:
        raise ValueError("min-valid-fraction must be in (0, 1]")
    if args.max_combinations_per_k is not None and args.max_combinations_per_k <= 0:
        raise ValueError("max-combinations-per-k must be positive")
    for k in args.k_values:
        if not 1 <= k <= agent_count:
            raise ValueError(f"K={k} is outside [1, {agent_count}]")
    if len(set(args.k_values)) != len(args.k_values):
        raise ValueError("k-values must not contain duplicates")
    if args.profile_constraints and min(args.k_values) < len(REQUIRED_ROLE_CATEGORIES):
        raise ValueError(
            "K<4 cannot cover all required role categories; use "
            "--no-profile-constraints for K=1..3"
        )


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    data = load_data(args.dataset_dir)
    if args.all_k:
        args.k_values = list(range(1, len(data.agent_ids)))
    validate_args(args, len(data.agent_ids))
    if args.folds > len(data.scenarios):
        raise ValueError("folds cannot exceed the number of scenarios")
    device = choose_device(args.device)
    dtype = torch_dtype(args.dtype)
    tensors = prepare_tensors(data, device, dtype)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / "run_status.json"
    with status_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "status": "running",
                "analysis_version": ANALYSIS_VERSION,
                "requested_k_values": list(args.k_values),
                "started_epoch_seconds": time.time(),
            },
            handle,
            indent=2,
        )

    k_summaries: list[dict[str, Any]] = []
    top_frames: list[pd.DataFrame] = []
    pareto_frames: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    k_iterator = tqdm(
        args.k_values,
        desc="K values",
        unit="K",
        disable=not args.progress,
    )
    for k in k_iterator:
        scores, k_summary, current_fold_rows = enumerate_k(
            data, tensors, args, int(k)
        )
        k_summaries.append(k_summary)
        fold_rows.extend(current_fold_rows)
        top_frames.append(scores.nsmallest(args.top_per_k, "overall_score"))
        pareto_frames.append(scores[scores["pareto_front"]].copy())
        if args.write_all_scores:
            scores.to_csv(
                args.output_dir / f"scores_k{k:02d}.csv.gz",
                index=False,
                compression="gzip",
                float_format="%.10g",
            )

    top_candidates = pd.concat(top_frames, ignore_index=True)
    pareto_candidates = pd.concat(pareto_frames, ignore_index=True)
    fold_results = pd.DataFrame(fold_rows)
    k_summary_frame = pd.DataFrame(k_summaries)
    top_candidates.to_csv(args.output_dir / "top_candidates.csv", index=False)
    pareto_candidates.to_csv(args.output_dir / "pareto_candidates.csv", index=False)
    fold_results.to_csv(args.output_dir / "fold_results.csv", index=False)
    k_summary_frame.to_csv(args.output_dir / "k_summary.csv", index=False)

    summary: dict[str, Any] = {
        "analysis_version": ANALYSIS_VERSION,
        "configuration": {
            "dataset_dir": str(args.dataset_dir.resolve()),
            "output_dir": str(args.output_dir.resolve()),
            "k_values": list(args.k_values),
            "all_k": args.all_k,
            "batch_size": args.batch_size,
            "top_per_k": args.top_per_k,
            "folds": args.folds,
            "min_majority_agreement": args.min_majority_agreement,
            "min_valid_fraction": args.min_valid_fraction,
            "profile_constraints": args.profile_constraints,
            "min_risk_categories": args.min_risk_categories,
            "min_horizon_categories": args.min_horizon_categories,
            "dtype": args.dtype,
            "write_all_scores": args.write_all_scores,
            "max_combinations_per_k": args.max_combinations_per_k,
        },
        "data": {
            "agent_ids": [int(value) for value in data.agent_ids],
            "agent_count": len(data.agent_ids),
            "scenario_count": len(data.scenarios),
            "scenarios": list(data.scenarios),
            "social_round_count": 4,
            "stage_count": len(data.stage_scenario_indices),
            "graph_communities": [list(group) for group in data.graph_communities],
        },
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": str(device),
            "cuda_device": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
        },
        "input_files": {
            name: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
            for name, path in data.input_paths.items()
        },
        "metric_weights": {
            "overall": {
                "belief": 0.35,
                "outcome": 0.15,
                "diversity": 0.15,
                "social": 0.15,
                "profile": 0.10,
                "graph": 0.10,
            },
            "profile": dict(PROFILE_FIELDS),
            "graph": {
                "node_strength": 0.40,
                "community": 0.25,
                "source_reach": 0.20,
                "topology": 0.15,
            },
        },
        "k_results": k_summaries,
        "elapsed_seconds": time.perf_counter() - started,
        "offline_screening_warning": (
            "Deleting rows from full-system logs does not reproduce a reduced-Agent rerun."
        ),
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    write_readme(args.output_dir / "README.md", summary)
    with status_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "status": "complete",
                "analysis_version": ANALYSIS_VERSION,
                "requested_k_values": list(args.k_values),
                "completed_epoch_seconds": time.time(),
                "elapsed_seconds": summary["elapsed_seconds"],
            },
            handle,
            indent=2,
        )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run(args)
    print(
        json.dumps(
            {
                "output_dir": summary["configuration"]["output_dir"],
                "device": summary["environment"]["device"],
                "elapsed_seconds": summary["elapsed_seconds"],
                "best_subsets": [
                    {
                        "k": item["k"],
                        "agent_ids": item["best_agent_ids"],
                        "overall_score": item["best_overall_score"],
                    }
                    for item in summary["k_results"]
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
