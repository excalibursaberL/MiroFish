#!/usr/bin/env python3
"""Exact offline enumeration for the five-seed, ten-Agent S1 dataset.

The implementation follows the finalized screening protocol:

* structural data validity and four role categories are hard constraints;
* belief, direction trajectory, social process, and graph errors are hard
  fidelity constraints;
* seed aggregation is P80, then worst round per scenario, then scenario P90;
* social and graph statistics are recomputed on the induced subgraph;
* Pareto fronts and explanatory indicators are reported without a weighted
  total score.

This is an offline screen. Deleting rows from a ten-Agent trace cannot replace
an actual reduced-Agent rerun, so shortlisted subsets must be rerun later.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable=None, **_kwargs):
        return iterable


ANALYSIS_VERSION = "agent_subset_enumeration_v2"
DEFAULT_DATASET = (
    Path(__file__).resolve().parents[3]
    / "Dataset"
    / "s1_multiseed_k10_round6_seeds4004_42_3407_999_2887_v2"
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[3]
    / "Dataset"
    / "agent_subset_enumeration_k10_v2"
)
AGENT_COUNT_EXPECTED = 10
ROUND_VALUES = tuple(range(7))
SOCIAL_ROUNDS = tuple(range(1, 7))
DIRECTIONS = ("up", "neutral", "down")
STANCE_LABELS = ("positive", "mixed", "negative", "neutral", "uncertain")
ROLE_CATEGORIES = ("institution", "retail_mature", "retail_basic", "retail_novice")
ACTION_LABELS = (
    "create_comment",
    "create_post",
    "like_comment",
    "like_post",
    "dislike_comment",
    "dislike_post",
    "follow",
)
TIE_EPSILON = 0.02
INVALID_OBSERVATION = {("4004", "SCN_006", 0, 7)}
EPS = 1e-12
STANCE_SCORE_EDGES = np.linspace(-1.0, 1.0, 11)
STANCE_SCORE_CENTERS = 0.5 * (STANCE_SCORE_EDGES[:-1] + STANCE_SCORE_EDGES[1:])

# These are frozen before candidate evaluation. Epsilon is an engineering
# tolerance, while domain_cap prevents unstable K=10 noise from making a gate
# arbitrarily permissive.
THRESHOLD_SPECS = {
    "belief_js": (0.02, 0.25),
    "majority_final_error": (0.10, 0.50),
    "majority_trajectory_error": (0.10, 0.50),
    "trajectory_return_error": (0.02, 0.25),
    "social_stance_js": (0.03, 0.35),
    "social_stance_score_error": (0.08, 1.00),
    "social_action_js": (0.08, 0.80),
    "social_content_rate_error": (0.20, 1.00),
    "social_participation_error": (0.20, 1.00),
    "social_interaction_rate_error": (0.20, 1.00),
    "social_source_reach_error": (0.20, 1.00),
    "graph_strength_error": (0.12, 1.00),
    "graph_active_pair_error": (0.20, 1.00),
    "graph_facility_error": (0.15, 0.80),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--k-values", nargs="+", type=int, default=list(range(4, 10)))
    parser.add_argument("--top-per-k", type=int, default=50)
    parser.add_argument("--profile-constraints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write-all-scores", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse an existing scores_kXX.csv.gz and continue missing K values",
    )
    return parser.parse_args(argv)


def require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def numeric(values: pd.Series) -> np.ndarray:
    return pd.to_numeric(values, errors="coerce").to_numpy(float)


def js_divergence(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Base-2 Jensen-Shannon divergence; zero vectors are allowed."""
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    left_sum = left.sum(axis=-1, keepdims=True)
    right_sum = right.sum(axis=-1, keepdims=True)
    left = np.divide(left, left_sum, out=np.zeros_like(left), where=left_sum > EPS)
    right = np.divide(right, right_sum, out=np.zeros_like(right), where=right_sum > EPS)
    mid = 0.5 * (left + right)
    with np.errstate(divide="ignore", invalid="ignore"):
        lterm = np.where(left > 0, left * np.log2(np.maximum(left, EPS) / np.maximum(mid, EPS)), 0.0)
        rterm = np.where(right > 0, right * np.log2(np.maximum(right, EPS) / np.maximum(mid, EPS)), 0.0)
    return 0.5 * np.sum(lterm + rterm, axis=-1)


def quantile_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Distance between five quantiles, after row-wise probability normalization."""
    points = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    left_total = left.sum(axis=-1, keepdims=True)
    right_total = right.sum(axis=-1, keepdims=True)
    left = np.divide(left, left_total, out=np.zeros_like(left), where=left_total > EPS)
    right = np.divide(right, right_total, out=np.zeros_like(right), where=right_total > EPS)
    # Inputs are degree values rather than distributions. Callers pass a
    # sorted vector and this helper is used only for small per-group arrays.
    return np.mean(np.abs(left - right), axis=-1)


def histogram_quantiles(histogram: np.ndarray) -> np.ndarray:
    """Approximate P10/P50/P90 from a fixed score histogram."""
    values = np.asarray(histogram, dtype=float)
    cumulative = np.cumsum(values, axis=-1)
    total = cumulative[..., -1:]
    result = np.zeros(values.shape[:-1] + (3,), dtype=float)
    for index, quantile in enumerate((0.10, 0.50, 0.90)):
        target = total[..., 0] * quantile
        bucket = np.argmax(cumulative >= target[..., None], axis=-1)
        bucket_total = np.take_along_axis(cumulative, bucket[..., None], axis=-1)[..., 0]
        previous = np.where(bucket > 0, np.take_along_axis(cumulative, (bucket - 1)[..., None], axis=-1)[..., 0], 0.0)
        width = np.take_along_axis(values, bucket[..., None], axis=-1)[..., 0]
        fraction = np.divide(target - previous, width, out=np.zeros_like(target), where=width > EPS)
        interpolated = np.take(STANCE_SCORE_CENTERS, bucket) + (fraction - 0.5) * (STANCE_SCORE_EDGES[1] - STANCE_SCORE_EDGES[0])
        result[..., index] = np.where(total[..., 0] > EPS, interpolated, 0.0)
    return result


def normalize_direction(values: pd.Series) -> np.ndarray:
    result = np.full(len(values), -1, dtype=np.int8)
    labels = values.astype(str).str.strip().str.lower().to_numpy()
    for i, direction in enumerate(DIRECTIONS):
        result[labels == direction] = i
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def key_group(scenario: str, seed: int, round_number: int, lookup: dict[tuple[str, int, int], int]) -> int:
    try:
        return lookup[(str(scenario), int(seed), int(round_number))]
    except KeyError as exc:
        raise ValueError(f"missing group key {(scenario, seed, round_number)}") from exc


def tie_aware_majority(probabilities: np.ndarray) -> np.ndarray:
    """Return 0/1/2, or -1 for a near-tie."""
    values = np.asarray(probabilities, dtype=float)
    order = np.argsort(values, axis=-1)
    top = np.take_along_axis(values, order[..., -1:,], axis=-1)[..., 0]
    second = np.take_along_axis(values, order[..., -2:-1], axis=-1)[..., 0]
    result = order[..., -1].astype(np.int8)
    result[(top - second) <= TIE_EPSILON] = -1
    return result


def aggregate_p80_worst_p90(values: np.ndarray, scenario_count: int, seed_count: int, round_count: int, scenario_mask: np.ndarray | None = None) -> np.ndarray:
    """P80 over seeds, worst round, then P90 over scenarios."""
    values = np.asarray(values, dtype=float).reshape(-1, scenario_count, seed_count, round_count)
    p80 = np.nanquantile(values, 0.80, axis=2)
    worst = np.nanmax(p80, axis=2)
    if scenario_mask is not None:
        worst = worst[:, scenario_mask]
    return np.nanquantile(worst, 0.90, axis=1)


def aggregate_final(values: np.ndarray, scenario_count: int, seed_count: int, scenario_mask: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1, scenario_count, seed_count)
    p80 = np.nanquantile(values, 0.80, axis=2)
    if scenario_mask is not None:
        p80 = p80[:, scenario_mask]
    return np.nanquantile(p80, 0.90, axis=1)


def aggregate_direction_agreement(
    values: np.ndarray,
    scenario_count: int,
    seed_count: int,
    *,
    final: bool,
    scenario_mask: np.ndarray | None = None,
) -> np.ndarray:
    """P90 scenario disagreement, averaging the declared denominator first.

    A binary mismatch cannot use a worst-round operator without turning one
    local flip into a global failure. Trajectory agreement therefore averages
    scenario-seed-round units; final agreement averages scenario-seed units.
    """
    round_count = 1 if final else 7
    cube = np.asarray(values, dtype=float).reshape(
        -1, scenario_count, seed_count, round_count
    )
    scenario_error = np.nanmean(cube, axis=(2, 3))
    if scenario_mask is not None:
        scenario_error = scenario_error[:, scenario_mask]
    return np.nanquantile(scenario_error, 0.90, axis=1)


def pairwise_noise(values: np.ndarray, scenario_count: int, seed_count: int, round_count: int, metric: str) -> float:
    """95th percentile seed-pair noise for a group-level metric."""
    values = np.asarray(values, dtype=float).reshape(scenario_count, seed_count, round_count, -1)
    diffs: list[np.ndarray] = []
    for i in range(seed_count):
        for j in range(i + 1, seed_count):
            a, b = values[:, i], values[:, j]
            if metric == "js":
                diffs.append(js_divergence(a, b).ravel())
            else:
                diffs.append(np.abs(a - b).ravel())
    flat = np.concatenate(diffs) if diffs else np.array([0.0])
    flat = flat[np.isfinite(flat)]
    return float(np.quantile(flat, 0.95)) if len(flat) else 0.0


def profile_order(belief: pd.DataFrame, agent_ids: np.ndarray) -> tuple[dict[int, dict[str, str]], dict[str, list[str]]]:
    columns = ["agent_role_category", "agent_analysis_style", "agent_risk_attitude", "agent_investment_horizon"]
    require_columns(belief, ["agent_id", *columns], "belief snapshots")
    profile: dict[int, dict[str, str]] = {}
    for agent_id in agent_ids:
        rows = belief[belief["agent_id"].astype(int).eq(int(agent_id))]
        if rows.empty:
            raise ValueError(f"missing Profile for Agent {agent_id}")
        item: dict[str, str] = {}
        for column in columns:
            unique = rows[column].fillna("").astype(str).str.strip().unique()
            if len(unique) != 1 or not unique[0]:
                raise ValueError(f"Profile field {column} is not unique for Agent {agent_id}")
            item[column] = unique[0]
        profile[int(agent_id)] = item
    categories = {column: sorted({values[column] for values in profile.values()}) for column in columns}
    return profile, categories


def load_data(dataset_dir: Path) -> dict[str, object]:
    paths = {
        "belief": dataset_dir / "merged_belief_snapshots.csv",
        "panel": dataset_dir / "merged_agent_round_panel.csv",
        "exposure": dataset_dir / "merged_agent_round_content_exposures.csv",
        "edges": dataset_dir / "merged_interaction_edges.csv",
        "scenario_metrics": dataset_dir / "merged_scenario_round_metrics.csv",
        "runs": dataset_dir / "merged_scenario_runs.csv",
        "pair_redundancy": dataset_dir / "agent_pair_redundancy.csv",
        "loo": dataset_dir / "agent_leave_one_out_metrics.csv",
        "contributions": dataset_dir / "agent_information_contributions.csv",
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"missing {name} input: {path}")
    belief = pd.read_csv(paths["belief"], low_memory=False)
    panel = pd.read_csv(paths["panel"], low_memory=False)
    exposure = pd.read_csv(paths["exposure"], low_memory=False)
    edges = pd.read_csv(paths["edges"], low_memory=False)
    scenario_metrics = pd.read_csv(paths["scenario_metrics"], low_memory=False)
    runs = pd.read_csv(paths["runs"], low_memory=False)
    require_columns(belief, ["scenario_id", "agent_id", "round", "seed", "status", "direction", "up_probability", "neutral_probability", "down_probability", "expected_return", "confidence"], "belief snapshots")
    require_columns(panel, ["scenario_id", "agent_id", "round", "seed", "action_count", "authored_content_count"], "agent round panel")
    require_columns(exposure, ["scenario_id", "viewer_agent_id", "author_agent_id", "author_class", "round", "seed", "content_stance", "stance_score", "interacted_any", "is_self_authored"], "exposures")
    require_columns(edges, ["scenario_id", "round", "actor_agent_id", "actor_class", "target_agent_id", "target_class", "action_type", "seed"], "interaction edges")
    require_columns(scenario_metrics, ["scenario_id", "round", "seed", "majority_direction", "mean_expected_return"], "scenario metrics")
    for frame in (belief, panel, exposure, edges, scenario_metrics, runs):
        if "seed" in frame:
            frame["seed"] = pd.to_numeric(frame["seed"], errors="coerce").astype("Int64")
    belief["agent_id"] = pd.to_numeric(belief["agent_id"], errors="raise").astype(int)
    panel["agent_id"] = pd.to_numeric(panel["agent_id"], errors="raise").astype(int)
    exposure["viewer_agent_id"] = pd.to_numeric(exposure["viewer_agent_id"], errors="coerce").astype("Int64")
    exposure["author_agent_id"] = pd.to_numeric(exposure["author_agent_id"], errors="coerce").astype("Int64")
    edges["actor_agent_id"] = pd.to_numeric(edges["actor_agent_id"], errors="coerce").astype("Int64")
    edges["target_agent_id"] = pd.to_numeric(edges["target_agent_id"], errors="coerce").astype("Int64")
    for frame, col in ((belief, "round"), (panel, "round"), (exposure, "round"), (edges, "round"), (scenario_metrics, "round")):
        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype(int)
    agent_ids = np.array(sorted(belief["agent_id"].unique()), dtype=int)
    if len(agent_ids) != AGENT_COUNT_EXPECTED:
        raise ValueError(f"expected 10 Agents, found {len(agent_ids)}: {agent_ids.tolist()}")
    scenarios = tuple(sorted(belief["scenario_id"].astype(str).unique(), key=lambda s: int(s.split("_")[-1])))
    seeds = tuple(sorted(int(s) for s in belief["seed"].dropna().unique()))
    if len(scenarios) != 18 or len(seeds) != 5:
        raise ValueError(f"expected 18 scenarios and 5 seeds, found {len(scenarios)} and {len(seeds)}")
    profile, profile_categories = profile_order(belief, agent_ids)
    agent_index = {int(a): i for i, a in enumerate(agent_ids)}
    scenario_index = {s: i for i, s in enumerate(scenarios)}
    seed_index = {s: i for i, s in enumerate(seeds)}
    group_keys = [(s, seed, r) for s in scenarios for seed in seeds for r in ROUND_VALUES]
    group_lookup = {key: i for i, key in enumerate(group_keys)}
    social_keys = [(s, seed, r) for s in scenarios for seed in seeds for r in SOCIAL_ROUNDS]
    social_lookup = {key: i for i, key in enumerate(social_keys)}
    g_count, h_count, n = len(group_keys), len(social_keys), len(agent_ids)
    probs = np.full((g_count, n, 3), np.nan, dtype=float)
    expected = np.full((g_count, n), np.nan, dtype=float)
    confidence = np.full((g_count, n), np.nan, dtype=float)
    valid = np.zeros((g_count, n), dtype=bool)
    direction = np.full((g_count, n), -1, dtype=np.int8)
    for row in belief.itertuples(index=False):
        key = (str(row.scenario_id), int(row.seed), int(row.round))
        if key not in group_lookup or int(row.agent_id) not in agent_index:
            continue
        g, a = group_lookup[key], agent_index[int(row.agent_id)]
        p = np.array([row.up_probability, row.neutral_probability, row.down_probability], dtype=float)
        e, c = float(row.expected_return), float(row.confidence)
        d = normalize_direction(pd.Series([row.direction]))[0]
        ok = str(row.status).lower() == "ok" and np.isfinite(p).all() and np.isfinite(e) and np.isfinite(c) and np.isclose(p.sum(), 1.0, atol=1e-6) and d >= 0
        if (str(row.seed), str(row.scenario_id), int(row.round), int(row.agent_id)) in INVALID_OBSERVATION:
            ok = False
        probs[g, a] = p
        expected[g, a] = e
        confidence[g, a] = c
        direction[g, a] = d
        valid[g, a] = ok
    if not valid.any(axis=1).all():
        raise ValueError("at least one valid snapshot is required in every scenario-seed-round group")
    target_probs = np.nanmean(np.where(valid[..., None], probs, np.nan), axis=1)
    target_expected = np.nanmean(np.where(valid, expected, np.nan), axis=1)
    target_confidence = np.nanmean(np.where(valid, confidence, np.nan), axis=1)
    # Use one tie rule for the full reference and every candidate. The stored
    # majority_direction remains an audit field but must not silently break a
    # probability tie.
    target_majority = tie_aware_majority(target_probs).astype(np.int8)
    # Social per viewer/author tensors.
    stance_pair = np.zeros((h_count, n, n, len(STANCE_LABELS)), dtype=float)
    score_pair_sum = np.zeros((h_count, n, n), dtype=float)
    score_pair_count = np.zeros((h_count, n, n), dtype=float)
    score_hist_pair = np.zeros((h_count, n, n, len(STANCE_SCORE_CENTERS)), dtype=float)
    interacted_pair = np.zeros((h_count, n, n), dtype=float)
    source_seen = np.zeros((h_count, n), dtype=float)
    source_count = np.zeros((h_count, n), dtype=float)
    stance_index = {label: i for i, label in enumerate(STANCE_LABELS)}
    for row in exposure.itertuples(index=False):
        key = (str(row.scenario_id), int(row.seed), int(row.round))
        if key not in social_lookup or pd.isna(row.viewer_agent_id):
            continue
        viewer = agent_index.get(int(row.viewer_agent_id))
        if viewer is None:
            continue
        h = social_lookup[key]
        author_class = str(row.author_class).lower()
        if author_class == "source":
            source_count[h, viewer] += 1.0
            source_seen[h, viewer] = 1.0
            continue
        if author_class != "investor" or pd.isna(row.author_agent_id):
            continue
        author = agent_index.get(int(row.author_agent_id))
        if author is None or author == viewer:
            continue
        label = str(row.content_stance).strip().lower()
        stance_pair[h, viewer, author, stance_index.get(label, stance_index["uncertain"])] += 1.0
        score = float(row.stance_score) if pd.notna(row.stance_score) else 0.0
        score_pair_sum[h, viewer, author] += score
        score_pair_count[h, viewer, author] += 1.0
        score_bucket = int(np.searchsorted(STANCE_SCORE_EDGES, np.clip(score, -1.0, 1.0), side="right") - 1)
        score_bucket = min(max(score_bucket, 0), len(STANCE_SCORE_CENTERS) - 1)
        score_hist_pair[h, viewer, author, score_bucket] += 1.0
        interacted = str(row.interacted_any).strip().lower() in {"true", "1", "yes"}
        interacted_pair[h, viewer, author] += float(interacted)
    action_categories = {name: i for i, name in enumerate(ACTION_LABELS)}
    edge_action = np.zeros((h_count, n, n + 1, len(ACTION_LABELS)), dtype=float)
    adjacency = np.zeros((h_count, n, n), dtype=float)
    for row in edges.itertuples(index=False):
        key = (str(row.scenario_id), int(row.seed), int(row.round))
        if key not in social_lookup or str(row.actor_class).lower() != "investor":
            continue
        actor = agent_index.get(int(row.actor_agent_id)) if pd.notna(row.actor_agent_id) else None
        if actor is None:
            continue
        target_class = str(row.target_class).lower()
        if target_class == "source":
            target = n
        elif target_class == "investor" and pd.notna(row.target_agent_id):
            target = agent_index.get(int(row.target_agent_id))
            if target is None:
                continue
        else:
            continue
        action_name = str(row.action_type).strip().lower()
        if action_name not in action_categories:
            continue
        h = social_lookup[key]
        edge_action[h, actor, target, action_categories[action_name]] += 1.0
        if target < n:
            adjacency[h, actor, target] += 1.0
    panel_action = np.zeros((h_count, n), dtype=float)
    panel_content = np.zeros((h_count, n), dtype=float)
    for row in panel.itertuples(index=False):
        key = (str(row.scenario_id), int(row.seed), int(row.round))
        if key not in social_lookup:
            continue
        agent = agent_index.get(int(row.agent_id))
        if agent is None:
            continue
        h = social_lookup[key]
        panel_action[h, agent] = float(row.action_count) if pd.notna(row.action_count) else 0.0
        panel_content[h, agent] = float(row.authored_content_count) if pd.notna(row.authored_content_count) else 0.0
    # Full-group social targets.
    full_stance = stance_pair.sum(axis=(1, 2))
    full_score = np.divide(score_pair_sum.sum(axis=(1, 2)), score_pair_count.sum(axis=(1, 2)), out=np.zeros(h_count), where=score_pair_count.sum(axis=(1, 2)) > EPS)
    full_score_quantiles = histogram_quantiles(score_hist_pair.sum(axis=(1, 2)))
    full_interact_rate = np.divide(interacted_pair.sum(axis=(1, 2)), score_pair_count.sum(axis=(1, 2)), out=np.zeros(h_count), where=score_pair_count.sum(axis=(1, 2)) > EPS)
    full_source_reach = source_seen.mean(axis=1)
    full_participation = (panel_action > 0).mean(axis=1)
    full_content_rate = panel_content.mean(axis=1)
    full_action = edge_action.sum(axis=(1, 2))
    graph_features = np.concatenate(
        [
            adjacency.sum(axis=2).T,
            adjacency.sum(axis=1).T,
            source_seen.sum(axis=0, keepdims=True).T,
        ],
        axis=1,
    )
    graph_features = graph_features / np.maximum(np.linalg.norm(graph_features, axis=1, keepdims=True), EPS)
    similarity = graph_features @ graph_features.T
    pair_nmi = np.zeros((n, n), dtype=float)
    pair_dir = np.zeros((n, n), dtype=float)
    pair_path = paths["pair_redundancy"]
    pair_frame = pd.read_csv(pair_path)
    for row in pair_frame.itertuples(index=False):
        i, j = agent_index.get(int(row.agent_i)), agent_index.get(int(row.agent_j))
        if i is None or j is None:
            continue
        pair_nmi[i, j] = pair_nmi[j, i] = float(row.normalized_mutual_information)
        pair_dir[i, j] = pair_dir[j, i] = float(row.direction_agreement_rate)
    loo_frame = pd.read_csv(paths["loo"])
    loo_by_agent = np.zeros(n, dtype=float)
    for row in loo_frame.itertuples(index=False):
        if int(row.agent_id) in agent_index:
            loo_by_agent[agent_index[int(row.agent_id)]] = float(row.mean_leave_one_out_group_js_bits)
    mi_frame = pd.read_csv(paths["contributions"])
    mi_by_agent = np.zeros(n, dtype=float)
    if "predictive_incremental_cmi_bias_corrected_bits" in mi_frame.columns:
        for row in mi_frame.itertuples(index=False):
            if int(row.agent_id) in agent_index:
                value = getattr(row, "predictive_incremental_cmi_bias_corrected_bits")
                mi_by_agent[agent_index[int(row.agent_id)]] = float(value) if pd.notna(value) else 0.0
    return {
        "paths": paths,
        "agent_ids": agent_ids,
        "scenarios": scenarios,
        "seeds": seeds,
        "group_keys": group_keys,
        "social_keys": social_keys,
        "scenario_index": scenario_index,
        "seed_index": seed_index,
        "profile": profile,
        "profile_categories": profile_categories,
        "probs": probs,
        "expected": expected,
        "confidence": confidence,
        "valid": valid,
        "target_probs": target_probs,
        "target_expected": target_expected,
        "target_confidence": target_confidence,
        "target_majority": target_majority,
        "stance_pair": stance_pair,
        "score_pair_sum": score_pair_sum,
        "score_pair_count": score_pair_count,
        "score_hist_pair": score_hist_pair,
        "interacted_pair": interacted_pair,
        "source_seen": source_seen,
        "source_count": source_count,
        "edge_action": edge_action,
        "adjacency": adjacency,
        "panel_action": panel_action,
        "panel_content": panel_content,
        "full_stance": full_stance,
        "full_score": full_score,
        "full_score_quantiles": full_score_quantiles,
        "full_interact_rate": full_interact_rate,
        "full_source_reach": full_source_reach,
        "full_participation": full_participation,
        "full_content_rate": full_content_rate,
        "full_action": full_action,
        "graph_features": graph_features,
        "graph_similarity": similarity,
        "pair_nmi": pair_nmi,
        "pair_dir": pair_dir,
        "loo_by_agent": loo_by_agent,
        "mi_by_agent": mi_by_agent,
        "invalid_observations": [
            {"seed": seed, "scenario_id": scenario, "round": round_number, "agent_id": agent, "reason": "fixed probability anomaly mask"}
            for seed, scenario, round_number, agent in sorted(INVALID_OBSERVATION)
        ],
    }


def threshold_from_noise(noise: float, epsilon: float, domain_cap: float) -> float:
    return float(min(domain_cap, (noise if np.isfinite(noise) else 0.0) + epsilon))


def compute_thresholds(
    data: dict[str, object],
    *,
    scenario_mask: np.ndarray | None = None,
    seed_mask: np.ndarray | None = None,
) -> dict[str, dict[str, float]]:
    scenarios = data["scenarios"]
    seeds = data["seeds"]
    full_s_count, full_seed_count = len(scenarios), len(seeds)
    if scenario_mask is None:
        scenario_mask = np.ones(full_s_count, dtype=bool)
    if seed_mask is None:
        seed_mask = np.ones(full_seed_count, dtype=bool)
    s_count, seed_count = int(scenario_mask.sum()), int(seed_mask.sum())
    if s_count < 2 or seed_count < 2:
        raise ValueError("threshold estimation requires at least two scenarios and seeds")
    probs = np.asarray(data["target_probs"]).reshape(
        full_s_count, full_seed_count, 7, 3
    )[scenario_mask][:, seed_mask].reshape(-1, 3)
    expected = np.asarray(data["target_expected"]).reshape(
        full_s_count, full_seed_count, 7
    )[scenario_mask][:, seed_mask].reshape(-1)
    majority = np.asarray(data["target_majority"]).reshape(
        full_s_count, full_seed_count, 7
    )[scenario_mask][:, seed_mask].reshape(-1)
    noise: dict[str, float] = {
        "belief_js": pairwise_noise(probs, s_count, seed_count, 7, "js"),
        "trajectory_return_error": pairwise_noise(expected[..., None], s_count, seed_count, 7, "abs"),
    }
    majority_cube = majority.reshape(s_count, seed_count, 7)
    majority_noise: list[float] = []
    majority_final_noise: list[float] = []
    for scenario in range(s_count):
        trajectory_pairs: list[float] = []
        final_pairs: list[float] = []
        for i in range(seed_count):
            for j in range(i + 1, seed_count):
                trajectory_pairs.append(
                    float(np.mean(majority_cube[scenario, i] != majority_cube[scenario, j]))
                )
                final_pairs.append(
                    float(majority_cube[scenario, i, 6] != majority_cube[scenario, j, 6])
                )
        majority_noise.append(float(np.mean(trajectory_pairs)))
        majority_final_noise.append(float(np.mean(final_pairs)))
    noise["majority_trajectory_error"] = float(np.quantile(majority_noise, 0.95))
    noise["majority_final_error"] = float(np.quantile(majority_final_noise, 0.95))
    def subset_social(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values)
        tail = values.shape[1:]
        reshaped = values.reshape(full_s_count, full_seed_count, 6, *tail)
        return reshaped[scenario_mask][:, seed_mask].reshape(-1, *tail)

    social_targets = {
        "social_stance_js": subset_social(np.asarray(data["full_stance"])),
        "social_stance_score_error": subset_social(np.asarray(data["full_score_quantiles"])),
        "social_action_js": subset_social(np.asarray(data["full_action"])),
        "social_content_rate_error": subset_social(np.asarray(data["full_content_rate"])[..., None]),
        "social_participation_error": subset_social(np.asarray(data["full_participation"])[..., None]),
        "social_interaction_rate_error": subset_social(np.asarray(data["full_interact_rate"])[..., None]),
        "social_source_reach_error": subset_social(np.asarray(data["full_source_reach"])[..., None]),
    }
    # For social targets, calculate pairwise seed noise at matching scenario/round.
    for name, values in social_targets.items():
        metric = "js" if name in {"social_stance_js", "social_action_js"} else "abs"
        noise[name] = pairwise_noise(values, s_count, seed_count, 6, metric)
    # Graph degree and active-pair noise are conservative, seed-pair based.
    adjacency = np.asarray(data["adjacency"])
    adj = adjacency.reshape(full_s_count, full_seed_count, 6, 10, 10)
    adj = adj[scenario_mask][:, seed_mask]
    strength_noise: list[float] = []
    pair_noise: list[float] = []
    for si in range(s_count):
        for r in range(6):
            for i in range(seed_count):
                for j in range(i + 1, seed_count):
                    a, b = adj[si, i, r], adj[si, j, r]
                    a_out, b_out = a.sum(axis=1), b.sum(axis=1)
                    a_in, b_in = a.sum(axis=0), b.sum(axis=0)
                    a_out = a_out / max(float(a_out.sum()), EPS)
                    b_out = b_out / max(float(b_out.sum()), EPS)
                    a_in = a_in / max(float(a_in.sum()), EPS)
                    b_in = b_in / max(float(b_in.sum()), EPS)
                    q = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
                    strength_noise.append(
                        0.5
                        * (
                            float(np.mean(np.abs(np.quantile(a_out, q) - np.quantile(b_out, q))))
                            + float(np.mean(np.abs(np.quantile(a_in, q) - np.quantile(b_in, q))))
                        )
                    )
                    a_rate = (a > 0).sum() / 90.0
                    b_rate = (b > 0).sum() / 90.0
                    pair_noise.append(abs(float(a_rate - b_rate)))
    noise["graph_strength_error"] = float(np.quantile(strength_noise, 0.95)) if strength_noise else 0.0
    noise["graph_active_pair_error"] = float(np.quantile(pair_noise, 0.95)) if pair_noise else 0.0
    noise["graph_facility_error"] = 0.0
    thresholds: dict[str, dict[str, float]] = {}
    for name, (epsilon, cap) in THRESHOLD_SPECS.items():
        thresholds[name] = {"noise_q95": float(noise.get(name, 0.0)), "epsilon": epsilon, "domain_cap": cap, "threshold": threshold_from_noise(noise.get(name, 0.0), epsilon, cap)}
    return thresholds


def aggregate_metric(values: np.ndarray, data: dict[str, object], *, rounds: int, scenario_mask: np.ndarray | None = None, final: bool = False) -> np.ndarray:
    scenario_count, seed_count = len(data["scenarios"]), len(data["seeds"])
    if final:
        return aggregate_final(values, scenario_count, seed_count, scenario_mask)
    return aggregate_p80_worst_p90(values, scenario_count, seed_count, rounds, scenario_mask)


def selected_pair_mask(mask: np.ndarray) -> np.ndarray:
    pair = mask[:, :, None] * mask[:, None, :]
    indices = np.arange(pair.shape[1])
    pair[:, indices, indices] = 0.0
    return pair


def candidate_metrics(
    data: dict[str, object],
    masks: np.ndarray,
    thresholds: dict[str, dict[str, float]],
    *,
    profile_constraints: bool,
    fold_thresholds: Sequence[dict[str, dict[str, float]]],
) -> pd.DataFrame:
    c_count, n = masks.shape
    g_count, h_count = len(data["group_keys"]), len(data["social_keys"])
    s_count, seed_count = len(data["scenarios"]), len(data["seeds"])
    mask_float = masks.astype(float)
    probs = np.nan_to_num(np.asarray(data["probs"]), nan=0.0)
    valid = np.asarray(data["valid"], dtype=float)
    valid_count = mask_float @ valid.T
    invalid_group = list(data["group_keys"]).index(("SCN_006", 4004, 0))
    no_candidate_observation = valid_count <= 0
    cand_probs = np.einsum("ca,gad->cgd", mask_float, probs)
    cand_probs = np.divide(cand_probs, valid_count[..., None], out=np.zeros_like(cand_probs), where=valid_count[..., None] > EPS)
    cand_probs[no_candidate_observation] = np.nan
    target_probs = np.asarray(data["target_probs"])
    belief_js = js_divergence(cand_probs, target_probs[None, :, :])
    cand_expected = np.divide(mask_float @ np.nan_to_num(np.asarray(data["expected"]), nan=0.0).T, valid_count, out=np.zeros_like(valid_count), where=valid_count > EPS)
    cand_conf = np.divide(mask_float @ np.nan_to_num(np.asarray(data["confidence"]), nan=0.0).T, valid_count, out=np.zeros_like(valid_count), where=valid_count > EPS)
    cand_expected[no_candidate_observation] = np.nan
    cand_conf[no_candidate_observation] = np.nan
    expected_error = np.abs(cand_expected - np.asarray(data["target_expected"])[None, :])
    confidence_error = np.abs(cand_conf - np.asarray(data["target_confidence"])[None, :])
    cand_majority = tie_aware_majority(cand_probs)
    target_majority = np.asarray(data["target_majority"])
    direction_error = np.where((cand_majority == -1) | (target_majority[None, :] == -1), (cand_majority != target_majority[None, :]).astype(float), (cand_majority != target_majority[None, :]).astype(float))
    direction_error[no_candidate_observation] = np.nan
    social_pair = selected_pair_mask(mask_float)
    stance_pair = np.asarray(data["stance_pair"])
    cand_stance = np.einsum("cij,hijq->chq", social_pair, stance_pair)
    full_stance = np.asarray(data["full_stance"])
    stance_js = js_divergence(cand_stance, full_stance[None, :, :])
    score_sum = np.einsum("cij,hij->ch", social_pair, np.asarray(data["score_pair_sum"]))
    score_count = np.einsum("cij,hij->ch", social_pair, np.asarray(data["score_pair_count"]))
    cand_score = np.divide(score_sum, score_count, out=np.zeros_like(score_sum), where=score_count > EPS)
    cand_score_hist = np.einsum(
        "cij,hijq->chq",
        social_pair,
        np.asarray(data["score_hist_pair"]),
    )
    cand_score_quantiles = histogram_quantiles(cand_score_hist)
    score_error = np.mean(
        np.abs(cand_score_quantiles - np.asarray(data["full_score_quantiles"])[None, :, :]),
        axis=-1,
    )
    cand_interacted = np.einsum("cij,hij->ch", social_pair, np.asarray(data["interacted_pair"]))
    cand_interact_rate = np.divide(cand_interacted, score_count, out=np.zeros_like(cand_interacted), where=score_count > EPS)
    interact_error = np.abs(cand_interact_rate - np.asarray(data["full_interact_rate"])[None, :])
    source_reach = (np.asarray(data["source_seen"])[None, :, :] * mask_float[:, None, :]).sum(axis=2) / np.maximum(masks.sum(axis=1)[:, None], 1)
    source_error = np.abs(source_reach - np.asarray(data["full_source_reach"])[None, :])
    action_pair = np.asarray(data["edge_action"])
    target_mask = np.concatenate([mask_float, np.ones((c_count, 1))], axis=1)
    cand_action = np.empty((c_count, h_count, action_pair.shape[-1]), dtype=float)
    cand_adj = np.empty((c_count, h_count, n, n), dtype=float)
    graph_strength = np.empty((c_count, h_count), dtype=float)
    graph_pair_error = np.empty((c_count, h_count), dtype=float)
    full_adj = np.asarray(data["adjacency"])
    for ci in range(c_count):
        pair3 = mask_float[ci, :, None] * target_mask[ci, None, :]
        cand_action[ci] = np.einsum("ij,hija->ha", pair3, action_pair)
        pair2 = social_pair[ci]
        cand_adj[ci] = full_adj * pair2
        for hi in range(h_count):
            selected = np.flatnonzero(masks[ci])
            full_matrix = full_adj[hi]
            matrix = cand_adj[ci, hi]
            def shares(values: np.ndarray, ids: np.ndarray | None = None) -> np.ndarray:
                values = values if ids is None else values[ids]
                total = values.sum()
                return values / total if total > EPS else np.zeros_like(values)
            in_full, out_full = shares(full_matrix.sum(axis=0)), shares(full_matrix.sum(axis=1))
            in_c = shares(matrix.sum(axis=0), selected)
            out_c = shares(matrix.sum(axis=1), selected)
            q = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
            graph_strength[ci, hi] = 0.5 * (np.mean(np.abs(np.quantile(in_c, q) - np.quantile(in_full, q))) + np.mean(np.abs(np.quantile(out_c, q) - np.quantile(out_full, q))))
            active_full = float((full_matrix > 0).sum() / 90.0)
            active_c = float((matrix > 0).sum() / max(1, int(masks[ci].sum() * (masks[ci].sum() - 1))))
            graph_pair_error[ci, hi] = abs(active_c - active_full)
    full_action = np.asarray(data["full_action"])
    action_js = js_divergence(cand_action, full_action[None, :, :])
    cand_participation = ((np.asarray(data["panel_action"])[None, :, :] * mask_float[:, None, :]) > 0).sum(axis=2) / np.maximum(masks.sum(axis=1)[:, None], 1)
    participation_error = np.abs(cand_participation - np.asarray(data["full_participation"])[None, :])
    cand_content = (np.asarray(data["panel_content"])[None, :, :] * mask_float[:, None, :]).sum(axis=2) / np.maximum(masks.sum(axis=1)[:, None], 1)
    content_error = np.abs(cand_content - np.asarray(data["full_content_rate"])[None, :]) / np.maximum(1.0, np.abs(np.asarray(data["full_content_rate"])[None, :]))
    # Global graph facility-location coverage is a soft representation signal
    # and is also included in the graph hard gate with a predeclared threshold.
    similarity = np.asarray(data["graph_similarity"])
    facility = np.array([1.0 - float(np.max(similarity[:, np.flatnonzero(row)], axis=1).mean()) for row in masks], dtype=float)
    values: dict[str, np.ndarray] = {}
    values["belief_js"] = aggregate_metric(belief_js, data, rounds=7)
    values["trajectory_return_error"] = aggregate_metric(expected_error, data, rounds=7)
    values["confidence_error"] = aggregate_metric(confidence_error, data, rounds=7)
    values["majority_trajectory_error"] = aggregate_direction_agreement(
        direction_error,
        s_count,
        seed_count,
        final=False,
    )
    values["majority_final_error"] = aggregate_direction_agreement(
        direction_error[:, 6::7],
        s_count,
        seed_count,
        final=True,
    )
    social_raw = {
        "social_stance_js": stance_js,
        "social_stance_score_error": score_error,
        "social_action_js": action_js,
        "social_content_rate_error": content_error,
        "social_participation_error": participation_error,
        "social_interaction_rate_error": interact_error,
        "social_source_reach_error": source_error,
    }
    for name, raw in social_raw.items():
        values[name] = aggregate_metric(raw, data, rounds=6)
    values["graph_strength_error"] = aggregate_metric(graph_strength, data, rounds=6)
    values["graph_active_pair_error"] = aggregate_metric(graph_pair_error, data, rounds=6)
    values["graph_facility_error"] = facility
    result = pd.DataFrame({name: value for name, value in values.items()})
    result["social_error"] = np.maximum.reduce([result[name] / thresholds[name]["threshold"] for name in social_raw])
    result["graph_error"] = np.maximum.reduce([
        result["graph_strength_error"] / thresholds["graph_strength_error"]["threshold"],
        result["graph_active_pair_error"] / thresholds["graph_active_pair_error"]["threshold"],
        result["social_source_reach_error"] / thresholds["social_source_reach_error"]["threshold"],
        result["graph_facility_error"] / thresholds["graph_facility_error"]["threshold"],
    ])
    result["core_error"] = np.maximum.reduce([
        result["belief_js"] / thresholds["belief_js"]["threshold"],
        result["trajectory_return_error"] / thresholds["trajectory_return_error"]["threshold"],
        result["majority_trajectory_error"] / thresholds["majority_trajectory_error"]["threshold"],
        result["majority_final_error"] / thresholds["majority_final_error"]["threshold"],
        result["social_error"],
        result["graph_error"],
    ])
    # Sensitivity: omit the entire scenario-seed-round unit containing the one
    # malformed observation, while retaining the frozen thresholds.
    belief_sensitivity = belief_js.copy()
    return_sensitivity = expected_error.copy()
    direction_sensitivity = direction_error.copy()
    belief_sensitivity[:, invalid_group] = np.nan
    return_sensitivity[:, invalid_group] = np.nan
    direction_sensitivity[:, invalid_group] = np.nan
    sensitivity_belief = aggregate_metric(belief_sensitivity, data, rounds=7)
    sensitivity_return = aggregate_metric(return_sensitivity, data, rounds=7)
    sensitivity_direction = aggregate_direction_agreement(
        direction_sensitivity,
        s_count,
        seed_count,
        final=False,
    )
    sensitivity_final = result["majority_final_error"].to_numpy(float)
    result["core_error_drop_invalid_unit"] = np.maximum.reduce([
        sensitivity_belief / thresholds["belief_js"]["threshold"],
        sensitivity_return / thresholds["trajectory_return_error"]["threshold"],
        sensitivity_direction / thresholds["majority_trajectory_error"]["threshold"],
        sensitivity_final / thresholds["majority_final_error"]["threshold"],
        result["social_error"].to_numpy(float),
        result["graph_error"].to_numpy(float),
    ])
    fold_core_columns: list[str] = []
    for fold, current_thresholds in enumerate(fold_thresholds):
        holdout = np.arange(s_count) % len(fold_thresholds) == fold
        fold_belief = aggregate_metric(
            belief_js, data, rounds=7, scenario_mask=holdout
        )
        fold_return = aggregate_metric(
            expected_error, data, rounds=7, scenario_mask=holdout
        )
        fold_direction = aggregate_direction_agreement(
            direction_error,
            s_count,
            seed_count,
            final=False,
            scenario_mask=holdout,
        )
        fold_final = aggregate_direction_agreement(
            direction_error[:, 6::7],
            s_count,
            seed_count,
            final=True,
            scenario_mask=holdout,
        )
        fold_social_components = [
            aggregate_metric(raw, data, rounds=6, scenario_mask=holdout)
            / current_thresholds[name]["threshold"]
            for name, raw in social_raw.items()
        ]
        fold_social = np.maximum.reduce(fold_social_components)
        fold_graph = np.maximum.reduce([
            aggregate_metric(
                graph_strength, data, rounds=6, scenario_mask=holdout
            )
            / current_thresholds["graph_strength_error"]["threshold"],
            aggregate_metric(
                graph_pair_error, data, rounds=6, scenario_mask=holdout
            )
            / current_thresholds["graph_active_pair_error"]["threshold"],
            aggregate_metric(
                source_error, data, rounds=6, scenario_mask=holdout
            )
            / current_thresholds["social_source_reach_error"]["threshold"],
            facility / current_thresholds["graph_facility_error"]["threshold"],
        ])
        fold_core = np.maximum.reduce([
            fold_belief / current_thresholds["belief_js"]["threshold"],
            fold_return / current_thresholds["trajectory_return_error"]["threshold"],
            fold_direction
            / current_thresholds["majority_trajectory_error"]["threshold"],
            fold_final / current_thresholds["majority_final_error"]["threshold"],
            fold_social,
            fold_graph,
        ])
        result[f"fold_{fold}_belief_js"] = fold_belief
        result[f"fold_{fold}_social_error"] = fold_social
        result[f"fold_{fold}_graph_error"] = fold_graph
        result[f"fold_{fold}_core_error"] = fold_core
        fold_core_columns.append(f"fold_{fold}_core_error")
    result["fold_core_error_worst"] = result[fold_core_columns].max(axis=1)
    result["fold_core_error_mean"] = result[fold_core_columns].mean(axis=1)
    allowed_missing = np.zeros(g_count, dtype=bool)
    allowed_missing[invalid_group] = True
    result["data_constraint"] = np.all(
        (~no_candidate_observation) | allowed_missing[None, :],
        axis=1,
    )
    categories = np.array([data["profile"][int(a)]["agent_role_category"] for a in data["agent_ids"]], dtype=object)
    role_coverage = np.array(
        [set(categories[row]) >= set(ROLE_CATEGORIES) for row in masks],
        dtype=bool,
    )
    result["role_constraint"] = role_coverage if profile_constraints else True
    gate_columns = {
        "belief": result["belief_js"] <= thresholds["belief_js"]["threshold"] + 1e-9,
        "majority_final": result["majority_final_error"] <= thresholds["majority_final_error"]["threshold"] + 1e-9,
        "majority_trajectory": result["majority_trajectory_error"] <= thresholds["majority_trajectory_error"]["threshold"] + 1e-9,
        "trajectory_return": result["trajectory_return_error"] <= thresholds["trajectory_return_error"]["threshold"] + 1e-9,
        "social": result["social_error"] <= 1.0 + 1e-9,
        "graph": result["graph_error"] <= 1.0 + 1e-9,
    }
    for name, gate in gate_columns.items():
        result[f"gate_{name}_pass"] = gate
    result["failed_constraints"] = [
        "|".join(name for name, gate in gate_columns.items() if not bool(gate.iloc[index]))
        for index in range(len(result))
    ]
    result["hard_constraints_pass"] = result["data_constraint"] & result["role_constraint"] & (result["core_error"] <= 1.0 + 1e-9)
    result["hard_constraints_pass_drop_invalid_unit"] = (
        result["data_constraint"]
        & result["role_constraint"]
        & (result["core_error_drop_invalid_unit"] <= 1.0 + 1e-9)
    )
    for fold in range(len(fold_thresholds)):
        result[f"fold_{fold}_hard_constraints_pass"] = (
            result["data_constraint"]
            & result["role_constraint"]
            & (result[f"fold_{fold}_core_error"] <= 1.0 + 1e-9)
        )
    result["fold_pass_count"] = result[
        [f"fold_{fold}_hard_constraints_pass" for fold in range(len(fold_thresholds))]
    ].sum(axis=1)
    # Explanatory soft fields, intentionally not combined into a final score.
    pair_similarity = np.asarray(data["pair_nmi"]) * np.asarray(data["pair_dir"])
    loo = np.asarray(data["loo_by_agent"])
    mi = np.asarray(data["mi_by_agent"])
    redundancy: list[float] = []
    for row in masks:
        selected = np.flatnonzero(row)
        block = pair_similarity[np.ix_(selected, selected)]
        redundancy.append(
            float(block[np.triu_indices(len(selected), 1)].mean())
            if len(selected) > 1
            else 0.0
        )
    result["mean_selected_pair_redundancy"] = redundancy
    result["mean_selected_loo_js_bits"] = masks @ loo / np.maximum(masks.sum(axis=1), 1)
    result["mean_selected_incremental_cmi_bits"] = masks @ mi / np.maximum(masks.sum(axis=1), 1)
    result["graph_facility_coverage"] = 1.0 - result["graph_facility_error"]
    result["agent_ids"] = ["|".join(str(int(data["agent_ids"][i])) for i in np.flatnonzero(row)) for row in masks]
    result["subset_mask"] = [int(sum(1 << int(data["agent_ids"][i]) for i in np.flatnonzero(row))) for row in masks]
    result["k"] = masks.sum(axis=1).astype(int)
    return result


def pareto_mask(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    if frame.empty:
        return np.zeros(0, dtype=bool)
    values = frame.loc[:, columns].to_numpy(float)
    result = np.ones(len(values), dtype=bool)
    for i, point in enumerate(values):
        others = np.delete(values, i, axis=0)
        if len(others) == 0:
            continue
        dominated = np.all(others <= point, axis=1) & np.any(others < point, axis=1)
        result[i] = not bool(dominated.any())
    return result


def build_fold_columns(result: pd.DataFrame, data: dict[str, object], thresholds: dict[str, dict[str, float]], masks: np.ndarray) -> pd.DataFrame:
    # Fold reports are calculated from the same per-group errors in a separate
    # function in the full implementation. Keep the output explicit and avoid
    # treating the 90 scenario-seed rows as independent test samples.
    scenarios = np.array(data["scenarios"])
    for fold in range(3):
        result[f"fold_{fold}_scenario_count"] = int(np.sum(np.arange(len(scenarios)) % 3 == fold))
    return result


def write_readme(output_dir: Path, summary: dict[str, object]) -> None:
    text = f"""# K=10 Agent 子集离线枚举（修订方案）

本目录由 `{ANALYSIS_VERSION}` 生成，参考系统为 10 个 Agent、5 个 seed、18 个场景。
主实验枚举 K=4..9 的全部 847 个组合；Profile 消融可单独使用 `--no-profile-constraints --k-values 1 ... 9`。

## 硬约束

1. Profile-ID 唯一、场景×seed×轮次键完整、字段可计算。已固定屏蔽
   `seed=4004, SCN_006, round=0, Agent=7` 的一条无效概率观测；不会删除 Agent 7。
2. 至少包含四类角色：institution、retail_mature、retail_basic、retail_novice。
3. BeliefJS、最终/轨迹多数方向、预期收益轨迹、社会过程和诱导图误差均不超过
   `thresholds.json` 中的门槛。连续误差的聚合顺序为 seed P80 → 场景最坏轮次 →
   场景 P90；多数方向是离散一致率，轨迹分母为场景-seed-round，最终分母为
   场景-seed，再对场景误差取 P90。

社会误差和图误差均为各分量归一化后的最大值，分量之间不互相抵消。社区划分没有作为约束。

## 诱导子图口径

候选 Agent 发出的内容和指向未入选投资者的边被删除；信息源内容和指向信息源的边保留；
曝光机会、社会互动率、动作分布和节点强度均在过滤后重新计算。

## 输出

- `scores_kXX.csv.gz`：该 K 的全部组合及硬约束、Pareto 辅助字段；
- `top_candidates.csv`：每个 K 的可行候选（按最坏核心误差、稳定性、图覆盖等字典序）；
- `pareto_candidates.csv`：同一 K 内的多目标 Pareto 前沿；
- `k_summary.csv`：每个 K 的组合数、可行数和最优核心误差；
- `thresholds.json`：冻结的噪声 Q95、epsilon、领域上限和最终阈值；
- `fold_thresholds.json`：每个场景 holdout 折仅用另外 12 个训练场景估计的阈值；
- `leave_one_seed_out_thresholds.json`：删除各 seed 后的阈值敏感性；
- `summary.json`：输入哈希、配置、质量检查和指标定义。

每个候选还包含删除完整异常单元后的 `core_error_drop_invalid_unit` 与敏感性可行标记。

离线删除日志行不能替代真实缩减重跑，最终仍需重跑 K*、K*+1、同规模随机和 Graph-only 基线。
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.top_per_k <= 0 or not args.k_values:
        raise ValueError("top-per-k and k-values must be positive")
    if len(set(args.k_values)) != len(args.k_values):
        raise ValueError("duplicate K values")
    if min(args.k_values) < 1 or max(args.k_values) >= AGENT_COUNT_EXPECTED:
        raise ValueError("K values must be in [1, 9]")
    started = time.perf_counter()
    data = load_data(args.dataset_dir)
    thresholds = compute_thresholds(data)
    scenario_positions = np.arange(len(data["scenarios"]))
    fold_thresholds = [
        compute_thresholds(
            data,
            scenario_mask=(scenario_positions % 3 != holdout_fold),
        )
        for holdout_fold in range(3)
    ]
    seed_positions = np.arange(len(data["seeds"]))
    leave_one_seed_out_thresholds = {
        str(int(data["seeds"][holdout_seed])): compute_thresholds(
            data,
            seed_mask=(seed_positions != holdout_seed),
        )
        for holdout_seed in range(len(data["seeds"]))
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_status.json").write_text(json.dumps({"status": "running", "analysis_version": ANALYSIS_VERSION, "started_epoch_seconds": time.time()}, indent=2), encoding="utf-8")
    (args.output_dir / "thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    (args.output_dir / "fold_thresholds.json").write_text(
        json.dumps(
            {
                str(fold): {
                    "training_scenarios": [
                        scenario
                        for index, scenario in enumerate(data["scenarios"])
                        if index % 3 != fold
                    ],
                    "holdout_scenarios": [
                        scenario
                        for index, scenario in enumerate(data["scenarios"])
                        if index % 3 == fold
                    ],
                    "thresholds": fold_thresholds[fold],
                }
                for fold in range(3)
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (args.output_dir / "leave_one_seed_out_thresholds.json").write_text(
        json.dumps(leave_one_seed_out_thresholds, indent=2),
        encoding="utf-8",
    )
    all_frames: list[pd.DataFrame] = []
    k_summaries: list[dict[str, object]] = []
    for k in tqdm(args.k_values, desc="K values", unit="K", disable=not args.progress):
        combinations = np.asarray(list(itertools.combinations(range(len(data["agent_ids"])), int(k))), dtype=int)
        masks = np.zeros((len(combinations), len(data["agent_ids"])), dtype=bool)
        masks[np.arange(len(combinations))[:, None], combinations] = True
        if args.profile_constraints:
            categories = np.array([data["profile"][int(a)]["agent_role_category"] for a in data["agent_ids"]], dtype=object)
            role_mask = np.array([set(categories[row]) >= set(ROLE_CATEGORIES) for row in masks], dtype=bool)
            masks_eval = masks[role_mask]
        else:
            role_mask = np.ones(len(masks), dtype=bool)
            masks_eval = masks
        score_path = args.output_dir / f"scores_k{int(k):02d}.csv.gz"
        if args.resume and score_path.exists():
            frame = pd.read_csv(score_path)
        else:
            frame = candidate_metrics(
                data,
                masks_eval,
                thresholds,
                profile_constraints=bool(args.profile_constraints),
                fold_thresholds=fold_thresholds,
            )
        if len(frame):
            frame["role_constraints_enabled"] = bool(args.profile_constraints)
            pareto = pareto_mask(frame[frame["hard_constraints_pass"]], ["belief_js", "social_error", "graph_error", "core_error"])
            frame["pareto_front"] = False
            feasible_indices = frame.index[frame["hard_constraints_pass"]].to_numpy()
            if len(feasible_indices):
                frame.loc[feasible_indices, "pareto_front"] = pareto
            # Lexicographic rule, not a weighted total score.
            frame["selection_rank"] = np.nan
            ordered = frame[frame["hard_constraints_pass"]].sort_values(
                [
                    "core_error",
                    "fold_core_error_worst",
                    "graph_error",
                    "mean_selected_pair_redundancy",
                    "mean_selected_loo_js_bits",
                ],
                ascending=[True, True, True, True, False],
            )
            frame.loc[ordered.index, "selection_rank"] = np.arange(1, len(ordered) + 1)
        if args.write_all_scores:
            frame.to_csv(
                score_path,
                index=False,
                compression="gzip",
                float_format="%.10g",
            )
        feasible = frame[frame["hard_constraints_pass"]].sort_values(
            "selection_rank"
        ).copy()
        top = feasible.nsmallest(args.top_per_k, "selection_rank") if len(feasible) else frame.head(0)
        top = top.copy()
        top["k"] = int(k)
        all_frames.append(top)
        pareto_frame = frame[frame["pareto_front"]].copy()
        pareto_frame.to_csv(args.output_dir / f"pareto_k{int(k):02d}.csv", index=False)
        k_summaries.append({
            "k": int(k),
            "combination_count": int(math.comb(len(data["agent_ids"]), int(k))),
            "role_feasible_count": int(len(masks_eval)),
            "hard_feasible_count": int(len(feasible)),
            "pareto_count": int(len(pareto_frame)),
            "best_agent_ids": str(feasible.iloc[0]["agent_ids"]) if len(feasible) else None,
            "best_core_error": float(feasible.iloc[0]["core_error"]) if len(feasible) else None,
            "best_social_error": float(feasible.iloc[0]["social_error"]) if len(feasible) else None,
            "best_graph_error": float(feasible.iloc[0]["graph_error"]) if len(feasible) else None,
            "best_fold_pass_count": int(feasible.iloc[0]["fold_pass_count"]) if len(feasible) else None,
            "sensitivity_feasible_count": int(frame["hard_constraints_pass_drop_invalid_unit"].sum()),
        })
    top_candidates = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    top_candidates.to_csv(args.output_dir / "top_candidates.csv", index=False)
    pareto_candidates = []
    for k in args.k_values:
        path = args.output_dir / f"pareto_k{int(k):02d}.csv"
        if path.exists():
            part = pd.read_csv(path)
            if not part.empty:
                pareto_candidates.append(part)
    if pareto_candidates:
        pd.concat(pareto_candidates, ignore_index=True).to_csv(args.output_dir / "pareto_candidates.csv", index=False)
    pd.DataFrame(k_summaries).to_csv(args.output_dir / "k_summary.csv", index=False)
    summary = {
        "analysis_version": ANALYSIS_VERSION,
        "configuration": {"dataset_dir": str(args.dataset_dir.resolve()), "output_dir": str(args.output_dir.resolve()), "k_values": list(map(int, args.k_values)), "profile_constraints": bool(args.profile_constraints), "top_per_k": int(args.top_per_k), "write_all_scores": bool(args.write_all_scores), "resume": bool(args.resume), "tie_epsilon": TIE_EPSILON, "seed_p80": 0.80, "scenario_p90": 0.90, "scenario_worst_round": True},
        "data": {"agent_ids": [int(a) for a in data["agent_ids"]], "agent_count": len(data["agent_ids"]), "scenario_count": len(data["scenarios"]), "seed_count": len(data["seeds"]), "scenarios": list(data["scenarios"]), "seeds": list(map(int, data["seeds"])), "invalid_observations_screened": data["invalid_observations"]},
        "thresholds": thresholds,
        "fold_thresholds": fold_thresholds,
        "leave_one_seed_out_thresholds": leave_one_seed_out_thresholds,
        "k_results": k_summaries,
        "input_files": {name: {"path": str(path.resolve()), "sha256": sha256_file(path)} for name, path in data["paths"].items()},
        "metric_definitions": {"belief_js": "candidate mean probability vs full K=10 mean", "majority_final_error": "scenario P90 of tie-aware mismatch rate over scenario-seed units at round 6", "majority_trajectory_error": "scenario P90 of tie-aware mismatch rate over scenario-seed-round units for rounds 0-6", "social_error": "max of normalized stance/action/content/participation/interaction/source components", "graph_error": "max of normalized induced-subgraph strength, active-pair, and facility-location errors"},
        "selection_rule": "minimum feasible K; within K Pareto front then lexicographic core_error, graph_error, redundancy, LOO JS; MI is explanatory only",
        "offline_warning": "Offline induced-subgraph screening is not a substitute for reduced-Agent simulation reruns.",
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    write_readme(args.output_dir, summary)
    (args.output_dir / "run_status.json").write_text(json.dumps({"status": "complete", "analysis_version": ANALYSIS_VERSION, "completed_epoch_seconds": time.time(), "elapsed_seconds": summary["elapsed_seconds"]}, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "elapsed_seconds": summary["elapsed_seconds"], "k_results": k_summaries}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
