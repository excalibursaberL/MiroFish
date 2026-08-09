#!/usr/bin/env python3
"""Compare the preferred real K=8 rerun with the frozen K=10 reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_K10 = ROOT / "Dataset" / "s1_multiseed_k10_round6_seeds4004_42_3407_999_2887_v2"
DEFAULT_ENUM = ROOT / "Dataset" / "agent_subset_enumeration_k10_v2"
DEFAULT_OUTPUT = ROOT / "Dataset" / "s1_k8_rerun_validation_round6_seeds42_999_2887_3407_4004_v1"
SEEDS = (42, 999, 2887, 3407, 4004)
SCENARIOS = tuple(f"SCN_{index:03d}" for index in range(1, 19))
ROUNDS = tuple(range(7))
SOCIAL_ROUNDS = tuple(range(1, 7))
PROBABILITY_COLUMNS = ("up_probability", "neutral_probability", "down_probability")
DIRECTIONS = ("up", "neutral", "down")
STANCE_LABELS = ("positive", "mixed", "negative", "neutral", "uncertain")
ACTION_LABELS = (
    "create_comment", "create_post", "like_comment", "like_post",
    "dislike_comment", "dislike_post", "follow",
)
EXPECTED_FULL_IDS = (1, 3, 4, 5, 9, 13, 14, 17)
TIE_EPSILON = 0.02
EPSILON = 1e-12
STANCE_EDGES = np.linspace(-1.0, 1.0, 11)
STANCE_CENTERS = 0.5 * (STANCE_EDGES[:-1] + STANCE_EDGES[1:])
TABLES = {
    "belief": "belief_snapshots.csv",
    "panel": "agent_round_panel.csv",
    "exposure": "agent_round_content_exposures.csv",
    "edges": "interaction_edges.csv",
    "runs": "scenario_runs.csv",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k8-datasets", type=Path, nargs=5, required=True)
    parser.add_argument("--k10-dataset", type=Path, default=DEFAULT_K10)
    parser.add_argument("--enumeration-dir", type=Path, default=DEFAULT_ENUM)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--random-seed", type=int, default=20260809)
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=json_value) + "\n",
        encoding="utf-8",
    )


def json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def load_k8(paths: Sequence[Path]) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    merged: dict[str, list[pd.DataFrame]] = {name: [] for name in TABLES}
    sources: list[dict[str, Any]] = []
    found_seeds: list[int] = []
    for path in paths:
        path = path.resolve()
        summary_path = path / "analysis_summary.json"
        quality_path = path / "quality_report.json"
        if not summary_path.exists() or not quality_path.exists():
            raise FileNotFoundError(f"incomplete K=8 dataset: {path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
        seed = int(summary["design"]["random_seed"])
        if int(summary["design"]["agent_count_per_scenario"]) != 8:
            raise ValueError(f"expected K=8: {path}")
        if tuple(summary["design"].get("selected_full_population_agent_ids", [])) != EXPECTED_FULL_IDS:
            raise ValueError(f"unexpected Agent subset: {path}")
        if not quality.get("passed"):
            raise ValueError(f"K=8 source failed quality checks: {path}")
        found_seeds.append(seed)
        files = []
        for name, filename in TABLES.items():
            source = path / filename
            frame = pd.read_csv(source, low_memory=False)
            frame["seed"] = seed
            merged[name].append(frame)
            files.append({"name": filename, "bytes": source.stat().st_size, "sha256": sha256(source)})
        sources.append({"seed": seed, "path": str(path), "quality_passed": True, "files": files})
    if tuple(sorted(found_seeds)) != SEEDS:
        raise ValueError(f"expected seeds {SEEDS}, found {sorted(found_seeds)}")
    return {name: pd.concat(frames, ignore_index=True) for name, frames in merged.items()}, sources


def load_k10(path: Path) -> dict[str, pd.DataFrame]:
    path = path.resolve()
    result = {}
    for name, filename in TABLES.items():
        source = path / f"merged_{filename}"
        result[name] = pd.read_csv(source, low_memory=False)
    return result


def normalize_frames(frames: dict[str, pd.DataFrame]) -> None:
    for frame in frames.values():
        frame["scenario_id"] = frame["scenario_id"].astype(str)
        frame["seed"] = pd.to_numeric(frame["seed"], errors="raise").astype(int)
    for name in ("belief", "panel", "exposure", "edges"):
        frames[name]["round"] = pd.to_numeric(frames[name]["round"], errors="raise").astype(int)


def js_divergence(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    left_sum = left.sum(axis=-1, keepdims=True)
    right_sum = right.sum(axis=-1, keepdims=True)
    left = np.divide(left, left_sum, out=np.zeros_like(left), where=left_sum > EPSILON)
    right = np.divide(right, right_sum, out=np.zeros_like(right), where=right_sum > EPSILON)
    middle = 0.5 * (left + right)
    with np.errstate(divide="ignore", invalid="ignore"):
        left_term = np.where(left > 0, left * np.log2(np.maximum(left, EPSILON) / np.maximum(middle, EPSILON)), 0.0)
        right_term = np.where(right > 0, right * np.log2(np.maximum(right, EPSILON) / np.maximum(middle, EPSILON)), 0.0)
    return 0.5 * np.sum(left_term + right_term, axis=-1)


def tie_majority(probabilities: np.ndarray) -> np.ndarray:
    order = np.argsort(probabilities, axis=-1)
    top = np.take_along_axis(probabilities, order[..., -1:], axis=-1)[..., 0]
    second = np.take_along_axis(probabilities, order[..., -2:-1], axis=-1)[..., 0]
    result = order[..., -1].astype(np.int8)
    result[(top - second) <= TIE_EPSILON] = -1
    return result


def histogram_quantiles(histogram: np.ndarray) -> np.ndarray:
    cumulative = np.cumsum(histogram, axis=-1)
    total = cumulative[..., -1:]
    result = np.zeros(histogram.shape[:-1] + (3,), dtype=float)
    for index, quantile in enumerate((0.10, 0.50, 0.90)):
        target = total[..., 0] * quantile
        bucket = np.argmax(cumulative >= target[..., None], axis=-1)
        previous = np.where(
            bucket > 0,
            np.take_along_axis(cumulative, (bucket - 1)[..., None], axis=-1)[..., 0],
            0.0,
        )
        width = np.take_along_axis(histogram, bucket[..., None], axis=-1)[..., 0]
        fraction = np.divide(target - previous, width, out=np.zeros_like(target), where=width > EPSILON)
        result[..., index] = np.where(
            total[..., 0] > EPSILON,
            np.take(STANCE_CENTERS, bucket) + (fraction - 0.5) * (STANCE_EDGES[1] - STANCE_EDGES[0]),
            0.0,
        )
    return result


def aggregate(values: np.ndarray, round_count: int) -> float:
    cube = np.asarray(values, dtype=float).reshape(len(SCENARIOS), len(SEEDS), round_count)
    seed_p80 = np.nanquantile(cube, 0.80, axis=1)
    scenario_worst = np.nanmax(seed_p80, axis=1)
    return float(np.nanquantile(scenario_worst, 0.90))


def aggregate_direction(values: np.ndarray, *, final: bool) -> float:
    round_count = 1 if final else len(ROUNDS)
    cube = np.asarray(values, dtype=float).reshape(len(SCENARIOS), len(SEEDS), round_count)
    scenario_error = np.nanmean(cube, axis=(1, 2))
    return float(np.nanquantile(scenario_error, 0.90))


def ordered_group_keys(rounds: Sequence[int]) -> list[tuple[str, int, int]]:
    return [(scenario, seed, round_number) for scenario in SCENARIOS for seed in SEEDS for round_number in rounds]


def build_system(frames: dict[str, pd.DataFrame], agent_count: int) -> dict[str, Any]:
    belief = frames["belief"].copy()
    require_columns(belief, ["status", *PROBABILITY_COLUMNS, "expected_return", "confidence", "agent_id"], "belief")
    belief = belief[belief["round"].isin(ROUNDS)]
    for column in (*PROBABILITY_COLUMNS, "expected_return", "confidence"):
        belief[column] = pd.to_numeric(belief[column], errors="coerce")
    belief["agent_id"] = pd.to_numeric(belief["agent_id"], errors="raise").astype(int)
    valid = belief["status"].astype(str).str.lower().eq("ok")
    valid &= np.isclose(belief[list(PROBABILITY_COLUMNS)].sum(axis=1), 1.0, atol=1e-6)
    valid &= belief[[*PROBABILITY_COLUMNS, "expected_return", "confidence"]].notna().all(axis=1)
    belief["valid"] = valid
    valid_belief = belief[belief["valid"]]
    grouped = valid_belief.groupby(["scenario_id", "seed", "round"], sort=False)
    group_means = grouped[[*PROBABILITY_COLUMNS, "expected_return", "confidence"]].mean()
    group_keys = ordered_group_keys(ROUNDS)
    group_means = group_means.reindex(pd.MultiIndex.from_tuples(group_keys, names=["scenario_id", "seed", "round"]))
    if group_means.isna().any().any():
        raise ValueError("missing valid belief group")

    social_keys = ordered_group_keys(SOCIAL_ROUNDS)
    social_lookup = {key: index for index, key in enumerate(social_keys)}
    stance = np.zeros((len(social_keys), len(STANCE_LABELS)), dtype=float)
    score_hist = np.zeros((len(social_keys), len(STANCE_CENTERS)), dtype=float)
    interaction_numerator = np.zeros(len(social_keys), dtype=float)
    interaction_denominator = np.zeros(len(social_keys), dtype=float)
    source_viewers = [set() for _ in social_keys]
    exposure = frames["exposure"]
    for row in exposure.itertuples(index=False):
        key = (str(row.scenario_id), int(row.seed), int(row.round))
        index = social_lookup.get(key)
        if index is None:
            continue
        if str(row.author_class).lower() == "source":
            source_viewers[index].add(int(row.viewer_agent_id))
            continue
        if str(row.author_class).lower() != "investor" or bool(row.is_self_authored):
            continue
        label = str(row.content_stance).strip().lower()
        stance[index, STANCE_LABELS.index(label) if label in STANCE_LABELS else STANCE_LABELS.index("uncertain")] += 1
        score = float(row.stance_score) if pd.notna(row.stance_score) else 0.0
        bucket = int(np.searchsorted(STANCE_EDGES, np.clip(score, -1, 1), side="right") - 1)
        score_hist[index, min(max(bucket, 0), len(STANCE_CENTERS) - 1)] += 1
        interaction_denominator[index] += 1
        interaction_numerator[index] += str(row.interacted_any).lower() in {"true", "1", "yes"}

    panel = frames["panel"]
    panel_group = panel.groupby(["scenario_id", "seed", "round"], sort=False)
    participation_lookup = panel_group["action_count"].apply(lambda values: float((pd.to_numeric(values, errors="coerce").fillna(0) > 0).mean()))
    content_lookup = panel_group["authored_content_count"].apply(lambda values: float(pd.to_numeric(values, errors="coerce").fillna(0).mean()))
    social_index = pd.MultiIndex.from_tuples(social_keys, names=["scenario_id", "seed", "round"])
    participation = participation_lookup.reindex(social_index).to_numpy(float)
    content_rate = content_lookup.reindex(social_index).to_numpy(float)
    if np.isnan(participation).any() or np.isnan(content_rate).any():
        raise ValueError("missing social panel group")

    action = np.zeros((len(social_keys), len(ACTION_LABELS)), dtype=float)
    adjacency = np.zeros((len(social_keys), agent_count, agent_count), dtype=float)
    action_index = {name: index for index, name in enumerate(ACTION_LABELS)}
    for row in frames["edges"].itertuples(index=False):
        key = (str(row.scenario_id), int(row.seed), int(row.round))
        index = social_lookup.get(key)
        if index is None or str(row.actor_class).lower() != "investor":
            continue
        name = str(row.action_type).strip().lower()
        if name in action_index:
            action[index, action_index[name]] += 1
        if str(row.target_class).lower() == "investor":
            actor, target = int(row.actor_agent_id), int(row.target_agent_id)
            if 0 <= actor < agent_count and 0 <= target < agent_count:
                adjacency[index, actor, target] += 1

    return {
        "belief": belief,
        "probabilities": group_means[list(PROBABILITY_COLUMNS)].to_numpy(float),
        "expected_return": group_means["expected_return"].to_numpy(float),
        "confidence": group_means["confidence"].to_numpy(float),
        "majority": tie_majority(group_means[list(PROBABILITY_COLUMNS)].to_numpy(float)),
        "stance": stance,
        "score_quantiles": histogram_quantiles(score_hist),
        "interaction_rate": np.divide(interaction_numerator, interaction_denominator, out=np.zeros_like(interaction_numerator), where=interaction_denominator > 0),
        "source_reach": np.array([len(viewers) / agent_count for viewers in source_viewers]),
        "participation": participation,
        "content_rate": content_rate,
        "action": action,
        "adjacency": adjacency,
        "valid_snapshot_count": int(valid.sum()),
        "snapshot_count": int(len(belief)),
    }


def graph_errors(
    left: np.ndarray,
    right: np.ndarray,
    reference_selected_indices: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    strength = np.zeros(len(left), dtype=float)
    active = np.zeros(len(left), dtype=float)
    quantiles = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    selected_indices = (
        list(range(left.shape[1]))
        if reference_selected_indices is None
        else [int(value) for value in reference_selected_indices]
    )
    for index, (candidate, reference) in enumerate(zip(left, right)):
        def share(values: np.ndarray) -> np.ndarray:
            total = values.sum()
            return values / total if total > EPSILON else np.zeros_like(values)
        candidate_in, candidate_out = share(candidate.sum(axis=0)), share(candidate.sum(axis=1))
        reference_in, reference_out = share(reference.sum(axis=0)), share(reference.sum(axis=1))
        strength[index] = 0.5 * (
            np.mean(np.abs(np.quantile(candidate_in, quantiles) - np.quantile(reference_in[selected_indices], quantiles)))
            + np.mean(np.abs(np.quantile(candidate_out, quantiles) - np.quantile(reference_out[selected_indices], quantiles)))
        )
        candidate_density = float((candidate > 0).sum() / max(candidate.shape[0] * (candidate.shape[0] - 1), 1))
        reference_density = float((reference > 0).sum() / max(reference.shape[0] * (reference.shape[0] - 1), 1))
        active[index] = abs(candidate_density - reference_density)
    return strength, active


def actual_labels(frames: dict[str, pd.DataFrame]) -> dict[str, str]:
    values = frames["runs"][["scenario_id", "actual_five_day_close_direction"]].drop_duplicates()
    grouped = values.groupby("scenario_id")["actual_five_day_close_direction"].nunique()
    if (grouped != 1).any():
        raise ValueError("conflicting actual labels")
    return dict(zip(values["scenario_id"].astype(str), values["actual_five_day_close_direction"].astype(str)))


def forecast_metrics(system: dict[str, Any], labels: dict[str, str], scenarios: Sequence[str] | None = None) -> dict[str, float]:
    frame = system["belief"]
    frame = frame[(frame["round"] == 6) & frame["valid"]].copy()
    if scenarios is not None:
        frame = frame[frame["scenario_id"].isin(scenarios)]
    truth = frame["scenario_id"].map(labels)
    accuracy = float((frame["direction"].astype(str) == truth).mean())
    recalls = [float((frame.loc[truth.eq(label), "direction"].astype(str) == label).mean()) for label in DIRECTIONS]
    target = np.stack([(truth.to_numpy() == label).astype(float) for label in DIRECTIONS], axis=1)
    probabilities = frame[list(PROBABILITY_COLUMNS)].to_numpy(float)
    brier = float(np.mean(np.sum((probabilities - target) ** 2, axis=1)))
    majority_rows = []
    for (scenario, seed), rows in frame.groupby(["scenario_id", "seed"]):
        probabilities = rows[list(PROBABILITY_COLUMNS)].mean().to_numpy(float)[None, :]
        majority = int(tie_majority(probabilities)[0])
        predicted = "tie" if majority < 0 else DIRECTIONS[majority]
        majority_rows.append(predicted == labels[str(scenario)])
    return {
        "individual_accuracy": accuracy,
        "balanced_accuracy": float(np.mean(recalls)),
        "brier_score": brier,
        "majority_accuracy": float(np.mean(majority_rows)),
    }


def bootstrap_forecast_metrics(
    system: dict[str, Any],
    labels: dict[str, str],
    selected_scenarios: Sequence[str],
) -> dict[str, float]:
    """Compute metrics on a scenario bootstrap sample without NaN class recalls."""
    frame = system["belief"]
    frame = frame[(frame["round"] == 6) & frame["valid"]]
    pieces = [frame[frame["scenario_id"].eq(scenario)] for scenario in selected_scenarios]
    sampled = pd.concat(pieces, ignore_index=True)
    truth = sampled["scenario_id"].map(labels)
    recalls = [
        float((sampled.loc[truth.eq(label), "direction"].astype(str) == label).mean())
        for label in DIRECTIONS
        if bool(truth.eq(label).any())
    ]
    target = np.stack([(truth.to_numpy() == label).astype(float) for label in DIRECTIONS], axis=1)
    probabilities = sampled[list(PROBABILITY_COLUMNS)].to_numpy(float)
    majority_rows = []
    for scenario in selected_scenarios:
        rows = sampled[sampled["scenario_id"].eq(scenario)]
        for seed, seed_rows in rows.groupby("seed"):
            predicted_index = int(tie_majority(seed_rows[list(PROBABILITY_COLUMNS)].mean().to_numpy()[None, :])[0])
            majority_rows.append(
                ("tie" if predicted_index < 0 else DIRECTIONS[predicted_index]) == labels[scenario]
            )
    return {
        "balanced_accuracy": float(np.mean(recalls)),
        "brier_score": float(np.mean(np.sum((probabilities - target) ** 2, axis=1))),
        "majority_accuracy": float(np.mean(majority_rows)),
    }


def bootstrap_prediction_delta(k8: dict[str, Any], k10: dict[str, Any], labels: dict[str, str], replicates: int, seed: int) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    samples = {"balanced_accuracy": [], "brier_score": []}
    for _ in range(replicates):
        selected = rng.choice(SCENARIOS, size=len(SCENARIOS), replace=True)
        # Preserve cluster multiplicity by assigning a synthetic cluster key.
        k8_values = bootstrap_forecast_metrics(k8, labels, selected)
        k10_values = bootstrap_forecast_metrics(k10, labels, selected)
        for metric in samples:
            samples[metric].append(float(k8_values[metric] - k10_values[metric]))
    return {
        metric: {
            "delta": float(forecast_metrics(k8, labels)[metric] - forecast_metrics(k10, labels)[metric]),
            "cluster_bootstrap_ci95_low": float(np.quantile(values, 0.025)),
            "cluster_bootstrap_ci95_high": float(np.quantile(values, 0.975)),
        }
        for metric, values in samples.items()
    }


def token_total(dataset_paths: Sequence[Path], max_round: int = 6) -> int:
    total = 0
    for path in dataset_paths:
        source = path / "llm_token_usage.jsonl"
        if not source.exists():
            continue
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            round_value = row.get("round")
            if round_value is not None and int(round_value) > max_round:
                continue
            total += int(row.get("total_tokens") or 0)
    return total


def k10_source_paths(dataset_dir: Path) -> list[Path]:
    manifest = json.loads((dataset_dir / "source_manifest.json").read_text(encoding="utf-8"))
    return [ROOT / row["path"] for row in manifest["sources"]]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.bootstrap_replicates < 100:
        raise ValueError("bootstrap-replicates must be at least 100")
    k8_paths = [path.resolve() for path in args.k8_datasets]
    k8_frames, sources = load_k8(k8_paths)
    k10_frames = load_k10(args.k10_dataset)
    normalize_frames(k8_frames)
    normalize_frames(k10_frames)
    k8 = build_system(k8_frames, 8)
    k10 = build_system(k10_frames, 10)
    thresholds = json.loads((args.enumeration_dir / "thresholds.json").read_text(encoding="utf-8"))
    candidate_rows = pd.read_csv(args.enumeration_dir / "top_candidates.csv")
    offline = candidate_rows[candidate_rows["agent_ids"].astype(str).eq("0|1|2|3|4|7|8|9")].iloc[0]

    belief_js = js_divergence(k8["probabilities"], k10["probabilities"])
    return_error = np.abs(k8["expected_return"] - k10["expected_return"])
    majority_error = (k8["majority"] != k10["majority"]).astype(float)
    social_raw = {
        "social_stance_js": js_divergence(k8["stance"], k10["stance"]),
        "social_stance_score_error": np.mean(np.abs(k8["score_quantiles"] - k10["score_quantiles"]), axis=1),
        "social_action_js": js_divergence(k8["action"], k10["action"]),
        "social_content_rate_error": np.abs(k8["content_rate"] - k10["content_rate"]) / np.maximum(1.0, np.abs(k10["content_rate"])),
        "social_participation_error": np.abs(k8["participation"] - k10["participation"]),
        "social_interaction_rate_error": np.abs(k8["interaction_rate"] - k10["interaction_rate"]),
        "social_source_reach_error": np.abs(k8["source_reach"] - k10["source_reach"]),
    }
    strength_raw, active_raw = graph_errors(
        k8["adjacency"],
        k10["adjacency"],
        reference_selected_indices=[0, 1, 2, 3, 4, 7, 8, 9],
    )
    values = {
        "belief_js": aggregate(belief_js, 7),
        "trajectory_return_error": aggregate(return_error, 7),
        "majority_trajectory_error": aggregate_direction(majority_error, final=False),
        "majority_final_error": aggregate_direction(majority_error.reshape(len(SCENARIOS), len(SEEDS), 7)[:, :, 6].ravel(), final=True),
        **{name: aggregate(raw, 6) for name, raw in social_raw.items()},
        "graph_strength_error": aggregate(strength_raw, 6),
        "graph_active_pair_error": aggregate(active_raw, 6),
        "graph_facility_error": float(offline["graph_facility_error"]),
    }
    normalized = {name: values[name] / float(thresholds[name]["threshold"]) for name in values}
    social_error = max(normalized[name] for name in social_raw)
    graph_error = max(
        normalized["graph_strength_error"], normalized["graph_active_pair_error"],
        normalized["social_source_reach_error"], normalized["graph_facility_error"],
    )
    core_error = max(
        normalized["belief_js"], normalized["trajectory_return_error"],
        normalized["majority_trajectory_error"], normalized["majority_final_error"],
        social_error, graph_error,
    )
    gates = {
        "belief": values["belief_js"] <= thresholds["belief_js"]["threshold"],
        "trajectory_return": values["trajectory_return_error"] <= thresholds["trajectory_return_error"]["threshold"],
        "majority_trajectory": values["majority_trajectory_error"] <= thresholds["majority_trajectory_error"]["threshold"],
        "majority_final": values["majority_final_error"] <= thresholds["majority_final_error"]["threshold"],
        "social": social_error <= 1.0,
        "graph": graph_error <= 1.0,
    }
    labels = actual_labels(k10_frames)
    predictions = {"k8": forecast_metrics(k8, labels), "k10": forecast_metrics(k10, labels)}
    prediction_delta = bootstrap_prediction_delta(k8, k10, labels, args.bootstrap_replicates, args.random_seed)
    k8_tokens = token_total(k8_paths)
    k10_tokens = token_total(k10_source_paths(args.k10_dataset.resolve()))
    quality = {
        "passed": True,
        "k8_snapshot_count": k8["snapshot_count"],
        "k8_expected_snapshot_count": 18 * 5 * 7 * 8,
        "k8_valid_snapshot_count": k8["valid_snapshot_count"],
        "k8_invalid_snapshot_count": k8["snapshot_count"] - k8["valid_snapshot_count"],
        "k10_snapshot_count": k10["snapshot_count"],
        "k10_valid_snapshot_count": k10["valid_snapshot_count"],
        "scenario_count": len(labels),
        "seeds": list(SEEDS),
        "role_categories": sorted(k8_frames["belief"]["agent_role_category"].dropna().astype(str).unique()),
    }
    quality["passed"] = all(
        [quality["k8_snapshot_count"] == quality["k8_expected_snapshot_count"],
         quality["k8_valid_snapshot_count"] >= quality["k8_expected_snapshot_count"] - 1,
         quality["scenario_count"] == 18,
         set(quality["role_categories"]) >= {"institution", "retail_mature", "retail_basic", "retail_novice"}]
    )
    conclusion = (
        "K=8 passes the predeclared real-rerun fidelity gates and is reasonable as the primary reduced candidate."
        if quality["passed"] and all(gates.values())
        else "K=8 does not pass every predeclared real-rerun fidelity gate; retain K=10 and inspect the failed components."
    )
    result = {
        "analysis_version": "s1_k8_rerun_validation_v1",
        "candidate": {"runtime_ids_in_k10": "0|1|2|3|4|7|8|9", "full_population_agent_ids": list(EXPECTED_FULL_IDS)},
        "quality": quality,
        "fidelity_metrics": {
            name: {
                "value": value,
                "threshold": float(thresholds[name]["threshold"]),
                "normalized_error": normalized[name],
                "pass": value <= float(thresholds[name]["threshold"]),
            }
            for name, value in values.items()
        },
        "social_error": social_error,
        "graph_error": graph_error,
        "core_error": core_error,
        "gates": gates,
        "all_hard_gates_pass": bool(quality["passed"] and all(gates.values())),
        "prediction_metrics": predictions,
        "prediction_delta_k8_minus_k10": prediction_delta,
        "tokens": {
            "k8_total_round_0_to_6": k8_tokens,
            "k10_total_round_0_to_6": k10_tokens,
            "relative_reduction": 1.0 - k8_tokens / k10_tokens if k10_tokens else None,
        },
        "offline_candidate_reference": {
            "core_error": float(offline["core_error"]),
            "social_error": float(offline["social_error"]),
            "graph_error": float(offline["graph_error"]),
        },
        "conclusion": conclusion,
        "method_note": "Thresholds were loaded unchanged from agent_subset_enumeration_k10_v2; actual labels are external descriptive checks only.",
        "sources": sources,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in k8_frames.items():
        frame.to_csv(args.output_dir / f"merged_{TABLES[name]}", index=False, encoding="utf-8-sig")
    write_json(args.output_dir / "validation_summary.json", result)
    write_json(args.output_dir / "quality_report.json", quality)
    metric_rows = [
        {"metric": name, **row}
        for name, row in result["fidelity_metrics"].items()
    ]
    pd.DataFrame(metric_rows).to_csv(args.output_dir / "fidelity_metrics.csv", index=False, encoding="utf-8-sig")
    report_rows = "\n".join(
        f"| {name} | {row['value']:.6f} | {row['threshold']:.6f} | {row['normalized_error']:.3f} | {'pass' if row['pass'] else 'fail'} |"
        for name, row in result["fidelity_metrics"].items()
    )
    report = f"""# K=8 首选候选真实重跑验证

## 结论

{conclusion}

- 数据完整性：{quality['k8_valid_snapshot_count']}/{quality['k8_expected_snapshot_count']} 条有效 K=8 信念快照；异常条数 {quality['k8_invalid_snapshot_count']}，按预先规定屏蔽而不删除 Agent。
- 核心归一化最坏误差：{core_error:.3f}；社会误差：{social_error:.3f}；图误差：{graph_error:.3f}。
- 硬门槛全部通过：{result['all_hard_gates_pass']}。
- token 相对 K=10 变化：{result['tokens']['relative_reduction']:.2%}。

## 冻结门槛复核

| 指标 | K=8 真重跑误差 | 冻结阈值 | 归一化 | 结果 |
|---|---:|---:|---:|---|
{report_rows}

连续过程误差按 seed P80、场景内最坏轮次、场景 P90 聚合；多数方向误差先按场景的 seed/轮次单元求平均，再取场景 P90。阈值直接读取 `Dataset/agent_subset_enumeration_k10_v2/thresholds.json`，没有根据本次结果调整。

## 预测外部检查

| 系统 | 个体准确率 | Balanced Accuracy | Brier | 多数方向准确率 |
|---|---:|---:|---:|---:|
| K=8 | {predictions['k8']['individual_accuracy']:.4f} | {predictions['k8']['balanced_accuracy']:.4f} | {predictions['k8']['brier_score']:.4f} | {predictions['k8']['majority_accuracy']:.4f} |
| K=10 | {predictions['k10']['individual_accuracy']:.4f} | {predictions['k10']['balanced_accuracy']:.4f} | {predictions['k10']['brier_score']:.4f} | {predictions['k10']['majority_accuracy']:.4f} |

预测指标包含真实标签，只作为外部描述性验证，不参与候选选择。bootstrap 以 18 个场景为聚类单位；5 个 seed 不作为 90 个独立市场样本。

## 边界

本次使用的 18 个场景参与过离线候选选择，因此属于开发集真重跑，而不是最终独立测试集。通过本报告只能确认 K=8 是否保留当前系统在这些场景上的主要行为，最终结论仍需新场景以及同规模随机、Graph-only 基线。
"""
    (args.output_dir / "analysis_report.md").write_text(report, encoding="utf-8")
    checksums = []
    for path in sorted(args.output_dir.iterdir()):
        if path.name == "CHECKSUMS.sha256" or not path.is_file():
            continue
        checksums.append(f"{sha256(path)}  {path.name}")
    (args.output_dir / "CHECKSUMS.sha256").write_text("\n".join(checksums) + "\n", encoding="ascii")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=json_value))
    return 0 if result["all_hard_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
