#!/usr/bin/env python3
"""Merge fixed-round S1 seed replicates and estimate Agent information value."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


ANALYSIS_VERSION = "s1_multiseed_k10_round6_seeds4004_42_3407_v2"
MAX_ROUND = 6
AGENT_COUNT = 10
SCENARIO_COUNT = 18
EPSILON = 1e-12
DIRECTIONS = ("up", "neutral", "down")

MERGED_TABLES = {
    "belief_snapshots.csv": "belief_snapshots.csv",
    "agent_round_panel.csv": "agent_round_panel.csv",
    "agent_round_content_exposures.csv": "agent_round_content_exposures.csv",
    "interaction_edges.csv": "interaction_edges.csv",
    "content_stance_catalog.csv": "content_stance_catalog.csv",
    "scenario_round_metrics.csv": "scenario_round_metrics.csv",
    "round_metrics.csv": "round_metrics.csv",
    "scenario_runs.csv": "scenario_runs.csv",
}

ROUND_TABLES = {
    "belief_snapshots.csv",
    "agent_round_panel.csv",
    "agent_round_content_exposures.csv",
    "interaction_edges.csv",
    "content_stance_catalog.csv",
    "scenario_round_metrics.csv",
    "round_metrics.csv",
}


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_sources() -> list[Path]:
    root = workspace_root() / "Dataset"
    return [
        root / "s1_round_selection_10rounds_k10_seed4004_v2",
        root / "s1_round_selection_6rounds_k10_seed42_v2",
        root / "s1_round_selection_6rounds_k10_seed3407_v2",
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources",
        type=Path,
        nargs="+",
        default=default_sources(),
        help="round-selection dataset directories",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=workspace_root() / "Dataset" / ANALYSIS_VERSION,
    )
    parser.add_argument("--permutations", type=int, default=1_000)
    parser.add_argument("--random-seed", type=int, default=20260808)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=json_scalar) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def json_scalar(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def normalized_labels(values: Iterable[Any]) -> list[str]:
    return ["<missing>" if pd.isna(value) else str(value) for value in values]


def entropy_bits(values: Iterable[Any]) -> float:
    labels = normalized_labels(values)
    if not labels:
        return 0.0
    counts = Counter(labels)
    total = len(labels)
    return -sum(
        (count / total) * math.log2(count / total)
        for count in counts.values()
        if count
    )


def mutual_information_bits(left: Iterable[Any], right: Iterable[Any]) -> float:
    pairs = list(zip(normalized_labels(left), normalized_labels(right)))
    if not pairs:
        return 0.0
    joint = Counter(pairs)
    left_counts = Counter(value[0] for value in pairs)
    right_counts = Counter(value[1] for value in pairs)
    total = len(pairs)
    result = 0.0
    for (left_value, right_value), count in joint.items():
        probability = count / total
        result += probability * math.log2(
            (count * total) / (left_counts[left_value] * right_counts[right_value])
        )
    return result


def conditional_mutual_information_bits(
    left: Iterable[Any], right: Iterable[Any], condition: Iterable[Any]
) -> float:
    triples = list(
        zip(
            normalized_labels(left),
            normalized_labels(right),
            normalized_labels(condition),
        )
    )
    if not triples:
        return 0.0
    condition_counts = Counter(value[2] for value in triples)
    total = len(triples)
    result = 0.0
    for condition_value, count in condition_counts.items():
        subset = [
            (left_value, right_value)
            for left_value, right_value, current in triples
            if current == condition_value
        ]
        result += (count / total) * mutual_information_bits(
            (value[0] for value in subset),
            (value[1] for value in subset),
        )
    return result


def normalized_mutual_information(
    left: Iterable[Any], right: Iterable[Any]
) -> float:
    left_values = list(left)
    right_values = list(right)
    denominator = math.sqrt(
        entropy_bits(left_values) * entropy_bits(right_values)
    )
    if denominator <= 0:
        return 0.0
    return mutual_information_bits(left_values, right_values) / denominator


def js_divergence_bits(left: Sequence[float], right: Sequence[float]) -> float:
    middle = [(a + b) / 2 for a, b in zip(left, right)]

    def kl(values: Sequence[float], target: Sequence[float]) -> float:
        return sum(
            value * math.log2(value / reference)
            for value, reference in zip(values, target)
            if value > 0 and reference > 0
        )

    return 0.5 * kl(left, middle) + 0.5 * kl(right, middle)


def majority_direction(values: Iterable[Any]) -> str:
    labels = [str(value) for value in values if str(value) in DIRECTIONS]
    counts = Counter(labels)
    if not counts:
        return "missing"
    top = max(counts.values())
    leaders = sorted(label for label, count in counts.items() if count == top)
    return leaders[0] if len(leaders) == 1 else "tie"


def stance_bin(value: Any) -> str:
    if pd.isna(value):
        return "none"
    score = float(value)
    if score < -0.1:
        return "negative"
    if score > 0.1:
        return "positive"
    return "mixed_neutral"


def amount_bin(value: Any) -> str:
    count = 0.0 if pd.isna(value) else max(0.0, float(value))
    if count <= 0:
        return "none"
    if count < 10:
        return "low_1_9"
    if count < 18:
        return "medium_10_17"
    return "high_18_plus"


def return_change_bin(value: Any) -> str:
    if pd.isna(value):
        return "missing"
    change = float(value)
    if change < -0.005:
        return "down"
    if change > 0.005:
        return "up"
    return "stable"


def bool_label(value: Any) -> str:
    if isinstance(value, bool):
        return "changed" if value else "stable"
    return "changed" if str(value).strip().lower() == "true" else "stable"


def permutation_summary(observed: float, null_values: Sequence[float]) -> dict[str, float]:
    null_array = np.asarray(null_values, dtype=float)
    return {
        "observed_bits": float(observed),
        "null_mean_bits": float(null_array.mean()),
        "null_std_bits": float(null_array.std(ddof=1)) if len(null_array) > 1 else 0.0,
        "bias_corrected_bits": float(observed - null_array.mean()),
        "one_sided_p_value": float(
            (1 + int(np.sum(null_array >= observed))) / (len(null_array) + 1)
        ),
    }


def bh_qvalues(values: Sequence[float]) -> list[float]:
    count = len(values)
    order = sorted(range(count), key=lambda index: values[index])
    result = [1.0] * count
    running = 1.0
    for position in range(count - 1, -1, -1):
        index = order[position]
        adjusted = min(1.0, values[index] * count / (position + 1))
        running = min(running, adjusted)
        result[index] = running
    return result


def brier_score(row: Mapping[str, Any]) -> float:
    actual = str(row["actual_direction"])
    return sum(
        (float(row[f"final_{direction}_probability"]) - float(direction == actual)) ** 2
        for direction in DIRECTIONS
    )


def merge_sources(
    sources: Sequence[Path], output_dir: Path
) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    merged: dict[str, list[pd.DataFrame]] = {name: [] for name in MERGED_TABLES}
    provenance: list[dict[str, Any]] = []
    seen_seeds: set[int] = set()

    for source in sources:
        source = source.resolve()
        summary = read_json(source / "analysis_summary.json")
        quality = read_json(source / "quality_report.json")
        seed = int(summary["design"]["random_seed"])
        if seed in seen_seeds:
            raise ValueError(f"duplicate seed: {seed}")
        seen_seeds.add(seed)
        if int(summary["design"]["agent_count_per_scenario"]) != AGENT_COUNT:
            raise ValueError(f"expected K={AGENT_COUNT}: {source}")
        if int(summary["design"]["rounds"]) < MAX_ROUND:
            raise ValueError(f"source has fewer than {MAX_ROUND} rounds: {source}")
        if not quality.get("passed"):
            raise ValueError(f"source quality check failed: {source}")

        source_record: dict[str, Any] = {
            "seed": seed,
            "dataset_version": summary["dataset_version"],
            "source_batch_id": summary["source_batch_id"],
            "path": str(source.relative_to(workspace_root())).replace("\\", "/"),
            "files": [],
        }
        for source_name, output_name in MERGED_TABLES.items():
            path = source / source_name
            frame = pd.read_csv(path, encoding="utf-8-sig")
            if source_name in ROUND_TABLES and "round" in frame.columns:
                frame["round"] = pd.to_numeric(frame["round"], errors="raise").astype(int)
                frame = frame.loc[frame["round"] <= MAX_ROUND].copy()
            frame["seed"] = seed
            merged[output_name].append(frame)
            source_record["files"].append(
                {
                    "name": source_name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        provenance.append(source_record)

    result = {
        name: pd.concat(frames, ignore_index=True, sort=False)
        for name, frames in merged.items()
    }
    for name, frame in result.items():
        write_csv(output_dir / f"merged_{name}", frame)
    return result, provenance


def build_final_observations(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    snapshots = tables["belief_snapshots.csv"].copy()
    snapshots["round"] = pd.to_numeric(snapshots["round"], errors="raise").astype(int)
    snapshots["agent_id"] = pd.to_numeric(snapshots["agent_id"], errors="raise").astype(int)
    valid = snapshots.loc[snapshots["status"].astype(str) == "ok"].copy()

    identity_columns = [
        "seed",
        "scenario_id",
        "agent_id",
        "full_population_agent_id",
        "agent_role",
        "agent_role_category",
        "agent_role_label",
    ]
    belief_columns = [
        "direction",
        "expected_return",
        "up_probability",
        "neutral_probability",
        "down_probability",
        "confidence",
    ]
    pre = valid.loc[valid["round"] == 0, identity_columns + belief_columns].copy()
    final = valid.loc[valid["round"] == MAX_ROUND, identity_columns + belief_columns].copy()
    pre = pre.rename(columns={name: f"pre_{name}" for name in belief_columns})
    final = final.rename(columns={name: f"final_{name}" for name in belief_columns})
    merge_keys = identity_columns
    observations = final.merge(pre, on=merge_keys, how="left", validate="one_to_one")

    runs = tables["scenario_runs.csv"][[
        "seed",
        "scenario_id",
        "actual_five_day_close_direction",
        "actual_five_day_close_return",
    ]].drop_duplicates(["seed", "scenario_id"])
    runs = runs.rename(
        columns={
            "actual_five_day_close_direction": "actual_direction",
            "actual_five_day_close_return": "actual_return",
        }
    )
    observations = observations.merge(
        runs, on=["seed", "scenario_id"], how="left", validate="many_to_one"
    )

    peer_majorities: list[str] = []
    full_majorities: list[str] = []
    for _, row in observations.iterrows():
        group = observations.loc[
            (observations["seed"] == row["seed"])
            & (observations["scenario_id"] == row["scenario_id"])
        ]
        full_majorities.append(majority_direction(group["final_direction"]))
        peer_majorities.append(
            majority_direction(
                group.loc[group["agent_id"] != row["agent_id"], "final_direction"]
            )
        )
    observations["full_majority_direction"] = full_majorities
    observations["peer_majority_direction"] = peer_majorities
    observations["final_direction_correct"] = (
        observations["final_direction"] == observations["actual_direction"]
    )
    observations["final_brier_score"] = [
        brier_score(row) for row in observations.to_dict("records")
    ]
    return observations.sort_values(["seed", "scenario_id", "agent_id"]).reset_index(drop=True)


def prepare_panel(panel: pd.DataFrame) -> pd.DataFrame:
    result = panel.copy()
    result["round"] = pd.to_numeric(result["round"], errors="raise").astype(int)
    result["agent_id"] = pd.to_numeric(result["agent_id"], errors="raise").astype(int)
    result["stance_bin"] = result["exposure_social_mean_stance_score"].map(stance_bin)
    result["amount_bin"] = result["exposure_social_unique_count"].map(amount_bin)
    result["return_change_bin"] = result["expected_return_delta"].map(return_change_bin)
    result["direction_flip_bin"] = result["direction_flip"].map(bool_label)
    result["round_condition"] = result["round"].astype(str)
    result["round_amount_condition"] = (
        result["round_condition"] + "|" + result["amount_bin"]
    )
    return result


def predictive_permutation_metrics(
    observations: pd.DataFrame,
    agent_id: int,
    permutations: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    rows = observations.loc[observations["agent_id"] == agent_id].copy()
    predictive = mutual_information_bits(rows["final_direction"], rows["actual_direction"])
    conditional_rows = rows.loc[rows["pre_direction"].notna()].copy()
    incremental = conditional_mutual_information_bits(
        conditional_rows["actual_direction"],
        conditional_rows["final_direction"],
        conditional_rows["pre_direction"],
    )
    predictive_null: list[float] = []
    incremental_null: list[float] = []
    for mapping in permutations:
        permuted = rows["scenario_id"].map(mapping)
        predictive_null.append(mutual_information_bits(rows["final_direction"], permuted))
        conditional_permuted = conditional_rows["scenario_id"].map(mapping)
        incremental_null.append(
            conditional_mutual_information_bits(
                conditional_permuted,
                conditional_rows["final_direction"],
                conditional_rows["pre_direction"],
            )
        )
    return {
        "predictive_sample_count": len(rows),
        "incremental_sample_count": len(conditional_rows),
        **{
            f"predictive_mi_{key}": value
            for key, value in permutation_summary(predictive, predictive_null).items()
        },
        **{
            f"predictive_incremental_cmi_{key}": value
            for key, value in permutation_summary(incremental, incremental_null).items()
        },
    }


def social_permutation_metrics(
    panel: pd.DataFrame,
    agent_id: int,
    permutation_count: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    rows = panel.loc[panel["agent_id"] == agent_id].copy().reset_index(drop=True)
    observed = {
        "social_amount_return_cmi": conditional_mutual_information_bits(
            rows["amount_bin"], rows["return_change_bin"], rows["round_condition"]
        ),
        "social_stance_return_cmi": conditional_mutual_information_bits(
            rows["stance_bin"],
            rows["return_change_bin"],
            rows["round_amount_condition"],
        ),
        "social_stance_flip_cmi": conditional_mutual_information_bits(
            rows["stance_bin"],
            rows["direction_flip_bin"],
            rows["round_amount_condition"],
        ),
    }
    null: dict[str, list[float]] = {name: [] for name in observed}
    blocks = [
        np.asarray(indices, dtype=int)
        for _, indices in rows.groupby(["seed", "round"], sort=True).groups.items()
    ]
    original_stance = rows["stance_bin"].to_numpy(copy=True)
    original_amount = rows["amount_bin"].to_numpy(copy=True)
    for _ in range(permutation_count):
        stance = original_stance.copy()
        amount = original_amount.copy()
        for indices in blocks:
            shuffled = rng.permutation(indices)
            stance[indices] = original_stance[shuffled]
            amount[indices] = original_amount[shuffled]
        round_amount = rows["round_condition"].astype(str).to_numpy() + "|" + amount
        null["social_amount_return_cmi"].append(
            conditional_mutual_information_bits(
                amount, rows["return_change_bin"], rows["round_condition"]
            )
        )
        null["social_stance_return_cmi"].append(
            conditional_mutual_information_bits(
                stance, rows["return_change_bin"], round_amount
            )
        )
        null["social_stance_flip_cmi"].append(
            conditional_mutual_information_bits(
                stance, rows["direction_flip_bin"], round_amount
            )
        )
    result: dict[str, Any] = {"social_sample_count": len(rows)}
    for name, observed_value in observed.items():
        result.update(
            {
                f"{name}_{key}": value
                for key, value in permutation_summary(
                    observed_value, null[name]
                ).items()
            }
        )
    return result


def individualized_social_permutation_metrics(
    panel: pd.DataFrame,
    *,
    permutation_count: int,
    rng: np.random.Generator,
) -> dict[int, dict[str, Any]]:
    """Test whether an Agent's assigned exposure matters beyond common context."""
    data = panel.sort_values(
        ["seed", "scenario_id", "round", "agent_id"]
    ).reset_index(drop=True)
    original_stance = data["stance_bin"].to_numpy(copy=True)
    original_amount = data["amount_bin"].to_numpy(copy=True)
    return_change = data["return_change_bin"].to_numpy(copy=True)
    direction_flip = data["direction_flip_bin"].to_numpy(copy=True)
    round_condition = data["round_condition"].astype(str).to_numpy(copy=True)
    agent_indices = {
        agent_id: np.flatnonzero(data["agent_id"].to_numpy() == agent_id)
        for agent_id in range(AGENT_COUNT)
    }
    blocks = [
        np.asarray(indices, dtype=int)
        for _, indices in data.groupby(
            ["seed", "scenario_id", "round"], sort=True
        ).groups.items()
    ]
    metric_names = (
        "social_amount_return_cmi",
        "social_stance_return_cmi",
        "social_stance_flip_cmi",
    )
    null = {
        agent_id: {name: [] for name in metric_names}
        for agent_id in range(AGENT_COUNT)
    }
    for _ in range(permutation_count):
        stance = original_stance.copy()
        amount = original_amount.copy()
        for indices in blocks:
            shuffled = rng.permutation(indices)
            stance[indices] = original_stance[shuffled]
            amount[indices] = original_amount[shuffled]
        round_amount = np.char.add(
            np.char.add(round_condition.astype(str), "|"), amount.astype(str)
        )
        for agent_id, indices in agent_indices.items():
            null[agent_id]["social_amount_return_cmi"].append(
                conditional_mutual_information_bits(
                    amount[indices],
                    return_change[indices],
                    round_condition[indices],
                )
            )
            null[agent_id]["social_stance_return_cmi"].append(
                conditional_mutual_information_bits(
                    stance[indices],
                    return_change[indices],
                    round_amount[indices],
                )
            )
            null[agent_id]["social_stance_flip_cmi"].append(
                conditional_mutual_information_bits(
                    stance[indices],
                    direction_flip[indices],
                    round_amount[indices],
                )
            )

    result: dict[int, dict[str, Any]] = {}
    for agent_id, indices in agent_indices.items():
        observed = {
            "social_amount_return_cmi": conditional_mutual_information_bits(
                original_amount[indices],
                return_change[indices],
                round_condition[indices],
            ),
            "social_stance_return_cmi": conditional_mutual_information_bits(
                original_stance[indices],
                return_change[indices],
                np.char.add(
                    np.char.add(round_condition.astype(str), "|"),
                    original_amount.astype(str),
                )[indices],
            ),
            "social_stance_flip_cmi": conditional_mutual_information_bits(
                original_stance[indices],
                direction_flip[indices],
                np.char.add(
                    np.char.add(round_condition.astype(str), "|"),
                    original_amount.astype(str),
                )[indices],
            ),
        }
        result[agent_id] = {}
        for name in metric_names:
            result[agent_id].update(
                {
                    f"individualized_{name}_{key}": value
                    for key, value in permutation_summary(
                        observed[name], null[agent_id][name]
                    ).items()
                }
            )
    return result


def agent_seed_metrics(
    observations: pd.DataFrame, panel: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for agent_id in range(AGENT_COUNT):
        agent_observations = observations.loc[observations["agent_id"] == agent_id]
        agent_panel = panel.loc[panel["agent_id"] == agent_id]
        for seed in sorted(observations["seed"].unique()):
            final = agent_observations.loc[agent_observations["seed"] == seed]
            social = agent_panel.loc[agent_panel["seed"] == seed]
            conditional = final.loc[final["pre_direction"].notna()]
            rows.append(
                {
                    "agent_id": agent_id,
                    "seed": int(seed),
                    "agent_role_label": final["agent_role_label"].iloc[0],
                    "sample_count": len(final),
                    "final_accuracy": float(final["final_direction_correct"].mean()),
                    "mean_brier_score": float(final["final_brier_score"].mean()),
                    "predictive_mi_bits": mutual_information_bits(
                        final["final_direction"], final["actual_direction"]
                    ),
                    "predictive_incremental_cmi_bits": conditional_mutual_information_bits(
                        conditional["actual_direction"],
                        conditional["final_direction"],
                        conditional["pre_direction"],
                    ),
                    "social_amount_return_cmi_bits": conditional_mutual_information_bits(
                        social["amount_bin"],
                        social["return_change_bin"],
                        social["round_condition"],
                    ),
                    "social_stance_return_cmi_bits": conditional_mutual_information_bits(
                        social["stance_bin"],
                        social["return_change_bin"],
                        social["round_amount_condition"],
                    ),
                    "social_stance_flip_cmi_bits": conditional_mutual_information_bits(
                        social["stance_bin"],
                        social["direction_flip_bin"],
                        social["round_amount_condition"],
                    ),
                }
            )
    return pd.DataFrame(rows)


def seed_rank_stability(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "final_accuracy",
        "predictive_mi_bits",
        "predictive_incremental_cmi_bits",
        "social_amount_return_cmi_bits",
        "social_stance_return_cmi_bits",
        "social_stance_flip_cmi_bits",
    )
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        pivot = seed_metrics.pivot(index="agent_id", columns="seed", values=metric)
        correlation = pivot.corr(method="spearman")
        seeds = sorted(int(value) for value in pivot.columns)
        pair_values: list[float] = []
        for left_index, left in enumerate(seeds):
            for right in seeds[left_index + 1 :]:
                value = float(correlation.loc[left, right])
                pair_values.append(value)
                rows.append(
                    {
                        "metric": metric,
                        "seed_i": left,
                        "seed_j": right,
                        "spearman_rank_correlation": value,
                        "mean_pairwise_spearman_for_metric": None,
                    }
                )
        metric_mean = float(np.mean(pair_values))
        for row in rows:
            if row["metric"] == metric:
                row["mean_pairwise_spearman_for_metric"] = metric_mean
    return pd.DataFrame(rows)


def pair_redundancy(observations: pd.DataFrame) -> pd.DataFrame:
    pivot = observations.pivot(
        index=["seed", "scenario_id"], columns="agent_id", values="final_direction"
    ).sort_index()
    rows: list[dict[str, Any]] = []
    for left in range(AGENT_COUNT):
        for right in range(left + 1, AGENT_COUNT):
            valid = pivot[[left, right]].dropna()
            rows.append(
                {
                    "agent_i": left,
                    "agent_j": right,
                    "sample_count": len(valid),
                    "mutual_information_bits": mutual_information_bits(
                        valid[left], valid[right]
                    ),
                    "normalized_mutual_information": normalized_mutual_information(
                        valid[left], valid[right]
                    ),
                    "direction_agreement_rate": float((valid[left] == valid[right]).mean()),
                }
            )
    return pd.DataFrame(rows)


def leave_one_out_metrics(snapshots: pd.DataFrame) -> pd.DataFrame:
    data = snapshots.loc[snapshots["status"].astype(str) == "ok"].copy()
    data["round"] = pd.to_numeric(data["round"], errors="raise").astype(int)
    data["agent_id"] = pd.to_numeric(data["agent_id"], errors="raise").astype(int)
    accumulators = {
        agent_id: {"js": [], "changed": [], "final_changed": []}
        for agent_id in range(AGENT_COUNT)
    }
    probability_columns = [f"{direction}_probability" for direction in DIRECTIONS]
    for (_, _, round_number), group in data.groupby(
        ["seed", "scenario_id", "round"], sort=True
    ):
        full_probabilities = group[probability_columns].astype(float).mean().to_numpy()
        full_majority = majority_direction(group["direction"])
        for agent_id in range(AGENT_COUNT):
            if agent_id not in set(group["agent_id"]):
                continue
            peers = group.loc[group["agent_id"] != agent_id]
            if peers.empty:
                continue
            peer_probabilities = peers[probability_columns].astype(float).mean().to_numpy()
            changed = full_majority != majority_direction(peers["direction"])
            accumulators[agent_id]["js"].append(
                js_divergence_bits(full_probabilities, peer_probabilities)
            )
            accumulators[agent_id]["changed"].append(changed)
            if int(round_number) == MAX_ROUND:
                accumulators[agent_id]["final_changed"].append(changed)

    rows: list[dict[str, Any]] = []
    for agent_id, values in accumulators.items():
        js_values = np.asarray(values["js"], dtype=float)
        rows.append(
            {
                "agent_id": agent_id,
                "snapshot_group_count": len(js_values),
                "mean_leave_one_out_group_js_bits": float(js_values.mean()),
                "p95_leave_one_out_group_js_bits": float(np.quantile(js_values, 0.95)),
                "majority_change_rate_all_rounds": float(np.mean(values["changed"])),
                "majority_change_rate_final_round": float(
                    np.mean(values["final_changed"])
                ),
            }
        )
    return pd.DataFrame(rows)


def rank_percentile(values: pd.Series, *, higher_is_better: bool = True) -> pd.Series:
    source = values if higher_is_better else -values
    if len(source) <= 1:
        return pd.Series([1.0] * len(source), index=source.index)
    ranks = source.rank(method="average", ascending=True)
    return (ranks - 1) / (len(source) - 1)


def pareto_front(frame: pd.DataFrame) -> pd.Series:
    benefits = [
        "predictive_incremental_cmi_bias_corrected_bits",
        "social_stance_return_cmi_bias_corrected_bits",
        "mean_leave_one_out_group_js_bits",
        "peer_representation_nmi",
    ]
    cost = "average_peer_redundancy_nmi"
    result: list[bool] = []
    for index, row in frame.iterrows():
        dominated = False
        for other_index, other in frame.iterrows():
            if index == other_index:
                continue
            no_worse = all(other[name] >= row[name] for name in benefits) and other[cost] <= row[cost]
            strictly_better = any(other[name] > row[name] for name in benefits) or other[cost] < row[cost]
            if no_worse and strictly_better:
                dominated = True
                break
        result.append(not dominated)
    return pd.Series(result, index=frame.index)


def build_information_contributions(
    observations: pd.DataFrame,
    panel: pd.DataFrame,
    snapshots: pd.DataFrame,
    *,
    permutation_count: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scenario_actual = (
        observations[["scenario_id", "actual_direction"]]
        .drop_duplicates("scenario_id")
        .sort_values("scenario_id")
    )
    scenarios = scenario_actual["scenario_id"].tolist()
    labels = scenario_actual["actual_direction"].tolist()
    rng = np.random.default_rng(random_seed)
    label_permutations: list[dict[str, str]] = []
    for _ in range(permutation_count):
        shuffled = rng.permutation(labels)
        label_permutations.append(dict(zip(scenarios, shuffled)))

    seed_metrics = agent_seed_metrics(observations, panel)
    pair_metrics = pair_redundancy(observations)
    leave_out = leave_one_out_metrics(snapshots)
    individualized_social = individualized_social_permutation_metrics(
        panel,
        permutation_count=permutation_count,
        rng=np.random.default_rng(random_seed + 5000),
    )
    rows: list[dict[str, Any]] = []
    for agent_id in range(AGENT_COUNT):
        agent_rows = observations.loc[observations["agent_id"] == agent_id]
        predictive = predictive_permutation_metrics(
            observations, agent_id, label_permutations
        )
        social = social_permutation_metrics(
            panel,
            agent_id,
            permutation_count,
            np.random.default_rng(random_seed + 1000 + agent_id),
        )
        peers = pair_metrics.loc[
            (pair_metrics["agent_i"] == agent_id)
            | (pair_metrics["agent_j"] == agent_id)
        ]
        per_seed = seed_metrics.loc[seed_metrics["agent_id"] == agent_id]
        peer_representation_mi = mutual_information_bits(
            agent_rows["final_direction"], agent_rows["peer_majority_direction"]
        )
        rows.append(
            {
                "agent_id": agent_id,
                "full_population_agent_id": int(agent_rows["full_population_agent_id"].iloc[0]),
                "agent_role": agent_rows["agent_role"].iloc[0],
                "agent_role_category": agent_rows["agent_role_category"].iloc[0],
                "agent_role_label": agent_rows["agent_role_label"].iloc[0],
                "final_accuracy_mean": float(per_seed["final_accuracy"].mean()),
                "final_accuracy_seed_sd": float(per_seed["final_accuracy"].std(ddof=1)),
                "mean_brier_score": float(per_seed["mean_brier_score"].mean()),
                "predictive_mi_seed_sd": float(per_seed["predictive_mi_bits"].std(ddof=1)),
                "predictive_incremental_cmi_seed_sd": float(
                    per_seed["predictive_incremental_cmi_bits"].std(ddof=1)
                ),
                "social_stance_return_cmi_seed_sd": float(
                    per_seed["social_stance_return_cmi_bits"].std(ddof=1)
                ),
                "social_stance_flip_cmi_seed_sd": float(
                    per_seed["social_stance_flip_cmi_bits"].std(ddof=1)
                ),
                "peer_representation_mi_bits": peer_representation_mi,
                "peer_representation_nmi": normalized_mutual_information(
                    agent_rows["final_direction"],
                    agent_rows["peer_majority_direction"],
                ),
                "average_peer_redundancy_nmi": float(
                    peers["normalized_mutual_information"].mean()
                ),
                "max_peer_redundancy_nmi": float(
                    peers["normalized_mutual_information"].max()
                ),
                **predictive,
                **social,
                **individualized_social[agent_id],
            }
        )
    contributions = pd.DataFrame(rows).merge(leave_out, on="agent_id", how="left")

    p_columns = [
        "predictive_mi_one_sided_p_value",
        "predictive_incremental_cmi_one_sided_p_value",
        "social_amount_return_cmi_one_sided_p_value",
        "social_stance_return_cmi_one_sided_p_value",
        "social_stance_flip_cmi_one_sided_p_value",
        "individualized_social_amount_return_cmi_one_sided_p_value",
        "individualized_social_stance_return_cmi_one_sided_p_value",
        "individualized_social_stance_flip_cmi_one_sided_p_value",
    ]
    for column in p_columns:
        q_column = column.replace("_one_sided_p_value", "_bh_q_value")
        contributions[q_column] = bh_qvalues(
            contributions[column].tolist()
        )

    contributions["predictive_component"] = rank_percentile(
        contributions["predictive_incremental_cmi_bias_corrected_bits"]
    )
    contributions["social_component"] = 0.25 * (
        rank_percentile(contributions["social_stance_return_cmi_bias_corrected_bits"])
        + rank_percentile(contributions["social_stance_flip_cmi_bias_corrected_bits"])
        + rank_percentile(
            contributions[
                "individualized_social_stance_return_cmi_bias_corrected_bits"
            ]
        )
        + rank_percentile(
            contributions[
                "individualized_social_stance_flip_cmi_bias_corrected_bits"
            ]
        )
    )
    contributions["group_impact_component"] = rank_percentile(
        contributions["mean_leave_one_out_group_js_bits"]
    )
    contributions["representation_component"] = rank_percentile(
        contributions["peer_representation_nmi"]
    )
    contributions["nonredundancy_component"] = rank_percentile(
        contributions["average_peer_redundancy_nmi"], higher_is_better=False
    )
    stability_cost = (
        contributions["predictive_incremental_cmi_seed_sd"]
        + contributions["social_stance_return_cmi_seed_sd"]
        + contributions["social_stance_flip_cmi_seed_sd"]
    )
    contributions["stability_component"] = rank_percentile(
        stability_cost, higher_is_better=False
    )
    component_columns = [
        "predictive_component",
        "social_component",
        "group_impact_component",
        "representation_component",
        "nonredundancy_component",
        "stability_component",
    ]
    contributions["balanced_reference_score"] = contributions[component_columns].mean(axis=1)
    contributions["balanced_reference_rank"] = contributions[
        "balanced_reference_score"
    ].rank(method="min", ascending=False).astype(int)
    contributions["pareto_front"] = pareto_front(contributions)
    contributions = contributions.sort_values("balanced_reference_rank").reset_index(drop=True)
    return contributions, seed_metrics, pair_metrics, leave_out


def render_report(contributions: pd.DataFrame, quality: Mapping[str, Any]) -> str:
    ranked = contributions.sort_values("balanced_reference_rank")
    table_lines = [
        "| 排名 | Agent | 角色 | 预测增量 CMI（校正） | 社会立场-改判 CMI（校正） | 个体化立场-改判 CMI（校正） | 群体扰动 JS | 平均冗余 NMI | 参考分 |",
        "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in ranked.to_dict("records"):
        table_lines.append(
            "| {balanced_reference_rank} | {agent_id} | {agent_role_label} | "
            "{predictive_incremental_cmi_bias_corrected_bits:.4f} | "
            "{social_stance_flip_cmi_bias_corrected_bits:.4f} | "
            "{individualized_social_stance_flip_cmi_bias_corrected_bits:.4f} | "
            "{mean_leave_one_out_group_js_bits:.6f} | "
            "{average_peer_redundancy_nmi:.3f} | {balanced_reference_score:.3f} |".format(**row)
        )
    predictive_significant = int(
        (
            (contributions["predictive_incremental_cmi_bias_corrected_bits"] > 0)
            & (contributions["predictive_incremental_cmi_bh_q_value"] <= 0.05)
        ).sum()
    )
    social_significant = int(
        (
            (contributions["social_stance_return_cmi_bias_corrected_bits"] > 0)
            & (contributions["social_stance_return_cmi_bh_q_value"] <= 0.05)
        ).sum()
    )
    social_flip_significant = int(
        (
            (contributions["social_stance_flip_cmi_bias_corrected_bits"] > 0)
            & (contributions["social_stance_flip_cmi_bh_q_value"] <= 0.05)
        ).sum()
    )
    individualized_flip_significant = int(
        (
            (
                contributions[
                    "individualized_social_stance_flip_cmi_bias_corrected_bits"
                ]
                > 0
            )
            & (
                contributions[
                    "individualized_social_stance_flip_cmi_bh_q_value"
                ]
                <= 0.05
            )
        ).sum()
    )
    top_agents = ", ".join(
        f"Agent {int(row.agent_id)}" for row in ranked.head(4).itertuples()
    )
    rank_stability = quality["mean_seed_rank_spearman"]
    return f"""# 三 seed Agent 多类信息贡献分析

## 结论

多类信息贡献可以作为降采样的**候选筛选参考**，但不适合作为单一删除规则。本分析将预测增量信息、社会响应信息、群体边际扰动、群体代表性、跨 Agent 冗余和跨 seed 稳定性分开计算；这些维度衡量的机制不同，不能把原始 bit 数直接相加。

等权百分位 `balanced_reference_score` 仅用于浏览候选，其当前前四名为 **{top_agents}**。权重没有外部理论标定，因此正式选 K 时应优先使用各分项、Pareto 前沿和角色约束，而不是机械采用该总排名。

## Agent 结果

{chr(10).join(table_lines)}

预测增量 CMI 经场景标签置换和 10-Agent BH 校正后显著的 Agent 数为 **{predictive_significant}/10**；社会立场-收益变化 CMI 为 **{social_significant}/10**；社会立场-方向改判 CMI 为 **{social_flip_significant}/10**。这三个口径检验的是跨场景关联。

进一步在同一 `seed + 场景 + 轮次` 内交换 Agent 曝光后，个体化立场-改判 CMI 显著的 Agent 数为 **{individualized_flip_significant}/10**。如果跨场景显著而个体化检验不显著，说明信息主要来自场景共同背景，不能归因于某个 Agent 得到了特殊内容。未显著也不表示 Agent 无用，只表示当前证据不足以把该项与置换偏差区分开。

## 跨 seed 排名稳定性

- 最终准确率排名平均 Spearman：**{rank_stability['final_accuracy']:.3f}**。
- 预测方向 MI 排名平均 Spearman：**{rank_stability['predictive_mi_bits']:.3f}**。
- 控制 pre-social 后的预测增量 CMI 排名平均 Spearman：**{rank_stability['predictive_incremental_cmi_bits']:.3f}**。
- 社会立场-收益变化 CMI 排名平均 Spearman：**{rank_stability['social_stance_return_cmi_bits']:.3f}**。
- 社会立场-方向改判 CMI 排名平均 Spearman：**{rank_stability['social_stance_flip_cmi_bits']:.3f}**。

除未条件预测 MI 外，逐 seed 排名稳定性普遍较弱。这意味着合并数据适合估计总体候选价值，但不能把某个 seed 内的 Agent 排名当作稳定个人属性。

## 指标定义

1. `predictive_mi`：第 6 轮方向与真实 T+5 方向的经验互信息。
2. `predictive_incremental_cmi`：控制 pre-social 方向后，第 6 轮方向对真实方向增加的信息。
3. `social_amount_return_cmi`：控制轮次后，社会曝光量分箱与收益预测变化的条件互信息。
4. `social_stance_return_cmi`：继续控制曝光量后，所见内容立场与收益预测变化的条件互信息。
5. `social_stance_flip_cmi`：继续控制曝光量后，所见立场与方向改判的条件互信息。
6. `individualized_*`：同场景、同 seed、同轮次交换 Agent 曝光后的个体化信息检验。
7. `peer_representation_nmi`：Agent 最终方向与其余 9 个 Agent 多数方向的标准化互信息。
8. `average_peer_redundancy_nmi`：该 Agent 与其他单个 Agent 最终方向的平均标准化互信息；越高通常越可替代。
9. `leave_one_out_group_js`：离线移除该 Agent 前后，群体平均三分类概率的 JS；越高说明它对完整群体聚合更独特。

## 为什么只能作为参考

- 真实标签只有 18 个场景，3 个 seed 是同一场景的随机重复，不是 54 个独立市场事件。
- DeepSeek 没有确定性 seed 合约，跨 seed 波动同时包含本地互动随机性和模型输出随机性。
- 互信息用于筛选时接触了真实 T+5 标签；最终评估必须按场景留出，否则会发生选择泄漏。
- 社会曝光不是随机分配，社会响应 CMI 是预测关联，不是因果影响或传递熵。
- leave-one-out 是对完整日志的离线聚合，真实删去 Agent 后互动网络会重新生成。

## 下一步降采样

1. 用 `agent_information_contributions.csv` 生成 K=8、K=6 候选，保留机构、成熟散户、基础散户和新手角色覆盖。
2. 候选目标采用 mRMR/图覆盖：提高预测与社会信息，降低 Agent 间冗余，同时约束群体 JS、熵和 Profile 误差。
3. 使用按场景分组的交叉验证；每一折只在训练场景计算 MI 和选 Agent，再在留出场景评估。
4. 对候选子集真实重跑三个 seed。只有重跑后仍保留 K=10 的群体分布和社会互动指标，才能确认降采样有效。

## 数据质量

- seed：{quality['seeds']}
- 合并快照：{quality['snapshot_count']}，有效 {quality['valid_snapshot_count']}，无效 {quality['invalid_snapshot_count']}
- Agent-round 面板：{quality['panel_count']}
- 最终 Agent 观测：{quality['final_observation_count']}
- 质量检查：{'通过' if quality['passed'] else '未通过'}
"""


def render_readme(quality: Mapping[str, Any], settings: Mapping[str, Any]) -> str:
    return f"""# 三 seed S1 联合信息贡献数据集

该目录合并 seed 4004、42、3407 的 K=10、前 6 轮 S1 数据，用于 Agent 多类信息贡献和下一阶段降采样候选分析。seed=4004 原始实验有 10 轮，本目录仅保留第 0-6 轮。

## 推荐入口

- `analysis_report.md`：结论、指标边界与下一步方案。
- `agent_information_contributions.csv`：每个 Agent 的多类信息贡献、置换校正、BH q 值和探索性参考排名。
- `agent_seed_information_contributions.csv`：逐 Agent、逐 seed 的原始信息量和预测表现。
- `agent_pair_redundancy.csv`：45 对 Agent 的方向 MI、NMI 和一致率。
- `agent_seed_rank_stability.csv`：不同 seed 下逐 Agent 指标排名的两两 Spearman 相关。
- `agent_leave_one_out_metrics.csv`：离线移除单个 Agent 后的群体概率和多数方向扰动。
- `final_agent_observations.csv`：第 0/6 轮配对后的 540 条 Agent-场景-seed 观测。
- `merged_agent_round_panel.csv`：3240 行社会响应分析主表。
- `merged_agent_round_content_exposures.csv`：三 seed 的逐条去重曝光关系；feed、直接互动、自身内容及首次曝光均已分层。
- `merged_interaction_edges.csv`：三 seed 的显式有向互动关系，与 feed 曝光分开。

## 统计口径

- 预测 MI 的置换单元是场景标签：同一场景的三个 seed 始终共享同一个置换标签。
- 社会响应 CMI 在每个 `Agent + seed + round` 内交换社会特征，保持轮次和 seed 的边际分布。
- 个体化社会响应检验在每个 `seed + scenario + round` 内交换 10 个 Agent 的社会特征。
- 所有 MI 以 bits 表示；`bias_corrected_bits = observed - permutation_null_mean`，允许为负。
- p 值为单侧置换 p，q 值使用 10 个 Agent 内的 Benjamini-Hochberg 校正。
- `balanced_reference_score` 是六个分项百分位的等权平均，只是候选浏览指标，不是最终科学结论。

## 重要边界

本目录包含真实 T+5 标签，只能用于离线评估和训练场景内筛选，不能输入预测 Agent。三个 seed 不是独立市场样本；正式验证必须按 `scenario_id` 分组。离线移除 Agent 不会重建社会网络，因此不能替代 K=8/K=6 的真实重跑。

用于 Agent 节点优化的图应从 `merged_interaction_edges.csv` 构造，并筛选 `actor_class=investor`、`target_class=investor`。指向 source 的边和 feed 可见机会应分别建层，不得合并解释为 Agent 间影响。

## 数据质量

- seed：{quality['seeds']}
- 场景-seed：{quality['scenario_seed_count']}
- 信念快照：{quality['snapshot_count']}，有效 {quality['valid_snapshot_count']}
- Agent-round 面板：{quality['panel_count']}
- 内容立场：{quality['content_count']}
- 去重曝光边：{quality['exposure_count']}
- 显式互动边：{quality['interaction_count']}
- 总体检查：{'通过' if quality['passed'] else '未通过'}

## 复现

```powershell
MiroFish/backend/.venv/Scripts/python.exe MiroFish/backend/scripts/build_multiseed_information_dataset.py --permutations {settings['permutations']} --random-seed {settings['random_seed']}
```
"""


def build_dataset(
    sources: Sequence[Path],
    output_dir: Path,
    *,
    permutation_count: int,
    random_seed: int,
) -> dict[str, Any]:
    if permutation_count < 99:
        raise ValueError("at least 99 permutations are required")
    output_dir.mkdir(parents=True, exist_ok=True)
    tables, provenance = merge_sources(sources, output_dir)
    observations = build_final_observations(tables)
    panel = prepare_panel(tables["agent_round_panel.csv"])
    contributions, seed_metrics, pair_metrics, leave_out = build_information_contributions(
        observations,
        panel,
        tables["belief_snapshots.csv"],
        permutation_count=permutation_count,
        random_seed=random_seed,
    )
    rank_stability = seed_rank_stability(seed_metrics)

    write_csv(output_dir / "final_agent_observations.csv", observations)
    write_csv(output_dir / "agent_information_contributions.csv", contributions)
    write_csv(output_dir / "agent_seed_information_contributions.csv", seed_metrics)
    write_csv(output_dir / "agent_pair_redundancy.csv", pair_metrics)
    write_csv(output_dir / "agent_leave_one_out_metrics.csv", leave_out)
    write_csv(output_dir / "agent_seed_rank_stability.csv", rank_stability)

    snapshots = tables["belief_snapshots.csv"]
    invalid = snapshots.loc[snapshots["status"].astype(str) != "ok"]
    seeds = sorted(int(value) for value in snapshots["seed"].unique())
    key_specs = {
        "snapshots": (snapshots, ["seed", "scenario_id", "round", "agent_id"]),
        "panel": (panel, ["seed", "scenario_id", "round", "agent_id"]),
        "final": (observations, ["seed", "scenario_id", "agent_id"]),
    }
    duplicate_counts = {
        name: int(frame.duplicated(keys).sum())
        for name, (frame, keys) in key_specs.items()
    }
    actual_counts = (
        observations[["scenario_id", "actual_direction"]]
        .drop_duplicates()
        .groupby("scenario_id")["actual_direction"]
        .nunique()
    )
    exposure_validation = tables["agent_round_content_exposures.csv"].copy()
    exposure_validation["is_self_authored"] = (
        exposure_validation["is_self_authored"]
        .astype(str)
        .str.lower()
        .eq("true")
    )
    corrected_social_counts = (
        exposure_validation.loc[
            exposure_validation["author_class"].astype(str).eq("investor")
            & ~exposure_validation["is_self_authored"]
        ]
        .groupby(["seed", "scenario_id", "round", "viewer_agent_id"])
        .size()
        .rename("expected_social_unique_count")
        .reset_index()
        .rename(columns={"viewer_agent_id": "agent_id"})
    )
    panel_validation = tables["agent_round_panel.csv"][
        ["seed", "scenario_id", "round", "agent_id", "exposure_social_unique_count"]
    ].copy()
    for column in ("seed", "round", "agent_id"):
        panel_validation[column] = pd.to_numeric(
            panel_validation[column], errors="raise"
        ).astype(int)
        corrected_social_counts[column] = pd.to_numeric(
            corrected_social_counts[column], errors="raise"
        ).astype(int)
    panel_validation = panel_validation.merge(
        corrected_social_counts,
        on=["seed", "scenario_id", "round", "agent_id"],
        how="left",
        validate="one_to_one",
    )
    panel_validation["expected_social_unique_count"] = (
        panel_validation["expected_social_unique_count"].fillna(0).astype(int)
    )
    panel_validation["exposure_social_unique_count"] = pd.to_numeric(
        panel_validation["exposure_social_unique_count"], errors="raise"
    ).astype(int)
    quality: dict[str, Any] = {
        "passed": True,
        "seeds": seeds,
        "scenario_seed_count": int(
            tables["scenario_runs.csv"][["seed", "scenario_id"]].drop_duplicates().shape[0]
        ),
        "snapshot_count": len(snapshots),
        "valid_snapshot_count": int((snapshots["status"].astype(str) == "ok").sum()),
        "invalid_snapshot_count": len(invalid),
        "invalid_snapshot_records": invalid[
            ["seed", "scenario_id", "round", "agent_id", "error"]
        ].to_dict("records"),
        "panel_count": len(panel),
        "final_observation_count": len(observations),
        "content_count": len(tables["content_stance_catalog.csv"]),
        "exposure_count": len(tables["agent_round_content_exposures.csv"]),
        "interaction_count": len(tables["interaction_edges.csv"]),
        "self_authored_exposure_count": int(
            exposure_validation["is_self_authored"].sum()
        ),
        "social_self_exposure_leak_count": int((
            panel_validation["exposure_social_unique_count"]
            != panel_validation["expected_social_unique_count"]
        ).sum()),
        "round_metric_count": len(tables["round_metrics.csv"]),
        "duplicate_key_counts": duplicate_counts,
        "actual_label_conflict_count": int((actual_counts != 1).sum()),
        "contribution_agent_count": len(contributions),
        "pair_count": len(pair_metrics),
        "seed_rank_stability_row_count": len(rank_stability),
        "mean_seed_rank_spearman": {
            metric: float(rows["mean_pairwise_spearman_for_metric"].iloc[0])
            for metric, rows in rank_stability.groupby("metric", sort=True)
        },
    }
    quality["passed"] = all(
        (
            seeds == [42, 3407, 4004],
            quality["scenario_seed_count"] == 3 * SCENARIO_COUNT,
            quality["snapshot_count"] == 3 * SCENARIO_COUNT * (MAX_ROUND + 1) * AGENT_COUNT,
            quality["valid_snapshot_count"] >= quality["snapshot_count"] - 1,
            quality["panel_count"] == 3 * SCENARIO_COUNT * MAX_ROUND * AGENT_COUNT,
            quality["final_observation_count"] == 3 * SCENARIO_COUNT * AGENT_COUNT,
            quality["round_metric_count"] == 3 * (MAX_ROUND + 1),
            not any(duplicate_counts.values()),
            quality["actual_label_conflict_count"] == 0,
            quality["contribution_agent_count"] == AGENT_COUNT,
            quality["pair_count"] == AGENT_COUNT * (AGENT_COUNT - 1) // 2,
            quality["seed_rank_stability_row_count"] == 18,
            quality["social_self_exposure_leak_count"] == 0,
        )
    )

    settings = {
        "analysis_version": ANALYSIS_VERSION,
        "max_round": MAX_ROUND,
        "agent_count": AGENT_COUNT,
        "permutations": permutation_count,
        "random_seed": random_seed,
        "stance_bins": {
            "negative": "score < -0.1",
            "mixed_neutral": "-0.1 <= score <= 0.1",
            "positive": "score > 0.1",
            "none": "no social exposure stance",
        },
        "amount_bins": ["none=0", "low=1..9", "medium=10..17", "high>=18"],
        "return_change_bins": ["down<-0.005", "stable[-0.005,0.005]", "up>0.005"],
    }
    write_json(output_dir / "quality_report.json", quality)
    write_json(output_dir / "analysis_settings.json", settings)
    write_json(
        output_dir / "source_manifest.json",
        {"analysis_version": ANALYSIS_VERSION, "sources": provenance},
    )

    report = render_report(contributions, quality)
    (output_dir / "analysis_report.md").write_text(report, encoding="utf-8")
    (output_dir / "README.md").write_text(
        render_readme(quality, settings), encoding="utf-8"
    )
    checksums = []
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "CHECKSUMS.sha256":
            checksums.append(f"{sha256_file(path)}  {path.name}")
    (output_dir / "CHECKSUMS.sha256").write_text(
        "\n".join(checksums) + "\n", encoding="ascii"
    )
    return {
        "output_dir": str(output_dir),
        "quality": quality,
        "top_agents": contributions[
            ["balanced_reference_rank", "agent_id", "balanced_reference_score"]
        ].head(10).to_dict("records"),
    }


def main() -> None:
    args = parse_args()
    result = build_dataset(
        [path.resolve() for path in args.sources],
        args.output_dir.resolve(),
        permutation_count=args.permutations,
        random_seed=args.random_seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=json_scalar))


if __name__ == "__main__":
    main()
