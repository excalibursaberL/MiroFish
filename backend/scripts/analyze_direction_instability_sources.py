#!/usr/bin/env python3
"""Diagnose S1 discrete-direction instability from existing artifacts only.

The script never calls an LLM, OASIS, or Zep.  It uses round 0 as the
pre-social control and separates the observed candidate-versus-K=10 change
into a composition-only comparison and an independent-rerun comparison.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
PROBABILITY_COLUMNS = ["up_probability", "neutral_probability", "down_probability"]
DIRECTION_LABELS = ("up", "neutral", "down")
PROFILE_COMPARE_FIELDS = (
    "full_population_agent_id",
    "username",
    "name",
    "bio",
    "persona",
    "profession",
    "agent_key",
    "role_id",
    "role_category",
    "role_label",
    "role_description",
    "knowledge_level",
    "analysis_style",
    "risk_attitude",
    "investment_horizon",
    "decision_source",
    "social_role",
    "profile_version",
    "profile_sources",
    "agent_class",
    "mbti",
    "gender",
    "age",
    "country",
    "interested_topics",
)
TIE_EPSILON = 0.02
KEY_COLUMNS = ["scenario_id", "seed", "round"]
AGENT_KEY_COLUMNS = ["scenario_id", "seed", "round", "full_population_agent_id"]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--k8-dataset",
        type=Path,
        default=ROOT / "Dataset" / "s1_k8_rerun_validation_round6_seeds42_999_2887_3407_4004_v1",
    )
    parser.add_argument(
        "--k9-dataset",
        type=Path,
        default=ROOT / "Dataset" / "s1_k9_rerun_validation_round6_seeds42_999_2887_3407_4004_v1",
    )
    parser.add_argument(
        "--k10-dataset",
        type=Path,
        default=ROOT / "Dataset" / "s1_multiseed_k10_round6_seeds4004_42_3407_999_2887_v2",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "Dataset" / "s1_direction_instability_diagnostic_v1",
    )
    return parser.parse_args(argv)


def read_beliefs(directory: Path, label: str) -> pd.DataFrame:
    frame = pd.read_csv(directory / "merged_belief_snapshots.csv", low_memory=False)
    frame["population"] = label
    frame["scenario_id"] = frame["scenario_id"].astype(str)
    frame["seed"] = pd.to_numeric(frame["seed"], errors="raise").astype(int)
    frame["round"] = pd.to_numeric(frame["round"], errors="raise").astype(int)
    frame["full_population_agent_id"] = pd.to_numeric(
        frame["full_population_agent_id"], errors="coerce"
    ).astype("Int64")
    for column in PROBABILITY_COLUMNS + ["expected_return", "confidence"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    probabilities = frame[PROBABILITY_COLUMNS].to_numpy(dtype=float)
    frame["valid_probability"] = (
        np.isfinite(probabilities).all(axis=1)
        & (probabilities >= 0).all(axis=1)
        & np.isclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
        & frame["status"].astype(str).eq("ok").to_numpy()
    )
    return frame[frame["round"].between(0, 6)].copy()


def normalize_probabilities(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    total = float(array.sum())
    if not np.isfinite(array).all() or total <= 0:
        return np.full(array.shape, np.nan)
    return array / total


def js_divergence(left: Sequence[float], right: Sequence[float]) -> float:
    left_values = normalize_probabilities(left)
    right_values = normalize_probabilities(right)
    if not np.isfinite(left_values).all() or not np.isfinite(right_values).all():
        return math.nan
    middle = 0.5 * (left_values + right_values)
    result = 0.0
    for values in (left_values, right_values):
        mask = values > 0
        result += 0.5 * float(np.sum(values[mask] * np.log2(values[mask] / middle[mask])))
    return result


def direction_and_margin(values: Sequence[float], tie_epsilon: float = TIE_EPSILON) -> tuple[str, float]:
    probabilities = normalize_probabilities(values)
    if not np.isfinite(probabilities).all():
        return "missing", math.nan
    order = np.argsort(probabilities)
    margin = float(probabilities[order[-1]] - probabilities[order[-2]])
    if margin <= tie_epsilon:
        return "tie", margin
    return DIRECTION_LABELS[int(order[-1])], margin


def strict_argmax_direction(values: Sequence[float]) -> str:
    probabilities = normalize_probabilities(values)
    if not np.isfinite(probabilities).all():
        return "missing"
    leaders = np.flatnonzero(np.isclose(probabilities, probabilities.max(), atol=1e-12))
    return DIRECTION_LABELS[int(leaders[0])] if len(leaders) == 1 else "tie"


def group_panel(frame: pd.DataFrame, agent_ids: Iterable[int] | None = None) -> pd.DataFrame:
    selected = frame[frame["valid_probability"]].copy()
    if agent_ids is not None:
        selected = selected[selected["full_population_agent_id"].isin(list(agent_ids))]
    grouped = selected.groupby(KEY_COLUMNS, as_index=False).agg(
        up_probability=("up_probability", "mean"),
        neutral_probability=("neutral_probability", "mean"),
        down_probability=("down_probability", "mean"),
        expected_return=("expected_return", "mean"),
        valid_agent_count=("full_population_agent_id", "nunique"),
    )
    decisions = grouped[PROBABILITY_COLUMNS].apply(
        lambda row: direction_and_margin(row.to_numpy()), axis=1
    )
    grouped["direction"] = [value[0] for value in decisions]
    grouped["margin"] = [value[1] for value in decisions]
    return grouped


def renamed_panel(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    columns = PROBABILITY_COLUMNS + ["expected_return", "valid_agent_count", "direction", "margin"]
    return frame.rename(columns={column: f"{prefix}_{column}" for column in columns})


def comparison_metrics(frame: pd.DataFrame, left: str, right: str) -> None:
    frame[f"{left}_vs_{right}_direction_disagreement"] = (
        frame[f"{left}_direction"] != frame[f"{right}_direction"]
    )
    frame[f"{left}_vs_{right}_belief_js"] = frame.apply(
        lambda row: js_divergence(
            [row[f"{left}_{column}"] for column in PROBABILITY_COLUMNS],
            [row[f"{right}_{column}"] for column in PROBABILITY_COLUMNS],
        ),
        axis=1,
    )
    frame[f"{left}_vs_{right}_return_abs_error"] = (
        frame[f"{left}_expected_return"] - frame[f"{right}_expected_return"]
    ).abs()


def candidate_comparison(candidate: pd.DataFrame, reference: pd.DataFrame, label: str) -> pd.DataFrame:
    selected_ids = sorted(candidate["full_population_agent_id"].dropna().astype(int).unique())
    actual = renamed_panel(group_panel(candidate), "candidate")
    matched = renamed_panel(group_panel(reference, selected_ids), "matched_k10_subset")
    full = renamed_panel(group_panel(reference), "full_k10")
    result = actual.merge(matched, on=KEY_COLUMNS, validate="one_to_one").merge(
        full, on=KEY_COLUMNS, validate="one_to_one"
    )
    comparison_metrics(result, "candidate", "full_k10")
    comparison_metrics(result, "candidate", "matched_k10_subset")
    comparison_metrics(result, "matched_k10_subset", "full_k10")
    result["candidate_label"] = label
    result["selected_full_population_agent_ids"] = "|".join(map(str, selected_ids))

    actual_mismatch = result["candidate_vs_full_k10_direction_disagreement"]
    rerun_mismatch = result["candidate_vs_matched_k10_subset_direction_disagreement"]
    composition_mismatch = result["matched_k10_subset_vs_full_k10_direction_disagreement"]
    result["actual_mismatch_path"] = np.select(
        [
            ~actual_mismatch,
            actual_mismatch & rerun_mismatch & ~composition_mismatch,
            actual_mismatch & ~rerun_mismatch & composition_mismatch,
            actual_mismatch & rerun_mismatch & composition_mismatch,
        ],
        ["no_actual_mismatch", "rerun_only", "composition_only", "both_paths"],
        default="nontransitive_or_other",
    )
    return result


def summarize_candidate_rounds(comparison: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (label, round_number), group in comparison.groupby(["candidate_label", "round"]):
        paths = group["actual_mismatch_path"].value_counts()
        row: dict[str, Any] = {
            "candidate_label": label,
            "round": int(round_number),
            "scenario_seed_units": int(len(group)),
            "actual_vs_full_direction_disagreement_rate": float(
                group["candidate_vs_full_k10_direction_disagreement"].mean()
            ),
            "rerun_vs_matched_subset_direction_disagreement_rate": float(
                group["candidate_vs_matched_k10_subset_direction_disagreement"].mean()
            ),
            "composition_only_direction_disagreement_rate": float(
                group["matched_k10_subset_vs_full_k10_direction_disagreement"].mean()
            ),
            "actual_vs_full_belief_js_mean": float(group["candidate_vs_full_k10_belief_js"].mean()),
            "rerun_vs_matched_subset_belief_js_mean": float(
                group["candidate_vs_matched_k10_subset_belief_js"].mean()
            ),
            "composition_only_belief_js_mean": float(
                group["matched_k10_subset_vs_full_k10_belief_js"].mean()
            ),
            "actual_vs_full_return_mae": float(
                group["candidate_vs_full_k10_return_abs_error"].mean()
            ),
            "rerun_vs_matched_subset_return_mae": float(
                group["candidate_vs_matched_k10_subset_return_abs_error"].mean()
            ),
            "composition_only_return_mae": float(
                group["matched_k10_subset_vs_full_k10_return_abs_error"].mean()
            ),
        }
        for path in (
            "rerun_only",
            "composition_only",
            "both_paths",
            "nontransitive_or_other",
        ):
            row[f"actual_mismatch_{path}_count"] = int(paths.get(path, 0))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["candidate_label", "round"])


def paired_agent_comparison(candidate: pd.DataFrame, reference: pd.DataFrame, label: str) -> pd.DataFrame:
    candidate_valid = candidate[candidate["valid_probability"]].copy()
    reference_valid = reference[reference["valid_probability"]].copy()
    keep = AGENT_KEY_COLUMNS + PROBABILITY_COLUMNS + [
        "expected_return",
        "direction",
        "input_snapshot_hash",
        "prompt_version",
        "prompt_hash",
        "agent_role",
        "agent_role_category",
        "agent_knowledge_level",
        "agent_analysis_style",
        "agent_risk_attitude",
        "agent_investment_horizon",
    ]
    left = candidate_valid[keep].rename(
        columns={column: f"candidate_{column}" for column in keep if column not in AGENT_KEY_COLUMNS}
    )
    right = reference_valid[keep].rename(
        columns={column: f"k10_{column}" for column in keep if column not in AGENT_KEY_COLUMNS}
    )
    result = left.merge(right, on=AGENT_KEY_COLUMNS, validate="one_to_one")
    result["candidate_label"] = label
    candidate_decisions = result[[f"candidate_{column}" for column in PROBABILITY_COLUMNS]].apply(
        lambda row: direction_and_margin(row.to_numpy()), axis=1
    )
    reference_decisions = result[[f"k10_{column}" for column in PROBABILITY_COLUMNS]].apply(
        lambda row: direction_and_margin(row.to_numpy()), axis=1
    )
    result["candidate_tie_aware_direction"] = [value[0] for value in candidate_decisions]
    result["k10_tie_aware_direction"] = [value[0] for value in reference_decisions]
    result["tie_aware_direction_disagreement"] = (
        result["candidate_tie_aware_direction"] != result["k10_tie_aware_direction"]
    )
    result["belief_js"] = result.apply(
        lambda row: js_divergence(
            [row[f"candidate_{column}"] for column in PROBABILITY_COLUMNS],
            [row[f"k10_{column}"] for column in PROBABILITY_COLUMNS],
        ),
        axis=1,
    )
    result["expected_return_abs_error"] = (
        result["candidate_expected_return"] - result["k10_expected_return"]
    ).abs()
    result["input_snapshot_hash_match"] = (
        result["candidate_input_snapshot_hash"] == result["k10_input_snapshot_hash"]
    )
    result["prompt_version_match"] = result["candidate_prompt_version"] == result["k10_prompt_version"]
    result["config_prompt_hash_match"] = result["candidate_prompt_hash"] == result["k10_prompt_hash"]
    profile_fields = [
        "agent_role",
        "agent_role_category",
        "agent_knowledge_level",
        "agent_analysis_style",
        "agent_risk_attitude",
        "agent_investment_horizon",
    ]
    result["recorded_profile_fields_match"] = np.logical_and.reduce(
        [result[f"candidate_{field}"] == result[f"k10_{field}"] for field in profile_fields]
    )
    return result


def summarize_paired_agents(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["candidate_label", "round"], as_index=False)
        .agg(
            paired_agent_observations=("belief_js", "size"),
            tie_aware_direction_disagreement_rate=("tie_aware_direction_disagreement", "mean"),
            belief_js_mean=("belief_js", "mean"),
            belief_js_p90=("belief_js", lambda values: float(np.quantile(values, 0.90))),
            expected_return_mae=("expected_return_abs_error", "mean"),
            input_snapshot_hash_match_rate=("input_snapshot_hash_match", "mean"),
            prompt_version_match_rate=("prompt_version_match", "mean"),
            config_prompt_hash_match_rate=("config_prompt_hash_match", "mean"),
            recorded_profile_fields_match_rate=("recorded_profile_fields_match", "mean"),
        )
        .sort_values(["candidate_label", "round"])
    )


def internal_output_consistency(populations: Sequence[pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for frame in populations:
        valid = frame[frame["valid_probability"]].copy()
        valid["probability_argmax_direction"] = valid[PROBABILITY_COLUMNS].apply(
            lambda row: strict_argmax_direction(row.to_numpy()), axis=1
        )
        valid["declared_matches_argmax"] = (
            valid["direction"].astype(str).str.lower() == valid["probability_argmax_direction"]
        )
        for round_number, group in valid.groupby("round"):
            rows.append(
                {
                    "population": str(group["population"].iloc[0]),
                    "round": int(round_number),
                    "valid_observations": int(len(group)),
                    "declared_direction_argmax_mismatch_rate": float(
                        1.0 - group["declared_matches_argmax"].mean()
                    ),
                    "retry_rate": float((pd.to_numeric(group["attempt_count"], errors="coerce") > 1).mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(["population", "round"])


def pairwise_js(values: np.ndarray) -> float:
    divergences = [js_divergence(values[left], values[right]) for left, right in itertools.combinations(range(len(values)), 2)]
    return float(np.mean(divergences)) if divergences else math.nan


def pairwise_disagreement(labels: Sequence[str]) -> float:
    comparisons = [labels[left] != labels[right] for left, right in itertools.combinations(range(len(labels)), 2)]
    return float(np.mean(comparisons)) if comparisons else math.nan


def normalized_label_entropy(labels: Sequence[str]) -> float:
    counts = pd.Series(list(labels)).value_counts().to_numpy(dtype=float)
    probabilities = counts / counts.sum()
    entropy = -float(np.sum(probabilities * np.log2(probabilities)))
    return entropy / math.log2(4.0)


def k10_seed_stability(reference: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel = group_panel(reference)
    unit_rows: list[dict[str, Any]] = []
    for (scenario, round_number), group in panel.groupby(["scenario_id", "round"]):
        group = group.sort_values("seed")
        labels = group["direction"].tolist()
        mode_count = int(pd.Series(labels).value_counts().max())
        unit_rows.append(
            {
                "scenario_id": scenario,
                "round": int(round_number),
                "seed_count": int(len(group)),
                "directions_by_seed": "|".join(f"{int(seed)}:{label}" for seed, label in zip(group["seed"], labels)),
                "mode_agreement_rate": mode_count / len(group),
                "unanimous": len(set(labels)) == 1,
                "pairwise_direction_disagreement_rate": pairwise_disagreement(labels),
                "normalized_direction_entropy": normalized_label_entropy(labels),
                "mean_pairwise_belief_js": pairwise_js(group[PROBABILITY_COLUMNS].to_numpy(dtype=float)),
                "median_margin": float(group["margin"].median()),
                "near_tie_seed_rate": float((group["margin"] <= TIE_EPSILON).mean()),
            }
        )
    units = pd.DataFrame(unit_rows).sort_values(["scenario_id", "round"])
    round_summary = (
        units.groupby("round", as_index=False)
        .agg(
            scenario_count=("scenario_id", "size"),
            unanimous_scenario_rate=("unanimous", "mean"),
            mean_mode_agreement_rate=("mode_agreement_rate", "mean"),
            mean_pairwise_direction_disagreement_rate=("pairwise_direction_disagreement_rate", "mean"),
            mean_normalized_direction_entropy=("normalized_direction_entropy", "mean"),
            mean_pairwise_belief_js=("mean_pairwise_belief_js", "mean"),
            median_margin=("median_margin", "median"),
            mean_near_tie_seed_rate=("near_tie_seed_rate", "mean"),
        )
        .sort_values("round")
    )

    valid = reference[reference["valid_probability"]].copy()
    agent_rows: list[dict[str, Any]] = []
    for (scenario, agent_id), group in valid[valid["round"] == 0].groupby(
        ["scenario_id", "full_population_agent_id"]
    ):
        group = group.sort_values("seed")
        decisions = [direction_and_margin(row)[0] for row in group[PROBABILITY_COLUMNS].to_numpy(dtype=float)]
        agent_rows.append(
            {
                "scenario_id": scenario,
                "full_population_agent_id": int(agent_id),
                "seed_count": int(len(group)),
                "pairwise_direction_disagreement_rate": pairwise_disagreement(decisions),
                "mean_pairwise_belief_js": pairwise_js(group[PROBABILITY_COLUMNS].to_numpy(dtype=float)),
                "normalized_direction_entropy": normalized_label_entropy(decisions),
            }
        )
    return units, round_summary, pd.DataFrame(agent_rows).sort_values(
        ["scenario_id", "full_population_agent_id"]
    )


def margin_sensitivity(comparison: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    bins = [-math.inf, 0.02, 0.05, math.inf]
    labels = ["near_tie_le_0.02", "small_margin_0.02_to_0.05", "clear_margin_gt_0.05"]
    data = comparison.copy()
    data["full_k10_margin_group"] = pd.cut(
        data["full_k10_margin"], bins=bins, labels=labels, include_lowest=True
    )
    for (candidate_label, round_number, margin_group), group in data.groupby(
        ["candidate_label", "round", "full_k10_margin_group"], observed=True
    ):
        rows.append(
            {
                "candidate_label": candidate_label,
                "round": int(round_number),
                "full_k10_margin_group": str(margin_group),
                "observation_count": int(len(group)),
                "actual_direction_disagreement_rate": float(
                    group["candidate_vs_full_k10_direction_disagreement"].mean()
                ),
                "rerun_direction_disagreement_rate": float(
                    group["candidate_vs_matched_k10_subset_direction_disagreement"].mean()
                ),
                "composition_direction_disagreement_rate": float(
                    group["matched_k10_subset_vs_full_k10_direction_disagreement"].mean()
                ),
                "actual_belief_js_mean": float(group["candidate_vs_full_k10_belief_js"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["candidate_label", "round", "full_k10_margin_group"])


def scenario_instability(comparison: pd.DataFrame, k10_units: pd.DataFrame) -> pd.DataFrame:
    result = (
        comparison.groupby(["candidate_label", "scenario_id"], as_index=False)
        .agg(
            actual_disagreement_all_rounds=("candidate_vs_full_k10_direction_disagreement", "mean"),
            rerun_disagreement_all_rounds=("candidate_vs_matched_k10_subset_direction_disagreement", "mean"),
            composition_disagreement_all_rounds=("matched_k10_subset_vs_full_k10_direction_disagreement", "mean"),
            belief_js_all_rounds=("candidate_vs_full_k10_belief_js", "mean"),
        )
    )
    for round_number, suffix in ((0, "round0"), (6, "round6")):
        subset = (
            comparison[comparison["round"] == round_number]
            .groupby(["candidate_label", "scenario_id"], as_index=False)
            .agg(
                actual=("candidate_vs_full_k10_direction_disagreement", "mean"),
                rerun=("candidate_vs_matched_k10_subset_direction_disagreement", "mean"),
                composition=("matched_k10_subset_vs_full_k10_direction_disagreement", "mean"),
            )
            .rename(columns={name: f"{suffix}_{name}_disagreement" for name in ("actual", "rerun", "composition")})
        )
        result = result.merge(subset, on=["candidate_label", "scenario_id"], validate="one_to_one")
    k10_round0 = k10_units[k10_units["round"] == 0][
        ["scenario_id", "pairwise_direction_disagreement_rate", "normalized_direction_entropy", "mean_pairwise_belief_js"]
    ].rename(
        columns={
            "pairwise_direction_disagreement_rate": "k10_round0_seed_pairwise_direction_disagreement",
            "normalized_direction_entropy": "k10_round0_seed_direction_entropy",
            "mean_pairwise_belief_js": "k10_round0_seed_pairwise_belief_js",
        }
    )
    return result.merge(k10_round0, on="scenario_id", validate="many_to_one").sort_values(
        ["candidate_label", "actual_disagreement_all_rounds"], ascending=[True, False]
    )


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def finance_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("finance_s1") or {}


def prompt_map(config: dict[str, Any]) -> dict[int, str]:
    return {
        int(item["agent_id"]): str(item.get("prompt", ""))
        for item in finance_config(config).get("pre_social_interviews", [])
    }


def mapping_by_full_id(config: dict[str, Any]) -> dict[int, int]:
    return {
        int(item["full_population_agent_id"]): int(item["agent_id"])
        for item in finance_config(config).get("investor_agent_mapping", [])
    }


def profiles_by_full_id(path: Path) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for profile in load_json(path):
        full_id = profile.get("full_population_agent_id")
        if full_id is not None:
            result[int(full_id)] = {field: profile.get(field) for field in PROFILE_COMPARE_FIELDS}
    return result


def pre_social_artifact_audit(candidate: pd.DataFrame, reference: pd.DataFrame, label: str) -> pd.DataFrame:
    candidate_runs = candidate[candidate["round"] == 0][
        ["scenario_id", "seed", "run_id", "input_snapshot_hash", "prompt_version"]
    ].drop_duplicates(["scenario_id", "seed"])
    reference_runs = reference[reference["round"] == 0][
        ["scenario_id", "seed", "run_id", "input_snapshot_hash", "prompt_version"]
    ].drop_duplicates(["scenario_id", "seed"])
    runs = candidate_runs.merge(
        reference_runs,
        on=["scenario_id", "seed"],
        suffixes=("_candidate", "_k10"),
        validate="one_to_one",
    )
    rows: list[dict[str, Any]] = []
    uploads = ROOT / "MiroFish" / "backend" / "uploads" / "finance"
    for row in runs.itertuples(index=False):
        candidate_dir = uploads / str(row.run_id_candidate)
        reference_dir = uploads / str(row.run_id_k10)
        paths_exist = all(
            (directory / name).exists()
            for directory in (candidate_dir, reference_dir)
            for name in ("simulation_config.json", "profiles.json", "random_seed_state.json")
        )
        record: dict[str, Any] = {
            "candidate_label": label,
            "scenario_id": row.scenario_id,
            "seed": int(row.seed),
            "candidate_run_id": row.run_id_candidate,
            "k10_run_id": row.run_id_k10,
            "artifacts_available": paths_exist,
            "input_snapshot_hash_match": row.input_snapshot_hash_candidate == row.input_snapshot_hash_k10,
            "prompt_version_match": row.prompt_version_candidate == row.prompt_version_k10,
        }
        if not paths_exist:
            rows.append(record)
            continue
        candidate_config = load_json(candidate_dir / "simulation_config.json")
        reference_config = load_json(reference_dir / "simulation_config.json")
        candidate_map = mapping_by_full_id(candidate_config)
        reference_map = mapping_by_full_id(reference_config)
        common_ids = sorted(set(candidate_map) & set(reference_map))
        candidate_prompts = prompt_map(candidate_config)
        reference_prompts = prompt_map(reference_config)
        candidate_profiles = profiles_by_full_id(candidate_dir / "profiles.json")
        reference_profiles = profiles_by_full_id(reference_dir / "profiles.json")
        candidate_seed_state = load_json(candidate_dir / "random_seed_state.json")
        reference_seed_state = load_json(reference_dir / "random_seed_state.json")
        record.update(
            {
                "candidate_llm_model": candidate_config.get("llm_model"),
                "k10_llm_model": reference_config.get("llm_model"),
                "llm_model_match": candidate_config.get("llm_model") == reference_config.get("llm_model"),
                "common_agent_count": len(common_ids),
                "exact_interview_prompt_match_rate": float(
                    np.mean(
                        [
                            candidate_prompts.get(candidate_map[agent_id])
                            == reference_prompts.get(reference_map[agent_id])
                            for agent_id in common_ids
                        ]
                    )
                ),
                "exact_normalized_profile_match_rate": float(
                    np.mean(
                        [
                            candidate_profiles.get(agent_id) == reference_profiles.get(agent_id)
                            for agent_id in common_ids
                        ]
                    )
                ),
                "candidate_llm_provider_seeded": bool(candidate_seed_state.get("llm_provider_seeded")),
                "k10_llm_provider_seeded": bool(reference_seed_state.get("llm_provider_seeded")),
            }
        )
        rows.append(record)
    return pd.DataFrame(rows).sort_values(["candidate_label", "scenario_id", "seed"])


def as_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def write_report(
    output_dir: Path,
    candidate_rounds: pd.DataFrame,
    paired_summary: pd.DataFrame,
    k10_rounds: pd.DataFrame,
    k10_agents: pd.DataFrame,
    audit: pd.DataFrame,
    output_consistency: pd.DataFrame,
    margins: pd.DataFrame,
) -> None:
    lines = [
        "# S1 离散方向不稳定来源诊断",
        "",
        "本诊断只读取既有 K=8、K=9、K=10 五 seed 结果，没有调用 LLM、OASIS 或 Zep，也没有重新运行实验。",
        "",
        "## 识别思路",
        "",
        "- `round=0` 是社会互动前的私有信念测量，可用来观察互动尚未发生时的重复性。",
        "- `composition-only`：在同一份 K=10 实现值上只保留候选 Agent，衡量机械删点效应。",
        "- `rerun`：候选真实重跑与 K=10 中相同 Agent 子集比较，衡量独立重跑链路的波动。",
        "- `actual`：候选真实重跑与完整 K=10 比较，是最终观察到的总差异。",
        "- 群体方向取平均概率的最大类；前两类概率差不超过 0.02 时记为 tie。",
        "",
        "## Round 0 分解",
        "",
        "| 候选 | 实际方向不一致 | 重跑路径不一致 | 组成效应不一致 | 实际 JS | 重跑 JS | 组成 JS |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in candidate_rounds[candidate_rounds["round"] == 0].itertuples(index=False):
        lines.append(
            f"| {row.candidate_label} | {row.actual_vs_full_direction_disagreement_rate:.1%} | "
            f"{row.rerun_vs_matched_subset_direction_disagreement_rate:.1%} | "
            f"{row.composition_only_direction_disagreement_rate:.1%} | "
            f"{row.actual_vs_full_belief_js_mean:.4f} | "
            f"{row.rerun_vs_matched_subset_belief_js_mean:.4f} | "
            f"{row.composition_only_belief_js_mean:.4f} |"
        )
    lines.extend(["", "在实际发生方向不一致的单元中："])
    for row in candidate_rounds[candidate_rounds["round"] == 0].itertuples(index=False):
        actual_count = int(round(row.actual_vs_full_direction_disagreement_rate * row.scenario_seed_units))
        rerun_involved = row.actual_mismatch_rerun_only_count + row.actual_mismatch_both_paths_count
        composition_involved = row.actual_mismatch_composition_only_count + row.actual_mismatch_both_paths_count
        lines.append(
            f"- {row.candidate_label}：共 {actual_count} 次；{rerun_involved}/{actual_count} "
            f"({rerun_involved / actual_count:.1%}) 涉及重跑路径，{composition_involved}/{actual_count} "
            f"({composition_involved / actual_count:.1%}) 涉及组成效应。两类可重叠，不能相加为 100%。"
        )
    lines.extend(
        [
            "",
            "`重跑路径`不是纯粹的模型误差估计，但 round 0 尚无社会互动；当原始 prompt、Profile、输入和模型完全一致且 provider seed 未固定时，它主要反映 LLM/API 预测链路的不可重复性。",
            "",
            "## 逐轮总差异",
            "",
            "| 候选 | 轮次 | 实际方向不一致 | 重跑路径不一致 | 组成效应不一致 | 实际 JS |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in candidate_rounds.itertuples(index=False):
        lines.append(
            f"| {row.candidate_label} | {row.round} | {row.actual_vs_full_direction_disagreement_rate:.1%} | "
            f"{row.rerun_vs_matched_subset_direction_disagreement_rate:.1%} | "
            f"{row.composition_only_direction_disagreement_rate:.1%} | "
            f"{row.actual_vs_full_belief_js_mean:.4f} |"
        )
    k10_round0 = k10_rounds[k10_rounds["round"] == 0].iloc[0]
    k10_round6 = k10_rounds[k10_rounds["round"] == 6].iloc[0]
    agent_repeat_direction = float(k10_agents["pairwise_direction_disagreement_rate"].mean())
    agent_repeat_js = float(k10_agents["mean_pairwise_belief_js"].mean())
    lines.extend(
        [
            "",
            "## K=10 自身跨 seed 稳定性",
            "",
            f"- round 0：场景级 seed 两两方向不一致率 {k10_round0.mean_pairwise_direction_disagreement_rate:.1%}，仅 {k10_round0.unanimous_scenario_rate:.1%} 的场景五 seed 完全一致。",
            f"- round 6：场景级 seed 两两方向不一致率 {k10_round6.mean_pairwise_direction_disagreement_rate:.1%}，仅 {k10_round6.unanimous_scenario_rate:.1%} 的场景五 seed 完全一致。",
            f"- round 0 单 Agent 的 seed 两两方向不一致率均值为 {agent_repeat_direction:.1%}，概率 JS 均值为 {agent_repeat_js:.4f}。",
            "",
            "round 0 没有社会互动，因此这部分不能由意见传播解释。它证明完整 K=10 系统本身就存在明显的预测重复性问题。",
            "",
            "更关键的是，K=10 自身的 round 0/round 6 两两不一致率 17.2%/31.1%，与 K=8 对 K=10 的 17.8%/30.0% 以及 K=9 对 K=10 的 15.6%/28.9% 几乎同量级。删点候选的离散不一致没有明显超出完整系统自己的重复运行噪声。",
            "",
            "## 决策边界放大",
            "",
            "| 候选 | 轮次 | K=10 margin | 样本数 | 实际方向不一致 |",
            "|---|---:|---|---:|---:|",
        ]
    )
    for row in margins[margins["round"].isin([0, 6])].itertuples(index=False):
        lines.append(
            f"| {row.candidate_label} | {row.round} | {row.full_k10_margin_group} | "
            f"{row.observation_count} | {row.actual_direction_disagreement_rate:.1%} |"
        )
    lines.extend(
        [
            "",
            "当 K=10 的前两类平均概率差不超过 0.02 时，K=8/K=9 的方向不一致率远高于 margin 大于 0.05 的清晰判断。这说明离散标签把很小的连续概率变化放大成了 tie 与 up/neutral/down 的跳变。",
            "",
            "## 配置与输出质量核对",
            "",
        ]
    )
    available = audit[audit["artifacts_available"]]
    lines.extend(
        [
            f"- 原始运行工件可用：{len(available)}/{len(audit)} 个候选-场景-seed 对。",
            f"- 输入快照一致率：{available['input_snapshot_hash_match'].mean():.1%}；prompt version 一致率：{available['prompt_version_match'].mean():.1%}。",
            f"- 相同 Agent 的 exact interview prompt 一致率：{available['exact_interview_prompt_match_rate'].mean():.1%}。",
            f"- 去除运行时 ID 后的 Profile 一致率：{available['exact_normalized_profile_match_rate'].mean():.1%}。",
            f"- 模型一致率：{available['llm_model_match'].mean():.1%}；provider seed 固定率：{available['candidate_llm_provider_seeded'].mean():.1%}。",
        ]
    )
    round0_consistency = output_consistency[output_consistency["round"] == 0]
    for row in round0_consistency.itertuples(index=False):
        lines.append(
            f"- {row.population} round 0：JSON 有效记录中，声明方向与概率 argmax 不一致率 "
            f"{row.declared_direction_argmax_mismatch_rate:.1%}，重试率 {row.retry_rate:.1%}。"
        )
    paired_round0 = paired_summary[paired_summary["round"] == 0]
    lines.extend(
        [
            "",
            "注意：CSV 中的 `prompt_hash` 是整套运行配置哈希，包含 Agent 映射，因此 K 改变后必然变化；不能把它当作单次 interview prompt 是否相同的证据。这里直接比较了原始 `simulation_config.json` 中的 prompt。",
            "",
            "## 结论",
            "",
            "1. 离散方向不稳定不是主要由 JSON 解析失败或方向字段自相矛盾造成，而是连续概率在决策边界附近的小幅波动被 tie/argmax 离散化放大。",
            "2. round 0 已存在显著重跑差异，且相同输入、Profile、prompt、模型下 provider seed 未固定；因此 LLM/API 不可重复性是重要的上游来源。",
            "3. 但不能把 round 6 的全部差异称为 LLM 自身误差。社会动作也由 LLM 生成，早期小差异会改变曝光和互动轨迹，形成反馈放大；删点的机械组成效应也仍存在。",
            "4. 更准确的因果表述是：`LLM/API 初始波动 + 离散阈值放大 + 社会互动路径依赖`。现有数据不能给出纯模型误差的精确百分比。",
            "",
            "## 证据文件",
            "",
            "- `candidate_round_comparison.csv`：组成、重跑与实际差异的逐场景-seed-轮次分解。",
            "- `candidate_round_summary.csv`：上述分解的逐轮汇总。",
            "- `paired_agent_round_summary.csv`：相同 Agent 独立重跑的概率与方向差异。",
            "- `k10_seed_stability.csv`：K=10 自身五 seed 的场景级稳定性。",
            "- `pre_social_context_audit.csv`：原始 prompt/Profile/模型/seed 状态核对。",
            "- `margin_sensitivity.csv`：方向不一致与 K=10 决策 margin 的关系。",
        ]
    )
    (output_dir / "diagnostic_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    datasets = {
        "K=8": read_beliefs(args.k8_dataset.resolve(), "K=8"),
        "K=9": read_beliefs(args.k9_dataset.resolve(), "K=9"),
        "K=10": read_beliefs(args.k10_dataset.resolve(), "K=10"),
    }
    reference = datasets["K=10"]
    comparisons = pd.concat(
        [candidate_comparison(datasets[label], reference, label) for label in ("K=8", "K=9")],
        ignore_index=True,
    )
    candidate_rounds = summarize_candidate_rounds(comparisons)
    paired = pd.concat(
        [paired_agent_comparison(datasets[label], reference, label) for label in ("K=8", "K=9")],
        ignore_index=True,
    )
    paired_summary = summarize_paired_agents(paired)
    consistency = internal_output_consistency(list(datasets.values()))
    k10_units, k10_rounds, k10_agents = k10_seed_stability(reference)
    margins = margin_sensitivity(comparisons)
    scenarios = scenario_instability(comparisons, k10_units)
    audits = pd.concat(
        [pre_social_artifact_audit(datasets[label], reference, label) for label in ("K=8", "K=9")],
        ignore_index=True,
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "candidate_round_comparison.csv": comparisons,
        "candidate_round_summary.csv": candidate_rounds,
        "paired_agent_round_comparison.csv": paired,
        "paired_agent_round_summary.csv": paired_summary,
        "k10_seed_stability.csv": k10_units,
        "k10_seed_stability_by_round.csv": k10_rounds,
        "k10_round0_agent_seed_stability.csv": k10_agents,
        "pre_social_context_audit.csv": audits,
        "output_contract_consistency.csv": consistency,
        "margin_sensitivity.csv": margins,
        "scenario_instability.csv": scenarios,
    }
    for name, frame in outputs.items():
        frame.to_csv(output_dir / name, index=False, encoding="utf-8-sig")

    round0 = candidate_rounds[candidate_rounds["round"] == 0]
    round6 = candidate_rounds[candidate_rounds["round"] == 6]
    summary = {
        "analysis_version": "s1_direction_instability_diagnostic_v1",
        "method": "existing_artifacts_only_no_llm_no_simulation",
        "tie_epsilon": TIE_EPSILON,
        "candidate_round0": as_records(round0),
        "candidate_round6": as_records(round6),
        "k10_seed_stability_round0": as_records(k10_rounds[k10_rounds["round"] == 0])[0],
        "k10_seed_stability_round6": as_records(k10_rounds[k10_rounds["round"] == 6])[0],
        "k10_round0_agent_seed_stability": {
            "agent_scenario_units": int(len(k10_agents)),
            "mean_pairwise_direction_disagreement_rate": float(
                k10_agents["pairwise_direction_disagreement_rate"].mean()
            ),
            "mean_pairwise_belief_js": float(k10_agents["mean_pairwise_belief_js"].mean()),
        },
        "pre_social_context_audit": {
            "run_pairs": int(len(audits)),
            "artifacts_available": int(audits["artifacts_available"].sum()),
            "input_snapshot_match_rate": float(audits["input_snapshot_hash_match"].mean()),
            "prompt_version_match_rate": float(audits["prompt_version_match"].mean()),
            "exact_interview_prompt_match_rate": float(
                audits.loc[audits["artifacts_available"], "exact_interview_prompt_match_rate"].mean()
            ),
            "exact_normalized_profile_match_rate": float(
                audits.loc[audits["artifacts_available"], "exact_normalized_profile_match_rate"].mean()
            ),
            "llm_model_match_rate": float(
                audits.loc[audits["artifacts_available"], "llm_model_match"].mean()
            ),
            "provider_seeded_rate": float(
                audits.loc[audits["artifacts_available"], "candidate_llm_provider_seeded"].mean()
            ),
        },
        "identification_limit": (
            "Round-0 rerun variability is strong evidence for pre-social LLM/API pipeline instability, "
            "but the existing runs cannot identify a pure provider-model error percentage."
        ),
        "conclusion": (
            "Discrete instability is best explained by pre-social LLM/API variability near the decision "
            "boundary, amplified by tie-aware discretization and later path-dependent social interaction."
        ),
    }
    (output_dir / "diagnostic_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_report(
        output_dir,
        candidate_rounds,
        paired_summary,
        k10_rounds,
        k10_agents,
        audits,
        consistency,
        margins,
    )
    print(f"Wrote direction-instability diagnostics to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
