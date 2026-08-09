#!/usr/bin/env python3
"""Diagnose scenario-level K=9 fidelity failures without calling an LLM."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
SCENARIOS = [f"SCN_{index:03d}" for index in range(1, 19)]
SEEDS = [42, 999, 2887, 3407, 4004]
ROUNDS = list(range(7))
SOCIAL_ROUNDS = list(range(1, 7))
PROBABILITY_COLUMNS = ["up_probability", "neutral_probability", "down_probability"]
STANCE_LABELS = ["positive", "mixed", "negative", "neutral", "uncertain"]
ACTION_LABELS = [
    "create_comment", "create_post", "like_comment", "like_post",
    "dislike_comment", "dislike_post", "follow",
]
TIE_EPSILON = 0.02
GROUP_COLUMNS = ["scenario_id", "seed", "round"]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
        default=ROOT / "Dataset" / "s1_k9_scenario_diagnostic_v1",
    )
    parser.add_argument("--candidate-label", default="K=9")
    return parser.parse_args(argv)


def read_csv(path: Path, *, merged: bool = False) -> pd.DataFrame:
    name = f"merged_{path.name}" if merged else path.name
    actual = path.parent / name
    return pd.read_csv(actual, low_memory=False)


def key_frame(rounds: Iterable[int]) -> pd.DataFrame:
    return pd.DataFrame(
        [(scenario, seed, round_number) for scenario in SCENARIOS for seed in SEEDS for round_number in rounds],
        columns=GROUP_COLUMNS,
    )


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def direction_and_margin(probabilities: Sequence[float]) -> tuple[str, float, bool]:
    values = np.asarray(probabilities, dtype=float)
    order = np.argsort(values)
    margin = float(values[order[-1]] - values[order[-2]])
    tie = margin <= TIE_EPSILON
    return ("tie" if tie else ("up", "neutral", "down")[int(order[-1])]), margin, tie


def js_divergence(left: Sequence[float], right: Sequence[float]) -> float:
    left_values = np.asarray(left, dtype=float)
    right_values = np.asarray(right, dtype=float)
    left_total = left_values.sum()
    right_total = right_values.sum()
    if left_total > 0:
        left_values = left_values / left_total
    else:
        left_values = np.zeros_like(left_values)
    if right_total > 0:
        right_values = right_values / right_total
    else:
        right_values = np.zeros_like(right_values)
    middle = 0.5 * (left_values + right_values)
    result = 0.0
    for values, target in ((left_values, middle), (right_values, middle)):
        for value, reference in zip(values, target):
            if value > 0 and reference > 0:
                result += 0.5 * value * math.log2(value / reference)
    return float(result)


def probability_panel(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame[frame["round"].isin(ROUNDS)].copy()
    for column in PROBABILITY_COLUMNS + ["expected_return", "confidence"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame[frame[PROBABILITY_COLUMNS + ["expected_return", "confidence"]].notna().all(axis=1)]
    grouped = frame.groupby(GROUP_COLUMNS, as_index=False)[PROBABILITY_COLUMNS + ["expected_return", "confidence"]].mean()
    directions = grouped[PROBABILITY_COLUMNS].apply(lambda row: direction_and_margin(row.to_numpy()), axis=1)
    grouped["majority"] = [value[0] for value in directions]
    grouped["margin"] = [value[1] for value in directions]
    grouped["near_tie"] = [value[2] for value in directions]
    return key_frame(ROUNDS).merge(grouped, on=GROUP_COLUMNS, how="left", validate="one_to_one")


def attach_probability_comparison(k9: pd.DataFrame, k10: pd.DataFrame) -> pd.DataFrame:
    left = k9.rename(columns={column: f"k9_{column}" for column in PROBABILITY_COLUMNS + ["expected_return", "confidence", "majority", "margin", "near_tie"]})
    right = k10.rename(columns={column: f"k10_{column}" for column in PROBABILITY_COLUMNS + ["expected_return", "confidence", "majority", "margin", "near_tie"]})
    merged = left.merge(right, on=GROUP_COLUMNS, how="inner", validate="one_to_one")
    merged["majority_disagreement"] = merged["k9_majority"] != merged["k10_majority"]
    merged["belief_js"] = merged.apply(
        lambda row: js_divergence(
            [row[f"k9_{column}"] for column in PROBABILITY_COLUMNS],
            [row[f"k10_{column}"] for column in PROBABILITY_COLUMNS],
        ),
        axis=1,
    )
    merged["expected_return_abs_error"] = (merged["k9_expected_return"] - merged["k10_expected_return"]).abs()
    merged["margin_abs_delta"] = (merged["k9_margin"] - merged["k10_margin"]).abs()
    return merged


def distribution_panel(frame: pd.DataFrame, kind: str) -> pd.DataFrame:
    keys = key_frame(SOCIAL_ROUNDS)
    if kind == "stance":
        eligible = frame[
            frame["round"].isin(SOCIAL_ROUNDS)
            & frame["author_class"].astype(str).str.lower().eq("investor")
            & ~frame["is_self_authored"].astype(bool)
        ].copy()
        eligible["label"] = eligible["content_stance"].astype(str).str.lower().where(
            eligible["content_stance"].astype(str).str.lower().isin(STANCE_LABELS), "uncertain"
        )
        labels = STANCE_LABELS
    else:
        eligible = frame[frame["round"].isin(SOCIAL_ROUNDS) & frame["actor_class"].astype(str).str.lower().eq("investor")].copy()
        eligible["label"] = eligible["action_type"].astype(str).str.lower()
        labels = ACTION_LABELS
    counts = eligible.groupby(GROUP_COLUMNS + ["label"]).size().unstack(fill_value=0).reset_index()
    for label in labels:
        if label not in counts:
            counts[label] = 0
    result = keys.merge(counts, on=GROUP_COLUMNS, how="left", validate="one_to_one").fillna(0)
    values = result[labels].to_numpy(dtype=float)
    totals = values.sum(axis=1, keepdims=True)
    values = np.divide(values, totals, out=np.zeros_like(values), where=totals > 0)
    for index, label in enumerate(labels):
        result[f"p_{label}"] = values[:, index]
    result["observation_count"] = totals[:, 0]
    return result[GROUP_COLUMNS + [f"p_{label}" for label in labels] + ["observation_count"]]


def compare_distributions(k9: pd.DataFrame, k10: pd.DataFrame, kind: str) -> pd.DataFrame:
    left = distribution_panel(k9, kind)
    right = distribution_panel(k10, kind)
    labels = STANCE_LABELS if kind == "stance" else ACTION_LABELS
    left = left.rename(columns={f"p_{label}": f"k9_p_{label}" for label in labels}).rename(columns={"observation_count": "k9_observation_count"})
    right = right.rename(columns={f"p_{label}": f"k10_p_{label}" for label in labels}).rename(columns={"observation_count": "k10_observation_count"})
    result = left.merge(right, on=GROUP_COLUMNS, how="inner", validate="one_to_one")
    result[f"{kind}_js"] = result.apply(
        lambda row: js_divergence(
            [row[f"k9_p_{label}"] for label in labels],
            [row[f"k10_p_{label}"] for label in labels],
        ),
        axis=1,
    )
    return result


def gate_scenario_summary(
    frame: pd.DataFrame,
    value_column: str,
    prefix: str,
    *,
    observation_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Reproduce the frozen gate order: seed P80, then worst round per scenario."""
    grouped = frame.groupby(["scenario_id", "round"], as_index=False).agg(
        gate_value=(value_column, lambda values: float(np.nanquantile(values, 0.80))),
        **{
            f"mean_{column}": (column, "mean")
            for column in observation_columns
        },
    )
    worst_indices = grouped.groupby("scenario_id")["gate_value"].idxmax()
    worst = grouped.loc[worst_indices].copy()
    rename = {
        "round": f"{prefix}_gate_worst_round",
        "gate_value": f"{prefix}_gate_scenario_error",
    }
    rename.update(
        {
            f"mean_{column}": f"{prefix}_gate_worst_round_{column}"
            for column in observation_columns
        }
    )
    return worst.rename(columns=rename)[["scenario_id", *rename.values()]]


def aggregate_scenario(probabilities: pd.DataFrame, stance: pd.DataFrame, action: pd.DataFrame) -> pd.DataFrame:
    final = probabilities[probabilities["round"] == 6].groupby("scenario_id", as_index=False).agg(
        final_majority_disagreement_rate=("majority_disagreement", "mean"),
        final_majority_disagreement_count=("majority_disagreement", "sum"),
        final_belief_js_mean=("belief_js", "mean"),
        final_belief_js_p90=("belief_js", lambda values: float(np.quantile(values, 0.90))),
        final_return_abs_error_mean=("expected_return_abs_error", "mean"),
        k9_final_near_tie_rate=("k9_near_tie", "mean"),
        k10_final_near_tie_rate=("k10_near_tie", "mean"),
        k9_final_margin_median=("k9_margin", "median"),
        k10_final_margin_median=("k10_margin", "median"),
    )
    trajectory = probabilities.groupby(["scenario_id", "round"], as_index=False).agg(
        disagreement_rate=("majority_disagreement", "mean"),
        belief_js_mean=("belief_js", "mean"),
        return_abs_error_mean=("expected_return_abs_error", "mean"),
    )
    worst = trajectory.groupby("scenario_id", as_index=False).agg(
        trajectory_disagreement_rate=("disagreement_rate", "mean"),
        trajectory_worst_round_disagreement_rate=("disagreement_rate", "max"),
        trajectory_worst_belief_js=("belief_js_mean", "max"),
        trajectory_worst_return_error=("return_abs_error_mean", "max"),
    )
    social = stance.groupby(["scenario_id", "round"], as_index=False).agg(
        stance_js_mean=("stance_js", "mean"),
        stance_js_max=("stance_js", "max"),
        stance_observations_k9=("k9_observation_count", "mean"),
        stance_observations_k10=("k10_observation_count", "mean"),
    )
    action_summary = action.groupby(["scenario_id", "round"], as_index=False).agg(
        action_js_mean=("action_js", "mean"),
        action_js_max=("action_js", "max"),
        action_observations_k9=("k9_observation_count", "mean"),
        action_observations_k10=("k10_observation_count", "mean"),
    )
    social_summary = social.groupby("scenario_id", as_index=False).agg(
        stance_js_trajectory_mean=("stance_js_mean", "mean"),
        stance_js_worst_round=("stance_js_mean", "max"),
        stance_js_p90_seed=("stance_js_max", lambda values: float(np.quantile(values, 0.90))),
        stance_observations_k9=("stance_observations_k9", "mean"),
        stance_observations_k10=("stance_observations_k10", "mean"),
    )
    action_summary = action_summary.groupby("scenario_id", as_index=False).agg(
        action_js_trajectory_mean=("action_js_mean", "mean"),
        action_js_worst_round=("action_js_mean", "max"),
        action_js_p90_seed=("action_js_max", lambda values: float(np.quantile(values, 0.90))),
        action_observations_k9=("action_observations_k9", "mean"),
        action_observations_k10=("action_observations_k10", "mean"),
    )
    belief_gate = gate_scenario_summary(probabilities, "belief_js", "belief")
    return_gate = gate_scenario_summary(probabilities, "expected_return_abs_error", "return")
    stance_gate = gate_scenario_summary(
        stance,
        "stance_js",
        "stance",
        observation_columns=("k9_observation_count", "k10_observation_count"),
    )
    stance_gate_no_round1 = gate_scenario_summary(
        stance[stance["round"] >= 2],
        "stance_js",
        "stance_no_round1",
    )
    stance_supported = stance[
        (stance["k9_observation_count"] >= 10)
        & (stance["k10_observation_count"] >= 10)
    ]
    stance_gate_supported = gate_scenario_summary(
        stance_supported,
        "stance_js",
        "stance_min_support_10",
    )
    action_gate = gate_scenario_summary(
        action,
        "action_js",
        "action",
        observation_columns=("k9_observation_count", "k10_observation_count"),
    )
    return (
        final.merge(worst, on="scenario_id")
        .merge(social_summary, on="scenario_id")
        .merge(action_summary, on="scenario_id")
        .merge(belief_gate, on="scenario_id")
        .merge(return_gate, on="scenario_id")
        .merge(stance_gate, on="scenario_id")
        .merge(stance_gate_no_round1, on="scenario_id")
        .merge(stance_gate_supported, on="scenario_id")
        .merge(action_gate, on="scenario_id")
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    k9_dir = args.k9_dataset.resolve()
    k10_dir = args.k10_dataset.resolve()
    k9_belief = read_csv(k9_dir / "belief_snapshots.csv", merged=True)
    k10_belief = read_csv(k10_dir / "belief_snapshots.csv", merged=True)
    k9_exposure = read_csv(k9_dir / "agent_round_content_exposures.csv", merged=True)
    k10_exposure = read_csv(k10_dir / "agent_round_content_exposures.csv", merged=True)
    k9_edges = read_csv(k9_dir / "interaction_edges.csv", merged=True)
    k10_edges = read_csv(k10_dir / "interaction_edges.csv", merged=True)
    k10_runs = read_csv(k10_dir / "scenario_runs.csv", merged=True)
    for frame in (k9_belief, k10_belief, k9_exposure, k10_exposure, k9_edges, k10_edges, k10_runs):
        frame["scenario_id"] = frame["scenario_id"].astype(str)
        frame["seed"] = pd.to_numeric(frame["seed"], errors="raise").astype(int)
        if "round" in frame:
            frame["round"] = pd.to_numeric(frame["round"], errors="raise").astype(int)
    probability = attach_probability_comparison(probability_panel(k9_belief), probability_panel(k10_belief))
    stance = compare_distributions(k9_exposure, k10_exposure, "stance")
    action = compare_distributions(k9_edges, k10_edges, "action")
    scenario = aggregate_scenario(probability, stance, action)
    actual = k10_runs[["scenario_id", "actual_five_day_close_direction"]].drop_duplicates("scenario_id")
    scenario = actual.merge(scenario, on="scenario_id", how="right", validate="one_to_one")
    scenario["final_disagreement_rank"] = scenario["final_majority_disagreement_rate"].rank(method="min", ascending=False).astype(int)
    scenario["stance_js_rank"] = scenario["stance_js_worst_round"].rank(method="min", ascending=False).astype(int)
    scenario = scenario.sort_values(["final_majority_disagreement_rate", "stance_js_worst_round"], ascending=False)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    probability.to_csv(args.output_dir / "scenario_round_probability_diagnostic.csv", index=False, encoding="utf-8-sig")
    stance.to_csv(args.output_dir / "scenario_round_stance_diagnostic.csv", index=False, encoding="utf-8-sig")
    action.to_csv(args.output_dir / "scenario_round_action_diagnostic.csv", index=False, encoding="utf-8-sig")
    scenario.to_csv(args.output_dir / "scenario_diagnostic.csv", index=False, encoding="utf-8-sig")

    margin_groups = probability[probability["round"] == 6].copy()
    margin_groups["k10_margin_group"] = np.where(margin_groups["k10_margin"] <= TIE_EPSILON, "near_tie_le_0.02", "clear_margin_gt_0.02")
    margin_summary = margin_groups.groupby("k10_margin_group", as_index=False).agg(
        observation_count=("majority_disagreement", "size"),
        disagreement_rate=("majority_disagreement", "mean"),
        mean_k9_margin=("k9_margin", "mean"),
        mean_k10_margin=("k10_margin", "mean"),
    )
    top_final = scenario.head(5)[["scenario_id", "actual_five_day_close_direction", "final_majority_disagreement_rate", "final_belief_js_mean", "k10_final_margin_median"]].to_dict("records")
    top_stance = scenario.sort_values("stance_gate_scenario_error", ascending=False).head(5)[[
        "scenario_id",
        "actual_five_day_close_direction",
        "stance_gate_scenario_error",
        "stance_gate_worst_round",
        "stance_gate_worst_round_k9_observation_count",
        "stance_gate_worst_round_k10_observation_count",
    ]].to_dict("records")
    final_rows = probability[probability["round"] == 6]
    final_disagreements = final_rows[final_rows["majority_disagreement"]]
    stance_low_support = (
        (stance["k9_observation_count"] < 10)
        | (stance["k10_observation_count"] < 10)
    )
    stance_high_js = stance["stance_js"] >= 0.30
    round_direction = probability.groupby("round", as_index=False).agg(
        disagreement_rate=("majority_disagreement", "mean"),
        belief_js_mean=("belief_js", "mean"),
        expected_return_abs_error_mean=("expected_return_abs_error", "mean"),
        k10_near_tie_rate=("k10_near_tie", "mean"),
    )
    summary = {
        "candidate_label": args.candidate_label,
        "scenario_count": len(scenario),
        "round_seed_probability_rows": len(probability),
        "stance_rows": len(stance),
        "action_rows": len(action),
        "top_final_direction_disagreement": top_final,
        "top_social_stance_js": top_stance,
        "margin_sensitivity": margin_summary.to_dict("records"),
        "final_direction_disagreement": {
            "observation_count": int(len(final_rows)),
            "disagreement_count": int(len(final_disagreements)),
            "disagreement_rate": float(final_rows["majority_disagreement"].mean()),
            "either_system_tie_count": int(
                ((final_disagreements["k9_majority"] == "tie") | (final_disagreements["k10_majority"] == "tie")).sum()
            ),
            "direct_up_down_count": int(
                (
                    ((final_disagreements["k9_majority"] == "up") & (final_disagreements["k10_majority"] == "down"))
                    | ((final_disagreements["k9_majority"] == "down") & (final_disagreements["k10_majority"] == "up"))
                ).sum()
            ),
        },
        "round_direction": round_direction.to_dict("records"),
        "stance_support": {
            "observation_rows": int(len(stance)),
            "either_system_below_10_count": int(stance_low_support.sum()),
            "either_system_below_10_rate": float(stance_low_support.mean()),
            "stance_js_ge_0_30_count": int(stance_high_js.sum()),
            "stance_js_ge_0_30_and_low_support_count": int((stance_high_js & stance_low_support).sum()),
            "formal_gate_p90": float(np.nanquantile(scenario["stance_gate_scenario_error"], 0.90)),
            "diagnostic_p90_excluding_round1": float(
                np.nanquantile(scenario["stance_no_round1_gate_scenario_error"], 0.90)
            ),
            "diagnostic_p90_min_support_10": float(
                np.nanquantile(scenario["stance_min_support_10_gate_scenario_error"], 0.90)
            ),
        },
        "tie_epsilon": TIE_EPSILON,
        "interpretation": "Scenario-level diagnostics are descriptive comparisons of independently rerun K=9 and K=10 systems; they do not identify causal influence.",
    }
    (args.output_dir / "diagnostic_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        f"# {args.candidate_label} Scenario-Level Diagnostic",
        "",
        "This report compares the independently rerun K=9 candidate with K=10 at scenario-seed-round resolution.",
        "It is a descriptive diagnostic and does not identify causal influence.",
        "",
        "## Highest final majority-direction disagreement",
        "",
        "| Scenario | Actual direction | Final disagreement rate | Final belief JS mean | K=10 final margin median |",
        "|---|---|---:|---:|---:|",
    ]
    for row in top_final:
        lines.append(f"| {row['scenario_id']} | {row['actual_five_day_close_direction']} | {row['final_majority_disagreement_rate']:.3f} | {row['final_belief_js_mean']:.4f} | {row['k10_final_margin_median']:.4f} |")
    lines.extend([
        "",
        "## Highest social-stance JS",
        "",
        "| Scenario | Actual direction | Gate-consistent stance JS | Worst round | Mean K=9 observations | Mean K=10 observations |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in top_stance:
        lines.append(
            f"| {row['scenario_id']} | {row['actual_five_day_close_direction']} | "
            f"{row['stance_gate_scenario_error']:.4f} | {int(row['stance_gate_worst_round'])} | "
            f"{row['stance_gate_worst_round_k9_observation_count']:.1f} | "
            f"{row['stance_gate_worst_round_k10_observation_count']:.1f} |"
        )
    lines.extend(["", "## Margin sensitivity", "", "| K=10 final margin group | Observations | Disagreement rate | Mean K=9 margin | Mean K=10 margin |", "|---|---:|---:|---:|---:|"])
    for row in margin_summary.to_dict("records"):
        lines.append(f"| {row['k10_margin_group']} | {int(row['observation_count'])} | {row['disagreement_rate']:.3f} | {row['mean_k9_margin']:.4f} | {row['mean_k10_margin']:.4f} |")
    lines.extend([
        "",
        "## Support diagnostics",
        "",
        f"- Final K=9/K=10 direction disagreements: {len(final_disagreements)}/{len(final_rows)}; "
        f"{summary['final_direction_disagreement']['either_system_tie_count']} involve a tie label and "
        f"{summary['final_direction_disagreement']['direct_up_down_count']} are direct up/down reversals.",
        f"- Stance rows with fewer than 10 observations in either system: {int(stance_low_support.sum())}/{len(stance)}.",
        f"- Stance rows with JS >= 0.30: {int(stance_high_js.sum())}; "
        f"{int((stance_high_js & stance_low_support).sum())} of them have low support.",
        f"- Formal stance gate P90: {summary['stance_support']['formal_gate_p90']:.4f}; "
        f"diagnostic P90 excluding round 1: {summary['stance_support']['diagnostic_p90_excluding_round1']:.4f}; "
        f"diagnostic P90 using only rows with at least 10 observations in both systems: "
        f"{summary['stance_support']['diagnostic_p90_min_support_10']:.4f}.",
        "",
        "## Reading guide",
        "",
        "- A high final disagreement rate means K=9 and K=10 choose different tie-aware majority directions in the same scenario-seed units.",
        "- A small belief JS together with a high direction disagreement rate indicates decision-boundary sensitivity rather than a large probability-distribution shift.",
        "- A high stance JS identifies a scenario where the composition of visible investor content differs; it is not evidence that one Agent caused a change.",
        "- Near-tie comparisons are sensitivity diagnostics only; the frozen 0.02 tie rule is not changed after seeing these results.",
    ])
    (args.output_dir / "diagnostic_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
