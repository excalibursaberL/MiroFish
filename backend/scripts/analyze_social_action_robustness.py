#!/usr/bin/env python3
"""Support-aware robustness check for K=8/K=9 action distributions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
ACTION_LABELS = [
    "create_comment", "create_post", "like_comment", "like_post",
    "dislike_comment", "dislike_post", "follow",
]
ROUNDS = list(range(2, 7))
SEEDS = [42, 999, 2887, 3407, 4004]
SUPPORT_MIN = 10
HIGH_JS = 0.5083547580976512


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-label", required=True)
    return parser.parse_args(argv)


def js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    left_total = left.sum()
    right_total = right.sum()
    left = left / left_total if left_total > 0 else np.zeros_like(left)
    right = right / right_total if right_total > 0 else np.zeros_like(right)
    middle = 0.5 * (left + right)
    result = 0.0
    for values in (left, right):
        positive = values > 0
        result += 0.5 * float(np.sum(values[positive] * np.log2(values[positive] / middle[positive])))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    frame = pd.read_csv(
        args.diagnostic_dir.resolve() / "scenario_round_action_diagnostic.csv",
        low_memory=False,
    )
    frame["seed"] = pd.to_numeric(frame["seed"], errors="raise").astype(int)
    frame["round"] = pd.to_numeric(frame["round"], errors="raise").astype(int)
    for column in ("k9_observation_count", "k10_observation_count"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    frame = frame[frame["round"].isin(ROUNDS)].copy()

    seed_rows: list[dict[str, object]] = []
    for (scenario_id, seed), group in frame.groupby(["scenario_id", "seed"], sort=True):
        supported = group[
            (group["k9_observation_count"] >= SUPPORT_MIN)
            & (group["k10_observation_count"] >= SUPPORT_MIN)
        ]
        k9_counts = np.array([
            (group[f"k9_p_{label}"] * group["k9_observation_count"]).sum()
            for label in ACTION_LABELS
        ], dtype=float)
        k10_counts = np.array([
            (group[f"k10_p_{label}"] * group["k10_observation_count"]).sum()
            for label in ACTION_LABELS
        ], dtype=float)
        seed_rows.append({
            "scenario_id": scenario_id,
            "seed": int(seed),
            "supported_round_count": int(len(supported)),
            "mean_supported_js": float(supported["action_js"].mean()) if len(supported) else np.nan,
            "max_supported_js": float(supported["action_js"].max()) if len(supported) else np.nan,
            "high_js_supported_round_count": int((supported["action_js"] >= HIGH_JS).sum()),
            "pooled_k9_observation_count": float(k9_counts.sum()),
            "pooled_k10_observation_count": float(k10_counts.sum()),
            "pooled_action_js": js_divergence(k9_counts, k10_counts),
        })
    seed_summary = pd.DataFrame(seed_rows)

    scenario_rows: list[dict[str, object]] = []
    for scenario_id, group in seed_summary.groupby("scenario_id", sort=True):
        scenario_rows.append({
            "scenario_id": scenario_id,
            "seed_count_with_2_high_js_rounds": int((group["high_js_supported_round_count"] >= 2).sum()),
            "seed_count_with_pooled_js_ge_threshold": int((group["pooled_action_js"] >= HIGH_JS).sum()),
            "pooled_action_js_median": float(group["pooled_action_js"].median()),
            "pooled_action_js_p90_seed": float(group["pooled_action_js"].quantile(0.90)),
            "pooled_action_gate_scenario_error": float(group["pooled_action_js"].quantile(0.80)),
        })
    scenario_summary = pd.DataFrame(scenario_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_summary.to_csv(args.output_dir / "scenario_seed_action_robustness.csv", index=False, encoding="utf-8-sig")
    scenario_summary.to_csv(args.output_dir / "scenario_action_robustness.csv", index=False, encoding="utf-8-sig")

    pooled_p90 = float(scenario_summary["pooled_action_gate_scenario_error"].quantile(0.90))
    persistent = scenario_summary[
        (scenario_summary["seed_count_with_2_high_js_rounds"] >= 3)
        & (scenario_summary["seed_count_with_pooled_js_ge_threshold"] >= 3)
    ]
    summary = {
        "candidate_label": args.candidate_label,
        "rounds": ROUNDS,
        "support_min_per_system": SUPPORT_MIN,
        "high_js_threshold": HIGH_JS,
        "supported_rows": int(((frame["k9_observation_count"] >= SUPPORT_MIN) & (frame["k10_observation_count"] >= SUPPORT_MIN)).sum()),
        "total_rows": int(len(frame)),
        "persistent_scenarios": persistent["scenario_id"].tolist(),
        "pooled_action_gate_p90_across_scenarios": pooled_p90,
        "pooled_action_gate_pass": pooled_p90 <= HIGH_JS,
        "scenario_summary": scenario_summary.to_dict("records"),
    }
    (args.output_dir / "action_robustness_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        f"# {args.candidate_label} Social-Action Robustness Diagnostic",
        "",
        "Rounds 2-6 are used; both systems need at least 10 action observations per round.",
        f"The frozen action-JS threshold is {HIGH_JS:.4f}.",
        "",
        f"- Supported rows: {summary['supported_rows']}/{summary['total_rows']}.",
        f"- Persistent scenarios: {', '.join(summary['persistent_scenarios']) or 'none'}.",
        f"- Pooled action-JS P90: {pooled_p90:.4f}; pass: {summary['pooled_action_gate_pass']}.",
        "",
        "| Scenario | Seeds with >=2 high-JS rounds | Pooled JS median | Pooled JS P90 |",
        "|---|---:|---:|---:|",
    ]
    for row in scenario_summary.sort_values(
        ["seed_count_with_2_high_js_rounds", "pooled_action_js_p90_seed"], ascending=False
    ).to_dict("records"):
        lines.append(
            f"| {row['scenario_id']} | {int(row['seed_count_with_2_high_js_rounds'])}/5 | "
            f"{row['pooled_action_js_median']:.4f} | {row['pooled_action_js_p90_seed']:.4f} |"
        )
    (args.output_dir / "action_robustness_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
