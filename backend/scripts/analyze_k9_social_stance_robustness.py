#!/usr/bin/env python3
"""Check whether K=9 social-stance differences persist with adequate support."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
SCENARIOS = [f"SCN_{index:03d}" for index in range(1, 19)]
SEEDS = [42, 999, 2887, 3407, 4004]
LABELS = ["positive", "mixed", "negative", "neutral", "uncertain"]
SOCIAL_ROUNDS = list(range(2, 7))
SUPPORT_MIN = 10
HIGH_JS = 0.35


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diagnostic-dir",
        type=Path,
        default=ROOT / "Dataset" / "s1_k9_scenario_diagnostic_v1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "Dataset" / "s1_k9_social_stance_robustness_v1",
    )
    parser.add_argument("--candidate-label", default="K=9")
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


def read_stance(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path / "scenario_round_stance_diagnostic.csv", low_memory=False)
    frame["seed"] = pd.to_numeric(frame["seed"], errors="raise").astype(int)
    frame["round"] = pd.to_numeric(frame["round"], errors="raise").astype(int)
    for prefix in ("k9", "k10"):
        frame[f"{prefix}_observation_count"] = pd.to_numeric(
            frame[f"{prefix}_observation_count"], errors="coerce"
        ).fillna(0.0)
    return frame[frame["round"].isin(SOCIAL_ROUNDS)].copy()


def build_seed_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (scenario_id, seed), group in frame.groupby(["scenario_id", "seed"], sort=True):
        supported = group[
            (group["k9_observation_count"] >= SUPPORT_MIN)
            & (group["k10_observation_count"] >= SUPPORT_MIN)
        ].copy()
        k9_counts = np.zeros(len(LABELS), dtype=float)
        k10_counts = np.zeros(len(LABELS), dtype=float)
        for index, label in enumerate(LABELS):
            k9_counts[index] = float(
                (group[f"k9_p_{label}"] * group["k9_observation_count"]).sum()
            )
            k10_counts[index] = float(
                (group[f"k10_p_{label}"] * group["k10_observation_count"]).sum()
            )
        rows.append(
            {
                "scenario_id": scenario_id,
                "seed": int(seed),
                "social_round_count": len(group),
                "supported_round_count": len(supported),
                "supported_round_rate": len(supported) / len(group) if len(group) else 0.0,
                "mean_supported_js": float(supported["stance_js"].mean()) if len(supported) else np.nan,
                "max_supported_js": float(supported["stance_js"].max()) if len(supported) else np.nan,
                "high_js_supported_round_count": int((supported["stance_js"] >= HIGH_JS).sum()),
                "pooled_k9_observation_count": float(k9_counts.sum()),
                "pooled_k10_observation_count": float(k10_counts.sum()),
                "pooled_stance_js": js_divergence(k9_counts, k10_counts),
            }
        )
    return pd.DataFrame(rows)


def build_scenario_summary(seed_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scenario_id, group in seed_summary.groupby("scenario_id", sort=True):
        rows.append(
            {
                "scenario_id": scenario_id,
                "seed_count": len(group),
                "seed_count_with_2_high_js_rounds": int(
                    (group["high_js_supported_round_count"] >= 2).sum()
                ),
                "seed_rate_with_2_high_js_rounds": float(
                    (group["high_js_supported_round_count"] >= 2).mean()
                ),
                "seed_count_with_pooled_js_ge_0_35": int(
                    (group["pooled_stance_js"] >= HIGH_JS).sum()
                ),
                "pooled_js_median": float(group["pooled_stance_js"].median()),
                "pooled_js_p90_seed": float(group["pooled_stance_js"].quantile(0.90)),
                "pooled_gate_scenario_error": float(group["pooled_stance_js"].quantile(0.80)),
                "mean_supported_round_rate": float(group["supported_round_rate"].mean()),
                "mean_pooled_k9_observations": float(group["pooled_k9_observation_count"].mean()),
                "mean_pooled_k10_observations": float(group["pooled_k10_observation_count"].mean()),
            }
        )
    return pd.DataFrame(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    stance = read_stance(args.diagnostic_dir.resolve())
    seed_summary = build_seed_summary(stance)
    scenario_summary = build_scenario_summary(seed_summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_summary.to_csv(args.output_dir / "scenario_seed_stance_robustness.csv", index=False, encoding="utf-8-sig")
    scenario_summary.to_csv(args.output_dir / "scenario_stance_robustness.csv", index=False, encoding="utf-8-sig")

    persistent = scenario_summary[
        (scenario_summary["seed_count_with_2_high_js_rounds"] >= 3)
        & (scenario_summary["seed_count_with_pooled_js_ge_0_35"] >= 3)
    ]
    summary = {
        "candidate_label": args.candidate_label,
        "input_rows": int(len(stance)),
        "scenario_count": int(len(scenario_summary)),
        "scenario_seed_count": int(len(seed_summary)),
        "social_rounds": SOCIAL_ROUNDS,
        "support_min_per_system": SUPPORT_MIN,
        "high_js_threshold": HIGH_JS,
        "supported_rows": int(
            ((stance["k9_observation_count"] >= SUPPORT_MIN) & (stance["k10_observation_count"] >= SUPPORT_MIN)).sum()
        ),
        "persistent_rule": {
            "per_seed": "at least 2 supported rounds with stance JS >= 0.35",
            "per_scenario": "at least 3 of 5 seeds satisfy the per-seed rule and at least 3 seeds have pooled JS >= 0.35",
        },
        "persistent_scenarios": persistent["scenario_id"].tolist(),
        "pooled_gate_p90_across_scenarios": float(scenario_summary["pooled_gate_scenario_error"].quantile(0.90)),
        "pooled_gate_pass_at_0_35": bool(scenario_summary["pooled_gate_scenario_error"].quantile(0.90) <= HIGH_JS),
        "scenario_summary": scenario_summary.to_dict("records"),
    }
    (args.output_dir / "robustness_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        f"# {args.candidate_label} Social-Stance Robustness Diagnostic",
        "",
        "This is a support-aware, descriptive comparison. It does not change the frozen validation gate.",
        "",
        "## Definition",
        "",
        f"- Rounds used: {SOCIAL_ROUNDS}; both K=9 and K=10 need at least {SUPPORT_MIN} exposure observations.",
        f"- A persistent seed requires at least 2 supported rounds with JS >= {HIGH_JS:.2f}.",
        "- A persistent scenario requires this pattern in at least 3 of 5 seeds and pooled JS >= 0.35 in at least 3 seeds.",
        "",
        "## Result",
        "",
        f"- Supported rows: {summary['supported_rows']}/{summary['input_rows']}.",
        f"- Persistent scenarios under the predeclared diagnostic rule: {', '.join(summary['persistent_scenarios']) or 'none'}.",
        f"- Pooled stance gate P90 across scenarios: {summary['pooled_gate_p90_across_scenarios']:.4f}; "
        f"pass at 0.35: {summary['pooled_gate_pass_at_0_35']}.",
        "",
        "## Scenario table",
        "",
        "| Scenario | Seeds with >=2 high-JS rounds | Seed rate | Pooled JS median | Pooled JS P90 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in scenario_summary.sort_values(
        ["seed_count_with_2_high_js_rounds", "pooled_js_p90_seed"], ascending=False
    ).to_dict("records"):
        lines.append(
            f"| {row['scenario_id']} | {int(row['seed_count_with_2_high_js_rounds'])}/5 | "
            f"{row['seed_rate_with_2_high_js_rounds']:.2f} | {row['pooled_js_median']:.4f} | "
            f"{row['pooled_js_p90_seed']:.4f} |"
        )
    (args.output_dir / "robustness_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
