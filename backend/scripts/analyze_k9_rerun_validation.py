#!/usr/bin/env python3
"""Validate the preferred K=9 real rerun against the frozen K=10 baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from analyze_k8_rerun_validation import (
    DEFAULT_ENUM,
    DEFAULT_K10,
    DIRECTIONS,
    ROUNDS,
    SCENARIOS,
    SEEDS,
    SOCIAL_ROUNDS,
    TABLES,
    aggregate,
    aggregate_direction,
    actual_labels,
    bootstrap_prediction_delta,
    build_system,
    forecast_metrics,
    graph_errors,
    js_divergence,
    k10_source_paths,
    load_k10,
    normalize_frames,
    sha256,
    token_total,
    write_json,
)


ROOT = Path(__file__).resolve().parents[3]
K9 = 9
EXPECTED_FULL_IDS = (1, 3, 4, 5, 9, 11, 12, 14, 17)
EXPECTED_RUNTIME_IDS = "0|1|2|3|4|5|6|8|9"
REFERENCE_SELECTED_INDICES = (0, 1, 2, 3, 4, 5, 6, 8, 9)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k9-datasets", type=Path, nargs=5, required=True)
    parser.add_argument("--k10-dataset", type=Path, default=DEFAULT_K10)
    parser.add_argument("--enumeration-dir", type=Path, default=DEFAULT_ENUM)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "Dataset" / "s1_k9_rerun_validation_round6_seeds42_999_2887_3407_4004_v1",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--random-seed", type=int, default=20260809)
    return parser.parse_args(argv)


def load_k9(paths: Sequence[Path]) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    merged: dict[str, list[pd.DataFrame]] = {name: [] for name in TABLES}
    sources: list[dict[str, Any]] = []
    found_seeds: list[int] = []
    for raw_path in paths:
        path = raw_path.resolve()
        summary = json.loads((path / "analysis_summary.json").read_text(encoding="utf-8"))
        quality = json.loads((path / "quality_report.json").read_text(encoding="utf-8"))
        design = summary["design"]
        seed = int(design["random_seed"])
        if int(design["agent_count_per_scenario"]) != K9:
            raise ValueError(f"expected K=9 dataset: {path}")
        if tuple(design.get("selected_full_population_agent_ids", [])) != EXPECTED_FULL_IDS:
            raise ValueError(f"unexpected K=9 Agent subset: {path}")
        if not quality.get("passed"):
            raise ValueError(f"K=9 source failed quality checks: {path}")
        found_seeds.append(seed)
        files: list[dict[str, Any]] = []
        for name, filename in TABLES.items():
            source = path / filename
            frame = pd.read_csv(source, low_memory=False)
            frame["seed"] = seed
            merged[name].append(frame)
            files.append({"name": filename, "bytes": source.stat().st_size, "sha256": sha256(source)})
        sources.append(
            {
                "seed": seed,
                "path": str(path),
                "quality_passed": True,
                "stance_annotation_count": int(quality["stance_annotation_count"]),
                "stance_annotation_failure_count": int(quality["stance_annotation_failure_count"]),
                "files": files,
            }
        )
    if tuple(sorted(found_seeds)) != SEEDS:
        raise ValueError(f"expected seeds {SEEDS}, found {sorted(found_seeds)}")
    return {name: pd.concat(frames, ignore_index=True) for name, frames in merged.items()}, sources


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.bootstrap_replicates < 100:
        raise ValueError("bootstrap-replicates must be at least 100")
    candidate_paths = [path.resolve() for path in args.k9_datasets]
    candidate, sources = load_k9(candidate_paths)
    reference = load_k10(args.k10_dataset.resolve())
    normalize_frames(candidate)
    normalize_frames(reference)
    k9 = build_system(candidate, K9)
    k10 = build_system(reference, 10)

    thresholds = json.loads((args.enumeration_dir / "thresholds.json").read_text(encoding="utf-8"))
    candidates = pd.read_csv(args.enumeration_dir / "top_candidates.csv")
    offline = candidates[candidates["agent_ids"].astype(str).eq(EXPECTED_RUNTIME_IDS)].iloc[0]

    belief_js = js_divergence(k9["probabilities"], k10["probabilities"])
    return_error = np.abs(k9["expected_return"] - k10["expected_return"])
    majority_error = (k9["majority"] != k10["majority"]).astype(float)
    social_raw = {
        "social_stance_js": js_divergence(k9["stance"], k10["stance"]),
        "social_stance_score_error": np.mean(np.abs(k9["score_quantiles"] - k10["score_quantiles"]), axis=1),
        "social_action_js": js_divergence(k9["action"], k10["action"]),
        "social_content_rate_error": np.abs(k9["content_rate"] - k10["content_rate"]) / np.maximum(1.0, np.abs(k10["content_rate"])),
        "social_participation_error": np.abs(k9["participation"] - k10["participation"]),
        "social_interaction_rate_error": np.abs(k9["interaction_rate"] - k10["interaction_rate"]),
        "social_source_reach_error": np.abs(k9["source_reach"] - k10["source_reach"]),
    }
    strength_raw, active_raw = graph_errors(
        k9["adjacency"], k10["adjacency"],
        reference_selected_indices=REFERENCE_SELECTED_INDICES,
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
        normalized["belief_js"],
        normalized["trajectory_return_error"],
        normalized["majority_trajectory_error"],
        normalized["majority_final_error"],
        social_error,
        graph_error,
    )
    gates = {
        "belief": values["belief_js"] <= thresholds["belief_js"]["threshold"],
        "trajectory_return": values["trajectory_return_error"] <= thresholds["trajectory_return_error"]["threshold"],
        "majority_trajectory": values["majority_trajectory_error"] <= thresholds["majority_trajectory_error"]["threshold"],
        "majority_final": values["majority_final_error"] <= thresholds["majority_final_error"]["threshold"],
        "social": social_error <= 1.0,
        "graph": graph_error <= 1.0,
    }
    labels = actual_labels(reference)
    predictions = {"k9": forecast_metrics(k9, labels), "k10": forecast_metrics(k10, labels)}
    prediction_delta = bootstrap_prediction_delta(k9, k10, labels, args.bootstrap_replicates, args.random_seed)
    k9_tokens = token_total(candidate_paths)
    k10_tokens = token_total(k10_source_paths(args.k10_dataset.resolve()))
    stance_annotation_count = sum(row["stance_annotation_count"] for row in sources)
    stance_annotation_failure_count = sum(row["stance_annotation_failure_count"] for row in sources)
    quality = {
        "passed": (
            k9["snapshot_count"] == 18 * 5 * 7 * K9
            and k9["valid_snapshot_count"] >= k9["snapshot_count"] - 1
            and stance_annotation_failure_count == 0
        ),
        "k9_snapshot_count": k9["snapshot_count"],
        "k9_expected_snapshot_count": 18 * 5 * 7 * K9,
        "k9_valid_snapshot_count": k9["valid_snapshot_count"],
        "k9_invalid_snapshot_count": k9["snapshot_count"] - k9["valid_snapshot_count"],
        "k10_snapshot_count": k10["snapshot_count"],
        "k10_valid_snapshot_count": k10["valid_snapshot_count"],
        "scenario_count": len(SCENARIOS),
        "scenario_seed_count": len(SCENARIOS) * len(SEEDS),
        "seeds": list(SEEDS),
        "role_categories": sorted(candidate["belief"]["agent_role_category"].dropna().astype(str).unique()),
        "stance_annotation_count": stance_annotation_count,
        "stance_annotation_failure_count": stance_annotation_failure_count,
    }
    all_hard_gates_pass = bool(quality["passed"] and all(gates.values()))
    conclusion = (
        "K=9 passes all predeclared real-rerun fidelity gates and is a viable reduced candidate."
        if all_hard_gates_pass
        else "K=9 does not pass every predeclared real-rerun fidelity gate; retain K=10 pending diagnosis."
    )
    result = {
        "analysis_version": "s1_k9_rerun_validation_v1",
        "candidate": {"runtime_ids_in_k10": EXPECTED_RUNTIME_IDS, "full_population_agent_ids": list(EXPECTED_FULL_IDS)},
        "quality": quality,
        "fidelity_metrics": {name: {"value": value, "threshold": float(thresholds[name]["threshold"]), "normalized_error": normalized[name], "pass": value <= float(thresholds[name]["threshold"])} for name, value in values.items()},
        "social_error": social_error,
        "graph_error": graph_error,
        "core_error": core_error,
        "gates": gates,
        "all_hard_gates_pass": all_hard_gates_pass,
        "prediction_metrics": predictions,
        "prediction_delta_k9_minus_k10": prediction_delta,
        "tokens": {"k9_total_round_0_to_6": k9_tokens, "k10_total_round_0_to_6": k10_tokens, "relative_reduction": 1.0 - k9_tokens / k10_tokens if k10_tokens else None},
        "offline_candidate_reference": {"core_error": float(offline["core_error"]), "social_error": float(offline["social_error"]), "graph_error": float(offline["graph_error"])},
        "conclusion": conclusion,
        "method_note": "Frozen thresholds from agent_subset_enumeration_k10_v2; external labels are descriptive checks only.",
        "bootstrap_replicates": args.bootstrap_replicates,
        "sources": sources,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in candidate.items():
        frame.to_csv(args.output_dir / f"merged_{TABLES[name]}", index=False, encoding="utf-8-sig")
    write_json(args.output_dir / "validation_summary.json", result)
    write_json(args.output_dir / "quality_report.json", quality)
    pd.DataFrame([{"metric": name, **row} for name, row in result["fidelity_metrics"].items()]).to_csv(args.output_dir / "fidelity_metrics.csv", index=False, encoding="utf-8-sig")
    lines = ["# K=9 Preferred Candidate Real-Rerun Validation", "", conclusion, "", "| Metric | K=9 | Threshold | Result |", "|---|---:|---:|---|"]
    lines.extend(f"| {name} | {row['value']:.6f} | {row['threshold']:.6f} | {'pass' if row['pass'] else 'fail'} |" for name, row in result["fidelity_metrics"].items())
    lines.extend(
        [
            "",
            f"- core_error: {core_error:.3f}",
            f"- social_error: {social_error:.3f}",
            f"- graph_error: {graph_error:.3f}",
            f"- all_hard_gates_pass: {all_hard_gates_pass}",
            f"- token_reduction_vs_k10: {result['tokens']['relative_reduction']:.2%}",
            f"- stance_annotations: {stance_annotation_count}, failures: {stance_annotation_failure_count}",
            "",
            "| System | Individual accuracy | Balanced accuracy | Brier | Majority accuracy |",
            "|---|---:|---:|---:|---:|",
            f"| K=9 | {predictions['k9']['individual_accuracy']:.4f} | {predictions['k9']['balanced_accuracy']:.4f} | {predictions['k9']['brier_score']:.4f} | {predictions['k9']['majority_accuracy']:.4f} |",
            f"| K=10 | {predictions['k10']['individual_accuracy']:.4f} | {predictions['k10']['balanced_accuracy']:.4f} | {predictions['k10']['brier_score']:.4f} | {predictions['k10']['majority_accuracy']:.4f} |",
            "",
            f"Prediction intervals use {args.bootstrap_replicates} scenario-cluster bootstrap replicates. Prediction metrics are external descriptive checks; the 18 scenarios are development scenarios and seeds are repeated simulations, not independent markets.",
        ]
    )
    (args.output_dir / "analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    checksums = []
    for path in sorted(args.output_dir.iterdir()):
        if path.name == "CHECKSUMS.sha256" or not path.is_file():
            continue
        checksums.append(f"{sha256(path)}  {path.name}")
    (args.output_dir / "CHECKSUMS.sha256").write_text("\n".join(checksums) + "\n", encoding="ascii")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if all_hard_gates_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
