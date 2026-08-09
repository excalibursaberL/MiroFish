#!/usr/bin/env python3
"""Compare baseline and Profile-ID-permuted S1 interaction graphs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


SEEDS = (42, 999, 2887, 3407, 4004)
MAX_ROUND = 6
AGENT_COUNT = 10
SCENARIO_COUNT = 18
METRICS = (
    "weighted_in_degree",
    "weighted_out_degree",
    "unique_in_neighbors",
    "unique_out_neighbors",
)
CORE_ACTION_GROUPS = (
    "all_explicit",
    "like_comment",
    "like_post",
    "create_comment",
    "follow",
    "negative",
)


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_baselines() -> list[Path]:
    root = workspace_root() / "Dataset"
    return [
        root / "s1_round_selection_6rounds_k10_seed42_v2",
        root / "s1_round_selection_6rounds_k10_seed999_v2",
        root / "s1_round_selection_6rounds_k10_seed2887_v2",
        root / "s1_round_selection_6rounds_k10_seed3407_v2",
        root / "s1_round_selection_10rounds_k10_seed4004_v2",
    ]


def default_permuted() -> list[Path]:
    root = workspace_root() / "Dataset"
    return [
        root / f"s1_profile_id_permutation_6rounds_k10_seed{seed}_v1"
        for seed in SEEDS
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, nargs=5, default=default_baselines())
    parser.add_argument("--permuted", type=Path, nargs=5, default=default_permuted())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=workspace_root()
        / "Dataset"
        / "s1_profile_id_permutation_paired_5seeds_v1",
    )
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
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def seed_from_dataset(path: Path) -> int:
    runs = pd.read_csv(path / "scenario_runs.csv", encoding="utf-8-sig")
    candidates: set[int] = set()
    for field in ("random_seed", "seed"):
        if field in runs:
            candidates.update(int(value) for value in runs[field].dropna().unique())
    if len(candidates) == 1:
        return candidates.pop()
    for seed in SEEDS:
        if f"seed{seed}" in path.name:
            return seed
    raise ValueError(f"cannot resolve one seed from {path}")


def load_profile_mapping(path: Path) -> pd.DataFrame:
    profiles = pd.read_json(path / "profiles.jsonl", lines=True)
    required = {"scenario_id", "user_id", "full_population_agent_id"}
    missing = required - set(profiles.columns)
    if missing:
        raise ValueError(f"{path}: profiles.jsonl lacks {sorted(missing)}")
    columns = [
        "scenario_id",
        "user_id",
        "full_population_agent_id",
        *(
            ["canonical_agent_id"]
            if "canonical_agent_id" in profiles.columns
            else []
        ),
        *(["role_id"] if "role_id" in profiles.columns else []),
        *(["agent_key"] if "agent_key" in profiles.columns else []),
    ]
    mapping = profiles[columns].copy()
    mapping = mapping.rename(
        columns={
            "user_id": "runtime_id",
            "full_population_agent_id": "profile_id",
        }
    )
    mapping["runtime_id"] = mapping["runtime_id"].astype(int)
    mapping["profile_id"] = mapping["profile_id"].astype(int)
    mapping = mapping.drop_duplicates(["scenario_id", "runtime_id"])
    for scenario_id, group in mapping.groupby("scenario_id"):
        if sorted(group["runtime_id"].tolist()) != list(range(AGENT_COUNT)):
            raise ValueError(f"{path}: invalid runtime IDs in {scenario_id}")
        if group["profile_id"].nunique() != AGENT_COUNT:
            raise ValueError(f"{path}: duplicate Profile identity in {scenario_id}")
    return mapping


def edge_groups(row: Mapping[str, Any]) -> set[str]:
    action_type = str(row.get("action_type", "")).lower()
    result = {"all_explicit", action_type}
    sign = pd.to_numeric(row.get("interaction_sign"), errors="coerce")
    if (pd.notna(sign) and float(sign) < 0) or action_type in {
        "dislike_post",
        "dislike_comment",
        "mute",
        "unfollow",
        "block",
    }:
        result.add("negative")
    return {value for value in result if value}


def load_degrees(path: Path, condition: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    seed = seed_from_dataset(path)
    mapping = load_profile_mapping(path)
    edges = pd.read_csv(path / "interaction_edges.csv", encoding="utf-8-sig")
    edges = edges[
        (edges["actor_class"] == "investor")
        & (edges["target_class"] == "investor")
        & (pd.to_numeric(edges["round"], errors="coerce").between(1, MAX_ROUND))
    ].copy()
    edges["actor_agent_id"] = edges["actor_agent_id"].astype(int)
    edges["target_agent_id"] = edges["target_agent_id"].astype(int)

    expanded: list[dict[str, Any]] = []
    for row in edges.to_dict("records"):
        for action_group in edge_groups(row):
            expanded.append({**row, "action_group": action_group})
    grouped_edges = pd.DataFrame(expanded)
    observed_groups = (
        set(grouped_edges["action_group"].unique()) if not grouped_edges.empty else set()
    )
    action_groups = sorted(observed_groups | set(CORE_ACTION_GROUPS))

    degree_rows: list[dict[str, Any]] = []
    scenario_ids = sorted(mapping["scenario_id"].unique())
    for scenario_id in scenario_ids:
        scenario_mapping = mapping[mapping["scenario_id"] == scenario_id]
        scenario_edges = (
            grouped_edges[grouped_edges["scenario_id"] == scenario_id]
            if not grouped_edges.empty
            else grouped_edges
        )
        for action_group in action_groups:
            current = (
                scenario_edges[scenario_edges["action_group"] == action_group]
                if not scenario_edges.empty
                else scenario_edges
            )
            for profile in scenario_mapping.to_dict("records"):
                runtime_id = int(profile["runtime_id"])
                inbound = current[current["target_agent_id"] == runtime_id]
                outbound = current[current["actor_agent_id"] == runtime_id]
                degree_rows.append(
                    {
                        "condition": condition,
                        "seed": seed,
                        "scenario_id": scenario_id,
                        "action_group": action_group,
                        "runtime_id": runtime_id,
                        "profile_id": int(profile["profile_id"]),
                        "canonical_agent_id": profile.get("canonical_agent_id"),
                        "role_id": profile.get("role_id"),
                        "agent_key": profile.get("agent_key"),
                        "weighted_in_degree": len(inbound),
                        "weighted_out_degree": len(outbound),
                        "unique_in_neighbors": inbound["actor_agent_id"].nunique(),
                        "unique_out_neighbors": outbound["target_agent_id"].nunique(),
                    }
                )
    dataset_quality = read_json(path / "quality_report.json")
    metadata = {
        "condition": condition,
        "seed": seed,
        "path": path,
        "scenario_count": len(scenario_ids),
        "investor_interaction_count": len(edges),
        "profile_mapping_count": len(mapping),
        "dataset_quality_passed": bool(dataset_quality.get("passed")),
        "invalid_snapshot_record_count": len(
            dataset_quality.get("invalid_snapshot_records", [])
        ),
        "stance_annotation_failure_count": int(
            dataset_quality.get("stance_annotation_failure_count", 0) or 0
        ),
    }
    return pd.DataFrame(degree_rows), metadata


def spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 3 or len(left) != len(right):
        return None
    left_series = pd.Series(left, dtype=float)
    right_series = pd.Series(right, dtype=float)
    if left_series.nunique() < 2 or right_series.nunique() < 2:
        return None
    value = left_series.rank(method="average").corr(
        right_series.rank(method="average"), method="pearson"
    )
    return None if pd.isna(value) else float(value)


def top_ids(frame: pd.DataFrame, id_field: str, metric: str, count: int) -> set[int]:
    ordered = frame.sort_values(
        [metric, id_field], ascending=[False, True], kind="stable"
    )
    return set(ordered.head(count)[id_field].astype(int))


def compare_frames(
    baseline: pd.DataFrame,
    permuted: pd.DataFrame,
    id_field: str,
    metric: str,
) -> dict[str, Any]:
    merged = baseline[[id_field, metric]].merge(
        permuted[[id_field, metric]], on=id_field, suffixes=("_baseline", "_permuted")
    )
    left = merged[f"{metric}_baseline"].astype(float)
    right = merged[f"{metric}_permuted"].astype(float)
    if merged.empty or baseline.empty or permuted.empty:
        return {
            "paired_agent_count": len(merged),
            "spearman": None,
            "mae": None,
            "top1_retained": None,
            "top3_jaccard": None,
        }
    top_baseline = top_ids(baseline, id_field, metric, 3)
    top_permuted = top_ids(permuted, id_field, metric, 3)
    union = top_baseline | top_permuted
    return {
        "paired_agent_count": len(merged),
        "spearman": spearman(left.tolist(), right.tolist()),
        "mae": float((left - right).abs().mean()),
        "top1_retained": int(
            next(iter(top_ids(baseline, id_field, metric, 1)))
            == next(iter(top_ids(permuted, id_field, metric, 1)))
        ),
        "top3_jaccard": len(top_baseline & top_permuted) / len(union) if union else None,
    }


def paired_comparisons(degrees: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["seed", "scenario_id", "action_group"]
    for key, group in degrees.groupby(keys, sort=True):
        baseline = group[group["condition"] == "baseline"]
        permuted = group[group["condition"] == "permuted"]
        if len(baseline) != AGENT_COUNT or len(permuted) != AGENT_COUNT:
            continue
        for metric in METRICS:
            runtime = compare_frames(baseline, permuted, "runtime_id", metric)
            profile = compare_frames(baseline, permuted, "profile_id", metric)
            rows.append(
                {
                    "seed": key[0],
                    "scenario_id": key[1],
                    "action_group": key[2],
                    "metric": metric,
                    **{f"runtime_{name}": value for name, value in runtime.items()},
                    **{f"profile_{name}": value for name, value in profile.items()},
                    "spearman_profile_minus_runtime": (
                        profile["spearman"] - runtime["spearman"]
                        if profile["spearman"] is not None
                        and runtime["spearman"] is not None
                        else None
                    ),
                }
            )
    return pd.DataFrame(rows)


def aggregate_seed_comparisons(degrees: pd.DataFrame) -> pd.DataFrame:
    aggregate = (
        degrees.groupby(
            ["condition", "seed", "action_group", "runtime_id", "profile_id"],
            as_index=False,
        )[list(METRICS)]
        .sum()
    )
    rows: list[dict[str, Any]] = []
    for (seed, action_group), group in aggregate.groupby(
        ["seed", "action_group"], sort=True
    ):
        baseline = group[group["condition"] == "baseline"]
        permuted = group[group["condition"] == "permuted"]
        if len(baseline) != AGENT_COUNT or len(permuted) != AGENT_COUNT:
            continue
        for metric in METRICS:
            runtime = compare_frames(baseline, permuted, "runtime_id", metric)
            profile = compare_frames(baseline, permuted, "profile_id", metric)
            rows.append(
                {
                    "seed": seed,
                    "action_group": action_group,
                    "metric": metric,
                    **{f"runtime_{name}": value for name, value in runtime.items()},
                    **{f"profile_{name}": value for name, value in profile.items()},
                    "spearman_profile_minus_runtime": (
                        profile["spearman"] - runtime["spearman"]
                        if profile["spearman"] is not None
                        and runtime["spearman"] is not None
                        else None
                    ),
                }
            )
    return pd.DataFrame(rows)


def categorical_design(values: Iterable[Any], prefix: str) -> pd.DataFrame:
    return pd.get_dummies(
        pd.Series(list(values), dtype="category"), prefix=prefix, drop_first=True, dtype=float
    )


def residual_sum_squares(y: np.ndarray, matrices: Sequence[pd.DataFrame]) -> float:
    parts = [np.ones((len(y), 1)), *(matrix.to_numpy(dtype=float) for matrix in matrices)]
    design = np.column_stack(parts)
    residual = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    return float(residual @ residual)


def partial_effects(degrees: pd.DataFrame) -> pd.DataFrame:
    source = degrees[
        (degrees["condition"] == "permuted")
        & (degrees["action_group"] == "all_explicit")
    ].copy()
    source["context"] = source["seed"].astype(str) + ":" + source["scenario_id"]
    context = categorical_design(source["context"], "context")
    runtime = categorical_design(source["runtime_id"], "runtime")
    profile = categorical_design(source["profile_id"], "profile")
    rows: list[dict[str, Any]] = []
    for metric in ("weighted_in_degree", "weighted_out_degree"):
        for scale in ("log1p_count", "within_context_rank"):
            if scale == "log1p_count":
                y = np.log1p(source[metric].to_numpy(dtype=float))
            else:
                y = (
                    source.groupby("context")[metric]
                    .rank(method="average", pct=True)
                    .to_numpy(dtype=float)
                )
            full = residual_sum_squares(y, (context, runtime, profile))
            drop_runtime = residual_sum_squares(y, (context, profile))
            drop_profile = residual_sum_squares(y, (context, runtime))
            runtime_added = max(0.0, drop_runtime - full)
            profile_added = max(0.0, drop_profile - full)
            rows.extend(
                [
                    {
                        "metric": metric,
                        "scale": scale,
                        "factor": "runtime_id",
                        "observation_count": len(source),
                        "partial_sum_squares": runtime_added,
                        "partial_eta_squared": runtime_added / (runtime_added + full)
                        if runtime_added + full
                        else None,
                    },
                    {
                        "metric": metric,
                        "scale": scale,
                        "factor": "profile_identity",
                        "observation_count": len(source),
                        "partial_sum_squares": profile_added,
                        "partial_eta_squared": profile_added / (profile_added + full)
                        if profile_added + full
                        else None,
                    },
                ]
            )
    return pd.DataFrame(rows)


def runtime_order_associations(degrees: pd.DataFrame) -> pd.DataFrame:
    source = degrees[
        (degrees["condition"] == "permuted")
        & (degrees["action_group"] == "all_explicit")
    ]
    rows: list[dict[str, Any]] = []
    for (seed, scenario_id), group in source.groupby(["seed", "scenario_id"]):
        for metric in ("weighted_in_degree", "weighted_out_degree"):
            rows.append(
                {
                    "seed": seed,
                    "scenario_id": scenario_id,
                    "metric": metric,
                    "runtime_id_spearman": spearman(
                        group["runtime_id"].tolist(), group[metric].tolist()
                    ),
                }
            )
    return pd.DataFrame(rows)


def finite_mean(values: Iterable[Any]) -> float | None:
    numbers = [float(value) for value in values if pd.notna(value)]
    return float(np.mean(numbers)) if numbers else None


def finite_median(values: Iterable[Any]) -> float | None:
    numbers = [float(value) for value in values if pd.notna(value)]
    return float(np.median(numbers)) if numbers else None


def summarize(
    paired: pd.DataFrame,
    aggregate: pd.DataFrame,
    effects: pd.DataFrame,
    runtime_order: pd.DataFrame,
    source_metadata: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    primary = paired[
        (paired["action_group"] == "all_explicit")
        & (paired["metric"].isin(["weighted_in_degree", "weighted_out_degree"]))
    ]
    aggregate_primary = aggregate[
        (aggregate["action_group"] == "all_explicit")
        & (aggregate["metric"].isin(["weighted_in_degree", "weighted_out_degree"]))
    ]
    by_metric: dict[str, Any] = {}
    for metric, group in primary.groupby("metric"):
        by_metric[str(metric)] = {
            "scenario_pair_count": len(group),
            "runtime_alignment_mean_spearman": finite_mean(group["runtime_spearman"]),
            "runtime_alignment_median_spearman": finite_median(group["runtime_spearman"]),
            "profile_alignment_mean_spearman": finite_mean(group["profile_spearman"]),
            "profile_alignment_median_spearman": finite_median(group["profile_spearman"]),
            "mean_profile_minus_runtime_spearman": finite_mean(
                group["spearman_profile_minus_runtime"]
            ),
            "runtime_top1_retention_rate": finite_mean(group["runtime_top1_retained"]),
            "profile_top1_retention_rate": finite_mean(group["profile_top1_retained"]),
        }
    aggregate_by_metric: dict[str, Any] = {}
    for metric, group in aggregate_primary.groupby("metric"):
        aggregate_by_metric[str(metric)] = {
            "seed_count": len(group),
            "runtime_alignment_mean_spearman": finite_mean(group["runtime_spearman"]),
            "profile_alignment_mean_spearman": finite_mean(group["profile_spearman"]),
            "mean_profile_minus_runtime_spearman": finite_mean(
                group["spearman_profile_minus_runtime"]
            ),
        }
    by_seed: dict[str, list[dict[str, Any]]] = {}
    for metric, metric_group in primary.groupby("metric"):
        by_seed[str(metric)] = []
        for seed, group in metric_group.groupby("seed"):
            by_seed[str(metric)].append(
                {
                    "seed": int(seed),
                    "scenario_pair_count": len(group),
                    "runtime_alignment_mean_spearman": finite_mean(
                        group["runtime_spearman"]
                    ),
                    "profile_alignment_mean_spearman": finite_mean(
                        group["profile_spearman"]
                    ),
                    "mean_profile_minus_runtime_spearman": finite_mean(
                        group["spearman_profile_minus_runtime"]
                    ),
                }
            )
    action_specific: dict[str, dict[str, Any]] = {}
    for action_group, group in aggregate[
        aggregate["metric"] == "weighted_in_degree"
    ].groupby("action_group"):
        action_specific[str(action_group)] = {
            "seed_count": len(group),
            "valid_correlation_seed_count": int(
                (group["runtime_spearman"].notna() & group["profile_spearman"].notna()).sum()
            ),
            "runtime_alignment_mean_spearman": finite_mean(group["runtime_spearman"]),
            "profile_alignment_mean_spearman": finite_mean(group["profile_spearman"]),
            "mean_profile_minus_runtime_spearman": finite_mean(
                group["spearman_profile_minus_runtime"]
            ),
        }
    return {
        "analysis_version": "s1_profile_id_permutation_paired_5seeds_v1",
        "design": {
            "seed_count": len(SEEDS),
            "scenario_count_per_seed": SCENARIO_COUNT,
            "agent_count": AGENT_COUNT,
            "rounds": MAX_ROUND,
            "paired_unit": "same seed and same scenario; independent LLM executions",
        },
        "source_metadata": list(source_metadata),
        "data_quality": {
            "invalid_snapshot_record_count": sum(
                int(item.get("invalid_snapshot_record_count", 0))
                for item in source_metadata
            ),
            "stance_annotation_failure_count": sum(
                int(item.get("stance_annotation_failure_count", 0))
                for item in source_metadata
            ),
            "all_source_dataset_quality_checks_passed": all(
                bool(item.get("dataset_quality_passed"))
                for item in source_metadata
            ),
        },
        "scenario_level_alignment": by_metric,
        "scenario_level_alignment_by_seed": by_seed,
        "seed_aggregate_alignment": aggregate_by_metric,
        "action_specific_seed_aggregate_in_degree": action_specific,
        "partial_effects": effects.to_dict("records"),
        "runtime_order": {
            metric: {
                "mean_spearman": finite_mean(group["runtime_id_spearman"]),
                "median_spearman": finite_median(group["runtime_id_spearman"]),
            }
            for metric, group in runtime_order.groupby("metric")
        },
        "caveats": [
            "Five seeds are repeated runs of the same 18 scenarios, not 90 independent market events.",
            "Fixed seeds do not guarantee deterministic external LLM outputs.",
            "Interaction edges describe observable actions, not causal opinion influence.",
            "The graph is dominated by reactions and comments; action-specific results must accompany the pooled graph.",
        ],
    }


def verdict(summary: Mapping[str, Any]) -> tuple[str, str]:
    primary = summary["scenario_level_alignment"]["weighted_in_degree"]
    runtime = primary["runtime_alignment_mean_spearman"]
    profile = primary["profile_alignment_mean_spearman"]
    if runtime is None or profile is None:
        return "证据不足", "有效分场景相关数量不足，不能判定入度跟随对象。"
    difference = profile - runtime
    if difference >= 0.15:
        return "更偏向 Profile 效应", "置换后入度排序按 Profile 身份对齐明显优于按运行时 ID 对齐。"
    if difference <= -0.15:
        return "更偏向运行时位置效应", "置换后入度排序按运行时 ID 对齐明显优于按 Profile 身份对齐。"
    if runtime >= 0.3 and profile >= 0.3:
        return "两类效应并存", "运行时位置与 Profile 身份都保留了部分入度结构。"
    return "旧稳定性主要来自聚合或随机波动", "两种对齐都未显示强稳定性，不能把汇总图排名视为稳定个体属性。"


def render_report(summary: Mapping[str, Any], effects: pd.DataFrame) -> str:
    label, explanation = verdict(summary)
    scenario = summary["scenario_level_alignment"]
    aggregate = summary["seed_aggregate_alignment"]

    def fmt(value: Any) -> str:
        return "NA" if value is None or pd.isna(value) else f"{float(value):.3f}"

    effect_lines = []
    for row in effects.to_dict("records"):
        effect_lines.append(
            f"| {row['metric']} | {row['scale']} | {row['factor']} | "
            f"{fmt(row['partial_eta_squared'])} |"
        )
    seed_lines = []
    for row in summary["scenario_level_alignment_by_seed"]["weighted_in_degree"]:
        seed_lines.append(
            f"| {row['seed']} | {fmt(row['runtime_alignment_mean_spearman'])} | "
            f"{fmt(row['profile_alignment_mean_spearman'])} | "
            f"{fmt(row['mean_profile_minus_runtime_spearman'])} |"
        )
    action_lines = []
    action_summary = summary["action_specific_seed_aggregate_in_degree"]
    for action in CORE_ACTION_GROUPS:
        row = action_summary.get(action)
        if row is None:
            continue
        action_lines.append(
            f"| {action} | {row['valid_correlation_seed_count']} | "
            f"{fmt(row['runtime_alignment_mean_spearman'])} | "
            f"{fmt(row['profile_alignment_mean_spearman'])} |"
        )
    return f"""# Profile-ID 随机置换实验分析

## 结论

**{label}。** {explanation}

这一结论针对的是可观察互动图中的度数结构，不等价于因果影响力，也不能直接决定降采样名单。

## 核心配对结果

| 指标 | 分场景 runtime 对齐 Spearman | 分场景 Profile 对齐 Spearman | Profile-runtime | 跨 18 场景汇总 runtime | 跨 18 场景汇总 Profile |
| --- | ---: | ---: | ---: | ---: | ---: |
| 加权入度 | {fmt(scenario['weighted_in_degree']['runtime_alignment_mean_spearman'])} | {fmt(scenario['weighted_in_degree']['profile_alignment_mean_spearman'])} | {fmt(scenario['weighted_in_degree']['mean_profile_minus_runtime_spearman'])} | {fmt(aggregate['weighted_in_degree']['runtime_alignment_mean_spearman'])} | {fmt(aggregate['weighted_in_degree']['profile_alignment_mean_spearman'])} |
| 加权出度 | {fmt(scenario['weighted_out_degree']['runtime_alignment_mean_spearman'])} | {fmt(scenario['weighted_out_degree']['profile_alignment_mean_spearman'])} | {fmt(scenario['weighted_out_degree']['mean_profile_minus_runtime_spearman'])} | {fmt(aggregate['weighted_out_degree']['runtime_alignment_mean_spearman'])} | {fmt(aggregate['weighted_out_degree']['profile_alignment_mean_spearman'])} |

分场景均值是主要证据；跨 18 场景汇总只作为辅助，因为汇总会抹平场景差异并制造表面稳定性。

### 分 seed 的分场景入度结果

| seed | runtime 对齐均值 | Profile 对齐均值 | Profile-runtime |
| ---: | ---: | ---: | ---: |
{chr(10).join(seed_lines)}

五个 seed 的 Profile-runtime 差值方向一致时，才把结果解释为跨 seed 可重复；绝对相关仍以分场景数值为准。

### 动作分层的汇总入度敏感性分析

| 动作层 | 有效 seed 数 | runtime 对齐 | Profile 对齐 |
| --- | ---: | ---: | ---: |
{chr(10).join(action_lines)}

`follow` 和 `negative` 样本极少，只能作为数据质量提示；总图结论主要由点赞和回复动作支撑。

## 双因素效应分解

模型控制 `seed × scenario` 后，同时放入运行时 ID 和 Profile 身份。`partial_eta_squared` 越大，表示该因素在另一个因素已进入模型后仍解释更多变异。

| 度指标 | 分析尺度 | 因素 | partial eta squared |
| --- | --- | --- | ---: |
{chr(10).join(effect_lines)}

## 如何读取文件

- `baseline_vs_permuted_paired_metrics.csv`：每个 seed、场景、动作类型、度指标的两种对齐结果。
- `seed_aggregate_alignment_metrics.csv`：先汇总 18 个场景，再做两种对齐的敏感性分析。
- `permuted_runtime_degree_metrics.csv`：按运行时 ID 表示的置换图节点度数。
- `permuted_profile_degree_metrics.csv`：同一批节点改按 Profile 身份表示。
- `partial_effect_decomposition.csv`：运行时位置和 Profile 身份的联合效应分解。
- `runtime_order_associations.csv`：运行时 ID 数值与入度/出度的分场景相关，仅用于检查执行顺序偏差。

## 设计边界

1. 五个 seed 共享同一组 18 场景，不能当作 90 个独立市场事件。
2. 随机种子控制 OASIS 侧随机过程，但外部 LLM 响应不保证逐 token 确定，因此这是配对重复实验，不是完全确定性反事实。
3. 互动边是点赞、回复、关注等显式行为关联，不证明某个 Agent 导致另一个 Agent 改变判断。
4. 总图受 `like_comment` 等高频动作支配，必须同时查看动作分层结果。
5. 全部十组源数据共存在 {summary['data_quality']['invalid_snapshot_record_count']} 条无效信念快照；它不改变互动边，但信念变化分析必须按缺失值处理。立场标注失败数为 {summary['data_quality']['stance_annotation_failure_count']}。
"""


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_by_seed = {seed_from_dataset(path.resolve()): path.resolve() for path in args.baseline}
    permuted_by_seed = {seed_from_dataset(path.resolve()): path.resolve() for path in args.permuted}
    if set(baseline_by_seed) != set(SEEDS) or set(permuted_by_seed) != set(SEEDS):
        raise ValueError(f"both conditions must contain exactly seeds {SEEDS}")

    frames: list[pd.DataFrame] = []
    metadata: list[dict[str, Any]] = []
    for condition, sources in (
        ("baseline", baseline_by_seed),
        ("permuted", permuted_by_seed),
    ):
        for seed in SEEDS:
            frame, details = load_degrees(sources[seed], condition)
            if details["scenario_count"] != SCENARIO_COUNT:
                raise ValueError(f"{sources[seed]}: expected {SCENARIO_COUNT} scenarios")
            frames.append(frame)
            metadata.append(details)
    degrees = pd.concat(frames, ignore_index=True)

    paired = paired_comparisons(degrees)
    aggregate = aggregate_seed_comparisons(degrees)
    effects = partial_effects(degrees)
    runtime_order = runtime_order_associations(degrees)
    permuted = degrees[degrees["condition"] == "permuted"].copy()
    profile_mapping = degrees[
        degrees["action_group"] == "all_explicit"
    ][
        [
            "condition",
            "seed",
            "runtime_id",
            "profile_id",
            "canonical_agent_id",
            "role_id",
            "agent_key",
        ]
    ].drop_duplicates()

    summary = summarize(paired, aggregate, effects, runtime_order, metadata)
    source_manifest = {
        "analysis_version": summary["analysis_version"],
        "sources": [
            {
                **dict(item),
                "path": str(Path(item["path"]).relative_to(workspace_root())),
                "interaction_edges_sha256": sha256_file(
                    Path(item["path"]) / "interaction_edges.csv"
                ),
                "profiles_sha256": sha256_file(Path(item["path"]) / "profiles.jsonl"),
            }
            for item in metadata
        ],
    }

    write_csv(output_dir / "all_condition_degree_metrics.csv", degrees)
    write_csv(output_dir / "baseline_vs_permuted_paired_metrics.csv", paired)
    write_csv(output_dir / "seed_aggregate_alignment_metrics.csv", aggregate)
    write_csv(
        output_dir / "scenario_level_rank_stability.csv",
        paired[
            (paired["action_group"] == "all_explicit")
            & (paired["metric"].isin(["weighted_in_degree", "weighted_out_degree"]))
        ],
    )
    write_csv(output_dir / "permuted_runtime_degree_metrics.csv", permuted)
    write_csv(
        output_dir / "permuted_profile_degree_metrics.csv",
        permuted.sort_values(["seed", "scenario_id", "action_group", "profile_id"]),
    )
    write_csv(output_dir / "partial_effect_decomposition.csv", effects)
    write_csv(output_dir / "runtime_order_associations.csv", runtime_order)
    write_csv(output_dir / "profile_runtime_mapping.csv", profile_mapping)
    write_json(output_dir / "analysis_summary.json", summary)
    write_json(output_dir / "source_manifest.json", source_manifest)
    (output_dir / "profile_id_permutation_analysis.md").write_text(
        render_report(summary, effects), encoding="utf-8"
    )

    expected_pairs = len(SEEDS) * SCENARIO_COUNT * len(CORE_ACTION_GROUPS) * len(METRICS)
    scenario_counts = degrees.groupby(["condition", "seed"])["scenario_id"].nunique()
    quality = {
        "status": "passed",
        "seed_count": len(set(degrees["seed"])),
        "condition_count": len(set(degrees["condition"])),
        "scenario_count_per_condition_seed": {
            f"{condition}:seed{seed}": int(count)
            for (condition, seed), count in scenario_counts.items()
        },
        "profile_mapping_is_bijective": True,
        "invalid_snapshot_record_count": summary["data_quality"][
            "invalid_snapshot_record_count"
        ],
        "stance_annotation_failure_count": summary["data_quality"][
            "stance_annotation_failure_count"
        ],
        "primary_pair_count": len(
            paired[
                (paired["action_group"].isin(CORE_ACTION_GROUPS))
                & (paired["metric"].isin(METRICS))
            ]
        ),
        "minimum_expected_primary_pair_count": expected_pairs,
    }
    if quality["primary_pair_count"] < expected_pairs:
        quality["status"] = "failed"
        write_json(output_dir / "quality_report.json", quality)
        raise ValueError("paired comparison coverage is incomplete")
    write_json(output_dir / "quality_report.json", quality)

    checksum_lines = []
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "CHECKSUMS.sha256":
            checksum_lines.append(f"{sha256_file(path)}  {path.name}")
    (output_dir / "CHECKSUMS.sha256").write_text(
        "\n".join(checksum_lines) + "\n", encoding="ascii"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=json_scalar))


if __name__ == "__main__":
    main()
