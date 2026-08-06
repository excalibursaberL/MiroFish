#!/usr/bin/env python3
"""Estimate pooled conditional information from S1 social exposure.

The analysis uses out-of-scenario predictive log-loss improvement as an
estimate of conditional mutual information.  It deliberately treats a whole
scenario as the validation and resampling unit; Agent-round rows within a
scenario are not assumed to be independent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ANALYSIS_VERSION = "pooled_cmi_v1"
TARGET = "direction_changed"
PRIMARY_C = 1.0
DEFAULT_C_VALUES = (0.1, 1.0, 10.0)
EPSILON = 1e-12

CATEGORICAL_FEATURES = (
    "round_category",
    "agent_role_category",
    "agent_analysis_style",
    "agent_risk_attitude",
    "agent_investment_horizon",
    "agent_decision_source",
)

BASE_NUMERIC_FEATURES = (
    "previous_up_probability",
    "previous_neutral_probability",
    "previous_expected_return",
    "previous_confidence",
    "source_has_exposure",
    "source_log_unique_count",
    "source_valence_positive",
    "source_valence_mixed",
    "source_valence_negative",
)

AMOUNT_FEATURES = (
    "social_has_exposure",
    "social_log_unique_count",
)

STANCE_FEATURES = (
    "social_stance_mean_filled",
    "social_stance_std_filled",
    "social_stance_mixed_proportion",
)

MODEL_NUMERIC_FEATURES = {
    "m0_baseline": BASE_NUMERIC_FEATURES,
    "m1_amount": BASE_NUMERIC_FEATURES + AMOUNT_FEATURES,
    "m2_stance": BASE_NUMERIC_FEATURES + AMOUNT_FEATURES + STANCE_FEATURES,
}

PERMUTED_SOCIAL_FEATURES = AMOUNT_FEATURES + STANCE_FEATURES

REQUIRED_COLUMNS = {
    "scenario_id",
    "agent_id",
    "round",
    TARGET,
    *CATEGORICAL_FEATURES[1:],
    "previous_up_probability",
    "previous_neutral_probability",
    "previous_expected_return",
    "previous_confidence",
    "exposure_social_unique_content_count",
    "exposure_social_mean_stance_score_unique",
    "exposure_social_stance_score_std_unique",
    "exposure_social_mixed_unique_proportion",
    "exposure_source_unique_content_count",
    "exposure_source_event_valence_positive_unique_proportion",
    "exposure_source_event_valence_mixed_unique_proportion",
    "exposure_source_event_valence_negative_unique_proportion",
}


@dataclass(frozen=True)
class Scope:
    name: str
    rounds: tuple[int, ...]


SCOPES = (
    Scope("all_rounds", (1, 2, 3, 4)),
    Scope("rounds_2_4", (2, 3, 4)),
)


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    root = workspace_root()
    default_dataset = root / "Dataset" / "downsampling_s1_rounds4_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=default_dataset / "agent_round_mi_features.csv",
        help="Agent-round feature table",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_dataset / ANALYSIS_VERSION,
        help="Destination for analysis artifacts",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=200,
        help="Permutations per null scheme for the primary analysis",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10_000,
        help="Scenario-level bootstrap samples",
    )
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument(
        "--c-values",
        type=float,
        nargs="+",
        default=list(DEFAULT_C_VALUES),
        help="Logistic-regression inverse regularization values",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_binary(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all() and set(numeric.unique()).issubset({0, 1}):
        return numeric.astype(int)
    normalized = series.astype(str).str.strip().str.lower()
    mapped = normalized.map({"true": 1, "false": 0, "yes": 1, "no": 0})
    if mapped.isna().any():
        invalid = sorted(normalized[mapped.isna()].unique().tolist())
        raise ValueError(f"{TARGET} contains non-binary values: {invalid}")
    return mapped.astype(int)


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    data = frame.copy().reset_index(drop=True)
    data[TARGET] = _as_binary(data[TARGET])
    data["round"] = pd.to_numeric(data["round"], errors="raise").astype(int)
    data["round_category"] = data["round"].astype(str)

    social_count = pd.to_numeric(
        data["exposure_social_unique_content_count"], errors="coerce"
    ).fillna(0.0).clip(lower=0.0)
    source_count = pd.to_numeric(
        data["exposure_source_unique_content_count"], errors="coerce"
    ).fillna(0.0).clip(lower=0.0)

    data["social_has_exposure"] = (social_count > 0).astype(float)
    data["social_log_unique_count"] = np.log1p(social_count)
    data["social_stance_mean_filled"] = pd.to_numeric(
        data["exposure_social_mean_stance_score_unique"], errors="coerce"
    ).fillna(0.0)
    data["social_stance_std_filled"] = pd.to_numeric(
        data["exposure_social_stance_score_std_unique"], errors="coerce"
    ).fillna(0.0)
    data["social_stance_mixed_proportion"] = pd.to_numeric(
        data["exposure_social_mixed_unique_proportion"], errors="coerce"
    ).fillna(0.0)

    data["source_has_exposure"] = (source_count > 0).astype(float)
    data["source_log_unique_count"] = np.log1p(source_count)
    for label in ("positive", "mixed", "negative"):
        source = f"exposure_source_event_valence_{label}_unique_proportion"
        data[f"source_valence_{label}"] = pd.to_numeric(
            data[source], errors="coerce"
        ).fillna(0.0)

    key_columns = ["scenario_id", "agent_id", "round"]
    if data.duplicated(key_columns).any():
        duplicates = int(data.duplicated(key_columns, keep=False).sum())
        raise ValueError(f"found {duplicates} rows with duplicate analysis keys")
    if data["scenario_id"].nunique() < 3:
        raise ValueError("at least three scenarios are required for grouped analysis")
    if data[TARGET].nunique() != 2:
        raise ValueError(f"{TARGET} must contain both outcome classes")
    return data


def make_model(model_name: str, *, c_value: float, seed: int) -> Pipeline:
    numeric_features = list(MODEL_NUMERIC_FEATURES[model_name])
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, list(CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
    )
    classifier = LogisticRegression(
        C=float(c_value),
        max_iter=2_000,
        solver="lbfgs",
        random_state=seed,
    )
    return Pipeline([("features", preprocessor), ("classifier", classifier)])


def cross_fit_predictions(
    data: pd.DataFrame,
    *,
    c_value: float,
    seed: int,
    model_names: Sequence[str] = tuple(MODEL_NUMERIC_FEATURES),
) -> pd.DataFrame:
    result = data[["scenario_id", "agent_id", "round", TARGET]].copy()
    scenarios = sorted(data["scenario_id"].astype(str).unique().tolist())
    for model_name in model_names:
        probabilities = np.full(len(data), np.nan, dtype=float)
        model_features = list(CATEGORICAL_FEATURES) + list(
            MODEL_NUMERIC_FEATURES[model_name]
        )
        for scenario in scenarios:
            test_mask = data["scenario_id"].astype(str).eq(scenario).to_numpy()
            train_mask = ~test_mask
            y_train = data.loc[train_mask, TARGET]
            if y_train.nunique() != 2:
                raise ValueError(
                    f"training fold excluding {scenario} does not contain both classes"
                )
            model = make_model(model_name, c_value=c_value, seed=seed)
            model.fit(data.loc[train_mask, model_features], y_train)
            probabilities[test_mask] = model.predict_proba(
                data.loc[test_mask, model_features]
            )[:, 1]
        if np.isnan(probabilities).any():
            raise RuntimeError(f"incomplete out-of-fold predictions for {model_name}")
        result[f"probability_{model_name}"] = probabilities
    return result


def binary_log_loss_nats(y_true: np.ndarray, probability: np.ndarray) -> np.ndarray:
    probability = np.clip(probability.astype(float), EPSILON, 1.0 - EPSILON)
    y_true = y_true.astype(float)
    return -(y_true * np.log(probability) + (1.0 - y_true) * np.log1p(-probability))


def add_row_losses(predictions: pd.DataFrame) -> pd.DataFrame:
    result = predictions.copy()
    y_true = result[TARGET].to_numpy(dtype=float)
    for model_name in MODEL_NUMERIC_FEATURES:
        probability_column = f"probability_{model_name}"
        if probability_column not in result:
            continue
        result[f"log_loss_bits_{model_name}"] = binary_log_loss_nats(
            y_true, result[probability_column].to_numpy()
        ) / math.log(2.0)
    return result


def information_from_losses(losses: dict[str, float]) -> dict[str, float]:
    return {
        "amount_bits": losses["m0_baseline"] - losses["m1_amount"],
        "stance_bits": losses["m1_amount"] - losses["m2_stance"],
        "total_bits": losses["m0_baseline"] - losses["m2_stance"],
    }


def summarize_predictions(predictions: pd.DataFrame) -> dict[str, Any]:
    with_losses = add_row_losses(predictions)
    losses = {
        model_name: float(with_losses[f"log_loss_bits_{model_name}"].mean())
        for model_name in MODEL_NUMERIC_FEATURES
    }
    brier = {
        model_name: float(
            np.mean(
                (
                    with_losses[f"probability_{model_name}"].to_numpy()
                    - with_losses[TARGET].to_numpy()
                )
                ** 2
            )
        )
        for model_name in MODEL_NUMERIC_FEATURES
    }
    return {
        "log_loss_bits": losses,
        "brier_score": brier,
        "information": information_from_losses(losses),
    }


def scenario_loss_table(predictions: pd.DataFrame, *, scope: str) -> pd.DataFrame:
    with_losses = add_row_losses(predictions)
    loss_columns = {
        f"log_loss_bits_{name}": f"log_loss_bits_{name}"
        for name in MODEL_NUMERIC_FEATURES
    }
    grouped = (
        with_losses.groupby("scenario_id", as_index=False)
        .agg(
            row_count=(TARGET, "size"),
            direction_change_rate=(TARGET, "mean"),
            **{column: (column, "mean") for column in loss_columns},
        )
        .sort_values("scenario_id")
    )
    grouped.insert(0, "scope", scope)
    grouped["amount_bits"] = (
        grouped["log_loss_bits_m0_baseline"]
        - grouped["log_loss_bits_m1_amount"]
    )
    grouped["stance_bits"] = (
        grouped["log_loss_bits_m1_amount"]
        - grouped["log_loss_bits_m2_stance"]
    )
    grouped["total_bits"] = (
        grouped["log_loss_bits_m0_baseline"]
        - grouped["log_loss_bits_m2_stance"]
    )
    return grouped


def bootstrap_scenarios(
    scenario_table: pd.DataFrame,
    *,
    samples: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    if samples <= 0:
        return {}
    rng = np.random.default_rng(seed)
    result: dict[str, dict[str, float]] = {}
    for metric in ("amount_bits", "stance_bits", "total_bits"):
        values = scenario_table[metric].to_numpy(dtype=float)
        draws = values[rng.integers(0, len(values), size=(samples, len(values)))].mean(
            axis=1
        )
        result[metric] = {
            "mean": float(values.mean()),
            "ci_2_5": float(np.quantile(draws, 0.025)),
            "ci_97_5": float(np.quantile(draws, 0.975)),
            "bootstrap_probability_above_zero": float(np.mean(draws > 0.0)),
        }
    return result


def permute_social_within_scenario_round(
    data: pd.DataFrame, *, rng: np.random.Generator
) -> pd.DataFrame:
    result = data.copy()
    columns = list(PERMUTED_SOCIAL_FEATURES)
    for indices in result.groupby(["scenario_id", "round"], sort=False).indices.values():
        index_array = np.asarray(indices, dtype=int)
        source_indices = rng.permutation(index_array)
        result.loc[index_array, columns] = data.loc[source_indices, columns].to_numpy()
    return result


def permute_social_scenario_trajectories(
    data: pd.DataFrame, *, rng: np.random.Generator
) -> pd.DataFrame:
    """Exchange complete social-exposure trajectories between scenarios.

    Matching by Agent and round preserves each donor scenario's temporal and
    cross-feature structure while breaking its association with the target
    scenario's beliefs, roles, source exposure, and outcome.
    """

    result = data.copy()
    columns = list(PERMUTED_SOCIAL_FEATURES)
    scenarios = np.asarray(sorted(data["scenario_id"].astype(str).unique()))
    donor_scenarios = rng.permutation(scenarios)
    key_columns = ["round", "agent_id"]
    for target, donor in zip(scenarios, donor_scenarios, strict=True):
        target_rows = (
            data[data["scenario_id"].astype(str).eq(target)]
            .sort_values(key_columns)
            .index.to_numpy()
        )
        donor_frame = data[data["scenario_id"].astype(str).eq(donor)].sort_values(
            key_columns
        )
        donor_rows = donor_frame.index.to_numpy()
        if len(target_rows) != len(donor_rows):
            raise ValueError(
                f"scenario trajectory sizes differ: {target} and {donor}"
            )
        target_keys = data.loc[target_rows, key_columns].to_numpy()
        donor_keys = data.loc[donor_rows, key_columns].to_numpy()
        if not np.array_equal(target_keys, donor_keys):
            raise ValueError(
                f"scenario trajectory keys differ: {target} and {donor}"
            )
        result.loc[target_rows, columns] = data.loc[donor_rows, columns].to_numpy()
    return result


def permutation_analysis(
    data: pd.DataFrame,
    *,
    baseline_predictions: pd.DataFrame,
    observed_information: dict[str, float],
    permutations: int,
    c_value: float,
    seed: int,
    scheme: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if permutations <= 0:
        return pd.DataFrame(), {}

    baseline_losses = add_row_losses(baseline_predictions)
    baseline_loss = float(baseline_losses["log_loss_bits_m0_baseline"].mean())
    rng = np.random.default_rng(seed)
    if scheme == "scenario_trajectory":
        permutation_function = permute_social_scenario_trajectories
        block_description = (
            "complete social-exposure trajectories exchanged between scenarios; "
            "matched by Agent and round"
        )
    elif scheme == "within_scenario_round_agents":
        permutation_function = permute_social_within_scenario_round
        block_description = (
            "social feature block jointly permuted among Agents within scenario and round"
        )
    else:
        raise ValueError(f"unknown permutation scheme: {scheme}")
    rows: list[dict[str, float | int]] = []
    for permutation in range(1, permutations + 1):
        permuted = permutation_function(data, rng=rng)
        predictions = cross_fit_predictions(
            permuted,
            c_value=c_value,
            seed=seed + permutation,
            model_names=("m1_amount", "m2_stance"),
        )
        predictions = add_row_losses(predictions)
        m1_loss = float(predictions["log_loss_bits_m1_amount"].mean())
        m2_loss = float(predictions["log_loss_bits_m2_stance"].mean())
        rows.append(
            {
                "scheme": scheme,
                "permutation": permutation,
                "amount_bits": baseline_loss - m1_loss,
                "stance_bits": m1_loss - m2_loss,
                "total_bits": baseline_loss - m2_loss,
            }
        )
    null = pd.DataFrame(rows)
    summary: dict[str, Any] = {
        "scheme": scheme,
        "count": permutations,
        "block": block_description,
        "metrics": {},
    }
    for metric in ("amount_bits", "stance_bits", "total_bits"):
        values = null[metric].to_numpy(dtype=float)
        observed = float(observed_information[metric])
        summary["metrics"][metric] = {
            "observed": observed,
            "null_mean": float(values.mean()),
            "null_std": float(values.std(ddof=1)),
            "null_95th_percentile": float(np.quantile(values, 0.95)),
            "bias_corrected": float(observed - values.mean()),
            "one_sided_p_value": float((1 + np.sum(values >= observed)) / (len(values) + 1)),
        }
    return null, summary


def serialize_frame(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.10g")


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _format_interval(value: dict[str, float]) -> str:
    return (
        f"{value['mean']:.5f} "
        f"[{value['ci_2_5']:.5f}, {value['ci_97_5']:.5f}]"
    )


def write_report(path: Path, summary: dict[str, Any]) -> None:
    primary = summary["scopes"]["all_rounds"]
    sensitivity = summary["scopes"]["rounds_2_4"]
    trajectory_permutation = summary["permutation"]["scenario_trajectory"][
        "metrics"
    ]
    personalized_permutation = summary["permutation"][
        "within_scenario_round_agents"
    ]["metrics"]
    lines = [
        "# S1 池化条件互信息初步分析",
        "",
        "## 口径",
        "",
        "使用按场景留一交叉验证，比较三个 L2 Logistic Regression：",
        "",
        "- M0：上一轮信念、角色、轮次和 source 曝光。",
        "- M1：M0 + 是否有社会曝光、社会曝光量。",
        "- M2：M1 + 社会内容立场均值、标准差和 mixed 占比。",
        "",
        "对数损失差除以 `ln(2)` 后以 bits 表示。场景内 Agent-round 行不视为独立样本。",
        "",
        "## 主要结果",
        "",
        "| 指标 | 场景均值和 95% bootstrap CI | 置换校正值 | 置换 p |",
        "| --- | ---: | ---: | ---: |",
    ]
    labels = {
        "amount_bits": "曝光量增量信息",
        "stance_bits": "立场内容增量信息",
        "total_bits": "社会曝光总增量信息",
    }
    for metric, label in labels.items():
        boot = primary["scenario_bootstrap"][metric]
        perm = trajectory_permutation[metric]
        lines.append(
            f"| {label} | {_format_interval(boot)} | "
            f"{perm['bias_corrected']:.5f} | {perm['one_sided_p_value']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 个体化曝光检验",
            "",
            "同一场景轮次内打乱 Agent 的曝光后，检验信息增量是否来自个体化分配。",
            "",
            "| 指标 | 个体化置换校正值 | 个体化置换 p |",
            "| --- | ---: | ---: |",
        ]
    )
    for metric, label in labels.items():
        perm = personalized_permutation[metric]
        lines.append(
            f"| {label} | {perm['bias_corrected']:.5f} | "
            f"{perm['one_sided_p_value']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 第 2-4 轮敏感性",
            "",
            "| 指标 | 场景均值和 95% bootstrap CI |",
            "| --- | ---: |",
        ]
    )
    for metric, label in labels.items():
        boot = sensitivity["scenario_bootstrap"][metric]
        lines.append(f"| {label} | {_format_interval(boot)} |")
    lines.extend(
        [
            "",
            "## 解释限制",
            "",
            "- 这是场景外预测信息增量，不是逐 Agent 因果传递熵。",
            "- 只有 18 个场景且共享一个随机种子，bootstrap 区间仅用于探索。",
            "- 跨场景轨迹置换检验总体社会曝光信息；场景轮次内 Agent 置换检验个体化曝光信息。",
            "- 同轮 action/interacted 字段可能是中介，因此没有进入主控制模型。",
            "- 立场标注模型与预测 Agent 属于同一模型家族，确认性分析仍需独立复标。",
            "",
            "详细结果见 `summary.json`、`scenario_logloss_deltas.csv`、"
            "`model_sensitivity.csv`、`oof_predictions.csv` 和 `permutation_null.csv`。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _scope_data(data: pd.DataFrame, scope: Scope) -> pd.DataFrame:
    return data[data["round"].isin(scope.rounds)].copy().reset_index(drop=True)


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    if args.permutations < 0 or args.bootstrap_samples < 0:
        raise ValueError("permutation and bootstrap counts must be non-negative")
    if not args.input.exists():
        raise FileNotFoundError(args.input)

    raw = pd.read_csv(args.input)
    data = prepare_features(raw)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scope_summaries: dict[str, Any] = {}
    scenario_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    sensitivity_rows: list[dict[str, Any]] = []
    prediction_cache: dict[tuple[str, float], pd.DataFrame] = {}

    c_values = sorted({float(value) for value in args.c_values} | {PRIMARY_C})
    for scope_index, scope in enumerate(SCOPES):
        scoped = _scope_data(data, scope)
        for c_value in c_values:
            predictions = cross_fit_predictions(
                scoped, c_value=c_value, seed=args.seed + scope_index
            )
            prediction_cache[(scope.name, c_value)] = predictions
            result = summarize_predictions(predictions)
            sensitivity_rows.append(
                {
                    "scope": scope.name,
                    "c_value": c_value,
                    "row_count": len(scoped),
                    "scenario_count": scoped["scenario_id"].nunique(),
                    "direction_change_rate": float(scoped[TARGET].mean()),
                    **result["log_loss_bits"],
                    **result["information"],
                }
            )

        primary_predictions = prediction_cache[(scope.name, PRIMARY_C)]
        primary_predictions.insert(0, "scope", scope.name)
        prediction_frames.append(add_row_losses(primary_predictions))
        scenario_table = scenario_loss_table(primary_predictions, scope=scope.name)
        scenario_frames.append(scenario_table)
        primary_result = summarize_predictions(primary_predictions)
        scope_summaries[scope.name] = {
            "rounds": list(scope.rounds),
            "row_count": len(scoped),
            "scenario_count": int(scoped["scenario_id"].nunique()),
            "agent_count": int(scoped["agent_id"].nunique()),
            "direction_change_rate": float(scoped[TARGET].mean()),
            "primary_c": PRIMARY_C,
            **primary_result,
            "scenario_bootstrap": bootstrap_scenarios(
                scenario_table,
                samples=args.bootstrap_samples,
                seed=args.seed + 100 + scope_index,
            ),
        }

    primary_data = _scope_data(data, SCOPES[0])
    primary_predictions = prediction_cache[(SCOPES[0].name, PRIMARY_C)]
    null_frames: list[pd.DataFrame] = []
    permutation_summaries: dict[str, Any] = {}
    for permutation_index, scheme in enumerate(
        ("scenario_trajectory", "within_scenario_round_agents")
    ):
        null_frame, permutation_summary = permutation_analysis(
            primary_data,
            baseline_predictions=primary_predictions,
            observed_information=scope_summaries[SCOPES[0].name]["information"],
            permutations=args.permutations,
            c_value=PRIMARY_C,
            seed=args.seed + 1_000 + permutation_index,
            scheme=scheme,
        )
        null_frames.append(null_frame)
        permutation_summaries[scheme] = permutation_summary

    summary = {
        "analysis_version": ANALYSIS_VERSION,
        "source": {
            "path": str(args.input.resolve()),
            "sha256": sha256_file(args.input),
        },
        "target": TARGET,
        "interpretation": (
            "Out-of-scenario predictive conditional information in bits; "
            "not causal or pairwise transfer entropy."
        ),
        "validation": "leave-one-scenario-out",
        "model": {
            "type": "L2 logistic regression",
            "class_weight": None,
            "categorical_features": list(CATEGORICAL_FEATURES),
            "numeric_features": {
                name: list(features) for name, features in MODEL_NUMERIC_FEATURES.items()
            },
        },
        "scopes": scope_summaries,
        "permutation": permutation_summaries,
        "settings": {
            "seed": args.seed,
            "bootstrap_samples": args.bootstrap_samples,
            "permutations": args.permutations,
            "c_values": c_values,
        },
        "warnings": [
            "The 1440 Agent-round rows form 18 scenario clusters, not 1440 independent samples.",
            "All source runs share one simulation seed.",
            "Social stance annotations use the same model family as the forecasting Agents.",
            "Negative out-of-sample information estimates are possible because these are finite-sample predictive estimates.",
        ],
    }

    serialize_frame(pd.concat(prediction_frames, ignore_index=True), args.output_dir / "oof_predictions.csv")
    serialize_frame(pd.concat(scenario_frames, ignore_index=True), args.output_dir / "scenario_logloss_deltas.csv")
    serialize_frame(pd.DataFrame(sensitivity_rows), args.output_dir / "model_sensitivity.csv")
    serialize_frame(pd.concat(null_frames, ignore_index=True), args.output_dir / "permutation_null.csv")
    write_json(args.output_dir / "summary.json", summary)
    write_report(args.output_dir / "README.md", summary)
    return summary


def main() -> int:
    args = parse_args()
    summary = run_analysis(args)
    compact = {
        "output_dir": str(args.output_dir.resolve()),
        "all_rounds": summary["scopes"]["all_rounds"]["information"],
        "rounds_2_4": summary["scopes"]["rounds_2_4"]["information"],
        "permutation": summary["permutation"],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
