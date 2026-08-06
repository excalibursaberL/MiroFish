"""C0 independent-forecast adapter.

This module deliberately does not create an OASIS environment. C0 is the
no-social-exposure control group, so an independent LLM call per Agent is the
smallest faithful implementation. The saved profile and scenario formats are
compatible with the existing MiroFish/OASIS files and can be reused by S1
later.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from openai import OpenAI

from ..config import Config
from ..utils.openai_chat_compat import (
    create_chat_completion,
    extract_chat_completion_finish_reason,
    extract_chat_completion_text,
    has_chat_completion_reasoning_content,
)
from ..utils.logger import get_logger
from .dataset import FinancialDatasetLoader, FinancialScenario
from .evaluator import (
    FIVE_DAY_DIRECTION_DEFINITION,
    FIVE_DAY_NEUTRAL_THRESHOLD,
    FinancialOutcomeEvaluator,
)
from .models import C0Forecast
from .roles import C0_AGENT_COUNT, build_c0_profiles, profile_prompt_text


logger = get_logger("mirofish.finance.c0")


def _forecast_profile_fields(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Copy active profile metadata into researcher-facing forecast records."""
    return {
        "agent_knowledge_level": profile.get("knowledge_level"),
        "agent_analysis_style": profile.get("analysis_style"),
        "agent_risk_attitude": profile.get("risk_attitude"),
        "agent_investment_horizon": profile.get("investment_horizon"),
        "profile_version": profile.get("profile_version"),
    }


class C0ExperimentService:
    """Prepare and run the C0 financial baseline."""

    GROUP = "C0"
    RUN_MODES = {"single", "all"}
    SEED_EVENT_COUNT = 5
    MAX_SCENARIOS_PER_RUN = 1
    MAX_FORECAST_ATTEMPTS = 2
    FORECAST_TEMPERATURE = 0.2
    FORECAST_MAX_TOKENS = 32000
    FORECAST_RESPONSE_FORMAT = {"type": "json_object"}
    FORECAST_THINKING_MODE = "disabled"
    ATOMIC_REPLACE_ATTEMPTS = 8
    ATOMIC_REPLACE_INITIAL_DELAY = 0.05
    ATOMIC_REPLACE_MAX_DELAY = 0.4
    PROMPT_VERSION = "finance_forecast_c0_v2"
    DEFAULT_AGENT_SET_VERSION = "n20_full"
    DEFAULT_SAMPLING_METHOD = "full"
    DEFAULT_DATA_SPLIT = "unspecified"
    RUN_METADATA_FIELDS = (
        "run_id",
        "replicate_id",
        "agent_set_version",
        "sampling_method",
        "data_split",
        "input_snapshot_hash",
        "prompt_version",
        "prompt_hash",
        "random_seed",
    )
    PREDICTION_CSV_FIELDS = (
        *RUN_METADATA_FIELDS,
        "scenario_id",
        "agent_id",
        "agent_role",
        "agent_role_category",
        "agent_role_label",
        "agent_knowledge_level",
        "agent_analysis_style",
        "agent_risk_attitude",
        "agent_investment_horizon",
        "profile_version",
        "as_of",
        "horizon",
        "direction",
        "up_probability",
        "neutral_probability",
        "down_probability",
        "expected_return",
        "expected_return_unit",
        "confidence",
        "evidence_event_ids",
        "reason",
        "raw_response",
        "status",
        "error",
        "attempt_count",
        "finish_reason",
        "response_content_length",
        "reasoning_content_present",
    )
    EVALUATION_CSV_FIELDS = PREDICTION_CSV_FIELDS + (
        "actual_astock_label",
        "actual_astock_direction",
        "actual_astock_change_return",
        "five_day_neutral_threshold",
        "five_day_direction_definition",
        "actual_five_day_close_direction",
        "actual_five_day_close_return",
        "five_day_direction_correct",
        "five_day_return_error",
    )

    def __init__(
        self,
        *,
        storage_dir: Optional[str | Path] = None,
        dataset_path: Optional[str | Path] = None,
    ):
        self.storage_dir = Path(
            storage_dir or getattr(Config, "FINANCE_ADAPTER_DATA_DIR")
        ).resolve()
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_path = dataset_path

    def _run_dir(self, run_id: str) -> Path:
        if not isinstance(run_id, str) or not re.fullmatch(
            r"c0_[A-Za-z0-9_-]{6,64}", run_id
        ):
            raise ValueError("invalid C0 run_id")
        return self.storage_dir / run_id

    def prepare(
        self,
        *,
        run_id: Optional[str] = None,
        dataset_path: Optional[str | Path] = None,
        scenario_ids: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
        run_mode: str = "single",
        replicate_id: Optional[str] = None,
        data_split: str = DEFAULT_DATA_SPLIT,
        agent_set_version: Optional[str] = None,
        sampling_method: str = DEFAULT_SAMPLING_METHOD,
        random_seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Freeze scenarios, roles, and prompts without calling the LLM."""
        if run_mode not in self.RUN_MODES:
            raise ValueError("run_mode must be 'single' or 'all'")
        if run_mode == "all" and (scenario_ids or limit is not None):
            raise ValueError("all-scenario mode cannot use scenario_ids or limit")
        loader = FinancialDatasetLoader(dataset_path or self.dataset_path)
        scenarios = (
            loader.load()
            if run_mode == "all"
            else loader.load(scenario_ids=scenario_ids, limit=limit)
        )
        if run_mode == "single" and len(scenarios) > self.MAX_SCENARIOS_PER_RUN:
            raise ValueError(
                "C0 原型每次只允许运行一个场景；请只选择一个 scenario_id"
            )
        invalid_seed_counts = sorted(
            {
                len(scenario.seed_events)
                for scenario in scenarios
                if len(scenario.seed_events) != self.SEED_EVENT_COUNT
            }
        )
        if invalid_seed_counts:
            raise ValueError(
                "C0 requires exactly five historical seed events per scenario; "
                f"found counts: {', '.join(map(str, invalid_seed_counts))}"
            )
        run_id = run_id or f"c0_{uuid.uuid4().hex[:12]}"
        replicate_id = str(replicate_id or run_id)
        data_split = str(data_split or self.DEFAULT_DATA_SPLIT)
        agent_set_version = str(agent_set_version or self.DEFAULT_AGENT_SET_VERSION)
        sampling_method = str(sampling_method or self.DEFAULT_SAMPLING_METHOD)
        run_dir = self._run_dir(run_id)
        if run_dir.exists() and any(run_dir.iterdir()):
            raise ValueError(f"C0 run already exists: {run_id}")
        run_dir.mkdir(parents=True, exist_ok=True)

        profiles = build_c0_profiles()
        if len(profiles) != C0_AGENT_COUNT:
            raise RuntimeError("C0 profile count does not match role configuration")

        self._write_json(run_dir / "profiles.json", profiles)
        self._write_jsonl(
            run_dir / "scenarios.jsonl",
            (scenario.to_safe_dict() for scenario in scenarios),
        )
        input_snapshot_hash = self._sha256_file(run_dir / "scenarios.jsonl")

        prompt_records = []
        for scenario in scenarios:
            for profile in profiles:
                system_prompt, user_prompt = self.build_prompt(scenario, profile)
                prompt_records.append(
                    {
                        "scenario_id": scenario.scenario_id,
                        "agent_id": profile["user_id"],
                        "agent_role": profile["role_id"],
                        "system": system_prompt,
                        "user": user_prompt,
                    }
                )
        prompt_hash = self._sha256_json(
            [
                {
                    "scenario_id": item["scenario_id"],
                    "agent_id": item["agent_id"],
                    "system": item["system"],
                    "user": item["user"],
                }
                for item in prompt_records
            ]
        )
        prompt_metadata = {
            "run_id": run_id,
            "replicate_id": replicate_id,
            "agent_set_version": agent_set_version,
            "sampling_method": sampling_method,
            "data_split": data_split,
            "input_snapshot_hash": input_snapshot_hash,
            "prompt_version": self.PROMPT_VERSION,
            "prompt_hash": prompt_hash,
            "random_seed": random_seed,
        }
        for record in prompt_records:
            record.update(prompt_metadata)
        self._write_jsonl(run_dir / "prompts.jsonl", prompt_records)

        manifest = {
            "run_id": run_id,
            "replicate_id": replicate_id,
            "agent_set_version": agent_set_version,
            "sampling_method": sampling_method,
            "data_split": data_split,
            "input_snapshot_hash": input_snapshot_hash,
            "prompt_version": self.PROMPT_VERSION,
            "prompt_hash": prompt_hash,
            "random_seed": random_seed,
            "group": self.GROUP,
            "run_mode": run_mode,
            "social_interaction": False,
            "prediction_target": {
                "horizon": "next_5_trading_days",
                "return_definition": "R5 = close5 / original_price - 1",
                "expected_return_unit": "decimal",
                "neutral_threshold": FIVE_DAY_NEUTRAL_THRESHOLD,
                "direction_definition": FIVE_DAY_DIRECTION_DEFINITION,
            },
            "dataset_path": str(loader.dataset_path),
            "scenario_count": len(scenarios),
            "scenario_ids": [scenario.scenario_id for scenario in scenarios],
            "agent_count": len(profiles),
            "expected_prediction_count": len(scenarios) * len(profiles),
            "completed_prediction_count": 0,
            "successful_prediction_count": 0,
            "failed_prediction_count": 0,
            "role_counts": self._role_counts(profiles),
            "status": "prepared",
            "prediction_count": 0,
            "created_at": self._now(),
            "updated_at": self._now(),
            "files": {
                "profiles": "profiles.json",
                "scenarios": "scenarios.jsonl",
                "prompts": "prompts.jsonl",
                "predictions": "predictions.jsonl",
                "predictions_csv": "predictions.csv",
                "evaluation_csv": "evaluation.csv",
                "llm_responses": "llm_responses.jsonl",
            },
        }
        self._write_json(run_dir / "manifest.json", manifest)
        return manifest

    def build_prompt(
        self,
        scenario: FinancialScenario,
        profile: Dict[str, Any],
    ) -> tuple[str, str]:
        """Build the same two-message contract for every independent Agent."""
        neutral_threshold_percent = f"{FIVE_DAY_NEUTRAL_THRESHOLD * 100:.1f}%"
        system_prompt = f"""你是一个用于研究的 A 股市场投资者 Agent。你现在属于 C0 独立判断组。

## 固定行为画像
{profile_prompt_text(profile)}

## C0 信息边界
C0 的关键要求是：不能读取、猜测或引用其他投资者的帖子、预测、回复、聚合意见或社会互动结果；只能依据本条 Prompt 中给出的信息作答。
不要调用外部搜索，不要补充输入之外的企业身份或事实。Profile 中的 decision_source 和 social_role 仅为后续 S1 保留，C0 不得据此假设已经看到任何社会观点。

## 五日预测标签（必须严格遵守）
目标收益 R5 是从当前截止点到未来第 5 个交易日收盘的累计收益率。请按以下固定区间选择 direction：
- R5 < -{neutral_threshold_percent}：down
- -{neutral_threshold_percent} <= R5 <= +{neutral_threshold_percent}：neutral
- R5 > +{neutral_threshold_percent}：up
neutral 只表示预计五日价格变化很小，不表示“信息矛盾”或“无法判断”。如果没有把握，请降低 confidence 并让三个概率更接近，而不要仅因为不确定就选择 neutral。
expected_return 必须使用小数收益率，例如 0.02 表示 +2%，并与 direction 所在区间保持一致。

## 输出格式
响应必须是一个可被 json.loads 解析的单个 JSON object，且只能输出 json，不要使用 Markdown 代码围栏或 JSON 之外的文字。
示例 json 输出：{{"direction":"neutral","up_probability":0.2,"neutral_probability":0.6,"down_probability":0.2,"expected_return":0.0,"confidence":0.5,"evidence_event_ids":[],"reason":"依据有限信息作出判断"}}。"""

        seed_lines = []
        for index, event in enumerate(scenario.seed_events, start=1):
            seed_lines.append(
                "历史种子 {index} | event_id={event_id} | event_time={event_time}\n"
                "文本：{text}\n"
                "READ={read} MARKET={market}".format(
                    index=index,
                    event_id=event.get("event_id", ""),
                    event_time=event.get("event_time", ""),
                    text=event.get("text", ""),
                    read=event.get("read", ""),
                    market=event.get("market", ""),
                )
            )
        current = scenario.current_event
        user_prompt = f"""# C0 独立预测任务

## Agent 角色
- agent_id: {profile['user_id']}
- role_id: {profile['role_id']}
- role: {profile['role_label']}
- role_description: {profile['role_description']}
- profile_version: {profile['profile_version']}

## 预测对象
- scenario_id: {scenario.scenario_id}
- anonymous_asset: {scenario.symbol}
- anonymous_company: {scenario.name}
- information_cutoff: {scenario.prediction_cutoff}
- horizon: {scenario.horizon}

## 截止点前的历史种子
{chr(10).join(seed_lines)}

## 当前已公开事件（截止点）
event_id={current.get('event_id', '')} | event_time={current.get('event_time', '')}
文本：{current.get('text', '')}
READ={current.get('read', '')} MARKET={current.get('market', '')}

请独立预测当前事件之后未来 5 个交易日的累计收盘收益。只预测方向和相对收益，不要预测绝对价格，也不要把“事件是否发生”作为目标。方向必须使用上文固定的 ±{neutral_threshold_percent} 区间；不确定性使用 confidence 和三个概率表达。请只输出一个合法的 json 对象，格式如下：
{{
  "direction": "up|neutral|down",
  "up_probability": 0.0,
  "neutral_probability": 0.0,
  "down_probability": 0.0,
  "expected_return": 0.0,
  "confidence": 0.0,
  "evidence_event_ids": ["EVT_..."],
  "reason": "只引用本 Prompt 中的信息，使用不超过 200 个汉字说明判断依据和不确定性"
}}
三个概率必须在 0 到 1 之间并且加总为 1；evidence_event_ids 只能使用本 Prompt 中出现的事件 ID；只返回一个完整 JSON 对象，不要输出 Markdown 或 JSON 之外的文字。"""
        return system_prompt, user_prompt

    def run(
        self,
        run_id: str,
        *,
        scenario_ids: Optional[Sequence[str]] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Run independent calls sequentially, or validate prompts in dry-run mode."""
        run_dir = self._run_dir(run_id)
        manifest = self.get_status(run_id)
        if manifest.get("run_mode") == "all" and scenario_ids:
            raise ValueError("all-scenario runs cannot be filtered during execution")
        if not dry_run and manifest.get("status") == "completed":
            raise ValueError("completed C0 runs cannot be started again")
        scenarios = self._read_scenarios(run_dir / "scenarios.jsonl", scenario_ids)
        profiles = self._read_json(run_dir / "profiles.json")

        if dry_run:
            records = []
            for scenario in scenarios:
                for profile in profiles:
                    system_prompt, user_prompt = self.build_prompt(scenario, profile)
                    records.append(
                        {
                            "scenario_id": scenario.scenario_id,
                            "agent_id": profile["user_id"],
                            "agent_role": profile["role_id"],
                            "status": "prompt_only",
                            "system": system_prompt,
                            "user": user_prompt,
                        }
                    )
            self._write_jsonl(run_dir / "dry_run_prompts.jsonl", records)
            prompt_check_at = self._now()
            if manifest.get("status") == "completed":
                # Prompt inspection is read-only once a real run has completed.
                manifest.update(
                    {
                        "last_prompt_check_at": prompt_check_at,
                        "updated_at": prompt_check_at,
                    }
                )
            else:
                manifest.update(
                    {
                        "status": "dry_run",
                        "prediction_count": 0,
                        "last_scenario_count": len(scenarios),
                        "last_prompt_check_at": prompt_check_at,
                        "updated_at": prompt_check_at,
                    }
                )
            self._write_json(run_dir / "manifest.json", manifest)
            return manifest

        if not Config.LLM_API_KEY:
            raise RuntimeError("LLM_API_KEY is required to run the C0 forecast")

        client = OpenAI(api_key=Config.LLM_API_KEY, base_url=Config.LLM_BASE_URL)
        prediction_path = run_dir / "predictions.jsonl"
        response_trace_path = run_dir / "llm_responses.jsonl"
        can_resume = manifest.get("run_mode") == "all" and prediction_path.exists()
        predictions: List[Dict[str, Any]] = (
            [
                self._with_run_metadata(record, manifest)
                for record in self._read_jsonl(prediction_path)
            ]
            if can_resume
            else []
        )
        expected_keys = {
            (scenario.scenario_id, int(profile["user_id"]))
            for scenario in scenarios
            for profile in profiles
        }
        completed_keys = [
            (str(record.get("scenario_id", "")), int(record.get("agent_id", -1)))
            for record in predictions
        ]
        if len(completed_keys) != len(set(completed_keys)):
            raise ValueError("existing batch predictions contain duplicate Agent records")
        unexpected_keys = sorted(set(completed_keys) - expected_keys)
        if unexpected_keys:
            raise ValueError(
                "existing batch predictions do not match the frozen run: "
                f"{unexpected_keys[:3]}"
            )
        completed_key_set = set(completed_keys)

        if predictions:
            self._write_csv(
                run_dir / "predictions.csv",
                predictions,
                self.PREDICTION_CSV_FIELDS,
            )
        else:
            self._write_prediction_artifacts(run_dir, predictions)
        if not response_trace_path.exists() or not predictions:
            self._write_jsonl(response_trace_path, [])
        manifest.setdefault("files", {}).update(
            {
                "predictions": "predictions.jsonl",
                "predictions_csv": "predictions.csv",
                "evaluation_csv": "evaluation.csv",
                "llm_responses": "llm_responses.jsonl",
            }
        )
        manifest.update(
            {
                "status": "running",
                "prediction_count": len(predictions),
                "completed_prediction_count": len(predictions),
                "successful_prediction_count": sum(
                    record.get("status") == "ok" for record in predictions
                ),
                "failed_prediction_count": sum(
                    record.get("status") != "ok" for record in predictions
                ),
                "resumed_prediction_count": len(predictions),
                "current_scenario_id": None,
                "current_agent_id": None,
                "updated_at": self._now(),
            }
        )
        self._write_json(run_dir / "manifest.json", manifest)

        for scenario in scenarios:
            for profile in profiles:
                prediction_key = (scenario.scenario_id, int(profile["user_id"]))
                if prediction_key in completed_key_set:
                    continue
                system_prompt, user_prompt = self.build_prompt(scenario, profile)
                forecast = self._request_forecast(
                    client=client,
                    scenario=scenario,
                    profile=profile,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    response_trace_path=response_trace_path,
                )
                record = self._with_run_metadata(forecast.to_dict(), manifest)
                predictions.append(record)
                completed_key_set.add(prediction_key)
                self._append_prediction_artifact(run_dir, record, predictions)
                manifest.update(
                    {
                        "prediction_count": len(predictions),
                        "completed_prediction_count": len(predictions),
                        "successful_prediction_count": sum(
                            record.get("status") == "ok" for record in predictions
                        ),
                        "failed_prediction_count": sum(
                            record.get("status") != "ok" for record in predictions
                        ),
                        "current_scenario_id": scenario.scenario_id,
                        "current_agent_id": profile["user_id"],
                        "updated_at": self._now(),
                    }
                )
                self._write_json(run_dir / "manifest.json", manifest)

        self._write_evaluation_csv(run_dir, predictions)
        manifest.update(
            {
                "status": "completed",
                "prediction_count": len(predictions),
                "successful_prediction_count": sum(
                    record.get("status") == "ok" for record in predictions
                ),
                "failed_prediction_count": sum(
                    record.get("status") != "ok" for record in predictions
                ),
                "last_scenario_count": len(scenarios),
                "current_scenario_id": None,
                "current_agent_id": None,
                "updated_at": self._now(),
            }
        )
        self._write_json(run_dir / "manifest.json", manifest)
        return manifest

    def mark_queued(self, run_id: str) -> Dict[str, Any]:
        """Mark a prepared run as owned by the background batch worker."""
        run_dir = self._run_dir(run_id)
        manifest = self.get_status(run_id)
        if manifest.get("status") == "completed":
            raise ValueError("completed C0 runs cannot be started again")
        manifest.update(
            {
                "status": "queued",
                "execution_mode": "background",
                "background_error": None,
                "updated_at": self._now(),
            }
        )
        self._write_json(run_dir / "manifest.json", manifest)
        return manifest

    def mark_failed(self, run_id: str, error: str) -> Dict[str, Any]:
        """Persist a fatal worker error without discarding partial predictions."""
        run_dir = self._run_dir(run_id)
        manifest = self._read_json(run_dir / "manifest.json")
        predictions = self.get_predictions(run_id)
        manifest.update(
            {
                "status": "failed",
                "prediction_count": len(predictions),
                "completed_prediction_count": len(predictions),
                "successful_prediction_count": sum(
                    record.get("status") == "ok" for record in predictions
                ),
                "failed_prediction_count": sum(
                    record.get("status") != "ok" for record in predictions
                ),
                "current_scenario_id": None,
                "current_agent_id": None,
                "background_error": str(error),
                "updated_at": self._now(),
            }
        )
        self._write_json(run_dir / "manifest.json", manifest)
        return manifest

    def _request_forecast(
        self,
        *,
        client: Any,
        scenario: FinancialScenario,
        profile: Dict[str, Any],
        system_prompt: str,
        user_prompt: str,
        response_trace_path: Optional[Path] = None,
    ) -> C0Forecast:
        """Request one forecast and retry once when JSON is empty or invalid."""
        last_forecast: Optional[C0Forecast] = None
        retry_instruction = (
            "上一次输出为空或不是完整有效的 JSON。请重新完成同一任务，只返回一个完整 "
            "JSON 对象，不要使用 Markdown，不要输出 JSON 之外的文字；reason 不超过 200 个汉字。"
        )

        for attempt_count in range(1, self.MAX_FORECAST_ATTEMPTS + 1):
            attempt_prompt = user_prompt
            if attempt_count > 1:
                attempt_prompt = f"{user_prompt}\n\n## 重试要求\n{retry_instruction}"
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": attempt_prompt},
            ]

            raw_response = ""
            finish_reason: Optional[str] = None
            reasoning_content_present = False
            response: Any = None
            try:
                response = create_chat_completion(
                    client,
                    model=Config.LLM_MODEL_NAME,
                    messages=messages,
                    temperature=self.FORECAST_TEMPERATURE,
                    max_tokens=self.FORECAST_MAX_TOKENS,
                    response_format=self.FORECAST_RESPONSE_FORMAT,
                    thinking_mode=self.FORECAST_THINKING_MODE,
                )
                raw_response = extract_chat_completion_text(response)
                finish_reason = extract_chat_completion_finish_reason(response)
                reasoning_content_present = has_chat_completion_reasoning_content(response)
                if finish_reason == "length":
                    raise ValueError(
                        "model response was truncated (finish_reason=length)"
                    )
                last_forecast = self.parse_forecast(
                    scenario=scenario,
                    profile=profile,
                    raw_response=raw_response,
                    attempt_count=attempt_count,
                    finish_reason=finish_reason,
                    reasoning_content_present=reasoning_content_present,
                )
                self._record_llm_attempt(
                    path=response_trace_path,
                    scenario=scenario,
                    profile=profile,
                    attempt_count=attempt_count,
                    messages=messages,
                    response=response,
                    forecast=last_forecast,
                )
                if last_forecast.status == "ok":
                    return last_forecast
                logger.warning(
                    "C0 Agent returned invalid JSON: scenario=%s agent=%s attempt=%s "
                    "error=%s raw_chars=%s finish_reason=%s reasoning_content=%s",
                    scenario.scenario_id,
                    profile["user_id"],
                    attempt_count,
                    last_forecast.error,
                    last_forecast.response_content_length,
                    last_forecast.finish_reason,
                    last_forecast.reasoning_content_present,
                )
            except Exception as error:
                logger.exception(
                    "C0 Agent call failed: scenario=%s agent=%s attempt=%s "
                    "raw_chars=%s finish_reason=%s reasoning_content=%s",
                    scenario.scenario_id,
                    profile["user_id"],
                    attempt_count,
                    len(raw_response),
                    finish_reason,
                    reasoning_content_present,
                )
                last_forecast = C0Forecast(
                    scenario_id=scenario.scenario_id,
                    agent_id=profile["user_id"],
                    agent_role=profile["role_id"],
                    agent_role_category=profile["role_category"],
                    agent_role_label=profile["role_label"],
                    as_of=scenario.prediction_cutoff,
                    horizon=scenario.horizon,
                    **_forecast_profile_fields(profile),
                    raw_response=raw_response,
                    status="error",
                    error=str(error),
                    attempt_count=attempt_count,
                    finish_reason=finish_reason,
                    response_content_length=len(raw_response),
                    reasoning_content_present=reasoning_content_present,
                )
                self._record_llm_attempt(
                    path=response_trace_path,
                    scenario=scenario,
                    profile=profile,
                    attempt_count=attempt_count,
                    messages=messages,
                    response=response,
                    forecast=last_forecast,
                    exception=error,
                )

        if last_forecast is None:  # Defensive guard; the loop always runs.
            raise RuntimeError("C0 forecast attempt loop produced no result")
        return last_forecast

    def _record_llm_attempt(
        self,
        *,
        path: Optional[Path],
        scenario: FinancialScenario,
        profile: Dict[str, Any],
        attempt_count: int,
        messages: List[Dict[str, Any]],
        response: Any,
        forecast: C0Forecast,
        exception: Optional[Exception] = None,
    ) -> None:
        """Persist one sanitized request/response trace without credentials."""
        if path is None:
            return
        record = {
            "recorded_at": self._now(),
            "scenario_id": scenario.scenario_id,
            "agent_id": profile["user_id"],
            "agent_role": profile["role_id"],
            "attempt_count": attempt_count,
            "request": {
                "model": Config.LLM_MODEL_NAME,
                "temperature": self.FORECAST_TEMPERATURE,
                "max_tokens": self.FORECAST_MAX_TOKENS,
                "response_format": self.FORECAST_RESPONSE_FORMAT,
                "thinking_mode": self.FORECAST_THINKING_MODE,
                "messages": messages,
            },
            "response": self._json_safe(response),
            "extracted": {
                "raw_response": forecast.raw_response,
                "finish_reason": forecast.finish_reason,
                "response_content_length": forecast.response_content_length,
                "reasoning_content_present": forecast.reasoning_content_present,
            },
            "parse_result": {
                "status": forecast.status,
                "error": forecast.error,
            },
            "exception": (
                {"type": type(exception).__name__, "message": str(exception)}
                if exception is not None
                else None
            ),
        }
        self._append_jsonl(path, record)

    def parse_forecast(
        self,
        *,
        scenario: FinancialScenario,
        profile: Dict[str, Any],
        raw_response: str,
        attempt_count: int = 1,
        finish_reason: Optional[str] = None,
        reasoning_content_present: bool = False,
    ) -> C0Forecast:
        """Parse and validate one model response without silently inventing values."""
        try:
            payload = self._parse_json_object(raw_response)
            probability_source = payload.get("probabilities", payload)
            up = self._probability(
                probability_source.get("up_probability", probability_source.get("up"))
            )
            neutral = self._probability(
                probability_source.get(
                    "neutral_probability", probability_source.get("neutral")
                )
            )
            down = self._probability(
                probability_source.get("down_probability", probability_source.get("down"))
            )
            probabilities = [up, neutral, down]
            if any(value is None for value in probabilities):
                raise ValueError("all three class probabilities are required")
            total = sum(probabilities)  # type: ignore[arg-type]
            if total <= 0:
                raise ValueError("probability sum must be positive")
            if abs(total - 1.0) > 1e-6:
                up, neutral, down = [value / total for value in probabilities]  # type: ignore[operator]

            direction = str(payload.get("direction", "")).lower().strip()
            direction = {
                "上涨": "up",
                "持平": "neutral",
                "中性": "neutral",
                "下跌": "down",
            }.get(direction, direction)
            if direction not in {"up", "neutral", "down"}:
                direction = ["up", "neutral", "down"][
                    [up, neutral, down].index(max(up, neutral, down))
                ]
            allowed_ids = {
                str(event.get("event_id"))
                for event in scenario.seed_events + [scenario.current_event]
            }
            evidence = payload.get("evidence_event_ids", [])
            if not isinstance(evidence, list):
                evidence = []
            evidence = [
                str(event_id)
                for event_id in evidence
                if str(event_id) in allowed_ids
            ]

            return C0Forecast(
                scenario_id=scenario.scenario_id,
                agent_id=profile["user_id"],
                agent_role=profile["role_id"],
                agent_role_category=profile["role_category"],
                agent_role_label=profile["role_label"],
                as_of=scenario.prediction_cutoff,
                horizon=scenario.horizon,
                **_forecast_profile_fields(profile),
                direction=direction,
                up_probability=up,
                neutral_probability=neutral,
                down_probability=down,
                expected_return=self._expected_return(
                    payload.get("expected_return"), direction=direction
                ),
                confidence=self._probability(payload.get("confidence")),
                evidence_event_ids=evidence,
                reason=str(payload.get("reason", "")),
                raw_response=raw_response,
                attempt_count=attempt_count,
                finish_reason=finish_reason,
                response_content_length=len(raw_response),
                reasoning_content_present=reasoning_content_present,
            )
        except Exception as error:
            return C0Forecast(
                scenario_id=scenario.scenario_id,
                agent_id=profile["user_id"],
                agent_role=profile["role_id"],
                agent_role_category=profile["role_category"],
                agent_role_label=profile["role_label"],
                as_of=scenario.prediction_cutoff,
                horizon=scenario.horizon,
                **_forecast_profile_fields(profile),
                raw_response=raw_response,
                status="parse_error",
                error=str(error),
                attempt_count=attempt_count,
                finish_reason=finish_reason,
                response_content_length=len(raw_response),
                reasoning_content_present=reasoning_content_present,
            )

    def get_status(self, run_id: str) -> Dict[str, Any]:
        run_dir = self._run_dir(run_id)
        manifest = self._read_json(run_dir / "manifest.json")
        prediction_path = run_dir / "predictions.jsonl"
        if prediction_path.exists():
            predictions = self._read_jsonl(prediction_path)
            expected = int(manifest.get("expected_prediction_count") or 0)
            if (
                manifest.get("status") not in {"queued", "running"}
                and expected > 0
                and len(predictions) == expected
            ):
                predictions_csv = run_dir / "predictions.csv"
                evaluation_csv = run_dir / "evaluation.csv"
                if not predictions_csv.exists():
                    self._write_csv(
                        predictions_csv,
                        predictions,
                        self.PREDICTION_CSV_FIELDS,
                    )
                if not evaluation_csv.exists():
                    self._write_evaluation_csv(run_dir, predictions)
                successful = sum(record.get("status") == "ok" for record in predictions)
                repaired = {
                    "status": "completed",
                    "prediction_count": len(predictions),
                    "completed_prediction_count": len(predictions),
                    "successful_prediction_count": successful,
                    "failed_prediction_count": len(predictions) - successful,
                    "current_scenario_id": None,
                    "current_agent_id": None,
                }
                if any(manifest.get(key) != value for key, value in repaired.items()):
                    manifest.update(repaired)
                    manifest["updated_at"] = self._now()
                    self._write_json(run_dir / "manifest.json", manifest)
        return manifest

    def get_outcome(self, run_id: str) -> Dict[str, Any]:
        """Return hidden ground truth to the researcher after completion only."""
        manifest = self.get_status(run_id)
        if manifest.get("status") != "completed":
            raise ValueError("ground truth is available only after the run is completed")
        scenario_ids = manifest.get("scenario_ids") or []
        if len(scenario_ids) != 1:
            raise ValueError("C0 outcome requires exactly one prepared scenario")
        return FinancialOutcomeEvaluator().get_outcome(str(scenario_ids[0]))

    def list_scenarios(self) -> List[Dict[str, Any]]:
        """Return safe summaries for the selector in the C0 workbench."""
        loader = FinancialDatasetLoader(self.dataset_path)
        summaries = []
        for scenario in loader.load():
            summaries.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "symbol": scenario.symbol,
                    "name": scenario.name,
                    "prediction_cutoff": scenario.prediction_cutoff,
                    "horizon": scenario.horizon,
                    "seed_count": len(scenario.seed_events),
                    "current_event_id": scenario.current_event.get("event_id"),
                    "current_event_text": scenario.current_event.get("text", ""),
                }
            )
        return summaries

    def get_preview(self, run_id: str) -> Dict[str, Any]:
        """Return one frozen prompt and safe scenario metadata for inspection."""
        run_dir = self._run_dir(run_id)
        manifest = self._read_json(run_dir / "manifest.json")
        prompts = self._read_jsonl(run_dir / "prompts.jsonl")
        scenarios = self._read_jsonl(run_dir / "scenarios.jsonl")
        return {
            "run_id": run_id,
            "manifest": manifest,
            "scenario": scenarios[0] if scenarios else None,
            "prompt": prompts[0] if prompts else None,
        }

    def get_predictions(self, run_id: str) -> List[Dict[str, Any]]:
        """Read the partial or completed prediction stream for a run."""
        path = self._run_dir(run_id) / "predictions.jsonl"
        if not path.exists():
            return []
        return self._read_jsonl(path)

    def get_csv_path(self, run_id: str, kind: str) -> Path:
        """Resolve a researcher CSV artifact using a strict file whitelist."""
        filenames = {
            "predictions": "predictions.csv",
            "evaluation": "evaluation.csv",
        }
        if kind not in filenames:
            raise ValueError("CSV kind must be 'predictions' or 'evaluation'")
        path = self._run_dir(run_id) / filenames[kind]
        if not path.exists():
            if kind == "evaluation":
                raise FileNotFoundError(
                    "evaluation CSV is available only after the run is completed"
                )
            raise FileNotFoundError(f"C0 CSV artifact not found: {path}")
        return path

    def _read_scenarios(
        self, path: Path, scenario_ids: Optional[Sequence[str]]
    ) -> List[FinancialScenario]:
        selected_values = list(scenario_ids or [])
        if any(not isinstance(value, str) or not value.strip() for value in selected_values):
            raise ValueError("scenario_ids must contain non-empty strings")
        selected = set(selected_values)
        scenarios = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                scenario = FinancialScenario.from_dict(json.loads(line))
                if not selected or scenario.scenario_id in selected:
                    scenarios.append(scenario)
        if selected and {s.scenario_id for s in scenarios} != selected:
            missing = sorted(selected - {s.scenario_id for s in scenarios})
            raise ValueError(f"scenario IDs not found in prepared run: {', '.join(missing)}")
        if not scenarios:
            raise ValueError("no scenarios selected in prepared run")
        return scenarios

    @staticmethod
    def _parse_json_object(raw_response: str) -> Dict[str, Any]:
        text = (raw_response or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.S)
            if not match:
                raise ValueError("model response is not a JSON object")
            payload = json.loads(match.group(0))
        if not isinstance(payload, dict):
            raise ValueError("model response must be a JSON object")
        return payload

    @staticmethod
    def _probability(value: Any) -> Optional[float]:
        if value is None:
            return None
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("probability must be finite")
        if number < 0 or number > 100:
            raise ValueError("probability must be between 0 and 1 (or 0 and 100)")
        return number / 100 if number > 1 else number

    @staticmethod
    def _number(value: Any) -> Optional[float]:
        if value is None or value == "":
            return None
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("number must be finite")
        return number

    @staticmethod
    def _expected_return(
        value: Any, *, direction: Optional[str] = None
    ) -> Optional[float]:
        """Parse a return as a decimal, accepting legacy percentage outputs.

        The experiment stores returns in decimal form (``0.03`` means +3%).
        Older prompts/models sometimes return ``3.5`` or ``"3.5%"`` for +3.5%,
        so values outside [-1, 1] are interpreted as percentage points. A
        neutral forecast also uses its fixed return band to disambiguate values
        such as ``0.5`` (0.5%, not 50%).
        """
        if value is None or value == "":
            return None
        if isinstance(value, str):
            text = value.strip()
            has_percent_sign = text.endswith("%")
            if has_percent_sign:
                text = text[:-1].strip()
            number = float(text)
        else:
            has_percent_sign = False
            number = float(value)
        if not math.isfinite(number):
            raise ValueError("expected_return must be finite")
        if has_percent_sign or abs(number) > 1:
            number /= 100.0
        elif (
            direction == "neutral"
            and abs(number) > FIVE_DAY_NEUTRAL_THRESHOLD
            and abs(number / 100.0) <= FIVE_DAY_NEUTRAL_THRESHOLD
        ):
            number /= 100.0
        return number

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _sha256_json(value: Any) -> str:
        payload = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _run_metadata(manifest: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: manifest.get(key)
            for key in C0ExperimentService.RUN_METADATA_FIELDS
        }

    @classmethod
    def _with_run_metadata(
        cls, record: Dict[str, Any], manifest: Dict[str, Any]
    ) -> Dict[str, Any]:
        enriched = dict(record)
        enriched.update(cls._run_metadata(manifest))
        enriched.setdefault("expected_return_unit", "decimal")
        return enriched

    @staticmethod
    def _role_counts(profiles: Iterable[Dict[str, Any]]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for profile in profiles:
            key = profile["role_category"]
            counts[key] = counts.get(key, 0) + 1
        return counts

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @classmethod
    def _write_json(cls, path: Path, payload: Any) -> None:
        temporary = cls._temporary_path(path)
        try:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            cls._replace_with_retry(temporary, path)
        finally:
            cls._remove_temporary(temporary)

    @classmethod
    def _write_jsonl(cls, path: Path, records: Iterable[Any]) -> None:
        temporary = cls._temporary_path(path)
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            cls._replace_with_retry(temporary, path)
        finally:
            cls._remove_temporary(temporary)

    def _write_prediction_artifacts(
        self, run_dir: Path, records: Sequence[Dict[str, Any]]
    ) -> None:
        self._write_jsonl(run_dir / "predictions.jsonl", records)
        self._write_csv(
            run_dir / "predictions.csv",
            records,
            self.PREDICTION_CSV_FIELDS,
        )

    def _append_prediction_artifact(
        self,
        run_dir: Path,
        record: Dict[str, Any],
        records: Sequence[Dict[str, Any]],
    ) -> None:
        # JSONL has one writer, so appending avoids replacing a file that the
        # frontend or an editor may briefly hold open on Windows.
        self._append_jsonl(run_dir / "predictions.jsonl", record)
        self._write_csv(
            run_dir / "predictions.csv",
            records,
            self.PREDICTION_CSV_FIELDS,
        )

    def _write_evaluation_csv(
        self, run_dir: Path, predictions: Sequence[Dict[str, Any]]
    ) -> None:
        evaluator = FinancialOutcomeEvaluator()
        outcomes: Dict[str, Dict[str, Any]] = {}
        records = []
        for prediction in predictions:
            scenario_id = str(prediction.get("scenario_id", ""))
            if scenario_id not in outcomes:
                outcomes[scenario_id] = evaluator.get_outcome(scenario_id)
            outcome = outcomes[scenario_id]
            expected_return = prediction.get("expected_return")
            return_error = ""
            if isinstance(expected_return, (int, float)) and math.isfinite(
                float(expected_return)
            ):
                return_error = float(expected_return) - float(
                    outcome["five_day_close_return"]
                )
            direction_correct: Any = ""
            if prediction.get("status") == "ok":
                direction_correct = (
                    prediction.get("direction")
                    == outcome["five_day_close_direction"]
                )
            records.append(
                {
                    **prediction,
                    "actual_astock_label": outcome["astock_label"],
                    "actual_astock_direction": outcome["astock_direction"],
                    "actual_astock_change_return": outcome["astock_change_return"],
                    "five_day_neutral_threshold": outcome[
                        "five_day_neutral_threshold"
                    ],
                    "five_day_direction_definition": outcome[
                        "five_day_direction_definition"
                    ],
                    "actual_five_day_close_direction": outcome[
                        "five_day_close_direction"
                    ],
                    "actual_five_day_close_return": outcome[
                        "five_day_close_return"
                    ],
                    "five_day_direction_correct": direction_correct,
                    "five_day_return_error": return_error,
                }
            )
        self._write_csv(
            run_dir / "evaluation.csv",
            records,
            self.EVALUATION_CSV_FIELDS,
        )

    @classmethod
    def _write_csv(
        cls,
        path: Path,
        records: Iterable[Dict[str, Any]],
        fieldnames: Sequence[str],
    ) -> None:
        temporary = cls._temporary_path(path)
        try:
            with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=fieldnames,
                    extrasaction="ignore",
                )
                writer.writeheader()
                for record in records:
                    writer.writerow(
                        {
                            field: cls._csv_value(record.get(field))
                            for field in fieldnames
                        }
                    )
            cls._replace_with_retry(temporary, path)
        finally:
            cls._remove_temporary(temporary)

    @classmethod
    def _temporary_path(cls, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        return path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")

    @classmethod
    def _replace_with_retry(cls, temporary: Path, path: Path) -> None:
        delay = cls.ATOMIC_REPLACE_INITIAL_DELAY
        for attempt in range(1, cls.ATOMIC_REPLACE_ATTEMPTS + 1):
            try:
                temporary.replace(path)
                return
            except PermissionError:
                if attempt >= cls.ATOMIC_REPLACE_ATTEMPTS:
                    raise
                if attempt == 1:
                    logger.warning(
                        "Atomic file replace was temporarily blocked; retrying: %s",
                        path,
                    )
                time.sleep(delay)
                delay = min(delay * 2, cls.ATOMIC_REPLACE_MAX_DELAY)

    @staticmethod
    def _remove_temporary(path: Path) -> None:
        if not path.exists():
            return
        try:
            path.unlink()
        except OSError:
            logger.warning("Could not remove temporary finance artifact: %s", path)

    @staticmethod
    def _csv_value(value: Any) -> Any:
        if value is None:
            return ""
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value, ensure_ascii=False)
        if isinstance(value, bool):
            return "true" if value else "false"
        return value

    @staticmethod
    def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

    @classmethod
    def _json_safe(cls, value: Any) -> Any:
        """Convert OpenAI/Pydantic response objects into JSON-safe values."""
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(key): cls._json_safe(child) for key, child in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._json_safe(child) for child in value]
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            try:
                return cls._json_safe(model_dump(mode="json"))
            except TypeError:
                return cls._json_safe(model_dump())
        attributes = getattr(value, "__dict__", None)
        if isinstance(attributes, dict):
            return {
                str(key): cls._json_safe(child)
                for key, child in attributes.items()
                if not str(key).startswith("_")
            }
        return str(value)

    @staticmethod
    def _read_json(path: Path) -> Any:
        if not path.exists():
            raise FileNotFoundError(f"C0 artifact not found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"C0 artifact not found: {path}")
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        # A status poll can overlap the final write of a record.
                        break
                    if isinstance(record, dict):
                        records.append(record)
        return records
