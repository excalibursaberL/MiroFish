"""Reddit-only S1 social-interaction adapter.

S1 deliberately reuses MiroFish' existing ``SimulationRunner`` and OASIS
Reddit environment.  This module only owns the finance-specific preparation,
event schedule, final forecast interview, and researcher-only artifacts.
"""

from __future__ import annotations

import json
import hashlib
import math
import re
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..config import Config
from ..models.project import ProjectManager
from ..services.simulation_manager import (
    SimulationManager,
    SimulationState,
    SimulationStatus,
)
from ..services.simulation_runner import RunnerStatus, SimulationRunner
from ..services.zep_entity_reader import EntityNode, ZepEntityReader
from ..utils.logger import get_logger
from .c0 import C0ExperimentService
from .dataset import FinancialDatasetLoader, FinancialScenario
from .evaluator import (
    FIVE_DAY_DIRECTION_DEFINITION,
    FIVE_DAY_NEUTRAL_THRESHOLD,
    FinancialOutcomeEvaluator,
)
from .models import C0Forecast
from .roles import C0_AGENT_COUNT, build_c0_profiles, profile_prompt_text
from .source_resolver import FinanceEventSourceResolver


logger = get_logger("mirofish.finance.s1")


class S1ExperimentService:
    """Prepare and run one Reddit S1 scenario through MiroFish/OASIS."""

    GROUP = "S1"
    PLATFORM = "reddit"
    MAX_SCENARIOS_PER_RUN = 1
    DEFAULT_SOCIAL_ROUNDS = 6
    MIN_SOCIAL_ROUNDS = 1
    MAX_SOCIAL_ROUNDS = 12
    # The generic runner still requires a time step. It is an internal pacing
    # value only; S1 rounds are interaction steps, not real-world minutes.
    DEFAULT_MINUTES_PER_ROUND = 30
    # OASIS Reddit currently assigns Agent IDs by profile-list position rather
    # than honoring a profile's user_id. Keep investors first (0-19); dynamic
    # source accounts then receive contiguous IDs starting at 20.
    SOURCE_AGENT_START = C0_AGENT_COUNT
    RUN_ID_PATTERN = re.compile(r"s1_reddit_[A-Za-z0-9_-]{6,64}")
    _background_lock = threading.Lock()
    _background_threads: Dict[str, threading.Thread] = {}
    PROMPT_VERSION = "finance_forecast_s1_v2"
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

    CSV_FIELDS = (
        *RUN_METADATA_FIELDS,
        "scenario_id",
        "condition",
        "prediction_stage",
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
        "social_action_count",
        "social_post_count",
        "social_comment_count",
        "social_like_count",
        "social_dislike_count",
        "attempt_count",
        "finish_reason",
        "response_content_length",
        "reasoning_content_present",
    )
    EVALUATION_FIELDS = CSV_FIELDS + (
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
    AGENT_CHANGE_FIELDS = (
        *RUN_METADATA_FIELDS,
        "scenario_id",
        "agent_id",
        "agent_role_label",
        "pre_direction",
        "post_direction",
        "direction_changed",
        "distribution_js_divergence",
        "pre_expected_return",
        "post_expected_return",
        "expected_return_delta",
        "pre_confidence",
        "post_confidence",
        "confidence_delta",
        "social_action_count",
        "social_post_count",
        "social_comment_count",
        "social_like_count",
        "social_dislike_count",
        "pre_evidence_event_ids",
        "post_evidence_event_ids",
        "evidence_changed",
        "pair_status",
    )
    ROUND_METRIC_FIELDS = (
        *RUN_METADATA_FIELDS,
        "scenario_id",
        "round",
        "action_count",
        "active_agent_count",
        "post_count",
        "comment_count",
        "like_count",
        "dislike_count",
        "refresh_count",
        "other_action_count",
        "unique_post_id_count",
        "unique_target_agent_count",
        "exposure_count",
    )
    BELIEF_SNAPSHOT_FIELDS = (
        *RUN_METADATA_FIELDS,
        "scenario_id",
        "round",
        "snapshot_type",
        "snapshot_source",
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
    EXPOSURE_FIELDS = (
        *RUN_METADATA_FIELDS,
        "scenario_id",
        "exposure_id",
        "trace_id",
        "round",
        "timestamp",
        "viewer_agent_id",
        "content_type",
        "content_id",
        "author_agent_id",
        "content_text",
        "content_stance",
        "stance_score",
        "stance_source",
        "exposure_type",
        "action_type",
        "interacted",
        "interaction_target_id",
        "first_seen_round",
    )

    def __init__(
        self,
        *,
        storage_dir: Optional[str | Path] = None,
        dataset_path: Optional[str | Path] = None,
    ) -> None:
        self.storage_dir = Path(
            storage_dir or getattr(Config, "FINANCE_ADAPTER_DATA_DIR")
        ).resolve()
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_path = dataset_path

    def _run_dir(self, run_id: str) -> Path:
        if not isinstance(run_id, str) or not self.RUN_ID_PATTERN.fullmatch(run_id):
            raise ValueError("invalid S1 Reddit run_id")
        return self.storage_dir / run_id

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _sha256_json(value: Any) -> str:
        payload = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @classmethod
    def _run_metadata(cls, manifest: Dict[str, Any]) -> Dict[str, Any]:
        return {key: manifest.get(key) for key in cls.RUN_METADATA_FIELDS}

    @classmethod
    def _with_run_metadata(
        cls, record: Dict[str, Any], manifest: Dict[str, Any]
    ) -> Dict[str, Any]:
        enriched = dict(record)
        enriched.update(cls._run_metadata(manifest))
        enriched.setdefault("expected_return_unit", "decimal")
        return enriched

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        C0ExperimentService._write_json(path, value)

    @staticmethod
    def _write_jsonl(path: Path, records: Sequence[Dict[str, Any]]) -> None:
        C0ExperimentService._write_jsonl(path, records)

    @staticmethod
    def _write_csv(path: Path, records: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
        C0ExperimentService._write_csv(path, records, fields)

    @staticmethod
    def _s1_investor_profiles(
        scenario: FinancialScenario,
    ) -> List[Dict[str, Any]]:
        history_lines = [
            f"{index}. [{event.get('event_id')} | {event.get('event_time')}] "
            f"{event.get('text', '')}"
            for index, event in enumerate(scenario.seed_events, start=1)
        ]
        history_memory = "\n".join(history_lines)
        profiles = []
        for profile in build_c0_profiles():
            role_text = profile_prompt_text(profile)
            persona = (
                f"你是{profile['role_label']}，编号为{profile['agent_key']}。"
                f"{profile['role_description']}\n固定行为画像：\n{role_text}\n"
                "本轮属于 S1 社会互动实验。你可以阅读其他投资者在讨论区公开发布的观点，"
                "也可以发布自己的分析、回复或评价。你的 Profile 不能预先决定股票涨跌；"
                "只能根据事件信息和互动中实际看到的内容形成判断。不要编造输入中没有的价格、"
                "指标或企业事实。\n\n"
                "以下是预测截止点之前已经发生、所有投资者都知道的只读历史记忆。"
                "它们不是本轮 Reddit 帖子，也不占用社会互动轮次：\n"
                f"{history_memory}"
            )
            item = dict(profile)
            item["persona"] = persona
            item["bio"] = f"S1 {profile['role_label']}（匿名投资者）"
            item["agent_class"] = "investor"
            item.setdefault("mbti", "ISTJ")
            item.setdefault("gender", "anonymous")
            item.setdefault("age", 35)
            item.setdefault("country", "CN")
            item.setdefault("interested_topics", ["A股", "公司公告"])
            profiles.append(item)
        if len(profiles) != C0_AGENT_COUNT:
            raise RuntimeError("S1 investor profile count does not equal C0 count")
        return profiles

    @staticmethod
    def _load_graph_entities(graph_id: str) -> List[EntityNode]:
        """Load every named Zep node for publisher matching.

        Some generated ontologies leave valid company nodes without a type
        label. The generic entity reader filters those nodes, but S1 publisher
        attribution depends on their names and must not silently discard them.
        Only entities actually matched as event publishers become source
        accounts, so retaining untyped nodes does not inflate active Agents.
        """
        nodes = ZepEntityReader().get_all_nodes(graph_id)
        return [
            EntityNode(
                uuid=str(node.get("uuid", "")),
                name=str(node.get("name", "")),
                labels=list(node.get("labels") or []),
                summary=str(node.get("summary", "")),
                attributes=dict(node.get("attributes") or {}),
            )
            for node in nodes
            if node.get("uuid") and str(node.get("name", "")).strip()
        ]

    def _build_current_event(
        self,
        scenario: FinancialScenario,
        event_sources: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        source_by_event = {str(item["event_id"]): item for item in event_sources}
        current = scenario.current_event
        source = source_by_event[str(current.get("event_id"))]
        return {
            "phase": "current",
            "round": 0,
            "event_id": current.get("event_id"),
            "event_time": current.get("event_time"),
            "text": current.get("text", ""),
            **source,
        }

    @staticmethod
    def _oasis_profiles(profiles: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return the minimal Reddit profile contract plus persona fields."""
        result = []
        for profile in profiles:
            result.append(
                {
                    "user_id": profile["user_id"],
                    "username": profile["username"],
                    "name": profile["name"],
                    "bio": profile["bio"],
                    "persona": profile["persona"],
                    "karma": profile.get("karma", 1000),
                    "created_at": profile.get("created_at", "1970-01-01"),
                    "mbti": profile.get("mbti", "ISTJ"),
                    "gender": profile.get("gender", "anonymous"),
                    "age": profile.get("age", 35),
                    "country": profile.get("country", "CN"),
                    "profession": profile.get("profession", "investor"),
                    "interested_topics": profile.get(
                        "interested_topics", ["A股", "公司公告"]
                    ),
                }
            )
        return result

    def prepare(
        self,
        *,
        run_id: Optional[str] = None,
        dataset_path: Optional[str | Path] = None,
        scenario_id: Optional[str] = None,
        graph_id: Optional[str] = None,
        project_id: Optional[str] = None,
        source_mode: str = "auto",
        social_rounds: int = DEFAULT_SOCIAL_ROUNDS,
        replicate_id: Optional[str] = None,
        data_split: str = DEFAULT_DATA_SPLIT,
        agent_set_version: Optional[str] = None,
        sampling_method: str = DEFAULT_SAMPLING_METHOD,
        random_seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        if isinstance(social_rounds, bool) or not isinstance(social_rounds, int):
            raise ValueError("social_rounds must be an integer")
        if not self.MIN_SOCIAL_ROUNDS <= social_rounds <= self.MAX_SOCIAL_ROUNDS:
            raise ValueError(
                f"social_rounds must be between {self.MIN_SOCIAL_ROUNDS} and "
                f"{self.MAX_SOCIAL_ROUNDS}"
            )
        if source_mode not in {"auto", "graph", "scenario"}:
            raise ValueError("source_mode must be auto, graph, or scenario")
        if project_id:
            project = ProjectManager.get_project(project_id)
            if project is None:
                raise ValueError(f"MiroFish project not found: {project_id}")
            if graph_id and project.graph_id and graph_id != project.graph_id:
                raise ValueError("graph_id does not match the MiroFish project")
            graph_id = graph_id or project.graph_id
            if not graph_id:
                raise ValueError("MiroFish project graph is not completed")
        resolved_source_mode = (
            "graph" if source_mode == "auto" and graph_id else
            "scenario" if source_mode == "auto" else
            source_mode
        )
        if resolved_source_mode == "graph" and not graph_id:
            raise ValueError("graph source mode requires graph_id")
        loader = FinancialDatasetLoader(dataset_path or self.dataset_path)
        scenarios = loader.load(scenario_ids=[scenario_id] if scenario_id else None, limit=None if scenario_id else 1)
        if len(scenarios) != 1 or len(scenarios[0].seed_events) != 5:
            raise ValueError("S1 prototype requires exactly one scenario with five seed events")
        scenario = scenarios[0]
        run_id = run_id or f"s1_reddit_{uuid.uuid4().hex[:12]}"
        replicate_id = str(replicate_id or run_id)
        data_split = str(data_split or self.DEFAULT_DATA_SPLIT)
        agent_set_version = str(agent_set_version or self.DEFAULT_AGENT_SET_VERSION)
        sampling_method = str(sampling_method or self.DEFAULT_SAMPLING_METHOD)
        run_dir = self._run_dir(run_id)
        if run_dir.exists() and any(run_dir.iterdir()):
            raise ValueError(f"S1 run already exists: {run_id}")

        investors = self._s1_investor_profiles(scenario)
        graph_entities = (
            self._load_graph_entities(str(graph_id))
            if resolved_source_mode == "graph"
            else []
        )
        sources, event_sources, entity_mapping = FinanceEventSourceResolver(
            graph_entities
        ).resolve(
            scenario,
            source_agent_start=self.SOURCE_AGENT_START,
            source_mode=resolved_source_mode,
            graph_id=graph_id,
        )
        all_profiles = investors + sources
        current_event = self._build_current_event(scenario, event_sources)
        run_dir.mkdir(parents=True, exist_ok=True)
        simulation_id = f"finance_{run_id[3:]}"
        total_rounds = social_rounds
        minutes_per_round = self.DEFAULT_MINUTES_PER_ROUND
        total_simulation_hours = max(
            1, math.ceil(total_rounds * minutes_per_round / 60)
        )
        simulation_dir = Path(SimulationManager.SIMULATION_DATA_DIR) / simulation_id
        simulation_dir.mkdir(parents=True, exist_ok=True)

        self._write_json(run_dir / "profiles.json", investors)
        self._write_json(run_dir / "source_profiles.json", sources)
        self._write_json(run_dir / "entity_agent_mapping.json", entity_mapping)
        self._write_jsonl(run_dir / "scenarios.jsonl", [scenario.to_safe_dict()])
        self._write_jsonl(run_dir / "history_memory.jsonl", scenario.seed_events)
        self._write_json(run_dir / "current_event.json", current_event)
        input_snapshot_hash = self._sha256_json(scenario.to_safe_dict())
        self._write_json(simulation_dir / "reddit_profiles.json", self._oasis_profiles(all_profiles))

        agent_configs = []
        for profile in all_profiles:
            is_investor = profile.get("agent_class") == "investor"
            agent_configs.append(
                {
                    "agent_id": profile["user_id"],
                    "entity_name": profile["name"],
                    "agent_class": profile.get("agent_class", "source"),
                    "activity_level": 1.0 if is_investor else 0.0,
                    "active_hours": list(range(24)) if is_investor else [],
                }
            )
        config = {
            "simulation_id": simulation_id,
            "llm_model": getattr(Config, "LLM_MODEL_NAME", None) or "deepseek-chat",
            "time_config": {
                "total_simulation_hours": total_simulation_hours,
                "minutes_per_round": minutes_per_round,
                "agents_per_hour_min": C0_AGENT_COUNT,
                "agents_per_hour_max": C0_AGENT_COUNT,
                "peak_hours": [],
                "off_peak_hours": [],
            },
            "agent_configs": agent_configs,
            "event_config": {
                "initial_posts": [
                    {
                        "poster_agent_id": current_event["publisher_agent_id"],
                        "content": (
                            f"[{current_event['event_id']} | "
                            f"{current_event['event_time']}] {current_event['text']}"
                        ),
                        "finance_event": current_event,
                    }
                ]
            },
            "finance_s1": {
                "scenario_id": scenario.scenario_id,
                "graph_id": graph_id or "",
                "source_mode": resolved_source_mode,
                "social_rounds": social_rounds,
                "total_rounds": total_rounds,
                "tracked_agent_ids": [p["user_id"] for p in investors],
                "source_agent_ids": [p["user_id"] for p in sources],
                "history_memory_event_ids": [
                    event.get("event_id") for event in scenario.seed_events
                ],
                "current_event_id": current_event.get("event_id"),
                "pre_social_interviews": [
                    {
                        "agent_id": int(profile["user_id"]),
                        "prompt": self.build_forecast_prompt(
                            scenario, stage="pre_social"
                        ),
                        "retry_prompt": self.build_retry_prompt(
                            scenario, stage="pre_social"
                        ),
                    }
                    for profile in investors
                ],
                "round_belief_snapshot_interviews": [
                    {
                        "agent_id": int(profile["user_id"]),
                        "prompt": self.build_belief_snapshot_prompt(
                            round_number="__ROUND_NUMBER__"
                        ),
                        "retry_prompt": (
                            "The previous private belief snapshot was invalid. "
                            "Return only the required JSON object.\n\n"
                            + self.build_belief_snapshot_prompt(
                                round_number="__ROUND_NUMBER__"
                            )
                        ),
                    }
                    for profile in investors
                ],
                "belief_snapshot_enabled": True,
                "anonymous": True,
                "prompt_version": self.PROMPT_VERSION,
            },
        }
        prompt_hash = self._sha256_json(
            {
                "pre_social": [
                    {
                        "agent_id": int(profile["user_id"]),
                        "prompt": self.build_forecast_prompt(
                            scenario, stage="pre_social"
                        ),
                    }
                    for profile in investors
                ],
                "post_social": [
                    {
                        "agent_id": int(profile["user_id"]),
                        "prompt": self.build_forecast_prompt(
                            scenario, stage="post_social"
                        ),
                    }
                    for profile in investors
                ],
                "round_belief_snapshot": self.build_belief_snapshot_prompt(
                    round_number="__ROUND_NUMBER__"
                ),
            }
        )
        config["finance_s1"].update(
            {
                "replicate_id": replicate_id,
                "agent_set_version": agent_set_version,
                "sampling_method": sampling_method,
                "data_split": data_split,
                "input_snapshot_hash": input_snapshot_hash,
                "prompt_hash": prompt_hash,
                "random_seed": random_seed,
            }
        )
        self._write_json(simulation_dir / "simulation_config.json", config)
        self._write_json(run_dir / "simulation_config.json", config)

        manager = SimulationManager()
        state = SimulationState(
            simulation_id=simulation_id,
            project_id=project_id or "finance_s1",
            graph_id=graph_id or "",
            enable_twitter=False,
            enable_reddit=True,
            status=SimulationStatus.READY,
            entities_count=len(sources),
            profiles_count=len(all_profiles),
            entity_types=[
                "finance_investor",
                *sorted({str(source.get("source_type", "organization")) for source in sources}),
            ],
            profiles_generated=True,
            config_generated=True,
        )
        manager._save_simulation_state(state)

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
            "platform": self.PLATFORM,
            "simulation_id": simulation_id,
            "scenario_id": scenario.scenario_id,
            "project_id": project_id or "",
            "graph_id": graph_id or "",
            "source_mode": resolved_source_mode,
            "graph_entity_count": len(graph_entities),
            "graph_resolved_event_count": sum(
                event.get("publisher_origin") == "zep_graph"
                for event in event_sources
            ),
            "public_feed_event_count": sum(
                event.get("publisher_origin") == "public_feed"
                for event in event_sources
            ),
            "dataset_path": str(loader.dataset_path),
            "scenario_count": 1,
            "investor_agent_count": C0_AGENT_COUNT,
            "source_agent_count": len(sources),
            "agent_count_total": len(all_profiles),
            "social_rounds": social_rounds,
            "total_rounds": total_rounds,
            "round_semantics": "interaction_step",
            "belief_snapshot_enabled": True,
            "expected_belief_snapshot_count": C0_AGENT_COUNT * (social_rounds + 1),
            "stance_annotation": {
                "status": "pending_offline_llm",
                "prompt_version": "finance_stance_annotation_v1",
            },
            "history_memory_event_count": len(scenario.seed_events),
            "current_public_event_count": 1,
            "prediction_target": {
                "horizon": "next_5_trading_days",
                "expected_return_unit": "decimal",
                "neutral_threshold": FIVE_DAY_NEUTRAL_THRESHOLD,
                "direction_definition": FIVE_DAY_DIRECTION_DEFINITION,
            },
            "status": "prepared",
            "created_at": self._now(),
            "updated_at": self._now(),
            "expected_prediction_count": C0_AGENT_COUNT * 2,
            "completed_prediction_count": 0,
            "successful_prediction_count": 0,
            "failed_prediction_count": 0,
            "files": {
                "profiles": "profiles.json",
                "source_profiles": "source_profiles.json",
                "entity_agent_mapping": "entity_agent_mapping.json",
                "scenarios": "scenarios.jsonl",
                "history_memory": "history_memory.jsonl",
                "current_event": "current_event.json",
                "simulation_config": "simulation_config.json",
                "round_belief_interviews": "round_belief_interviews.jsonl",
                "social_actions": "social_actions.jsonl",
                "agent_round_states": "agent_round_states.jsonl",
                "belief_snapshots": "belief_snapshots.jsonl",
                "exposure_edges": "exposure_edges.jsonl",
                "stance_annotations": "stance_annotations.jsonl",
                "stance_annotations_csv": "stance_annotations.csv",
                "social_actions_annotated": "social_actions_annotated.jsonl",
                "exposure_edges_annotated": "exposure_edges_annotated.jsonl",
                "interview_responses": "interview_responses.json",
                "predictions": "predictions.jsonl",
                "pre_social_predictions": "pre_social_predictions.jsonl",
                "post_social_predictions": "post_social_predictions.jsonl",
                "prediction_changes": "prediction_changes.jsonl",
                "predictions_csv": "predictions.csv",
                "evaluation_csv": "evaluation.csv",
                "agent_changes_csv": "agent_changes.csv",
                "round_metrics_csv": "round_metrics.csv",
                "social_metrics": "social_metrics.json",
            },
        }
        self._write_json(run_dir / "manifest.json", manifest)
        return manifest

    def build_forecast_prompt(
        self, scenario: FinancialScenario, *, stage: str = "post_social"
    ) -> str:
        if stage not in {"pre_social", "post_social"}:
            raise ValueError("stage must be pre_social or post_social")
        stage_text = (
            "expected_return must use decimal return units: 0.03 means +3%.\n"
            "这是社会互动开始前的第一次预测。当前事件已经公开，但你还没有看到其他投资者的观点。"
            "只能根据只读历史记忆、当前公开事件和自己的 Profile 判断。"
            if stage == "pre_social"
            else
            "这是社会互动结束后的第二次预测。你已经看过讨论区中其他投资者的公开内容，"
            "请把实际看到的帖子、评论和评价作为社会信息，并结合自己的 Profile、历史记忆和当前事件判断。"
        )
        return f"""你正在参加 A 股 S1 社会互动实验。
{stage_text}

场景：{scenario.scenario_id}
当前公开事件：[{scenario.current_event.get('event_id')} | {scenario.current_event.get('event_time')}] {scenario.current_event.get('text', '')}
预测窗口：未来 5 个交易日累计收盘收益
方向区间：R5 < -1.7% 为 down；-1.7% <= R5 <= +1.7% 为 neutral；R5 > +1.7% 为 up。
neutral 只表示预计价格变化很小，不表示没有把握；不确定性请用 confidence 和概率分布表达。
不要使用外部搜索，不要猜测匿名企业的真实身份，不要引用未来价格或评测答案。请只返回一个 JSON object，不要输出 Markdown 或 JSON 之外的文字：
{{"direction":"up|neutral|down","up_probability":0.0,"neutral_probability":0.0,"down_probability":0.0,"expected_return":0.0,"confidence":0.0,"evidence_event_ids":[],"reason":"不超过200字，说明你如何使用事件和社会信息"}}"""

    def build_retry_prompt(
        self, scenario: FinancialScenario, *, stage: str = "post_social"
    ) -> str:
        return (
            "这是最终研究采访，不是社交平台行动回合。你刚才没有返回有效 JSON。"
            "不要点赞、发帖、评论、搜索或调用任何工具；只回答预测 JSON。\n\n"
            + self.build_forecast_prompt(scenario, stage=stage)
        )

    @staticmethod
    def build_belief_snapshot_prompt(*, round_number: int) -> str:
        """Build a private, structured belief measurement prompt.

        The prompt is intentionally independent of the social-action protocol.
        It measures the Agent's current belief without asking it to publish,
        like, or reply.  ``round_number`` is substituted when the live runner
        executes the private interview.
        """
        return f"""Private belief measurement after interaction round {round_number}.
Do not publish a post, write a comment, like, search, or call any tool.  Return exactly one JSON object.
Use decimal return units: 0.03 means +3%.  Do not use future labels, actual prices, or information not visible in the current simulation.
The probabilities must be numbers in [0, 1] and sum to 1.
{{"direction":"up|neutral|down","up_probability":0.0,"neutral_probability":0.0,"down_probability":0.0,"expected_return":0.0,"confidence":0.0,"evidence_event_ids":[],"reason":"brief explanation"}}"""

    def run_sync(self, run_id: str, *, timeout: float = 900.0) -> Dict[str, Any]:
        run_dir = self._run_dir(run_id)
        manifest = self._read_json(run_dir / "manifest.json")
        if manifest.get("status") == "completed":
            return manifest
        simulation_id = manifest["simulation_id"]
        scenario = FinancialDatasetLoader(manifest.get("dataset_path") or self.dataset_path).load(
            scenario_ids=[manifest["scenario_id"]]
        )[0]
        manifest.update({"status": "running", "updated_at": self._now()})
        self._write_json(run_dir / "manifest.json", manifest)
        try:
            SimulationRunner.start_simulation(
                simulation_id,
                platform="reddit",
                max_rounds=int(manifest["social_rounds"]),
            )
            deadline = time.time() + timeout
            while time.time() < deadline:
                state = SimulationRunner.get_run_state(simulation_id)
                if state is None:
                    time.sleep(1)
                    continue
                if state.runner_status == RunnerStatus.INTERACTIVE_READY:
                    break
                if state.runner_status in {RunnerStatus.FAILED, RunnerStatus.STOPPED}:
                    raise RuntimeError(state.error or state.runner_status.value)
                time.sleep(1)
            else:
                raise TimeoutError("S1 Reddit simulation did not become interactive in time")

            simulation_dir = (
                Path(SimulationManager.SIMULATION_DATA_DIR) / simulation_id
            )
            pre_response_path = simulation_dir / "pre_social_interviews.json"
            if not pre_response_path.exists():
                raise RuntimeError(
                    "pre-social interview artifact is missing; social isolation cannot be verified"
                )
            pre_response = self._read_json(pre_response_path)
            if not pre_response.get("success"):
                raise RuntimeError(
                    pre_response.get("error") or "pre-social forecast interview failed"
                )
            self._write_json(
                run_dir / "interview_responses.json",
                {"pre_social": pre_response, "post_social": []},
            )

            self._export_social_actions(run_dir)
            actions = self.get_actions(run_id)
            exposure_edges = self._build_exposure_edges(run_dir, manifest, actions)
            self._write_jsonl(
                run_dir / "agent_round_states.jsonl",
                self._build_agent_round_states(run_dir, manifest, actions),
            )
            profiles = self._read_json(run_dir / "profiles.json")
            pre_predictions = self._parse_interviews(
                scenario,
                profiles,
                pre_response.get("results", {}),
                run_dir,
                stage="pre_social",
                attempt_count=int(pre_response.get("attempt_count", 1)),
            )
            pre_predictions = [
                self._with_run_metadata(prediction, manifest)
                for prediction in pre_predictions
            ]
            interviews = [
                {
                    "agent_id": int(p["user_id"]),
                    "prompt": self.build_forecast_prompt(
                        scenario, stage="post_social"
                    ),
                }
                for p in profiles
            ]
            response = SimulationRunner.interview_agents_batch(
                simulation_id, interviews, platform="reddit", timeout=timeout
            )
            interview_attempts = [{"attempt": 1, "response": response}]
            self._write_json(
                run_dir / "interview_responses.json",
                {
                    "pre_social": pre_response,
                    "post_social": interview_attempts,
                },
            )
            if not response.get("success"):
                raise RuntimeError(response.get("error") or "S1 forecast interview failed")
            post_predictions = self._parse_interviews(
                scenario,
                profiles,
                response.get("result", {}).get("results", {}),
                run_dir,
                stage="post_social",
                attempt_count=1,
            )
            failed_ids = {
                int(prediction["agent_id"])
                for prediction in post_predictions
                if prediction.get("status") != "ok"
            }
            if failed_ids:
                retry_profiles = [
                    profile
                    for profile in profiles
                    if int(profile["user_id"]) in failed_ids
                ]
                retry_response = SimulationRunner.interview_agents_batch(
                    simulation_id,
                    [
                        {
                            "agent_id": int(profile["user_id"]),
                            "prompt": self.build_retry_prompt(
                                scenario, stage="post_social"
                            ),
                        }
                        for profile in retry_profiles
                    ],
                    platform="reddit",
                    timeout=timeout,
                )
                interview_attempts.append({"attempt": 2, "response": retry_response})
                if retry_response.get("success"):
                    retried = self._parse_interviews(
                        scenario,
                        retry_profiles,
                        retry_response.get("result", {}).get("results", {}),
                        run_dir,
                        stage="post_social",
                        attempt_count=2,
                    )
                    retried_by_id = {int(item["agent_id"]): item for item in retried}
                    post_predictions = [
                        retried_by_id.get(int(item["agent_id"]), item)
                        for item in post_predictions
                    ]
            post_predictions = [
                self._with_run_metadata(prediction, manifest)
                for prediction in post_predictions
            ]
            raw_round_snapshot_path = simulation_dir / "round_belief_interviews.jsonl"
            if raw_round_snapshot_path.exists():
                # Keep a run-local copy so the manifest's artifact paths are
                # self-contained and can be archived without the simulation
                # manager directory.
                (run_dir / "round_belief_interviews.jsonl").write_text(
                    raw_round_snapshot_path.read_text(encoding="utf-8"),
                    encoding="utf-8",
                )
            belief_snapshots = self._parse_round_belief_snapshots(
                run_dir,
                manifest,
                scenario,
                profiles,
                pre_predictions,
            )
            self._write_jsonl(run_dir / "belief_snapshots.jsonl", belief_snapshots)
            self._write_json(
                run_dir / "interview_responses.json",
                {
                    "pre_social": pre_response,
                    "post_social": interview_attempts,
                },
            )
            combined_predictions = pre_predictions + post_predictions
            changes = self._build_prediction_changes(
                pre_predictions, post_predictions
            )
            social_metrics, round_metrics = self._build_social_metrics(
                run_dir, pre_predictions, post_predictions, changes
            )
            social_metrics["belief_snapshots"] = {
                "expected_count": int(manifest.get("expected_belief_snapshot_count", 0)),
                "record_count": len(belief_snapshots),
                "valid_count": sum(item.get("status") == "ok" for item in belief_snapshots),
                "round_counts": {
                    str(round_number): sum(
                        int(item.get("round", -1)) == round_number
                        and item.get("status") == "ok"
                        for item in belief_snapshots
                    )
                    for round_number in range(0, int(manifest.get("social_rounds", 0)) + 1)
                },
            }
            social_metrics["social_behavior"]["exposure_edge_count"] = len(exposure_edges)
            social_metrics["artifacts"].update(
                {
                    "belief_snapshots": "belief_snapshots.jsonl",
                    "exposure_edges": "exposure_edges.jsonl",
                }
            )
            self._write_jsonl(
                run_dir / "pre_social_predictions.jsonl", pre_predictions
            )
            self._write_jsonl(
                run_dir / "post_social_predictions.jsonl", post_predictions
            )
            # Compatibility alias for existing post-social analysis scripts.
            self._write_jsonl(run_dir / "predictions.jsonl", post_predictions)
            self._write_jsonl(run_dir / "prediction_changes.jsonl", changes)
            self._write_csv(
                run_dir / "predictions.csv", combined_predictions, self.CSV_FIELDS
            )
            self._write_csv(
                run_dir / "agent_changes.csv",
                changes,
                self.AGENT_CHANGE_FIELDS,
            )
            self._write_csv(
                run_dir / "round_metrics.csv",
                round_metrics,
                self.ROUND_METRIC_FIELDS,
            )
            self._write_json(run_dir / "social_metrics.json", social_metrics)
            self._write_evaluation(run_dir, combined_predictions)
            SimulationRunner.close_simulation_env(simulation_id, timeout=60)
            successful_count = sum(
                prediction.get("status") == "ok"
                for prediction in combined_predictions
            )
            manifest.update(
                {
                    "status": "completed",
                    "current_phase": "completed",
                    "updated_at": self._now(),
                    "completed_prediction_count": len(combined_predictions),
                    "successful_prediction_count": successful_count,
                    "failed_prediction_count": len(combined_predictions) - successful_count,
                    "prediction_count": len(combined_predictions),
                    "pre_social_prediction_count": len(pre_predictions),
                    "post_social_prediction_count": len(post_predictions),
                    "direction_flip_rate": social_metrics.get(
                        "group_change", {}
                    ).get("direction_flip_rate"),
                }
            )
            self._write_json(run_dir / "manifest.json", manifest)
            return manifest
        except Exception as error:
            logger.exception("S1 run failed: %s", run_id)
            manifest.update({"status": "failed", "error": str(error), "updated_at": self._now()})
            self._write_json(run_dir / "manifest.json", manifest)
            try:
                SimulationRunner.close_simulation_env(simulation_id, timeout=20)
            except Exception:
                pass
            raise

    def run_background(self, run_id: str) -> Dict[str, Any]:
        run_dir = self._run_dir(run_id)
        manifest = self._read_json(run_dir / "manifest.json")
        if manifest.get("status") != "prepared":
            raise ValueError("S1 run must be prepared before it can start")
        with self._background_lock:
            existing = self._background_threads.get(run_id)
            if existing is not None and existing.is_alive():
                raise ValueError(f"S1 run is already active: {run_id}")
            manifest.update({"status": "running", "updated_at": self._now()})
            self._write_json(run_dir / "manifest.json", manifest)
            thread = threading.Thread(
                target=self._run_background_worker,
                args=(run_id, self.storage_dir, self.dataset_path),
                name=f"finance-{run_id}",
                daemon=True,
            )
            self._background_threads[run_id] = thread
            thread.start()
        return manifest

    @classmethod
    def _run_background_worker(
        cls,
        run_id: str,
        storage_dir: Path,
        dataset_path: Optional[str | Path],
    ) -> None:
        try:
            cls(storage_dir=storage_dir, dataset_path=dataset_path).run_sync(run_id)
        except Exception:
            logger.exception("S1 background run failed: %s", run_id)
        finally:
            with cls._background_lock:
                current = cls._background_threads.get(run_id)
                if current is threading.current_thread():
                    cls._background_threads.pop(run_id, None)

    def update_settings(
        self,
        run_id: str,
        *,
        social_rounds: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Update interaction controls while a prepared run is still editable."""
        run_dir = self._run_dir(run_id)
        manifest_path = run_dir / "manifest.json"
        manifest = self._read_json(manifest_path)
        if manifest.get("status") != "prepared":
            raise ValueError(
                "S1 interaction settings can only be changed while the run is prepared"
            )

        social_rounds = (
            int(manifest.get("social_rounds", self.DEFAULT_SOCIAL_ROUNDS))
            if social_rounds is None else social_rounds
        )
        if isinstance(social_rounds, bool) or not isinstance(social_rounds, int):
            raise ValueError("social_rounds must be an integer")
        if not self.MIN_SOCIAL_ROUNDS <= social_rounds <= self.MAX_SOCIAL_ROUNDS:
            raise ValueError(
                f"social_rounds must be between {self.MIN_SOCIAL_ROUNDS} and "
                f"{self.MAX_SOCIAL_ROUNDS}"
            )
        total_rounds = social_rounds
        total_simulation_hours = max(
            1,
            math.ceil(
                total_rounds * self.DEFAULT_MINUTES_PER_ROUND / 60
            ),
        )

        config_paths = [
            run_dir / "simulation_config.json",
            Path(SimulationManager.SIMULATION_DATA_DIR)
            / manifest["simulation_id"]
            / "simulation_config.json",
        ]
        for config_path in config_paths:
            config = self._read_json(config_path)
            time_config = config.setdefault("time_config", {})
            time_config["total_simulation_hours"] = total_simulation_hours
            finance_config = config.setdefault("finance_s1", {})
            finance_config["social_rounds"] = social_rounds
            finance_config["total_rounds"] = total_rounds
            self._write_json(config_path, config)

        manifest.update(
            {
                "social_rounds": social_rounds,
                "total_rounds": total_rounds,
                "updated_at": self._now(),
            }
        )
        self._write_json(manifest_path, manifest)
        return manifest

    def get_status(self, run_id: str) -> Dict[str, Any]:
        run_dir = self._run_dir(run_id)
        manifest = self._read_json(run_dir / "manifest.json")
        if manifest.get("status") == "running":
            simulation_id = manifest.get("simulation_id")
            state = SimulationRunner.get_run_state(simulation_id) if simulation_id else None
            simulation_dir = (
                Path(SimulationManager.SIMULATION_DATA_DIR) / str(simulation_id)
            )
            pre_ready = (simulation_dir / "pre_social_interviews.json").exists()
            if not pre_ready:
                manifest["current_phase"] = "pre_social_prediction"
            elif state is not None and state.runner_status == RunnerStatus.INTERACTIVE_READY:
                manifest["current_phase"] = "post_social_prediction"
            else:
                manifest["current_phase"] = "social_interaction"
            if state is not None:
                manifest["current_social_round"] = state.reddit_current_round
                manifest["runner_status"] = state.runner_status.value
        return manifest

    def get_predictions(
        self, run_id: str, *, stage: str = "all"
    ) -> List[Dict[str, Any]]:
        if stage not in {"pre", "post", "all"}:
            raise ValueError("stage must be pre, post, or all")
        run_dir = self._run_dir(run_id)
        paths = []
        if stage in {"pre", "all"}:
            paths.append(run_dir / "pre_social_predictions.jsonl")
        if stage in {"post", "all"}:
            post_path = run_dir / "post_social_predictions.jsonl"
            paths.append(
                post_path if post_path.exists() else run_dir / "predictions.jsonl"
            )
        records = []
        for path in paths:
            if path.exists():
                records.extend(
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )
        return records

    def get_mapping(self, run_id: str) -> Dict[str, Any]:
        """Return the frozen entity/publisher mapping for the workbench."""
        run_dir = self._run_dir(run_id)
        mapping = self._read_json(run_dir / "entity_agent_mapping.json")
        history_path = run_dir / "history_memory.jsonl"
        mapping["history_memory"] = [
            json.loads(line)
            for line in history_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ] if history_path.exists() else []
        current_path = run_dir / "current_event.json"
        mapping["current_event"] = (
            self._read_json(current_path) if current_path.exists() else None
        )
        return mapping

    def get_scenario_seed_document(
        self,
        scenario_id: str,
        *,
        dataset_path: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        """Return safe events for the S1 page's graph-construction step."""
        scenario = FinancialDatasetLoader(dataset_path or self.dataset_path).load(
            scenario_ids=[scenario_id]
        )[0]
        events = []
        for index, event in enumerate(scenario.seed_events):
            events.append(
                {
                    "event_id": event.get("event_id"),
                    "phase": "history",
                    "seed_rank": index + 1,
                    "event_time": event.get("event_time"),
                    "text": event.get("text", ""),
                }
            )
        current = scenario.current_event
        events.append(
            {
                "event_id": current.get("event_id"),
                "phase": "current",
                "seed_rank": None,
                "event_time": current.get("event_time"),
                "text": current.get("text", ""),
            }
        )
        return {
            "scenario_id": scenario.scenario_id,
            "symbol": scenario.symbol,
            "name": scenario.name,
            "prediction_cutoff": scenario.prediction_cutoff,
            "horizon": scenario.horizon,
            "events": events,
        }

    def get_csv_path(self, run_id: str, kind: str) -> Path:
        if kind not in {
            "predictions", "evaluation", "agent_changes", "round_metrics"
        }:
            raise ValueError(
                "kind must be predictions, evaluation, agent_changes, or round_metrics"
            )
        path = self._run_dir(run_id) / f"{kind}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    def get_metrics(self, run_id: str) -> Dict[str, Any]:
        path = self._run_dir(run_id) / "social_metrics.json"
        return self._read_json(path) if path.exists() else {}

    def get_actions(
        self, run_id: str, *, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        run_dir = self._run_dir(run_id)
        # Once offline stance annotation has completed, expose the enriched
        # view by default while keeping the raw OASIS export immutable on disk.
        path = run_dir / "social_actions_annotated.jsonl"
        if not path.exists():
            path = run_dir / "social_actions.jsonl"
        if not path.exists():
            return []
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if limit is not None:
            if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
                raise ValueError("limit must be a positive integer")
            return records[-limit:]
        return records

    def get_agent_round_states(self, run_id: str) -> List[Dict[str, Any]]:
        path = self._run_dir(run_id) / "agent_round_states.jsonl"
        if not path.exists():
            return []
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def get_belief_snapshots(self, run_id: str) -> List[Dict[str, Any]]:
        path = self._run_dir(run_id) / "belief_snapshots.jsonl"
        if not path.exists():
            return []
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def get_exposure_edges(self, run_id: str) -> List[Dict[str, Any]]:
        run_dir = self._run_dir(run_id)
        path = run_dir / "exposure_edges_annotated.jsonl"
        if not path.exists():
            path = run_dir / "exposure_edges.jsonl"
        if not path.exists():
            return []
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(path)
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _parse_interviews(
        scenario: FinancialScenario,
        profiles: Sequence[Dict[str, Any]],
        results: Dict[str, Any],
        run_dir: Path,
        *,
        stage: str,
        attempt_count: int,
    ) -> List[Dict[str, Any]]:
        if stage not in {"pre_social", "post_social"}:
            raise ValueError("stage must be pre_social or post_social")
        parser = C0ExperimentService()
        social_counts = (
            S1ExperimentService._social_counts(run_dir)
            if stage == "post_social" else {}
        )
        records = []
        for profile in profiles:
            item = results.get(str(profile["user_id"]), results.get(profile["user_id"], {}))
            raw = item.get("response", "") if isinstance(item, dict) else item
            raw = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
            forecast = parser.parse_forecast(
                scenario=scenario,
                profile=profile,
                raw_response=raw,
                attempt_count=(
                    int(item.get("attempt_count", attempt_count))
                    if isinstance(item, dict) else attempt_count
                ),
                finish_reason="stop",
            )
            record = forecast.to_dict()
            record.update(
                {
                    "condition": f"S1_{stage.upper()}",
                    "prediction_stage": stage,
                    "social_action_count": social_counts.get(profile["user_id"], {}).get("total", 0),
                    "social_post_count": social_counts.get(profile["user_id"], {}).get("post", 0),
                    "social_comment_count": social_counts.get(profile["user_id"], {}).get("comment", 0),
                    "social_like_count": social_counts.get(profile["user_id"], {}).get("like", 0),
                    "social_dislike_count": social_counts.get(profile["user_id"], {}).get("dislike", 0),
                }
            )
            records.append(record)
        return records

    @staticmethod
    def _classify_content_stance(
        text: Any,
        explicit: Any = None,
    ) -> tuple[str, Optional[float], str]:
        """Return a transparent, non-future-looking stance annotation.

        OASIS normally stores only the content string for a post/comment.  We
        preserve an explicit ``stance``/``sentiment`` value when one exists;
        otherwise this small lexicon is a *heuristic annotation*, never a
        replacement for a human or LLM content coder.  The source and score
        are persisted so downstream analyses can filter or audit it.
        """
        if isinstance(explicit, str) and explicit.strip():
            value = explicit.strip().lower()
            aliases = {
                "bullish": "positive",
                "bearish": "negative",
                "up": "positive",
                "down": "negative",
                "neutral": "neutral",
                "mixed": "mixed",
                "positive": "positive",
                "negative": "negative",
            }
            normalized = aliases.get(value)
            if normalized:
                return normalized, None, "explicit_action_metadata"
        if not isinstance(text, str) or not text.strip():
            return "unknown", None, "unlabeled"
        positive_terms = (
            "利好", "积极", "乐观", "看好", "上行", "上涨", "增长", "放量",
            "落地", "改善", "突破", "正面", "good news", "bullish", "upside",
        )
        negative_terms = (
            "利空", "悲观", "风险", "亏损", "扩大", "回调", "下行", "下跌",
            "担忧", "不确定", "谨慎", "竞争激烈", "negative", "bearish", "downside",
        )
        lowered = text.lower()
        positive_count = sum(lowered.count(term.lower()) for term in positive_terms)
        negative_count = sum(lowered.count(term.lower()) for term in negative_terms)
        total = positive_count + negative_count
        if total == 0:
            return "unknown", None, "unlabeled"
        score = (positive_count - negative_count) / total
        if positive_count and negative_count:
            label = "mixed" if abs(score) < 0.6 else (
                "positive" if score > 0 else "negative"
            )
        elif positive_count:
            label = "positive"
        else:
            label = "negative"
        return label, round(score, 6), "lexicon_v1"

    @staticmethod
    def _snapshot_record_from_forecast(
        forecast: Dict[str, Any],
        manifest: Dict[str, Any],
        *,
        round_number: int,
        snapshot_type: str,
        snapshot_source: str,
    ) -> Dict[str, Any]:
        record = {
            **S1ExperimentService._run_metadata(manifest),
            **forecast,
            "scenario_id": forecast.get("scenario_id") or manifest.get("scenario_id"),
            "round": round_number,
            "snapshot_type": snapshot_type,
            "snapshot_source": snapshot_source,
        }
        # ``condition``/``prediction_stage`` are useful when loading this JSONL
        # beside predictions.csv, but are deliberately not part of the compact
        # snapshot CSV field list.
        record["condition"] = "S1_ROUND_SNAPSHOT"
        record["prediction_stage"] = f"round_{round_number}"
        return record

    def _parse_round_belief_snapshots(
        self,
        run_dir: Path,
        manifest: Dict[str, Any],
        scenario: FinancialScenario,
        profiles: Sequence[Dict[str, Any]],
        pre_predictions: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Parse round-0 and private post-round interviews into one JSONL.

        A row is emitted for every investor and every expected round, including
        an explicit ``status=missing`` row when a live snapshot was unavailable.
        This makes missingness measurable instead of silently shrinking the
        sample used by information-flow analysis.
        """
        rows: List[Dict[str, Any]] = []
        pre_by_id = {int(item["agent_id"]): item for item in pre_predictions}
        for profile in profiles:
            agent_id = int(profile["user_id"])
            forecast = pre_by_id.get(agent_id)
            if forecast is None:
                forecast = {
                    "scenario_id": manifest.get("scenario_id"),
                    "agent_id": agent_id,
                    "agent_role": profile.get("role_id", "investor"),
                    "agent_role_category": profile.get("role_category", ""),
                    "agent_role_label": profile.get("role_label", ""),
                    "as_of": scenario.prediction_cutoff,
                    "horizon": scenario.horizon,
                    "status": "missing",
                    "error": "pre-social prediction is missing",
                    "evidence_event_ids": [],
                }
            rows.append(
                self._snapshot_record_from_forecast(
                    forecast,
                    manifest,
                    round_number=0,
                    snapshot_type="pre_social",
                    snapshot_source="pre_social_interview",
                )
            )

        raw_path = (
            Path(SimulationManager.SIMULATION_DATA_DIR)
            / manifest["simulation_id"]
            / "round_belief_interviews.jsonl"
        )
        payloads: Dict[int, Dict[str, Any]] = {}
        if raw_path.exists():
            for line in raw_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                    payloads[int(payload["round"])] = payload
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    logger.warning("Ignoring malformed belief snapshot payload: %s", line[:200])

        parser = C0ExperimentService()
        for round_number in range(1, int(manifest.get("social_rounds", 0)) + 1):
            payload = payloads.get(round_number, {})
            results = payload.get("results") or {}
            attempt_count = int(payload.get("attempt_count", 1) or 1)
            for profile in profiles:
                agent_id = int(profile["user_id"])
                item = results.get(str(agent_id), results.get(agent_id, {}))
                raw = item.get("response", "") if isinstance(item, dict) else item
                raw = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
                if raw.strip():
                    forecast = parser.parse_forecast(
                        scenario=scenario,
                        profile=profile,
                        raw_response=raw,
                        attempt_count=(
                            int(item.get("attempt_count", attempt_count))
                            if isinstance(item, dict) else attempt_count
                        ),
                        finish_reason="stop",
                    ).to_dict()
                else:
                    forecast = {
                        "scenario_id": scenario.scenario_id,
                        "agent_id": agent_id,
                        "agent_role": profile.get("role_id", "investor"),
                        "agent_role_category": profile.get("role_category", ""),
                        "agent_role_label": profile.get("role_label", ""),
                        "as_of": scenario.prediction_cutoff,
                        "horizon": scenario.horizon,
                        "status": "missing",
                        "error": payload.get("error") or "round belief snapshot is missing",
                        "attempt_count": attempt_count,
                        "evidence_event_ids": [],
                        "raw_response": raw,
                    }
                rows.append(
                    self._snapshot_record_from_forecast(
                        forecast,
                        manifest,
                        round_number=round_number,
                        snapshot_type="post_round",
                        snapshot_source="private_round_interview",
                    )
                )
        return rows

    def _build_exposure_edges(
        self,
        run_dir: Path,
        manifest: Dict[str, Any],
        actions: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Normalize feed visibility and direct interactions into edge rows."""
        post_owners: Dict[Any, int] = {}
        comment_owners: Dict[Any, int] = {}
        content: Dict[tuple[str, Any], Dict[str, Any]] = {}
        for action in actions:
            agent_id = action.get("agent_id")
            action_type = str(action.get("action_type", "")).lower()
            args = action.get("action_args") or {}
            if action_type == "create_post" and action.get("post_id") is not None:
                post_id = action.get("post_id")
                post_owners[post_id] = int(agent_id)
                text = args.get("content", "")
                if action.get("agent_class") == "source":
                    label, score, source = "informational", 0.0, "source_event"
                else:
                    label, score, source = self._classify_content_stance(
                        text, args.get("stance", args.get("sentiment"))
                    )
                content[("post", post_id)] = {
                    "author_agent_id": int(agent_id),
                    "content_text": text,
                    "content_stance": label,
                    "stance_score": score,
                    "stance_source": source,
                }
            elif action_type == "create_comment" and action.get("comment_id") is not None:
                comment_id = action.get("comment_id")
                comment_owners[comment_id] = int(agent_id)
                text = args.get("content", "")
                label, score, source = self._classify_content_stance(
                    text, args.get("stance", args.get("sentiment"))
                )
                content[("comment", comment_id)] = {
                    "author_agent_id": int(agent_id),
                    "content_text": text,
                    "content_stance": label,
                    "stance_score": score,
                    "stance_source": source,
                }

        # Refresh/trend payloads contain the comments visible inside each
        # post.  Preserve those content objects even when the trace has no
        # standalone ``create_comment`` relation for them.
        for action in actions:
            visible_posts = (action.get("action_args") or {}).get("posts")
            if not isinstance(visible_posts, list):
                continue
            for post in visible_posts:
                if not isinstance(post, dict):
                    continue
                for comment in post.get("comments") or []:
                    if not isinstance(comment, dict) or comment.get("comment_id") is None:
                        continue
                    comment_id = comment.get("comment_id")
                    if ("comment", comment_id) in content:
                        continue
                    author_id = comment.get("user_id", comment.get("author_id"))
                    text = comment.get("content", "")
                    if author_id is not None and int(author_id) >= self.SOURCE_AGENT_START:
                        label, score, source = "informational", 0.0, "source_event"
                    else:
                        label, score, source = self._classify_content_stance(text)
                    content[("comment", comment_id)] = {
                        "author_agent_id": int(author_id) if author_id is not None else None,
                        "content_text": text,
                        "content_stance": label,
                        "stance_score": score,
                        "stance_source": source,
                    }

        edges: List[Dict[str, Any]] = []
        first_seen: Dict[tuple[int, str, Any], int] = {}

        def add_edge(
            action: Dict[str, Any],
            content_type: str,
            content_id: Any,
            exposure_type: str,
            interacted: bool,
        ) -> None:
            if content_id is None or action.get("agent_class") != "investor":
                return
            key = (int(action["agent_id"]), content_type, content_id)
            round_number = int(action.get("round", 0) or 0)
            first_seen[key] = min(round_number, first_seen.get(key, round_number))
            metadata = content.get((content_type, content_id), {})
            interaction_target = (
                action.get("target_post_id")
                if content_type == "post" else action.get("target_comment_id")
            )
            trace_id = action.get("trace_id")
            exposure_id = f"{trace_id}:{content_type}:{content_id}"
            edges.append(
                {
                    **self._run_metadata(manifest),
                    "scenario_id": manifest.get("scenario_id"),
                    "exposure_id": exposure_id,
                    "trace_id": trace_id,
                    "round": round_number,
                    "timestamp": action.get("timestamp"),
                    "viewer_agent_id": int(action["agent_id"]),
                    "content_type": content_type,
                    "content_id": content_id,
                    "author_agent_id": metadata.get("author_agent_id"),
                    "content_text": metadata.get("content_text", ""),
                    "content_stance": metadata.get("content_stance", "unknown"),
                    "stance_score": metadata.get("stance_score"),
                    "stance_source": metadata.get("stance_source", "unlabeled"),
                    "exposure_type": exposure_type,
                    "action_type": action.get("action_type"),
                    "interacted": bool(interacted),
                    "interaction_target_id": interaction_target,
                    "first_seen_round": round_number,
                }
            )

        for action in actions:
            if action.get("agent_class") != "investor":
                continue
            visible_ids = action.get("visible_post_ids") or []
            for post_id in visible_ids:
                add_edge(action, "post", post_id, "feed_visible", False)
            visible_posts = (action.get("action_args") or {}).get("posts")
            if isinstance(visible_posts, list):
                for post in visible_posts:
                    if not isinstance(post, dict):
                        continue
                    for comment in post.get("comments") or []:
                        if isinstance(comment, dict):
                            add_edge(
                                action,
                                "comment",
                                comment.get("comment_id"),
                                "feed_visible",
                                False,
                            )
            action_type = str(action.get("action_type", "")).lower()
            if action_type in {"like_post", "dislike_post", "create_comment"}:
                add_edge(
                    action,
                    "post",
                    action.get("target_post_id"),
                    "direct_action",
                    True,
                )
            if action_type in {"like_comment", "dislike_comment"}:
                add_edge(
                    action,
                    "comment",
                    action.get("target_comment_id"),
                    "direct_action",
                    True,
                )

        for edge in edges:
            edge["first_seen_round"] = first_seen.get(
                (
                    int(edge["viewer_agent_id"]),
                    str(edge["content_type"]),
                    edge["content_id"],
                ),
                edge["round"],
            )
        self._write_jsonl(run_dir / "exposure_edges.jsonl", edges)
        return edges

    @staticmethod
    def _social_counts(run_dir: Path) -> Dict[int, Dict[str, int]]:
        path = run_dir / "social_actions.jsonl"
        if not path.exists():
            return {}
        counts: Dict[int, Dict[str, int]] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            agent_id = int(record.get("agent_id", -1))
            if not 0 <= agent_id < S1ExperimentService.SOURCE_AGENT_START:
                continue
            action = str(record.get("action_type", "")).lower()
            item = counts.setdefault(
                agent_id,
                {"total": 0, "post": 0, "comment": 0, "like": 0, "dislike": 0},
            )
            item["total"] += 1
            if action == "create_post":
                item["post"] += 1
            elif action == "create_comment":
                item["comment"] += 1
            elif action in {"like_post", "like_comment"}:
                item["like"] += 1
            elif action in {"dislike_post", "dislike_comment"}:
                item["dislike"] += 1
        return counts

    def _build_agent_round_states(
        self,
        run_dir: Path,
        manifest: Dict[str, Any],
        actions: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Aggregate observed exposure/actions per investor and interaction round.

        This artifact intentionally contains no fabricated per-round beliefs. It
        records only what the OASIS trace makes observable, which can be joined
        with the actual pre/post forecasts for information-flow analysis.
        """
        profiles_path = run_dir / "profiles.json"
        profiles = self._read_json(profiles_path) if profiles_path.exists() else []
        investor_ids = [int(profile["user_id"]) for profile in profiles]
        rows: List[Dict[str, Any]] = []
        for round_number in range(1, int(manifest.get("social_rounds", 0)) + 1):
            for agent_id in investor_ids:
                records = [
                    item
                    for item in actions
                    if int(item.get("agent_id", -1)) == agent_id
                    and int(item.get("round", 0)) == round_number
                ]
                action_types = [
                    str(item.get("action_type", "")).lower() for item in records
                ]
                visible_post_ids = {
                    post_id
                    for item in records
                    for post_id in (item.get("visible_post_ids") or [])
                    if post_id is not None
                }
                visible_agent_ids = {
                    target_id
                    for item in records
                    for target_id in (item.get("visible_agent_ids") or [])
                    if target_id is not None
                }
                target_agent_ids = {
                    target_id
                    for item in records
                    for target_id in [item.get("target_agent_id")]
                    if target_id is not None
                }
                rows.append(
                    {
                        **self._run_metadata(manifest),
                        "scenario_id": manifest.get("scenario_id"),
                        "round": round_number,
                        "agent_id": agent_id,
                        "action_count": len(records),
                        "post_count": action_types.count("create_post"),
                        "comment_count": action_types.count("create_comment"),
                        "like_count": sum(
                            action in {"like_post", "like_comment"}
                            for action in action_types
                        ),
                        "dislike_count": sum(
                            action in {"dislike_post", "dislike_comment"}
                            for action in action_types
                        ),
                        "refresh_count": action_types.count("refresh"),
                        "visible_post_ids": sorted(visible_post_ids, key=str),
                        "visible_agent_ids": sorted(visible_agent_ids, key=str),
                        "target_agent_ids": sorted(target_agent_ids, key=str),
                        "exposure_count": len(visible_post_ids),
                        "state_source": "oasis_trace_actions",
                    }
                )
        return rows

    def _export_social_actions(self, run_dir: Path) -> List[Dict[str, Any]]:
        """Export the authoritative OASIS trace with an inferred social round."""
        manifest = self._read_json(run_dir / "manifest.json")
        simulation_dir = (
            Path(SimulationManager.SIMULATION_DATA_DIR) / manifest["simulation_id"]
        )
        db_path = simulation_dir / "reddit_simulation.db"
        if not db_path.exists():
            self._write_jsonl(run_dir / "social_actions.jsonl", [])
            return []

        intervals: List[tuple[int, datetime, datetime]] = []
        trace_ranges: List[tuple[int, int, int]] = []
        round_starts: Dict[int, datetime] = {}
        action_log = simulation_dir / "reddit" / "actions.jsonl"
        if action_log.exists():
            for line in action_log.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                event_type = record.get("event_type")
                if event_type not in {"round_start", "round_end"}:
                    continue
                stamp = datetime.fromisoformat(str(record["timestamp"]))
                round_number = int(record.get("round", 0))
                if event_type == "round_start":
                    round_starts[round_number] = stamp
                elif round_number in round_starts:
                    intervals.append((round_number, round_starts[round_number], stamp))
                if event_type == "round_end":
                    trace_start = record.get("trace_start_rowid")
                    trace_end = record.get("trace_end_rowid")
                    if isinstance(trace_start, int) and isinstance(trace_end, int):
                        trace_ranges.append(
                            (round_number, trace_start, trace_end)
                        )

        def infer_round(
            stamp: datetime, agent_id: int, trace_id: int
        ) -> tuple[int, str]:
            for round_number, start_rowid, end_rowid in trace_ranges:
                if start_rowid < trace_id <= end_rowid:
                    return round_number, "oasis_trace_rowid_range"
            if agent_id >= self.SOURCE_AGENT_START:
                return 0, "source_initialization"
            for round_number, start, end in intervals:
                if start <= stamp <= end:
                    return round_number, "oasis_trace_time_window"
            return 0, "unattributed"

        records = []
        with sqlite3.connect(db_path) as connection:
            rows = connection.execute(
                """
                SELECT rowid, user_id, created_at, action, info
                FROM trace
                WHERE action NOT IN ('sign_up', 'interview')
                ORDER BY created_at, rowid
                """
            ).fetchall()
        for trace_id, agent_id, created_at, action, info in rows:
            try:
                action_args = json.loads(info) if info else {}
            except json.JSONDecodeError:
                action_args = {"raw_info": info}
            stamp = datetime.fromisoformat(str(created_at))
            round_number, round_source = infer_round(
                stamp, int(agent_id), int(trace_id)
            )
            def optional_arg(*keys: str) -> Any:
                for key in keys:
                    value = action_args.get(key)
                    if value is not None and value != "":
                        return value
                return None

            action_type = str(action).lower()
            post_id = optional_arg("post_id", "target_post_id")
            comment_id = optional_arg("comment_id", "target_comment_id")
            target_post_id = optional_arg("target_post_id")
            target_comment_id = optional_arg("target_comment_id")
            if action_type not in {"create_post"} and target_post_id is None:
                target_post_id = post_id
            if action_type not in {"create_comment"} and target_comment_id is None:
                target_comment_id = comment_id
            if action_type == "create_comment" and target_comment_id is None:
                target_comment_id = optional_arg("parent_comment_id", "parent_id")
            visible_posts = action_args.get("posts")
            if not isinstance(visible_posts, list):
                visible_posts = []
            visible_post_ids = [
                item.get("post_id")
                for item in visible_posts
                if isinstance(item, dict) and item.get("post_id") is not None
            ]
            visible_agent_ids = [
                item.get("user_id")
                for item in visible_posts
                if isinstance(item, dict) and item.get("user_id") is not None
            ]

            record = {
                "trace_id": int(trace_id),
                **self._run_metadata(manifest),
                "scenario_id": manifest["scenario_id"],
                "round": round_number,
                "round_source": round_source,
                "timestamp": str(created_at),
                "platform": "reddit",
                "agent_id": int(agent_id),
                "agent_class": (
                    "investor"
                    if int(agent_id) < self.SOURCE_AGENT_START else "source"
                ),
                "action_type": action_type,
                "action_args": action_args,
                # These are populated only when OASIS explicitly records them;
                # an absent relation remains null instead of being inferred.
                "source_agent_id": optional_arg("source_agent_id", "author_id"),
                "target_agent_id": optional_arg("target_agent_id", "target_user_id", "parent_author_id"),
                "post_id": post_id,
                "comment_id": comment_id,
                "parent_comment_id": optional_arg("parent_comment_id", "parent_id"),
                "target_post_id": target_post_id,
                "target_comment_id": target_comment_id,
                "visibility": optional_arg("visibility"),
                "stance": optional_arg("stance", "sentiment", "direction"),
                "supports": optional_arg("supports"),
                "challenges": optional_arg("challenges"),
                "adopts": optional_arg("adopts"),
                "evidence_event_ids": optional_arg("evidence_event_ids", "event_ids"),
                "content_stance": None,
                "stance_score": None,
                "stance_source": "unlabeled",
                "visible_post_ids": visible_post_ids,
                "visible_agent_ids": visible_agent_ids,
                "exposure_count": len(visible_post_ids),
            }
            records.append(record)
        post_owners = {
            item["post_id"]: item["agent_id"]
            for item in records
            if item.get("action_type") == "create_post"
            and item.get("post_id") is not None
        }
        comment_owners = {
            item["comment_id"]: item["agent_id"]
            for item in records
            if item.get("action_type") == "create_comment"
            and item.get("comment_id") is not None
        }
        for item in records:
            if item.get("target_agent_id") is not None:
                continue
            if item.get("target_comment_id") in comment_owners:
                item["target_agent_id"] = comment_owners[item["target_comment_id"]]
            elif item.get("target_post_id") in post_owners:
                item["target_agent_id"] = post_owners[item["target_post_id"]]
        for item in records:
            action_type = str(item.get("action_type", "")).lower()
            if action_type not in {"create_post", "create_comment"}:
                if action_type in {"like_post", "like_comment"}:
                    item["interaction_stance"] = "supports_target"
                elif action_type in {"dislike_post", "dislike_comment"}:
                    item["interaction_stance"] = "challenges_target"
                else:
                    item["interaction_stance"] = None
                continue
            args = item.get("action_args") or {}
            if item.get("agent_class") == "source":
                label, score, source = "informational", 0.0, "source_event"
            else:
                label, score, source = self._classify_content_stance(
                    args.get("content", ""),
                    args.get("stance", args.get("sentiment", args.get("direction"))),
                )
            item["content_stance"] = label
            item["stance_score"] = score
            item["stance_source"] = source
            item["interaction_stance"] = "author_position"
        self._write_jsonl(run_dir / "social_actions.jsonl", records)
        return records

    @staticmethod
    def _probability_vector(record: Dict[str, Any]) -> Optional[List[float]]:
        values = [
            record.get("up_probability"),
            record.get("neutral_probability"),
            record.get("down_probability"),
        ]
        if not all(isinstance(value, (int, float)) for value in values):
            return None
        total = sum(max(0.0, float(value)) for value in values)
        if total <= 0:
            return None
        return [max(0.0, float(value)) / total for value in values]

    @staticmethod
    def _js_divergence(
        left: Optional[Sequence[float]], right: Optional[Sequence[float]]
    ) -> Optional[float]:
        if left is None or right is None or len(left) != len(right):
            return None
        midpoint = [(float(a) + float(b)) / 2 for a, b in zip(left, right)]

        def kl(values: Sequence[float], reference: Sequence[float]) -> float:
            return sum(
                value * math.log(value / ref, 2)
                for value, ref in zip(values, reference)
                if value > 0 and ref > 0
            )

        return (kl(left, midpoint) + kl(right, midpoint)) / 2

    @staticmethod
    def _numeric_delta(before: Any, after: Any) -> Optional[float]:
        if not isinstance(before, (int, float)) or not isinstance(after, (int, float)):
            return None
        if not math.isfinite(float(before)) or not math.isfinite(float(after)):
            return None
        return float(after) - float(before)

    def _build_prediction_changes(
        self,
        pre_predictions: Sequence[Dict[str, Any]],
        post_predictions: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        pre_by_id = {int(item["agent_id"]): item for item in pre_predictions}
        post_by_id = {int(item["agent_id"]): item for item in post_predictions}
        changes = []
        for agent_id in sorted(pre_by_id.keys() | post_by_id.keys()):
            before = pre_by_id.get(agent_id, {})
            after = post_by_id.get(agent_id, {})
            paired_ok = before.get("status") == "ok" and after.get("status") == "ok"
            before_evidence = list(before.get("evidence_event_ids") or [])
            after_evidence = list(after.get("evidence_event_ids") or [])
            metadata_source = after if after else before
            changes.append(
                {
                    **{
                        key: metadata_source.get(key)
                        for key in self.RUN_METADATA_FIELDS
                    },
                    "scenario_id": before.get("scenario_id") or after.get("scenario_id"),
                    "agent_id": agent_id,
                    "agent_role_label": before.get("agent_role_label") or after.get("agent_role_label"),
                    "pre_direction": before.get("direction"),
                    "post_direction": after.get("direction"),
                    "direction_changed": (
                        before.get("direction") != after.get("direction")
                        if paired_ok else None
                    ),
                    "distribution_js_divergence": self._js_divergence(
                        self._probability_vector(before),
                        self._probability_vector(after),
                    ) if paired_ok else None,
                    "pre_expected_return": before.get("expected_return"),
                    "post_expected_return": after.get("expected_return"),
                    "expected_return_delta": self._numeric_delta(
                        before.get("expected_return"), after.get("expected_return")
                    ) if paired_ok else None,
                    "pre_confidence": before.get("confidence"),
                    "post_confidence": after.get("confidence"),
                    "confidence_delta": self._numeric_delta(
                        before.get("confidence"), after.get("confidence")
                    ) if paired_ok else None,
                    "social_action_count": after.get("social_action_count", 0),
                    "social_post_count": after.get("social_post_count", 0),
                    "social_comment_count": after.get("social_comment_count", 0),
                    "social_like_count": self._social_counts_for_agent(
                        after, "like"
                    ),
                    "social_dislike_count": self._social_counts_for_agent(
                        after, "dislike"
                    ),
                    "pre_evidence_event_ids": json.dumps(before_evidence, ensure_ascii=False),
                    "post_evidence_event_ids": json.dumps(after_evidence, ensure_ascii=False),
                    "evidence_changed": before_evidence != after_evidence if paired_ok else None,
                    "pair_status": "ok" if paired_ok else "incomplete",
                }
            )
        return changes

    @staticmethod
    def _social_counts_for_agent(record: Dict[str, Any], kind: str) -> int:
        return int(record.get(f"social_{kind}_count", 0) or 0)

    @staticmethod
    def _mean(values: Sequence[Any]) -> Optional[float]:
        numeric = [
            float(value)
            for value in values
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        ]
        return sum(numeric) / len(numeric) if numeric else None

    def _prediction_stage_metrics(
        self, predictions: Sequence[Dict[str, Any]]
    ) -> Dict[str, Any]:
        valid = [item for item in predictions if item.get("status") == "ok"]
        direction_counts = {
            direction: sum(item.get("direction") == direction for item in valid)
            for direction in ("up", "neutral", "down")
        }
        total = len(valid)
        direction_proportions = {
            key: value / total if total else 0.0
            for key, value in direction_counts.items()
        }
        nonzero = [value for value in direction_proportions.values() if value > 0]
        entropy = -sum(value * math.log(value, 2) for value in nonzero)
        vectors = [
            vector
            for vector in (self._probability_vector(item) for item in valid)
            if vector is not None
        ]
        pairwise_js = []
        for index, left in enumerate(vectors):
            for right in vectors[index + 1:]:
                divergence = self._js_divergence(left, right)
                if divergence is not None:
                    pairwise_js.append(divergence)
        return {
            "prediction_count": len(predictions),
            "valid_prediction_count": total,
            "failed_prediction_count": len(predictions) - total,
            "direction_counts": direction_counts,
            "direction_proportions": direction_proportions,
            "mean_probabilities": {
                "up": self._mean([item.get("up_probability") for item in valid]),
                "neutral": self._mean([
                    item.get("neutral_probability") for item in valid
                ]),
                "down": self._mean([item.get("down_probability") for item in valid]),
            },
            "mean_expected_return": self._mean([
                item.get("expected_return") for item in valid
            ]),
            "mean_confidence": self._mean([
                item.get("confidence") for item in valid
            ]),
            "direction_entropy_bits": entropy,
            "consensus_rate": max(direction_proportions.values(), default=0.0),
            "mean_pairwise_js_divergence": self._mean(pairwise_js),
        }

    def _build_social_metrics(
        self,
        run_dir: Path,
        pre_predictions: Sequence[Dict[str, Any]],
        post_predictions: Sequence[Dict[str, Any]],
        changes: Sequence[Dict[str, Any]],
    ) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
        manifest = self._read_json(run_dir / "manifest.json")
        actions = self.get_actions(manifest["run_id"])
        investor_actions = [
            item for item in actions if item.get("agent_class") == "investor"
        ]
        round_metrics = []
        for round_number in range(1, int(manifest["social_rounds"]) + 1):
            records = [
                item for item in investor_actions
                if int(item.get("round", 0)) == round_number
            ]
            action_types = [str(item.get("action_type", "")).lower() for item in records]
            post_count = action_types.count("create_post")
            comment_count = action_types.count("create_comment")
            like_count = sum(action in {"like_post", "like_comment"} for action in action_types)
            dislike_count = sum(action in {"dislike_post", "dislike_comment"} for action in action_types)
            refresh_count = action_types.count("refresh")
            classified = post_count + comment_count + like_count + dislike_count + refresh_count
            post_ids = {
                item.get("post_id")
                for item in records
                if item.get("post_id") is not None
            }
            target_agent_ids = {
                item.get("target_agent_id")
                for item in records
                if item.get("target_agent_id") is not None
            }
            round_metrics.append(
                {
                    **self._run_metadata(manifest),
                    "scenario_id": manifest["scenario_id"],
                    "round": round_number,
                    "action_count": len(records),
                    "active_agent_count": len({item["agent_id"] for item in records}),
                    "post_count": post_count,
                    "comment_count": comment_count,
                    "like_count": like_count,
                    "dislike_count": dislike_count,
                    "refresh_count": refresh_count,
                    "other_action_count": len(records) - classified,
                    "unique_post_id_count": len(post_ids),
                    "unique_target_agent_count": len(target_agent_ids),
                    "exposure_count": sum(
                        int(item.get("exposure_count", 0) or 0) for item in records
                    ),
                }
            )

        paired = [item for item in changes if item.get("pair_status") == "ok"]
        action_type_counts: Dict[str, int] = {}
        for action in investor_actions:
            action_type = str(action.get("action_type", "")).lower()
            action_type_counts[action_type] = action_type_counts.get(action_type, 0) + 1
        pre_metrics = self._prediction_stage_metrics(pre_predictions)
        post_metrics = self._prediction_stage_metrics(post_predictions)
        metrics = {
            **self._run_metadata(manifest),
            "scenario_id": manifest["scenario_id"],
            "generated_at": self._now(),
            "flow": [
                "history_memory",
                "current_event",
                "pre_social_prediction",
                "social_interaction",
                "round_belief_snapshots",
                "post_social_prediction",
            ],
            "social_rounds": manifest["social_rounds"],
            "pre_social": pre_metrics,
            "post_social": post_metrics,
            "group_change": {
                "paired_agent_count": len(paired),
                "direction_flip_count": sum(
                    item.get("direction_changed") is True for item in paired
                ),
                "direction_flip_rate": (
                    sum(item.get("direction_changed") is True for item in paired)
                    / len(paired) if paired else None
                ),
                "mean_distribution_js_divergence": self._mean([
                    item.get("distribution_js_divergence") for item in paired
                ]),
                "mean_expected_return_delta": self._mean([
                    item.get("expected_return_delta") for item in paired
                ]),
                "mean_confidence_delta": self._mean([
                    item.get("confidence_delta") for item in paired
                ]),
                "mean_probability_deltas": {
                    direction: self._numeric_delta(
                        pre_metrics["mean_probabilities"].get(direction),
                        post_metrics["mean_probabilities"].get(direction),
                    )
                    for direction in ("up", "neutral", "down")
                },
                "consensus_rate_delta": self._numeric_delta(
                    pre_metrics.get("consensus_rate"),
                    post_metrics.get("consensus_rate"),
                ),
                "direction_entropy_delta": self._numeric_delta(
                    pre_metrics.get("direction_entropy_bits"),
                    post_metrics.get("direction_entropy_bits"),
                ),
                "polarization_delta": self._numeric_delta(
                    pre_metrics.get("mean_pairwise_js_divergence"),
                    post_metrics.get("mean_pairwise_js_divergence"),
                ),
            },
            "social_behavior": {
                "trace_record_count": len(actions),
                "investor_action_count": len(investor_actions),
                "active_investor_count": len({
                    item["agent_id"] for item in investor_actions
                }),
                "exposure_count": sum(
                    int(item.get("exposure_count", 0) or 0)
                    for item in investor_actions
                ),
                "observed_target_relation_count": sum(
                    item.get("target_agent_id") is not None
                    for item in investor_actions
                ),
                "action_type_counts": action_type_counts,
            },
            "artifacts": {
                "complete_trace": "social_actions.jsonl",
                "agent_round_states": "agent_round_states.jsonl",
                "round_metrics": "round_metrics.csv",
                "agent_changes": "agent_changes.csv",
            },
        }
        return metrics, round_metrics

    def _write_evaluation(self, run_dir: Path, predictions: Sequence[Dict[str, Any]]) -> None:
        evaluator = FinancialOutcomeEvaluator()
        outcomes: Dict[str, Dict[str, Any]] = {}
        rows = []
        for prediction in predictions:
            sid = str(prediction.get("scenario_id", ""))
            outcomes.setdefault(sid, evaluator.get_outcome(sid))
            outcome = outcomes[sid]
            expected = prediction.get("expected_return")
            error = ""
            if isinstance(expected, (int, float)) and math.isfinite(float(expected)):
                error = float(expected) - float(outcome["five_day_close_return"])
            rows.append(
                {
                    **prediction,
                    "actual_astock_label": outcome["astock_label"],
                    "actual_astock_direction": outcome["astock_direction"],
                    "actual_astock_change_return": outcome["astock_change_return"],
                    "five_day_neutral_threshold": outcome["five_day_neutral_threshold"],
                    "five_day_direction_definition": outcome["five_day_direction_definition"],
                    "actual_five_day_close_direction": outcome["five_day_close_direction"],
                    "actual_five_day_close_return": outcome["five_day_close_return"],
                    "five_day_direction_correct": prediction.get("direction") == outcome["five_day_close_direction"] if prediction.get("status") == "ok" else "",
                    "five_day_return_error": error,
                }
            )
        self._write_csv(run_dir / "evaluation.csv", rows, self.EVALUATION_FIELDS)
