"""Offline LLM annotation for social-content stance.

The live S1 simulation must remain unchanged by content coding.  This module
reads the completed OASIS trace after a run, deduplicates posts/comments, and
uses a separately configured OpenAI-compatible model to assign an auditable
stance label.  It writes a new annotated view instead of replacing the raw
trace or the lexical baseline produced during export.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import time
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


logger = get_logger("mirofish.finance.stance_annotator")


class StanceAnnotationError(ValueError):
    """Raised when an annotator response cannot satisfy the JSON contract."""


class OfflineStanceAnnotator:
    """Annotate a completed S1 run with an independent LLM.

    ``client`` is injectable for tests and dry-run tooling. In research runs,
    set STANCE_LLM_API_KEY/BASE_URL/MODEL_NAME explicitly. A primary-model
    fallback is available only when explicitly enabled for local debugging.
    """

    RUN_ID_PATTERN = re.compile(r"s1_reddit_[A-Za-z0-9_-]{6,64}")
    PROMPT_VERSION = "finance_stance_annotation_v1"
    MAX_ATTEMPTS = 2
    MAX_TOKENS = 1200
    RESPONSE_FORMAT = {"type": "json_object"}
    TEMPERATURE = 0.0
    CSV_FIELDS = (
        "run_id",
        "scenario_id",
        "content_type",
        "content_id",
        "author_agent_id",
        "author_class",
        "round",
        "parent_content_id",
        "content_text",
        "content_hash",
        "stance",
        "target",
        "event_valence",
        "stance_score",
        "confidence",
        "supports_content_id",
        "challenges_content_id",
        "reason",
        "annotator_model",
        "annotator_base_url",
        "config_source",
        "prompt_version",
        "attempt_count",
        "finish_reason",
        "response_content_length",
        "reasoning_content_present",
        "raw_response",
        "status",
        "error",
        "baseline_stance",
        "baseline_stance_score",
        "baseline_stance_source",
        "annotated_at",
    )
    STANCE_LABELS = {"positive", "negative", "mixed", "neutral", "uncertain"}
    TARGET_LABELS = {"stock", "event", "other_agent", "none", "uncertain"}

    def __init__(
        self,
        *,
        storage_dir: Optional[str | Path] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        client: Any = None,
        allow_primary_fallback: bool = False,
    ) -> None:
        self.storage_dir = Path(
            storage_dir or getattr(Config, "FINANCE_ADAPTER_DATA_DIR")
        ).resolve()
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        stance_key = getattr(Config, "STANCE_LLM_API_KEY", None)
        stance_url = getattr(Config, "STANCE_LLM_BASE_URL", None)
        stance_model = getattr(Config, "STANCE_LLM_MODEL_NAME", None)
        stance_configured = bool(stance_key and stance_url and stance_model)
        if allow_primary_fallback and not stance_configured:
            self.api_key = api_key or getattr(Config, "LLM_API_KEY", None)
            self.base_url = base_url or getattr(Config, "LLM_BASE_URL")
            self.model = model or getattr(Config, "LLM_MODEL_NAME")
            self.config_source = "primary_llm_fallback"
        else:
            self.api_key = api_key or stance_key
            self.base_url = base_url or stance_url
            self.model = model or stance_model
            self.config_source = "stance_llm_env" if stance_configured or api_key else "missing_independent_config"
        self.client = client

    def _get_client(self) -> Any:
        if self.client is not None:
            return self.client
        if not self.api_key or not self.base_url or not self.model:
            raise RuntimeError(
                "独立内容标注模型未配置，请设置 STANCE_LLM_API_KEY、"
                "STANCE_LLM_BASE_URL 和 STANCE_LLM_MODEL_NAME"
            )
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self.client

    def _run_dir(self, run_id: str) -> Path:
        if not isinstance(run_id, str) or not self.RUN_ID_PATTERN.fullmatch(run_id):
            raise ValueError("invalid S1 Reddit run_id")
        return self.storage_dir / run_id

    @staticmethod
    def _read_json(path: Path) -> Any:
        if not path.exists():
            raise FileNotFoundError(path)
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        records: List[Dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Ignoring malformed JSONL line in %s", path)
                continue
            if isinstance(value, dict):
                records.append(value)
        return records

    @staticmethod
    def _content_key(content_type: Any, content_id: Any) -> str:
        return f"{str(content_type).lower()}:{str(content_id)}"

    @staticmethod
    def _content_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @classmethod
    def _json_value(cls, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (list, tuple)):
            return [cls._json_value(item) for item in value]
        if isinstance(value, dict):
            return {str(key): cls._json_value(item) for key, item in value.items()}
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            try:
                return cls._json_value(model_dump(mode="json"))
            except TypeError:
                return cls._json_value(model_dump())
        return str(value)

    @classmethod
    def _extract_json_object(cls, raw: str) -> Dict[str, Any]:
        text = (raw or "").strip()
        if not text:
            raise StanceAnnotationError("empty model response")
        candidates = [text]
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.I | re.S)
        if fenced:
            candidates.insert(0, fenced.group(1))
        decoder = json.JSONDecoder()
        for candidate in candidates:
            try:
                value = json.loads(candidate)
                if isinstance(value, dict):
                    return value
            except json.JSONDecodeError:
                pass
        for index, char in enumerate(text):
            if char != "{":
                continue
            try:
                value, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
        raise StanceAnnotationError("model response is not a JSON object")

    @classmethod
    def _validate_payload(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
        aliases = {
            "bullish": "positive",
            "bearish": "negative",
            "up": "positive",
            "down": "negative",
            "uncertain": "uncertain",
        }
        stance = str(payload.get("stance", "uncertain")).strip().lower()
        stance = aliases.get(stance, stance)
        if stance not in cls.STANCE_LABELS:
            raise StanceAnnotationError(f"invalid stance: {stance}")
        target = str(payload.get("target", "uncertain")).strip().lower()
        if target not in cls.TARGET_LABELS:
            target = "uncertain"
        event_valence = str(payload.get("event_valence", "uncertain")).strip().lower()
        event_valence = aliases.get(event_valence, event_valence)
        if event_valence not in cls.STANCE_LABELS:
            event_valence = "uncertain"

        def bounded_number(name: str, lower: float, upper: float) -> Optional[float]:
            value = payload.get(name)
            if value is None or value == "":
                return None
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise StanceAnnotationError(f"{name} must be numeric") from exc
            if not lower <= number <= upper:
                raise StanceAnnotationError(f"{name} must be in [{lower}, {upper}]")
            return round(number, 6)

        return {
            "stance": stance,
            "target": target,
            "event_valence": event_valence,
            "stance_score": bounded_number("stance_score", -1.0, 1.0),
            "confidence": bounded_number("confidence", 0.0, 1.0),
            "supports_content_id": payload.get("supports_content_id"),
            "challenges_content_id": payload.get("challenges_content_id"),
            "reason": str(payload.get("reason", "")).strip()[:1000],
        }

    @classmethod
    def build_prompt(cls, item: Dict[str, Any]) -> List[Dict[str, str]]:
        author_class = item.get("author_class", "unknown")
        role_instruction = (
            "这是投资者发布的观点，请识别作者对股票/事件的立场。"
            if author_class == "investor"
            else "这是信息源发布的内容。请把作者立场设为 uncertain 或 neutral，另行判断事件本身的 event_valence。"
        )
        system = (
            "你是独立的社会媒体内容标注员，不是投资 Agent。"
            "只根据给定文本标注，不使用外部搜索、股票实际结果或未来信息。"
            "必须返回一个 JSON object，不要输出 Markdown 或解释文字。"
        )
        user = f"""请标注以下 A 股匿名社会媒体内容。

内容身份：{item.get('content_type')}:{item.get('content_id')}
作者类别：{author_class}
作者 Agent ID：{item.get('author_agent_id')}
互动轮次：{item.get('round')}
{role_instruction}

标签定义：
- stance：作者对股票或事件的态度，positive/negative/mixed/neutral/uncertain
- target：态度针对 stock/event/other_agent/none/uncertain
- event_valence：内容描述的事件对股票可能影响，positive/negative/mixed/neutral/uncertain
- stance_score：作者立场强度，[-1,1]，负数偏负面，正数偏正面；无法判断时填 0
- confidence：[0,1]

原文：
{item.get('content_text', '')}

仅返回：
{{"stance":"positive|negative|mixed|neutral|uncertain","target":"stock|event|other_agent|none|uncertain","event_valence":"positive|negative|mixed|neutral|uncertain","stance_score":0.0,"confidence":0.0,"supports_content_id":null,"challenges_content_id":null,"reason":"不超过1000字，列出文本证据"}}"""
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    @classmethod
    def _collect_contents(
        cls, actions: Sequence[Dict[str, Any]], exposures: Sequence[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        by_key: Dict[str, Dict[str, Any]] = {}
        for item in actions:
            action_type = str(item.get("action_type", "")).lower()
            if action_type not in {"create_post", "create_comment"}:
                continue
            content_type = "post" if action_type == "create_post" else "comment"
            content_id = item.get("post_id") if content_type == "post" else item.get("comment_id")
            if content_id is None:
                continue
            args = item.get("action_args") or {}
            text = args.get("content", item.get("content_text", ""))
            key = cls._content_key(content_type, content_id)
            by_key.setdefault(
                key,
                {
                    "content_key": key,
                    "content_type": content_type,
                    "content_id": content_id,
                    "author_agent_id": item.get("agent_id"),
                    "author_class": item.get("agent_class", "unknown"),
                    "round": item.get("round"),
                    "parent_content_id": item.get("parent_comment_id"),
                    "content_text": str(text or ""),
                    "baseline_stance": item.get("content_stance", "unknown"),
                    "baseline_stance_score": item.get("stance_score"),
                    "baseline_stance_source": item.get("stance_source", "unlabeled"),
                },
            )
        for item in exposures:
            content_type = item.get("content_type")
            content_id = item.get("content_id")
            if content_type not in {"post", "comment"} or content_id is None:
                continue
            key = cls._content_key(content_type, content_id)
            if key in by_key:
                continue
            by_key[key] = {
                "content_key": key,
                "content_type": content_type,
                "content_id": content_id,
                "author_agent_id": item.get("author_agent_id"),
                "author_class": "source" if (item.get("author_agent_id") or -1) >= 20 else "investor",
                "round": item.get("round"),
                "parent_content_id": None,
                "content_text": str(item.get("content_text", "") or ""),
                "baseline_stance": item.get("content_stance", "unknown"),
                "baseline_stance_score": item.get("stance_score"),
                "baseline_stance_source": item.get("stance_source", "unlabeled"),
            }
        return sorted(by_key.values(), key=lambda item: (str(item.get("round")), item["content_key"]))

    def _annotate_one(self, item: Dict[str, Any]) -> Dict[str, Any]:
        last_raw = ""
        last_error = ""
        finish_reason = None
        reasoning_present = False
        attempts = 0
        for attempts in range(1, self.MAX_ATTEMPTS + 1):
            prompt = self.build_prompt(item)
            if attempts > 1:
                prompt[-1]["content"] += "\n上一次响应无效。请严格返回合法 JSON object，不能省略字段。"
            try:
                response = create_chat_completion(
                    self._get_client(),
                    model=self.model,
                    messages=prompt,
                    temperature=self.TEMPERATURE,
                    max_tokens=self.MAX_TOKENS,
                    response_format=self.RESPONSE_FORMAT,
                    thinking_mode="disabled",
                )
                last_raw = extract_chat_completion_text(response)
                finish_reason = extract_chat_completion_finish_reason(response)
                reasoning_present = has_chat_completion_reasoning_content(response)
                parsed = self._validate_payload(self._extract_json_object(last_raw))
                return {
                    **item,
                    **parsed,
                    "annotator_model": self.model,
                    "annotator_base_url": self.base_url,
                    "config_source": self.config_source,
                    "prompt_version": self.PROMPT_VERSION,
                    "attempt_count": attempts,
                    "finish_reason": finish_reason,
                    "response_content_length": len(last_raw),
                    "reasoning_content_present": reasoning_present,
                    "raw_response": last_raw,
                    "status": "ok",
                    "error": None,
                    "content_hash": self._content_hash(item["content_text"]),
                    "annotated_at": datetime.now(timezone.utc).isoformat(),
                }
            except Exception as exc:
                last_error = str(exc)
                logger.warning("Stance annotation attempt %s failed for %s: %s", attempts, item["content_key"], exc)
                if attempts < self.MAX_ATTEMPTS:
                    time.sleep(0.1)
        return {
            **item,
            "stance": "uncertain",
            "target": "uncertain",
            "event_valence": "uncertain",
            "stance_score": None,
            "confidence": None,
            "supports_content_id": None,
            "challenges_content_id": None,
            "reason": "",
            "annotator_model": self.model,
            "annotator_base_url": self.base_url,
            "config_source": self.config_source,
            "prompt_version": self.PROMPT_VERSION,
            "attempt_count": attempts,
            "finish_reason": finish_reason,
            "response_content_length": len(last_raw),
            "reasoning_content_present": reasoning_present,
            "raw_response": last_raw,
            "status": "failed",
            "error": last_error,
            "content_hash": self._content_hash(item["content_text"]),
            "annotated_at": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    @classmethod
    def _write_csv(cls, path: Path, rows: Sequence[Dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=cls.CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({field: cls._csv_value(row.get(field)) for field in cls.CSV_FIELDS})

    @staticmethod
    def _csv_value(value: Any) -> Any:
        if value is None:
            return ""
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value, ensure_ascii=False)
        if isinstance(value, bool):
            return "true" if value else "false"
        return value

    @classmethod
    def _merge_annotations(
        cls, actions: Sequence[Dict[str, Any]], exposures: Sequence[Dict[str, Any]], annotations: Sequence[Dict[str, Any]]
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        by_key = {item["content_key"]: item for item in annotations}

        def enrich(record: Dict[str, Any], content_type: Any, content_id: Any) -> Dict[str, Any]:
            result = dict(record)
            annotation = by_key.get(cls._content_key(content_type, content_id))
            if not annotation:
                return result
            result["baseline_content_stance"] = result.get("content_stance")
            result["baseline_stance_score"] = result.get("stance_score")
            result["baseline_stance_source"] = result.get("stance_source")
            result["content_stance"] = annotation.get("stance", "uncertain")
            result["stance_score"] = annotation.get("stance_score")
            result["stance_confidence"] = annotation.get("confidence")
            result["event_valence"] = annotation.get("event_valence", "uncertain")
            result["stance_target"] = annotation.get("target", "uncertain")
            result["stance_source"] = "offline_llm"
            result["stance_annotation_status"] = annotation.get("status")
            result["stance_annotation_id"] = annotation.get("content_key")
            return result

        enriched_actions = []
        for record in actions:
            action_type = str(record.get("action_type", "")).lower()
            content_type = "post" if action_type == "create_post" else "comment" if action_type == "create_comment" else None
            content_id = record.get("post_id") if content_type == "post" else record.get("comment_id")
            enriched_actions.append(enrich(record, content_type, content_id) if content_type and content_id is not None else dict(record))
        enriched_exposures = [
            enrich(record, record.get("content_type"), record.get("content_id"))
            if record.get("content_type") in {"post", "comment"} and record.get("content_id") is not None
            else dict(record)
            for record in exposures
        ]
        return enriched_actions, enriched_exposures

    def annotate_run(self, run_id: str, *, force: bool = False) -> Dict[str, Any]:
        run_dir = self._run_dir(run_id)
        manifest = self._read_json(run_dir / "manifest.json")
        if manifest.get("status") not in {"completed", "failed", "stopped", "interrupted"}:
            raise ValueError("stance annotation requires a completed S1 run")
        actions = self._read_jsonl(run_dir / "social_actions.jsonl")
        exposures = self._read_jsonl(run_dir / "exposure_edges.jsonl")
        items = self._collect_contents(actions, exposures)
        items = [
            {
                **item,
                "run_id": run_id,
                "scenario_id": manifest.get("scenario_id"),
            }
            for item in items
        ]
        existing_rows = self._read_jsonl(run_dir / "stance_annotations.jsonl") if not force else []
        existing = {row.get("content_key"): row for row in existing_rows if row.get("status") == "ok"}
        rows: List[Dict[str, Any]] = []
        for item in items:
            cached = existing.get(item["content_key"])
            if cached and cached.get("content_hash") == self._content_hash(item["content_text"]):
                rows.append(cached)
            else:
                rows.append(self._annotate_one(item))
        self._write_jsonl(run_dir / "stance_annotations.jsonl", rows)
        self._write_csv(run_dir / "stance_annotations.csv", rows)
        enriched_actions, enriched_exposures = self._merge_annotations(actions, exposures, rows)
        self._write_jsonl(run_dir / "social_actions_annotated.jsonl", enriched_actions)
        self._write_jsonl(run_dir / "exposure_edges_annotated.jsonl", enriched_exposures)
        manifest.setdefault("files", {}).update(
            {
                "stance_annotations": "stance_annotations.jsonl",
                "stance_annotations_csv": "stance_annotations.csv",
                "social_actions_annotated": "social_actions_annotated.jsonl",
                "exposure_edges_annotated": "exposure_edges_annotated.jsonl",
            }
        )
        manifest["stance_annotation"] = {
            "status": "completed",
            "prompt_version": self.PROMPT_VERSION,
            "annotator_model": self.model,
            "annotator_base_url": self.base_url,
            "config_source": self.config_source,
            "content_count": len(rows),
            "success_count": sum(row.get("status") == "ok" for row in rows),
            "failed_count": sum(row.get("status") != "ok" for row in rows),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        run_dir.joinpath("manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return manifest["stance_annotation"] | {"run_id": run_id, "files": manifest["files"]}

    def get_annotations(self, run_id: str) -> List[Dict[str, Any]]:
        return self._read_jsonl(self._run_dir(run_id) / "stance_annotations.jsonl")

    def get_annotated_exposure_edges(self, run_id: str) -> List[Dict[str, Any]]:
        return self._read_jsonl(self._run_dir(run_id) / "exposure_edges_annotated.jsonl")
