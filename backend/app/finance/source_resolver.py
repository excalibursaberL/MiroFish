"""Resolve finance event publishers to scenario-aware OASIS source accounts.

The investor population is an experimental control and therefore remains
fixed.  Information-source accounts are different: they must be derived from
the entities and attribution language present in the frozen seed material.
When a Zep graph is supplied, its entity UUIDs and labels are retained.  A
small deterministic text resolver is kept as an explicit offline fallback; it
never invents media, exchange, or regulator accounts merely to fill a quota.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..services.zep_entity_reader import EntityNode
from .dataset import FinancialScenario


PUBLIC_FEED_ENTITY_ID = "system:public_disclosure_feed"


@dataclass(frozen=True)
class GraphEntityView:
    entity_id: str
    name: str
    labels: Tuple[str, ...]
    summary: str
    attributes: Dict[str, Any]
    aliases: Tuple[str, ...]


class FinanceEventSourceResolver:
    """Map each frozen event to one actual publisher identity.

    One publisher entity becomes one account even when it publishes several
    events.  Mentioned entities are recorded for auditability, but do not
    automatically become active or publishing Agents.
    """

    _MEDIA_PREFIX = re.compile(
        r"^\s*(?P<name>财经媒体[A-Za-z0-9_]+?)(?:T[+-]\d+d)?(?:电|讯)"
    )
    _LEADING_ANONYMOUS_COMPANY = re.compile(
        r"^\s*\*?(?:[一二两三四五六七八九十0-9]+连板)?(?P<name>COMPANY_\d+)"
    )
    _LEADING_PUBLISHER = re.compile(
        r"^\s*\*?(?P<name>[^，。；:：]{1,64}?)(?:公告|发布|披露|回复|表示|称)"
    )
    _COMPANY_ANNOUNCEMENT = re.compile(
        r"(?P<name>\*?COMPANY_\d+)\s*(?:公告|发布|披露|回复)"
    )
    _LOCAL_ENTITY = re.compile(
        r"COMPANY_\d+|证券交易所[A-Z]|财经媒体[A-Z]|"
        r"[\u4e00-\u9fff]{2,20}(?:公司|集团|机构|企业)[A-Z]"
    )
    _ALIAS_ATTRIBUTE_KEYS = {
        "alias",
        "aliases",
        "full_name",
        "org_name",
        "company_name",
        "short_name",
        "abbreviation",
        "简称",
        "别名",
    }

    def __init__(self, graph_entities: Optional[Sequence[EntityNode]] = None) -> None:
        self.graph_entities = [self._view(entity) for entity in graph_entities or []]

    def resolve(
        self,
        scenario: FinancialScenario,
        *,
        source_agent_start: int,
        source_mode: str,
        graph_id: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        if source_mode not in {"graph", "scenario"}:
            raise ValueError("source_mode must be graph or scenario")
        if source_mode == "graph" and not graph_id:
            raise ValueError("graph source mode requires graph_id")

        raw_events = list(scenario.seed_events) + [scenario.current_event]
        source_records: Dict[str, Dict[str, Any]] = {}
        event_records: List[Dict[str, Any]] = []
        unmatched_graph_publishers: List[str] = []

        for event in raw_events:
            text = str(event.get("text", ""))
            inferred_name, inference_method = self._infer_publisher_name(
                text, scenario.name
            )
            graph_match, graph_match_method = self._match_graph_entity(inferred_name)

            if graph_match is not None:
                source = self._source_from_graph(graph_match)
                resolution_method = f"{inference_method}+{graph_match_method}"
            elif inferred_name:
                if source_mode == "graph":
                    unmatched_graph_publishers.append(
                        f"{event.get('event_id')}={inferred_name}"
                    )
                source = self._source_from_text(inferred_name)
                resolution_method = inference_method
            else:
                source = self._public_feed_source()
                resolution_method = "public_feed_fallback"

            source_records.setdefault(source["source_entity_id"], source)
            mentioned_ids = self._mentioned_entity_ids(text)
            if source["source_entity_id"] not in mentioned_ids:
                mentioned_ids.insert(0, source["source_entity_id"])
            event_records.append(
                {
                    "event_id": event.get("event_id"),
                    "publisher_entity_id": source["source_entity_id"],
                    "publisher_name": source["name"],
                    "publisher_source_type": source["source_type"],
                    "publisher_origin": source["source_origin"],
                    "publisher_resolution": resolution_method,
                    "mentioned_entity_ids": mentioned_ids,
                }
            )

        if unmatched_graph_publishers:
            details = ", ".join(unmatched_graph_publishers)
            raise ValueError(
                "event publisher was not extracted into the supplied Zep graph: "
                + details
            )

        ordered_sources = list(source_records.values())
        entity_to_agent: Dict[str, int] = {}
        source_profiles = []
        for index, source in enumerate(ordered_sources):
            agent_id = source_agent_start + index
            entity_to_agent[source["source_entity_id"]] = agent_id
            source_profiles.append(self._to_profile(source, agent_id, index))

        for event in event_records:
            event["publisher_agent_id"] = entity_to_agent[event["publisher_entity_id"]]

        mapping = {
            "source_mode": source_mode,
            "graph_id": graph_id or "",
            "publisher_account_count": len(source_profiles),
            "graph_entity_count": len(self.graph_entities),
            "publishers": [
                {
                    "agent_id": profile["user_id"],
                    "source_entity_id": profile["source_entity_id"],
                    "name": profile["name"],
                    "source_type": profile["source_type"],
                    "source_origin": profile["source_origin"],
                    "source_entity_type": profile.get("source_entity_type"),
                }
                for profile in source_profiles
            ],
            "events": event_records,
        }
        return source_profiles, event_records, mapping

    @classmethod
    def _view(cls, entity: EntityNode) -> GraphEntityView:
        aliases = [entity.name]
        for key, value in (entity.attributes or {}).items():
            if str(key).lower() not in cls._ALIAS_ATTRIBUTE_KEYS and str(key) not in cls._ALIAS_ATTRIBUTE_KEYS:
                continue
            if isinstance(value, (list, tuple, set)):
                aliases.extend(str(item) for item in value if item)
            elif value:
                aliases.append(str(value))
        unique_aliases = []
        for alias in aliases:
            normalized = cls._normalize_name(alias)
            if normalized and normalized not in unique_aliases:
                unique_aliases.append(normalized)
        return GraphEntityView(
            entity_id=entity.uuid,
            name=entity.name,
            labels=tuple(entity.labels or []),
            summary=entity.summary or "",
            attributes=dict(entity.attributes or {}),
            aliases=tuple(unique_aliases),
        )

    @classmethod
    def _infer_publisher_name(
        cls, text: str, scenario_company: str
    ) -> Tuple[Optional[str], str]:
        media = cls._MEDIA_PREFIX.search(text)
        if media:
            return cls._normalize_name(media.group("name")), "text_media_prefix"

        anonymous_company = cls._LEADING_ANONYMOUS_COMPANY.search(text)
        if anonymous_company:
            return (
                cls._normalize_name(anonymous_company.group("name")),
                "text_company_prefix",
            )

        leading = cls._LEADING_PUBLISHER.search(text)
        if leading:
            publisher = cls._normalize_name(leading.group("name"))
            if publisher in {"公司", "本公司", "上市公司"} and scenario_company:
                return (
                    cls._normalize_name(scenario_company),
                    "scenario_company_attribution",
                )
            return publisher, "text_leading_attribution"

        company = cls._COMPANY_ANNOUNCEMENT.search(text)
        if company:
            return cls._normalize_name(company.group("name")), "text_company_attribution"

        if scenario_company and scenario_company in text and any(
            marker in text for marker in ("公告", "发布", "披露", "回复")
        ):
            return cls._normalize_name(scenario_company), "scenario_company_attribution"
        return None, "unresolved"

    def _match_graph_entity(
        self, inferred_name: Optional[str]
    ) -> Tuple[Optional[GraphEntityView], str]:
        normalized = self._normalize_name(inferred_name or "")
        if not normalized:
            return None, "graph_unmatched"
        for entity in self.graph_entities:
            if normalized in entity.aliases:
                return entity, "graph_exact"
        if len(normalized) >= 4:
            for entity in self.graph_entities:
                if any(
                    len(alias) >= 4 and (normalized in alias or alias in normalized)
                    for alias in entity.aliases
                ):
                    return entity, "graph_alias"
        return None, "graph_unmatched"

    def _mentioned_entity_ids(self, text: str) -> List[str]:
        result: List[str] = []
        for entity in self.graph_entities:
            if any(len(alias) >= 3 and alias in text for alias in entity.aliases):
                result.append(entity.entity_id)
        for match in self._LOCAL_ENTITY.finditer(text):
            entity_id = f"text:{self._normalize_name(match.group(0))}"
            if entity_id not in result:
                result.append(entity_id)
        return result

    @classmethod
    def _source_from_graph(cls, entity: GraphEntityView) -> Dict[str, Any]:
        entity_type = next(
            (label for label in entity.labels if label not in {"Entity", "Node"}),
            "Entity",
        )
        return {
            "source_entity_id": entity.entity_id,
            "source_entity_type": entity_type,
            "name": entity.name,
            "source_type": cls._classify_source_type(entity.name, entity.labels),
            "source_origin": "zep_graph",
            "summary": entity.summary,
            "labels": list(entity.labels),
        }

    @classmethod
    def _source_from_text(cls, name: str) -> Dict[str, Any]:
        normalized = cls._normalize_name(name)
        return {
            "source_entity_id": f"text:{normalized}",
            "source_entity_type": "TextAttributedEntity",
            "name": normalized,
            "source_type": cls._classify_source_type(normalized, ()),
            "source_origin": "scenario_text",
            "summary": "",
            "labels": [],
        }

    @staticmethod
    def _public_feed_source() -> Dict[str, Any]:
        return {
            "source_entity_id": PUBLIC_FEED_ENTITY_ID,
            "source_entity_type": "PublicDisclosureFeed",
            "name": "PUBLIC_DISCLOSURE_FEED",
            "source_type": "public_feed",
            "source_origin": "public_feed",
            "summary": "",
            "labels": [],
        }

    @classmethod
    def _to_profile(
        cls, source: Dict[str, Any], agent_id: int, index: int
    ) -> Dict[str, Any]:
        type_label = {
            "company": "上市公司公开账号",
            "media": "财经媒体公开账号",
            "exchange": "证券交易所公开账号",
            "regulator": "监管机构公开账号",
            "shareholder": "股东公开账号",
            "public_feed": "统一公开信息流",
        }.get(source["source_type"], "事件相关实体公开账号")
        return {
            "user_id": agent_id,
            "username": f"source_{source['source_type']}_{index + 1:03d}",
            "name": source["name"],
            "bio": f"{type_label}（从当前匿名场景解析，不参与预测）",
            "persona": (
                f"你是{source['name']}对应的{type_label}。"
                "你只发布实验调度中逐字提供的匿名公开事件，不补充、改写或推测事实；"
                "不预测收益，也不主动参与投资者讨论。"
            ),
            "karma": 5000,
            "created_at": "1970-01-01",
            "profession": type_label,
            "mbti": "ISTJ",
            "gender": "organization",
            "age": 0,
            "country": "CN",
            "interested_topics": ["A股", "公开信息"],
            "agent_key": f"source_{index + 1:03d}",
            "agent_class": "source",
            "source_type": source["source_type"],
            "source_origin": source["source_origin"],
            "source_entity_id": source["source_entity_id"],
            "source_entity_uuid": (
                source["source_entity_id"]
                if source["source_origin"] == "zep_graph"
                else None
            ),
            "source_entity_type": source["source_entity_type"],
            "source_labels": source.get("labels", []),
        }

    @classmethod
    def _classify_source_type(
        cls, name: str, labels: Iterable[str]
    ) -> str:
        text = " ".join([name, *[str(label) for label in labels]]).lower()
        if any(token in text for token in ("media", "news", "媒体", "新闻")):
            return "media"
        if any(token in text for token in ("exchange", "交易所")):
            return "exchange"
        if any(token in text for token in ("regulator", "governmentagency", "监管", "证监")):
            return "regulator"
        if any(token in text for token in ("shareholder", "股东", "资管", "资产管理")):
            return "shareholder"
        if any(token in text for token in ("company", "corporation", "上市公司", "企业", "公司")):
            return "company"
        return "organization"

    @staticmethod
    def _normalize_name(value: str) -> str:
        normalized = str(value or "").strip().lstrip("*").strip()
        normalized = re.sub(r"T[+-]\d+d$", "", normalized)
        return normalized.strip(" ，,:：。；;（）()[]【】")
