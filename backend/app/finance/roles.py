"""Deterministic, minimal investor profiles for the C0 experiment.

The population mix is an experimental voice allocation: 17 retail voices and
3 institutional voices.  Retail margins for knowledge, risk attitude,
investment horizon, and decision source are taken from the SIPF 2019 survey.
The fundamental/technical split is borrowed from TwinMarket as an explicitly
synthetic experimental prior, not as a claim about the real A-share population.

Only the fields relevant to event-based price-direction forecasting are put in
the C0 prompt.  Trading-account and social-propagation fields remain in the
profile record for later S1/trading experiments but are deliberately inactive
in C0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from .skill_registry import (
    assign_finance_skills,
    normalize_finance_skill_scope,
)


FULL_AGENT_COUNT = 20
SELECTED_AGENT_IDS = (1, 3, 4, 5, 9, 11, 12, 13, 14, 17)
C0_AGENT_COUNT = len(SELECTED_AGENT_IDS)
DEFAULT_AGENT_SET_VERSION = "n10_k10_exact_v1"
DEFAULT_SAMPLING_METHOD = "offline_exact_enumeration_k10"
PROFILE_VERSION = "survey2019_twinmarket_minimal_v1"


@dataclass(frozen=True)
class InvestorRole:
    role_id: str
    category: str
    label: str
    description: str
    count: int


C0_ROLE_TEMPLATES = (
    InvestorRole(
        role_id="institutional_active",
        category="institution",
        label="主动管理机构投资者",
        description=(
            "关注公司基本面、公告可信度和估值变化；会比较多个证据，"
            "但不会读取其他投资者的观点。"
        ),
        count=1,
    ),
    InvestorRole(
        role_id="institutional_quant",
        category="institution",
        label="量化与技术机构投资者",
        description=(
            "更重视结构化行情特征、短期价格行为和风险信号；避免把"
            "单条新闻直接等同于确定的涨跌。"
        ),
        count=1,
    ),
    InvestorRole(
        role_id="institutional_risk",
        category="institution",
        label="被动与风险控制机构投资者",
        description=(
            "重视流动性、下行风险和不确定性，证据不足时降低置信度，"
            "而不是追逐单一叙事。"
        ),
        count=1,
    ),
    InvestorRole(
        role_id="retail_mature",
        category="retail_mature",
        label="有经验的散户",
        description=(
            "能够阅读公告和基础行情，关注事件影响与市场预期的差异，"
            "但资金和信息工具有限。"
        ),
        count=6,
    ),
    InvestorRole(
        role_id="retail_basic",
        category="retail_basic",
        label="具备基础知识的散户",
        description=(
            "能理解公告中的主要事实和简单涨跌，但分析深度有限，容易"
            "受到明显叙事和近期行情的影响。"
        ),
        count=8,
    ),
    InvestorRole(
        role_id="retail_novice",
        category="retail_novice",
        label="投资新手散户",
        description=(
            "金融知识较少，只使用输入中直接可见的事实，保持较低置信度，"
            "不应凭空补充企业背景。"
        ),
        count=3,
    ),
)


# These are the survey margins used in Agent占比分析.md.  Keeping them in
# code makes the generated profiles reproducible and auditable.
RETAIL_MARGIN_COUNTS = {
    "knowledge_level": {"experienced": 6, "basic": 8, "novice": 3},
    "risk_attitude": {"low": 3, "medium": 10, "high": 3, "very_high": 1},
    "investment_horizon": {"long": 9, "mixed": 6, "short": 2},
    "decision_source": {
        "self_analysis": 11,
        "friends": 3,
        "online_expert": 1,
        "advisor": 2,
    },
    # TwinMarket's synthetic strategy distribution is approximately 40/60.
    # This is a controlled prior for analysis-style diversity, not a survey
    # estimate of the A-share population.
    "analysis_style": {"fundamental": 7, "technical": 10},
}


_RETAIL_PERMUTATIONS = {
    "risk_attitude": (10, 9, 16, 4, 11, 1, 8, 12, 6, 13, 0, 7, 14, 3, 2, 5, 15),
    "investment_horizon": (4, 12, 1, 9, 16, 6, 14, 3, 11, 0, 8, 15, 5, 13, 2, 10, 7),
    "decision_source": (1, 8, 15, 3, 10, 0, 7, 14, 5, 12, 2, 9, 16, 4, 11, 6, 13),
    "analysis_style": (0, 8, 16, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 15, 11, 7, 3),
}


def _expand_counts(counts: Dict[str, int]) -> List[str]:
    values: List[str] = []
    for label, count in counts.items():
        values.extend([label] * count)
    return values


def _spread_margin(field: str) -> List[str]:
    values = _expand_counts(RETAIL_MARGIN_COUNTS[field])
    permutation = _RETAIL_PERMUTATIONS[field]
    if len(values) != len(permutation):
        raise RuntimeError(f"invalid retail margin for {field}")
    if sorted(permutation) != list(range(len(values))):
        raise RuntimeError(f"invalid retail permutation for {field}")
    return [values[index] for index in permutation]


_RETAIL_ASSIGNMENTS = {
    field: _spread_margin(field)
    for field in RETAIL_MARGIN_COUNTS
    if field != "knowledge_level"
}


_INSTITUTION_ASSIGNMENTS = {
    "institutional_active": {
        "knowledge_level": "professional",
        "analysis_style": "fundamental",
        "risk_attitude": "medium",
        "investment_horizon": "long",
        "decision_source": "self_analysis",
        "social_role": "institutional",
    },
    "institutional_quant": {
        "knowledge_level": "professional",
        "analysis_style": "technical",
        "risk_attitude": "high",
        "investment_horizon": "mixed",
        "decision_source": "self_analysis",
        "social_role": "institutional",
    },
    "institutional_risk": {
        "knowledge_level": "professional",
        "analysis_style": "risk_control",
        "risk_attitude": "low",
        "investment_horizon": "long",
        "decision_source": "self_analysis",
        "social_role": "institutional",
    },
}


_RETAIL_ROLE_TO_KNOWLEDGE = {
    "retail_mature": "experienced",
    "retail_basic": "basic",
    "retail_novice": "novice",
}


_KNOWLEDGE_LABELS = {
    "experienced": "熟悉投资",
    "basic": "具备基本知识",
    "novice": "新手",
    "professional": "专业机构研究人员",
}
_ANALYSIS_LABELS = {
    "fundamental": "基本面与事件分析",
    "technical": "技术与价格行为分析",
    "risk_control": "风险控制与流动性分析",
}
_ANALYSIS_GUIDANCE = {
    "fundamental": (
        "优先比较公告事实、业绩和业务影响，同时检查正负证据是否相互矛盾。"
    ),
    "technical": (
        "优先关注输入中明确给出的时间、价格行为或热度信号；如果没有提供"
        "K 线或技术指标，不得编造均线、MACD、成交量等数据。"
    ),
    "risk_control": (
        "优先检查信息缺口、下行风险和事件兑现条件，证据不足时降低置信度。"
    ),
}
_RISK_LABELS = {
    "low": "低风险",
    "medium": "中等风险",
    "high": "高风险",
    "very_high": "极高风险/投机",
}
_HORIZON_LABELS = {
    "long": "较长期",
    "mixed": "中期或混合期限",
    "short": "较短期",
}


def _retail_assignment(index: int, field: str) -> str:
    try:
        return _RETAIL_ASSIGNMENTS[field][index]
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"invalid retail assignment {field}[{index}]") from exc


def iter_c0_roles() -> List[Dict[str, Any]]:
    """Expand role templates to the 20 deterministic investor records."""
    roles: List[Dict[str, Any]] = []
    next_id = 0
    retail_index = 0
    for template in C0_ROLE_TEMPLATES:
        for ordinal in range(1, template.count + 1):
            role = {
                "agent_id": next_id,
                "agent_key": f"investor_{next_id + 1:03d}",
                "role_id": template.role_id,
                "role_category": template.category,
                "role_label": template.label,
                "role_description": template.description,
                "role_ordinal": ordinal,
            }
            if template.category == "institution":
                role.update(_INSTITUTION_ASSIGNMENTS[template.role_id])
            else:
                role.update(
                    {
                        "knowledge_level": _RETAIL_ROLE_TO_KNOWLEDGE[template.role_id],
                        "analysis_style": _retail_assignment(
                            retail_index, "analysis_style"
                        ),
                        "risk_attitude": _retail_assignment(
                            retail_index, "risk_attitude"
                        ),
                        "investment_horizon": _retail_assignment(
                            retail_index, "investment_horizon"
                        ),
                        "decision_source": _retail_assignment(
                            retail_index, "decision_source"
                        ),
                        "social_role": "ordinary",
                    }
                )
                retail_index += 1
            roles.append(role)
            next_id += 1
    if len(roles) != FULL_AGENT_COUNT:
        raise RuntimeError(
            f"full role configuration must contain {FULL_AGENT_COUNT} agents"
        )
    if retail_index != 17:
        raise RuntimeError("C0 role configuration must contain 17 retail agents")
    return roles


def profile_prompt_text(
    profile: Dict[str, Any],
    *,
    include_finance_skill: bool = True,
) -> str:
    """Render stable profile fields, optionally omitting the Skill body.

    S1's pre-social-only Skill ablation keeps the assigned Skill metadata but
    builds the OASIS persona from the base profile.  The Skill is then added
    explicitly to the pre-social interview prompt only.
    """
    text = (
        "- 知识水平：{knowledge}\n"
        "- 分析方式：{analysis}\n"
        "- 分析规则：{analysis_guidance}\n"
        "- 风险态度：{risk}\n"
        "- 投资期限：{horizon}\n"
        "\n"
        "画像使用边界：风险态度只影响你对不确定性和下行风险的容忍度，"
        "投资期限只影响你理解信息时的关注重点；二者都不能预先决定股票涨跌。"
        "无论你的个人投资期限如何，都必须回答题目指定的预测 horizon。"
    ).format(
        knowledge=_KNOWLEDGE_LABELS[profile["knowledge_level"]],
        analysis=_ANALYSIS_LABELS[profile["analysis_style"]],
        analysis_guidance=_ANALYSIS_GUIDANCE[profile["analysis_style"]],
        risk=_RISK_LABELS[profile["risk_attitude"]],
        horizon=_HORIZON_LABELS[profile["investment_horizon"]],
    )
    skill_prompt = (
        str(profile.get("finance_skill_prompt") or "").strip()
        if include_finance_skill
        else ""
    )
    if skill_prompt:
        skill_names = ", ".join(profile.get("finance_skill_names") or [])
        text += (
            "\n\n## Enabled heterogeneous finance Skill\n"
            f"Skill IDs: {skill_names}\n"
            "Apply the following role-specific method while preserving every "
            "experiment information boundary and the common output contract.\n\n"
            f"{skill_prompt}"
        )
    return text


def build_full_c0_profiles(
    enabled_finance_skills: Optional[Sequence[str]] = None,
    finance_skill_scope: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Create the auditable 20-Agent source pool used by subset selection."""
    profiles: List[Dict[str, Any]] = []
    for role in iter_c0_roles():
        name = role["agent_key"]
        profile_text = profile_prompt_text(role)
        if role["role_category"] == "institution":
            profile_sources = {
                "knowledge_level": "institutional_role_design",
                "risk_attitude": "institutional_role_design",
                "investment_horizon": "institutional_role_design",
                "decision_source": "institutional_role_design_S1_only",
                "analysis_style": "institutional_role_design",
            }
        else:
            profile_sources = {
                "knowledge_level": "SIPF_2019_natural_person_survey",
                "risk_attitude": "SIPF_2019_natural_person_survey",
                "investment_horizon": "SIPF_2019_natural_person_survey",
                "decision_source": "SIPF_2019_natural_person_survey_S1_only",
                "analysis_style": (
                    "TwinMarket_strategy_category_40_60_experimental_prior"
                ),
            }
        persona = (
            f"你是{role['role_label']}，编号为{name}。{role['role_description']}\n"
            f"固定行为画像：\n{profile_text}\n"
            "本轮属于 C0 独立判断组。你不能看到其他投资者的帖子、预测、"
            "回复或任何聚合意见，只能依据给定的历史事件和当前公开事件作答。"
        )
        profiles.append(
            {
                "user_id": role["agent_id"],
                "full_population_agent_id": role["agent_id"],
                "username": name,
                "name": name,
                "bio": f"C0 {role['role_label']}（匿名实验角色）",
                "persona": persona,
                "karma": 1000,
                # Synthetic metadata only; it is not an event date and is
                # kept ISO-formatted for OASIS profile compatibility.
                "created_at": "1970-01-01",
                "profession": role["role_label"],
                "agent_key": role["agent_key"],
                "role_id": role["role_id"],
                "role_category": role["role_category"],
                "role_label": role["role_label"],
                "role_description": role["role_description"],
                "role_ordinal": role["role_ordinal"],
                "knowledge_level": role["knowledge_level"],
                "analysis_style": role["analysis_style"],
                "risk_attitude": role["risk_attitude"],
                "investment_horizon": role["investment_horizon"],
                # Reserved for S1; this field is not rendered in the C0
                # prompt because C0 has no social information to follow.
                "decision_source": role["decision_source"],
                "social_role": role["social_role"],
                "profile_version": PROFILE_VERSION,
                "profile_sources": profile_sources,
            }
        )
    scope = normalize_finance_skill_scope(finance_skill_scope)
    return [
        assign_finance_skills(
            profile,
            enabled_finance_skills,
            finance_skill_scope=scope,
        )
        for profile in profiles
    ]


def normalize_selected_agent_ids(
    selected_full_population_agent_ids: Optional[Sequence[int]] = None,
) -> tuple[int, ...]:
    """Validate a subset of the already selected K=10 source-pool IDs."""
    values = tuple(
        SELECTED_AGENT_IDS
        if selected_full_population_agent_ids is None
        else selected_full_population_agent_ids
    )
    if not values:
        raise ValueError("at least one selected Agent is required")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError("selected Agent IDs must be integers")
    if len(set(values)) != len(values):
        raise ValueError("selected Agent IDs must be unique")
    if not set(values).issubset(set(SELECTED_AGENT_IDS)):
        raise ValueError(
            "selected Agent IDs must be a subset of the configured K=10 source pool"
        )
    return values


def build_c0_profiles(
    selected_full_population_agent_ids: Optional[Sequence[int]] = None,
    enabled_finance_skills: Optional[Sequence[str]] = None,
    finance_skill_scope: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return selected source-pool profiles with contiguous runtime IDs."""
    selected_ids = normalize_selected_agent_ids(selected_full_population_agent_ids)
    full_profiles = {
        int(profile["full_population_agent_id"]): profile
        for profile in build_full_c0_profiles(
            enabled_finance_skills,
            finance_skill_scope=finance_skill_scope,
        )
    }
    profiles: List[Dict[str, Any]] = []
    for runtime_id, full_population_agent_id in enumerate(selected_ids):
        try:
            profile = dict(full_profiles[full_population_agent_id])
        except KeyError as exc:
            raise RuntimeError(
                f"selected Agent {full_population_agent_id} is absent from the full pool"
            ) from exc
        profile["user_id"] = runtime_id
        profile["full_population_agent_id"] = full_population_agent_id
        profiles.append(profile)
    if len(profiles) != len(selected_ids):
        raise RuntimeError(
            f"experiment profile configuration must contain {len(selected_ids)} agents"
        )
    return profiles
