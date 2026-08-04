"""Deterministic C0 investor roles.

The 17:3 composition follows the current experiment plan. It is an
experimental voice allocation, not a claim that it exactly reproduces the
number of people or the capital held by each market group.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


C0_AGENT_COUNT = 20


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
            "重视流动性、下行风险和不确定性，倾向于在证据不足时降低"
            "置信度，而不是追逐单一叙事。"
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


def iter_c0_roles() -> List[Dict[str, Any]]:
    """Expand role templates to the 20 deterministic investor records."""
    roles: List[Dict[str, Any]] = []
    next_id = 0
    for template in C0_ROLE_TEMPLATES:
        for ordinal in range(1, template.count + 1):
            roles.append(
                {
                    "agent_id": next_id,
                    "agent_key": f"investor_{next_id + 1:03d}",
                    "role_id": template.role_id,
                    "role_category": template.category,
                    "role_label": template.label,
                    "role_description": template.description,
                    "role_ordinal": ordinal,
                }
            )
            next_id += 1
    if len(roles) != C0_AGENT_COUNT:
        raise RuntimeError(f"C0 role configuration must contain {C0_AGENT_COUNT} agents")
    return roles


def build_c0_profiles() -> List[Dict[str, Any]]:
    """Create profiles compatible with the existing OASIS profile files.

    C0 does not start an OASIS social environment, but using the same profile
    shape keeps the adapter plug-and-play for a later S1 implementation.
    """
    profiles: List[Dict[str, Any]] = []
    for role in iter_c0_roles():
        name = role["agent_key"]
        persona = (
            f"你是{role['role_label']}，编号为{name}。{role['role_description']}"
            "本轮属于 C0 独立判断组。你不能看到其他投资者的帖子、预测、"
            "回复或任何聚合意见，只能依据给定的历史事件和当前公开事件作答。"
        )
        profiles.append(
            {
                "user_id": role["agent_id"],
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
            }
        )
    return profiles
