import hashlib

import pytest

from app.finance.roles import build_c0_profiles, build_full_c0_profiles
from app.finance.skill_registry import (
    INSTITUTIONAL_ANALYST_SKILL,
    load_finance_skill,
    normalize_finance_skill_names,
    normalize_finance_skill_stage,
)


def test_institutional_analyst_skill_is_auditable_and_role_scoped():
    skill = load_finance_skill(INSTITUTIONAL_ANALYST_SKILL)

    assert skill.name == INSTITUTIONAL_ANALYST_SKILL
    assert len(skill.sha256) == 64
    assert skill.sha256 == hashlib.sha256(
        skill.path.read_bytes()
    ).hexdigest()

    full_profiles = build_full_c0_profiles([INSTITUTIONAL_ANALYST_SKILL])
    institutional = [
        profile for profile in full_profiles if profile["role_category"] == "institution"
    ]
    retail = [
        profile for profile in full_profiles if profile["role_category"] != "institution"
    ]
    assert len(institutional) == 3
    assert all(
        profile["finance_skill_names"] == [INSTITUTIONAL_ANALYST_SKILL]
        and profile["finance_skill_bundle_hash"]
        and "事件与新闻 Analyst" in profile["finance_skill_prompt"]
        for profile in institutional
    )
    assert all(profile["finance_skill_names"] == [] for profile in retail)

    selected = build_c0_profiles(
        enabled_finance_skills=[INSTITUTIONAL_ANALYST_SKILL]
    )
    assert [
        profile["full_population_agent_id"]
        for profile in selected
        if profile["finance_skill_names"]
    ] == [1]


def test_finance_skill_is_disabled_by_default():
    profiles = build_c0_profiles()
    assert all(profile["finance_skill_names"] == [] for profile in profiles)
    assert all(profile["finance_skill_prompt"] == "" for profile in profiles)
    assert all(profile["finance_skill_bundle_hash"] == "" for profile in profiles)


def test_institutional_analyst_skill_can_be_assigned_to_all_agents_for_ablation():
    profiles = build_c0_profiles(
        enabled_finance_skills=[INSTITUTIONAL_ANALYST_SKILL],
        finance_skill_scope="all_agents",
    )

    assert len(profiles) == 10
    assert all(
        profile["finance_skill_names"] == [INSTITUTIONAL_ANALYST_SKILL]
        and profile["finance_skill_scope"] == "all_agents"
        and profile["finance_skill_bundle_hash"]
        for profile in profiles
    )


@pytest.mark.parametrize(
    "names, message",
    [
        ([INSTITUTIONAL_ANALYST_SKILL, INSTITUTIONAL_ANALYST_SKILL], "duplicates"),
        (["missing-finance-skill"], "unsupported"),
        ([1], "only Skill names"),
    ],
)
def test_finance_skill_names_are_validated(names, message):
    with pytest.raises(ValueError, match=message):
        normalize_finance_skill_names(names)


def test_finance_skill_stages_are_validated():
    assert normalize_finance_skill_stage() == "all_stages"
    assert normalize_finance_skill_stage("pre_social_only") == "pre_social_only"
    with pytest.raises(ValueError, match="finance_skill_stage"):
        normalize_finance_skill_stage("social_only")
