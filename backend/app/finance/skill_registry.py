"""Load and assign auditable finance Skills to eligible investor Profiles."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


SKILLS_DIR = Path(__file__).resolve().with_name("skills")
SKILL_NAME_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
INSTITUTIONAL_ANALYST_SKILL = "ashare-institutional-analyst"
FINANCE_SKILL_SCOPE_ELIGIBLE_ROLES = "eligible_roles"
FINANCE_SKILL_SCOPE_ALL_AGENTS = "all_agents"
FINANCE_SKILL_STAGE_ALL = "all_stages"
FINANCE_SKILL_STAGE_PRE_SOCIAL_ONLY = "pre_social_only"


@dataclass(frozen=True)
class FinanceSkill:
    name: str
    description: str
    body: str
    sha256: str
    path: Path

    def to_manifest_dict(self) -> Dict[str, str]:
        return {
            "name": self.name,
            "description": self.description,
            "sha256": self.sha256,
            "path": str(self.path),
        }


SKILL_ROLE_CATEGORIES = {
    INSTITUTIONAL_ANALYST_SKILL: frozenset({"institution"}),
}


def _parse_skill_document(path: Path) -> FinanceSkill:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if len(lines) < 4 or lines[0].strip() != "---":
        raise ValueError(f"finance Skill is missing YAML frontmatter: {path}")
    try:
        end = next(
            index for index, line in enumerate(lines[1:], start=1)
            if line.strip() == "---"
        )
    except StopIteration as exc:
        raise ValueError(f"finance Skill frontmatter is not closed: {path}") from exc

    metadata: Dict[str, str] = {}
    for line in lines[1:end]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, separator, value = line.partition(":")
        if not separator:
            raise ValueError(f"invalid finance Skill frontmatter line: {line}")
        metadata[key.strip()] = value.strip().strip('"').strip("'")

    name = metadata.get("name", "")
    description = metadata.get("description", "")
    if not SKILL_NAME_PATTERN.fullmatch(name) or not description:
        raise ValueError(f"invalid finance Skill metadata: {path}")
    body = "\n".join(lines[end + 1 :]).strip()
    if not body:
        raise ValueError(f"finance Skill body is empty: {path}")
    return FinanceSkill(
        name=name,
        description=description,
        body=body,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        path=path,
    )


def load_finance_skill(name: str) -> FinanceSkill:
    if not isinstance(name, str) or not SKILL_NAME_PATTERN.fullmatch(name):
        raise ValueError(f"invalid finance Skill name: {name!r}")
    if name not in SKILL_ROLE_CATEGORIES:
        raise ValueError(f"unsupported finance Skill: {name}")
    path = SKILLS_DIR / name / "SKILL.md"
    if not path.is_file():
        raise FileNotFoundError(path)
    skill = _parse_skill_document(path)
    if skill.name != name:
        raise ValueError(
            f"finance Skill folder/name mismatch: folder={name}, metadata={skill.name}"
        )
    return skill


def normalize_finance_skill_names(
    names: Optional[Sequence[str]] = None,
) -> tuple[str, ...]:
    if names is None:
        return ()
    if isinstance(names, (str, bytes)):
        raise ValueError("enabled_finance_skills must be a list of Skill names")
    values = tuple(names)
    if any(not isinstance(name, str) for name in values):
        raise ValueError("enabled_finance_skills must contain only Skill names")
    if len(values) != len(set(values)):
        raise ValueError("enabled_finance_skills must not contain duplicates")
    for name in values:
        load_finance_skill(name)
    return values


def normalize_finance_skill_scope(scope: Optional[str] = None) -> str:
    """Normalize the assignment policy without changing the default.

    ``eligible_roles`` preserves the original role-gated behavior.  The
    ``all_agents`` option is an explicit ablation for experiments that want
    every investor to receive the same Skill bundle.
    """
    value = FINANCE_SKILL_SCOPE_ELIGIBLE_ROLES if scope is None else str(scope)
    if value not in {
        FINANCE_SKILL_SCOPE_ELIGIBLE_ROLES,
        FINANCE_SKILL_SCOPE_ALL_AGENTS,
    }:
        raise ValueError(
            "finance_skill_scope must be 'eligible_roles' or 'all_agents'"
        )
    return value


def normalize_finance_skill_stage(stage: Optional[str] = None) -> str:
    """Normalize when an assigned Skill is rendered into S1 prompts.

    ``all_stages`` preserves the existing behavior.  ``pre_social_only`` is
    an explicit ablation: the Skill is added only to the initial private
    forecast interview; the OASIS persona and all round belief snapshots use
    the base prompt.
    """
    value = FINANCE_SKILL_STAGE_ALL if stage is None else str(stage)
    if value not in {
        FINANCE_SKILL_STAGE_ALL,
        FINANCE_SKILL_STAGE_PRE_SOCIAL_ONLY,
    }:
        raise ValueError(
            "finance_skill_stage must be 'all_stages' or 'pre_social_only'"
        )
    return value


def assign_finance_skills(
    profile: Dict[str, Any],
    enabled_skill_names: Optional[Sequence[str]] = None,
    finance_skill_scope: Optional[str] = None,
) -> Dict[str, Any]:
    names = normalize_finance_skill_names(enabled_skill_names)
    scope = normalize_finance_skill_scope(finance_skill_scope)
    assigned = [
        load_finance_skill(name)
        for name in names
        if scope == FINANCE_SKILL_SCOPE_ALL_AGENTS
        or str(profile.get("role_category", "")) in SKILL_ROLE_CATEGORIES[name]
    ]
    result = dict(profile)
    result["finance_skill_names"] = [skill.name for skill in assigned]
    result["finance_skill_hashes"] = {
        skill.name: skill.sha256 for skill in assigned
    }
    result["finance_skill_prompt"] = "\n\n".join(
        skill.body for skill in assigned
    )
    bundle_payload = "\n".join(
        f"{skill.name}:{skill.sha256}" for skill in assigned
    )
    result["finance_skill_bundle_hash"] = (
        hashlib.sha256(bundle_payload.encode("utf-8")).hexdigest()
        if bundle_payload else ""
    )
    result["finance_skill_scope"] = scope
    return result


def finance_skill_manifest(
    names: Optional[Sequence[str]] = None,
) -> list[Dict[str, str]]:
    return [
        load_finance_skill(name).to_manifest_dict()
        for name in normalize_finance_skill_names(names)
    ]
