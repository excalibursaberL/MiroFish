from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_round_selection_dataset.py"
)
SPEC = importlib.util.spec_from_file_location("round_selection_dataset", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_external_social_exposures_excludes_self_and_sources() -> None:
    rows = [
        {"author_class": "investor", "is_self_authored": False, "id": "peer"},
        {"author_class": "investor", "is_self_authored": True, "id": "self"},
        {"author_class": "source", "is_self_authored": False, "id": "source"},
    ]

    assert [
        row["id"] for row in MODULE.external_social_exposures(rows)
    ] == ["peer"]


def test_derive_interaction_edges_recovers_follow_target_and_sign() -> None:
    actions = [
        {
            "scenario_id": "SCN_001",
            "run_id": "run-1",
            "trace_id": 10,
            "round": 2,
            "timestamp": "2026-01-01T00:00:00",
            "agent_id": 1,
            "agent_class": "investor",
            "action_type": "follow",
            "target_agent_id": None,
            "action_args": {"follow_id": 4},
        }
    ]

    result = MODULE.derive_interaction_edges(actions, [])

    assert len(result) == 1
    assert result[0]["actor_agent_id"] == 1
    assert result[0]["target_agent_id"] == 4
    assert result[0]["interaction_sign"] == 1
    assert result[0]["interaction_kind"] == "agent_relation"


def test_comment_parent_is_recovered_as_typed_direct_interaction() -> None:
    actions = [
        {
            "scenario_id": "SCN_001",
            "run_id": "run-1",
            "trace_id": 11,
            "round": 2,
            "timestamp": "2026-01-01T00:00:00",
            "agent_id": 1,
            "agent_class": "investor",
            "action_type": "create_comment",
            "comment_id": 7,
            "target_agent_id": None,
            "target_post_id": None,
            "target_comment_id": None,
            "action_args": {"content": "reply", "comment_id": 7},
        }
    ]

    enriched = MODULE.enrich_action_targets(
        actions,
        post_owners={3: 12},
        comment_owners={7: 1},
        comment_posts={7: 3},
    )
    exposures = MODULE.restore_direct_content_exposures(enriched, [])
    interactions = MODULE.derive_interaction_edges(enriched, exposures)

    assert enriched[0]["target_post_id"] == 3
    assert enriched[0]["target_agent_id"] == 12
    assert len(exposures) == 1
    assert exposures[0]["edge_layer"] == "direct_interaction"
    assert exposures[0]["interaction_sign"] == 0
    assert len(interactions) == 1
    assert interactions[0]["interaction_kind"] == "comment"
    assert interactions[0]["target_agent_id"] == 12
    assert interactions[0]["content_type"] == "post"
    assert interactions[0]["content_id"] == 3
