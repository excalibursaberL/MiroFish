import asyncio
import json
import random
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.finance.s1 import S1ExperimentService
from app.finance.s1_batch import S1BatchRunner
from app.finance.source_resolver import FinanceEventSourceResolver
from app.finance.dataset import FinancialScenario
from app.services.simulation_manager import SimulationManager
from app.services.zep_entity_reader import EntityNode
from scripts.run_reddit_simulation import (
    AgentTokenUsageRecorder,
    RedditSimulationRunner,
)


def test_s1_records_provider_tokens_for_one_agent(tmp_path):
    class FakeAgent:
        def __init__(self):
            self.model_backend = SimpleNamespace(model_type="fallback-model")

        async def _aget_model_response(self, *_args, **_kwargs):
            return SimpleNamespace(
                usage_dict={
                    "prompt_tokens": 321,
                    "completion_tokens": 45,
                    "total_tokens": 366,
                },
                response=SimpleNamespace(model="deepseek-v4-flash"),
            )

    class FakeAgentGraph:
        def __init__(self, agent):
            self.agent = agent

        def get_agents(self):
            return [(0, self.agent)]

    agent = FakeAgent()
    recorder = AgentTokenUsageRecorder(
        str(tmp_path),
        [
            {
                "agent_id": 0,
                "full_population_agent_id": 1,
                "agent_class": "investor",
            }
        ],
    )
    recorder.set_context("social_interaction", 2)
    recorder.instrument(FakeAgentGraph(agent))
    recorder.instrument(FakeAgentGraph(agent))

    asyncio.run(agent._aget_model_response())

    records = [
        json.loads(line)
        for line in (tmp_path / "llm_token_usage.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(records) == 1
    assert records[0]["agent_id"] == 0
    assert records[0]["full_population_agent_id"] == 1
    assert records[0]["phase"] == "social_interaction"
    assert records[0]["round"] == 2
    assert records[0]["model"] == "deepseek-v4-flash"
    assert records[0]["prompt_tokens"] == 321
    assert records[0]["completion_tokens"] == 45
    assert records[0]["total_tokens"] == 366


def test_s1_reddit_prepare_uses_scenario_attributed_publishers_and_safe_events(
    monkeypatch, tmp_path
):
    simulation_dir = tmp_path / "simulations"
    monkeypatch.setattr(SimulationManager, "SIMULATION_DATA_DIR", str(simulation_dir))

    service = S1ExperimentService(storage_dir=tmp_path / "finance")
    manifest = service.prepare(scenario_id="SCN_009")

    assert manifest["status"] == "prepared"
    assert manifest["platform"] == "reddit"
    assert manifest["investor_agent_count"] == 10
    assert manifest["source_mode"] == "scenario"
    assert manifest["source_agent_count"] == 1
    assert manifest["agent_count_total"] == 11
    assert manifest["graph_entity_count"] == 0
    assert manifest["graph_resolved_event_count"] == 0
    assert manifest["public_feed_event_count"] == 0
    assert "social_start_round" not in manifest
    assert "minutes_per_round" not in manifest
    assert manifest["social_rounds"] == 6
    assert manifest["total_rounds"] == 6
    assert manifest["history_memory_event_count"] == 5
    assert manifest["current_public_event_count"] == 1
    assert manifest["expected_prediction_count"] == 20
    assert manifest["random_seed"] == 4004
    assert manifest["replicate_id"] == manifest["run_id"]
    assert manifest["agent_set_version"] == "n10_k10_exact_v1"
    assert manifest["sampling_method"] == "offline_exact_enumeration_k10"
    assert manifest["selected_full_population_agent_ids"] == [
        1, 3, 4, 5, 9, 11, 12, 13, 14, 17
    ]
    assert len(manifest["input_snapshot_hash"]) == 64
    assert len(manifest["prompt_hash"]) == 64
    assert manifest["files"]["agent_round_states"] == "agent_round_states.jsonl"
    assert manifest["prediction_target"]["expected_return_unit"] == "decimal"

    run_dir = tmp_path / "finance" / manifest["run_id"]
    investors = json.loads((run_dir / "profiles.json").read_text(encoding="utf-8"))
    sources = json.loads((run_dir / "source_profiles.json").read_text(encoding="utf-8"))
    history = [
        json.loads(line)
        for line in (run_dir / "history_memory.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    current = json.loads((run_dir / "current_event.json").read_text(encoding="utf-8"))

    assert [profile["user_id"] for profile in investors] == list(range(10))
    assert [profile["full_population_agent_id"] for profile in investors] == [
        1, 3, 4, 5, 9, 11, 12, 13, 14, 17
    ]
    assert all(profile["agent_class"] == "investor" for profile in investors)
    assert all("S1 社会互动实验" in profile["persona"] for profile in investors)
    assert all("只读历史记忆" in profile["persona"] for profile in investors)
    assert all(history[0]["event_id"] in profile["persona"] for profile in investors)
    assert [profile["user_id"] for profile in sources] == [10]
    assert all(profile["agent_class"] == "source" for profile in sources)
    assert sources[0]["name"] == "COMPANY_004"
    assert sources[0]["source_type"] == "company"
    assert sources[0]["source_origin"] == "scenario_text"
    assert len(history) == 5
    assert all("publisher_agent_id" not in event for event in history)
    assert current["phase"] == "current"
    assert current["round"] == 0
    assert current["publisher_agent_id"] == 10
    assert current["publisher_name"] == "COMPANY_004"

    mapping = json.loads(
        (run_dir / "entity_agent_mapping.json").read_text(encoding="utf-8")
    )
    assert mapping["publisher_account_count"] == 1
    assert mapping["publishers"][0]["source_entity_id"] == "text:COMPANY_004"
    assert len(mapping["events"]) == 6
    workbench_mapping = service.get_mapping(manifest["run_id"])
    assert len(workbench_mapping["history_memory"]) == 5
    assert workbench_mapping["current_event"]["publisher_name"] == "COMPANY_004"

    sim_config = json.loads(
        (
            simulation_dir
            / manifest["simulation_id"]
            / "simulation_config.json"
        ).read_text(encoding="utf-8")
    )
    assert len(sim_config["agent_configs"]) == 11
    assert sim_config["finance_s1"]["tracked_agent_ids"] == list(range(10))
    assert sim_config["random_seed"] == 4004
    assert sim_config["finance_s1"]["random_seed"] == 4004
    assert sim_config["finance_s1"]["source_agent_ids"] == [10]
    assert sim_config["finance_s1"]["source_mode"] == "scenario"
    assert "event_schedule" not in sim_config["finance_s1"]
    assert "social_start_round" not in sim_config["finance_s1"]
    assert len(sim_config["finance_s1"]["pre_social_interviews"]) == 10
    assert len(sim_config["event_config"]["initial_posts"]) == 1
    assert sim_config["event_config"]["initial_posts"][0]["finance_event"]["phase"] == "current"
    reddit_profiles = json.loads(
        (simulation_dir / manifest["simulation_id"] / "reddit_profiles.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(reddit_profiles) == 11
    assert [profile["user_id"] for profile in reddit_profiles] == list(range(11))
    assert all(
        {"persona", "mbti", "gender", "age", "country"} <= set(profile)
        for profile in reddit_profiles
    )

    safe_text = json.dumps(sim_config, ensure_ascii=False)
    for forbidden in (
        "actual_five_day_close_return",
        "actual_five_day_close_direction",
        "evaluator_targets",
        "normalized_close5",
        '"label"',
        '"CHANGE"',
    ):
        assert forbidden not in safe_text


def test_s1_prepare_applies_auditable_profile_id_permutation(monkeypatch, tmp_path):
    simulation_dir = tmp_path / "simulations"
    monkeypatch.setattr(SimulationManager, "SIMULATION_DATA_DIR", str(simulation_dir))
    permutation = list(reversed(range(10)))
    canonical_full_ids = [1, 3, 4, 5, 9, 11, 12, 13, 14, 17]

    service = S1ExperimentService(storage_dir=tmp_path / "finance")
    manifest = service.prepare(
        scenario_id="SCN_009",
        run_id="s1_reddit_profileperm",
        profile_id_permutation=permutation,
    )

    assert manifest["selected_full_population_agent_ids"] == canonical_full_ids
    assert manifest["profile_id_permutation"] == permutation
    assert [row["agent_id"] for row in manifest["runtime_agent_mapping"]] == list(
        range(10)
    )
    assert [
        row["canonical_agent_id"] for row in manifest["runtime_agent_mapping"]
    ] == permutation
    assert [
        row["full_population_agent_id"] for row in manifest["runtime_agent_mapping"]
    ] == [canonical_full_ids[index] for index in permutation]

    run_dir = tmp_path / "finance" / manifest["run_id"]
    profiles = json.loads((run_dir / "profiles.json").read_text(encoding="utf-8"))
    assert [profile["user_id"] for profile in profiles] == list(range(10))
    assert [profile["canonical_agent_id"] for profile in profiles] == permutation
    config = json.loads(
        (
            simulation_dir
            / manifest["simulation_id"]
            / "simulation_config.json"
        ).read_text(encoding="utf-8")
    )
    assert config["finance_s1"]["profile_id_permutation"] == permutation
    assert config["finance_s1"]["investor_agent_mapping"] == manifest[
        "runtime_agent_mapping"
    ]


def test_s1_prepare_supports_preferred_k8_candidate(monkeypatch, tmp_path):
    simulation_dir = tmp_path / "simulations"
    monkeypatch.setattr(SimulationManager, "SIMULATION_DATA_DIR", str(simulation_dir))
    selected_ids = [1, 3, 4, 5, 9, 13, 14, 17]

    service = S1ExperimentService(storage_dir=tmp_path / "finance")
    manifest = service.prepare(
        scenario_id="SCN_009",
        run_id="s1_reddit_k8candidate",
        social_rounds=6,
        selected_full_population_agent_ids=selected_ids,
        agent_set_version="n10_k8_enum_best_v1",
        sampling_method="offline_exact_enumeration_k10_v2_candidate",
    )

    assert manifest["investor_agent_count"] == 8
    assert manifest["selected_full_population_agent_ids"] == selected_ids
    assert manifest["expected_belief_snapshot_count"] == 56
    assert manifest["expected_prediction_count"] == 16
    assert [row["agent_id"] for row in manifest["runtime_agent_mapping"]] == list(
        range(8)
    )
    assert [
        row["full_population_agent_id"] for row in manifest["runtime_agent_mapping"]
    ] == selected_ids

    run_dir = tmp_path / "finance" / manifest["run_id"]
    profiles = json.loads((run_dir / "profiles.json").read_text(encoding="utf-8"))
    sources = json.loads((run_dir / "source_profiles.json").read_text(encoding="utf-8"))
    assert [profile["user_id"] for profile in profiles] == list(range(8))
    assert [source["user_id"] for source in sources] == [8]

    config = json.loads(
        (simulation_dir / manifest["simulation_id"] / "simulation_config.json")
        .read_text(encoding="utf-8")
    )
    assert config["finance_s1"]["tracked_agent_ids"] == list(range(8))
    assert config["finance_s1"]["source_agent_ids"] == [8]
    assert len(config["finance_s1"]["pre_social_interviews"]) == 8
    assert len(config["finance_s1"]["round_belief_snapshot_interviews"]) == 8


def test_s1_rejects_invalid_profile_id_permutation(tmp_path):
    service = S1ExperimentService(storage_dir=tmp_path / "finance")
    with pytest.raises(ValueError, match="every canonical Profile ID"):
        service.prepare(
            scenario_id="SCN_009",
            profile_id_permutation=[0] * 10,
        )


def test_s1_graph_mode_reuses_zep_entities_for_dynamic_source_profiles(
    monkeypatch, tmp_path
):
    simulation_dir = tmp_path / "simulations"
    monkeypatch.setattr(SimulationManager, "SIMULATION_DATA_DIR", str(simulation_dir))
    graph_entities = [
        EntityNode(
            uuid="zep-company-003",
            name="COMPANY_003",
            labels=["Entity", "ListedCompany"],
            summary="匿名上市公司",
            attributes={},
        ),
        EntityNode(
            uuid="zep-media-b",
            name="财经媒体B",
            labels=["Entity", "MediaOutlet"],
            summary="匿名财经媒体",
            attributes={},
        ),
    ]
    service = S1ExperimentService(storage_dir=tmp_path / "finance")
    monkeypatch.setattr(service, "_load_graph_entities", lambda _graph_id: graph_entities)

    manifest = service.prepare(
        scenario_id="SCN_018",
        graph_id="mirofish_test_graph",
        source_mode="graph",
    )

    assert manifest["source_mode"] == "graph"
    assert manifest["graph_id"] == "mirofish_test_graph"
    assert manifest["graph_entity_count"] == 2
    assert manifest["graph_resolved_event_count"] == 6
    assert manifest["source_agent_count"] == 2
    run_dir = tmp_path / "finance" / manifest["run_id"]
    sources = json.loads((run_dir / "source_profiles.json").read_text(encoding="utf-8"))
    assert [profile["user_id"] for profile in sources] == [10, 11]
    assert {profile["source_entity_uuid"] for profile in sources} == {
        "zep-company-003",
        "zep-media-b",
    }
    assert {profile["source_origin"] for profile in sources} == {"zep_graph"}
    current = json.loads((run_dir / "current_event.json").read_text(encoding="utf-8"))
    assert current["publisher_agent_id"] in {10, 11}
    # Date-bearing media identifiers are preserved during parsing.  The
    # fixture intentionally uses a generic media alias, so the resolver uses
    # its auditable substring match instead of pretending it was an exact ID.
    assert "+graph_alias" in current["publisher_resolution"]


def test_s1_graph_mode_requires_graph_id(tmp_path):
    service = S1ExperimentService(storage_dir=tmp_path)

    with pytest.raises(ValueError, match="requires graph_id"):
        service.prepare(scenario_id="SCN_001", source_mode="graph")


def test_s1_rounds_are_configurable_and_bounded(monkeypatch, tmp_path):
    simulation_dir = tmp_path / "simulations"
    monkeypatch.setattr(SimulationManager, "SIMULATION_DATA_DIR", str(simulation_dir))
    service = S1ExperimentService(storage_dir=tmp_path / "finance")
    manifest = service.prepare(
        scenario_id="SCN_009",
        source_mode="scenario",
        social_rounds=8,
    )
    assert manifest["social_rounds"] == 8
    assert manifest["total_rounds"] == 8
    assert "minutes_per_round" not in manifest
    config = json.loads(
        (tmp_path / "simulations" / manifest["simulation_id"] / "simulation_config.json")
        .read_text(encoding="utf-8")
    )
    assert config["finance_s1"]["social_rounds"] == 8
    assert config["time_config"]["minutes_per_round"] == 30

    with pytest.raises(ValueError, match="between 1 and 12"):
        service.prepare(scenario_id="SCN_009", source_mode="scenario", social_rounds=13)

    with pytest.raises(ValueError, match="random_seed must be an integer"):
        service.prepare(
            scenario_id="SCN_009",
            source_mode="scenario",
            random_seed=True,
        )


def test_s1_prepared_settings_update_both_config_copies(monkeypatch, tmp_path):
    simulation_dir = tmp_path / "simulations"
    monkeypatch.setattr(SimulationManager, "SIMULATION_DATA_DIR", str(simulation_dir))
    service = S1ExperimentService(storage_dir=tmp_path / "finance")
    manifest = service.prepare(scenario_id="SCN_009", source_mode="scenario")

    updated = service.update_settings(manifest["run_id"], social_rounds=12)

    assert updated["status"] == "prepared"
    assert updated["social_rounds"] == 12
    assert "minutes_per_round" not in updated
    assert updated["total_rounds"] == 12
    run_config = json.loads(
        (
            tmp_path
            / "finance"
            / manifest["run_id"]
            / "simulation_config.json"
        ).read_text(encoding="utf-8")
    )
    manager_config = json.loads(
        (
            simulation_dir
            / manifest["simulation_id"]
            / "simulation_config.json"
        ).read_text(encoding="utf-8")
    )
    for config in (run_config, manager_config):
        assert config["finance_s1"]["social_rounds"] == 12
        assert config["finance_s1"]["total_rounds"] == 12
        assert config["time_config"]["minutes_per_round"] == 30
        assert config["time_config"]["total_simulation_hours"] == 6

    manifest_path = tmp_path / "finance" / manifest["run_id"] / "manifest.json"
    completed = json.loads(manifest_path.read_text(encoding="utf-8"))
    completed["status"] = "completed"
    manifest_path.write_text(json.dumps(completed), encoding="utf-8")
    with pytest.raises(ValueError, match="only be changed while the run is prepared"):
        service.update_settings(manifest["run_id"], social_rounds=4)


def test_s1_graph_mode_rejects_publishers_missing_from_graph(monkeypatch, tmp_path):
    simulation_dir = tmp_path / "simulations"
    monkeypatch.setattr(SimulationManager, "SIMULATION_DATA_DIR", str(simulation_dir))
    service = S1ExperimentService(storage_dir=tmp_path / "finance")
    monkeypatch.setattr(service, "_load_graph_entities", lambda _graph_id: [])

    with pytest.raises(ValueError, match="not extracted into the supplied Zep graph"):
        service.prepare(
            scenario_id="SCN_009",
            graph_id="mirofish_empty_graph",
            source_mode="graph",
        )


def test_s1_graph_loader_keeps_named_nodes_without_ontology_labels(monkeypatch):
    class FakeReader:
        def get_all_nodes(self, _graph_id):
            return [
                {
                    "uuid": "untyped-company",
                    "name": "COMPANY_006",
                    "labels": [],
                    "summary": "anonymous listed company",
                    "attributes": {},
                }
            ]

    monkeypatch.setattr("app.finance.s1.ZepEntityReader", lambda: FakeReader())

    entities = S1ExperimentService._load_graph_entities("graph-test")

    assert len(entities) == 1
    assert entities[0].name == "COMPANY_006"
    assert entities[0].labels == []


def test_source_resolver_maps_legacy_issuer_alias_to_anonymous_graph_company():
    scenario = FinancialScenario(
        scenario_id="SCN_ALIAS",
        symbol="ASSET_014",
        name="COMPANY_014",
        prediction_cutoff="T+0d",
        horizon="next_5_trading_days",
        seed_events=[
            {
                "event_id": f"EVT_ALIAS_{index}",
                "text": "ST岩石公告，公司披露一项历史事项。",
            }
            for index in range(1, 6)
        ],
        current_event={
            "event_id": "EVT_ALIAS_6",
            "text": "COMPANY_014公告，公司披露当前事项。",
        },
    )
    graph_entity = EntityNode(
        uuid="zep-company-014",
        name="COMPANY_014",
        labels=["Company"],
        summary="anonymous company",
        attributes={},
    )

    sources, events, _mapping = FinanceEventSourceResolver([graph_entity]).resolve(
        scenario,
        source_agent_start=20,
        source_mode="graph",
        graph_id="graph-alias",
    )

    assert len(sources) == 1
    assert sources[0]["source_entity_uuid"] == "zep-company-014"
    assert all(event["publisher_origin"] == "zep_graph" for event in events)
    assert all(
        "graph_scenario_company_alias" in event["publisher_resolution"]
        for event in events[:5]
    )


def test_source_resolver_uses_unique_graph_type_when_display_name_differs():
    scenario = FinancialScenario(
        scenario_id="SCN_MEDIA_ALIAS",
        symbol="ASSET_MEDIA",
        name="COMPANY_999",
        prediction_cutoff="T+0d",
        horizon="next_5_trading_days",
        seed_events=[
                {
                    "event_id": f"EVT_MEDIA_{index}",
                    "text": "\u8d22\u7ecf\u5a92\u4f53B\u7535\uff0cCOMPANY_999\u516c\u544a\u3002",
                }
            for index in range(1, 6)
        ],
        current_event={"event_id": "EVT_MEDIA_6", "text": "COMPANY_999公告。"},
    )
    graph_entities = [
        EntityNode(
            uuid="zep-media",
            name="BT-13d",
            labels=["MediaOutlet"],
            summary="financial media",
            attributes={},
        ),
        EntityNode(
            uuid="zep-company",
            name="COMPANY_999",
            labels=["Company"],
            summary="anonymous company",
            attributes={},
        ),
    ]

    sources, events, _mapping = FinanceEventSourceResolver(graph_entities).resolve(
        scenario,
        source_agent_start=20,
        source_mode="graph",
        graph_id="graph-media-alias",
    )

    assert {source["source_entity_id"] for source in sources} == {
        "zep-media",
        "zep-company",
    }
    assert all("graph_unique_type" in event["publisher_resolution"] for event in events[:5])


def test_source_resolver_preserves_date_bearing_media_ids_and_audits_missing_graph_nodes():
    from app.finance.dataset import FinancialScenario

    scenario = FinancialScenario(
        scenario_id="SCN_MEDIA_DATED",
        symbol="ASSET_MEDIA_DATED",
        name="COMPANY_003",
        prediction_cutoff="T+0d",
        horizon="next_5_trading_days",
        seed_events=[
            {
                "event_id": "EVT_MEDIA_53",
                "text": "财经媒体BT-53d电，COMPANY_003公告。",
            },
            {
                "event_id": "EVT_MEDIA_9",
                "text": "财经媒体BT-9d电，COMPANY_003公告。",
            },
            *[
                {
                    "event_id": f"EVT_MEDIA_{index}",
                    "text": "COMPANY_003公告。",
                }
                for index in range(3)
            ],
        ],
        current_event={
            "event_id": "EVT_MEDIA_CURRENT",
            "text": "财经媒体BT+0d电，COMPANY_003公告。",
        },
    )
    graph_entities = [
        EntityNode(
            uuid="zep-company",
            name="COMPANY_003",
            labels=["Company"],
            summary="anonymous company",
            attributes={},
        ),
        EntityNode(
            uuid="zep-media-53",
            name="BT-53d",
            labels=["MediaOutlet"],
            summary="financial media",
            attributes={},
        ),
        EntityNode(
            uuid="zep-media-41",
            name="BT-41d",
            labels=["MediaOutlet"],
            summary="financial media",
            attributes={},
        ),
        EntityNode(
            uuid="zep-media-34",
            name="BT-34d",
            labels=["MediaOutlet"],
            summary="financial media",
            attributes={},
        ),
    ]

    sources, events, _mapping = FinanceEventSourceResolver(graph_entities).resolve(
        scenario,
        source_agent_start=20,
        source_mode="graph",
        graph_id="graph-media-dated",
    )

    by_event = {event["event_id"]: event for event in events}
    assert by_event["EVT_MEDIA_53"]["publisher_name"] == "BT-53d"
    assert by_event["EVT_MEDIA_53"]["publisher_origin"] == "zep_graph"
    assert "财经媒体BT-9d" in {source["name"] for source in sources}
    assert "财经媒体BT+0d" in {source["name"] for source in sources}
    assert "graph_missing_publisher_text_fallback" in by_event["EVT_MEDIA_9"]["publisher_resolution"]
    assert "graph_missing_publisher_text_fallback" in by_event["EVT_MEDIA_CURRENT"]["publisher_resolution"]


def test_s1_batch_summary_lock_does_not_raise(monkeypatch, tmp_path):
    runner = S1BatchRunner(storage_dir=tmp_path)
    manifest = {}
    monkeypatch.setattr(
        runner,
        "_write_summary",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(PermissionError("locked")),
    )

    assert runner._try_write_summary(tmp_path, manifest) is False
    assert "locked" in manifest["summary_write_error"]


def test_s1_batch_run_sync_keeps_worker_in_current_process(monkeypatch, tmp_path):
    runner = S1BatchRunner(storage_dir=tmp_path)
    batch_id = "s1_batch_sync123"
    batch_dir = tmp_path / batch_id
    batch_dir.mkdir()
    runner._write_json(
        batch_dir / "manifest.json",
        {"batch_id": batch_id, "status": "prepared", "runs": []},
    )

    def fake_execute(received_batch_id):
        assert received_batch_id == batch_id
        manifest = runner._read_json(batch_dir / "manifest.json")
        assert manifest["status"] == "queued"
        manifest["status"] = "completed"
        runner._write_json(batch_dir / "manifest.json", manifest)

    monkeypatch.setattr(runner, "_execute", fake_execute)

    result = runner.run_sync(batch_id)

    assert result["status"] == "completed"


def test_source_resolver_uses_one_public_feed_only_when_attribution_is_missing():
    from app.finance.dataset import FinancialScenario

    scenario = FinancialScenario(
        scenario_id="SCN_TEST",
        symbol="ASSET_TEST",
        name="COMPANY_TEST",
        prediction_cutoff="T+0d",
        horizon="next_5_trading_days",
        seed_events=[
            {"event_id": f"EVT_TEST_{index}", "text": "市场出现一条匿名公开信息。"}
            for index in range(5)
        ],
        current_event={"event_id": "EVT_TEST_5", "text": "又一条无法归因的信息。"},
    )

    sources, events, mapping = FinanceEventSourceResolver().resolve(
        scenario,
        source_agent_start=20,
        source_mode="scenario",
    )

    assert len(sources) == 1
    assert sources[0]["source_type"] == "public_feed"
    assert sources[0]["user_id"] == 20
    assert {event["publisher_agent_id"] for event in events} == {20}
    assert {event["publisher_resolution"] for event in events} == {
        "public_feed_fallback"
    }
    assert mapping["publisher_account_count"] == 1


def test_s1_seed_document_contains_only_safe_event_fields(tmp_path):
    service = S1ExperimentService(storage_dir=tmp_path)
    document = service.get_scenario_seed_document("SCN_009")
    assert len(document["events"]) == 6
    assert {event["phase"] for event in document["events"]} == {"history", "current"}
    text = json.dumps(document, ensure_ascii=False)
    for forbidden in (
        "actual_five_day_close_return",
        "evaluator_targets",
        '"label"',
        '"CHANGE"',
        "stock_factors",
    ):
        assert forbidden not in text


def test_s1_forecast_prompt_defines_social_boundary_and_neutral_band(tmp_path):
    service = S1ExperimentService(storage_dir=tmp_path)
    from app.finance.dataset import FinancialDatasetLoader

    prompt = service.build_forecast_prompt(
        FinancialDatasetLoader().load(scenario_ids=["SCN_001"])[0]
    )

    assert "S1 社会互动实验" in prompt
    assert "其他投资者" in prompt
    assert "-1.7% <= R5 <= +1.7%" in prompt
    assert "不要引用未来价格或评测答案" in prompt
    assert "JSON object" in prompt

    pre_prompt = service.build_forecast_prompt(
        FinancialDatasetLoader().load(scenario_ids=["SCN_001"])[0],
        stage="pre_social",
    )
    snapshot_prompt = service.build_belief_snapshot_prompt(
        FinancialDatasetLoader().load(scenario_ids=["SCN_001"])[0]
    )
    assert pre_prompt == snapshot_prompt
    assert "私有信念测量" in pre_prompt
    assert "此刻在模拟中实际可见的信息" in pre_prompt
    assert "社会互动开始前" not in pre_prompt
    assert "interaction round" not in pre_prompt


def test_s1_prepare_uses_identical_pre_and_round_belief_prompts(
    monkeypatch, tmp_path
):
    simulation_dir = tmp_path / "simulations"
    monkeypatch.setattr(SimulationManager, "SIMULATION_DATA_DIR", str(simulation_dir))
    service = S1ExperimentService(storage_dir=tmp_path / "finance")
    manifest = service.prepare(
        scenario_id="SCN_001",
        source_mode="scenario",
        social_rounds=2,
    )
    config = json.loads(
        (
            simulation_dir
            / manifest["simulation_id"]
            / "simulation_config.json"
        ).read_text(encoding="utf-8")
    )
    pre = config["finance_s1"]["pre_social_interviews"]
    snapshots = config["finance_s1"]["round_belief_snapshot_interviews"]

    assert {item["prompt"] for item in pre} == {
        item["prompt"] for item in snapshots
    }
    assert {item["retry_prompt"] for item in pre} == {
        item["retry_prompt"] for item in snapshots
    }
    assert "__ROUND_NUMBER__" not in pre[0]["prompt"]
    assert config["finance_s1"]["prompt_version"] == (
        "finance_forecast_s1_v3_unified_belief"
    )


def test_s1_prediction_changes_and_group_metrics_are_paired_by_agent(tmp_path):
    service = S1ExperimentService(storage_dir=tmp_path)
    common = {
        "scenario_id": "SCN_001",
        "agent_role_label": "测试投资者",
        "status": "ok",
        "social_action_count": 0,
        "social_post_count": 0,
        "social_comment_count": 0,
        "evidence_event_ids": ["EVT_0001"],
    }
    pre = [{
        **common,
        "agent_id": 0,
        "direction": "up",
        "up_probability": 0.6,
        "neutral_probability": 0.3,
        "down_probability": 0.1,
        "expected_return": 0.03,
        "confidence": 0.6,
    }]
    post = [{
        **common,
        "agent_id": 0,
        "direction": "down",
        "up_probability": 0.1,
        "neutral_probability": 0.2,
        "down_probability": 0.7,
        "expected_return": -0.02,
        "confidence": 0.8,
        "social_action_count": 4,
        "social_post_count": 1,
        "social_comment_count": 2,
    }]

    changes = service._build_prediction_changes(pre, post)
    assert len(changes) == 1
    assert changes[0]["direction_changed"] is True
    assert changes[0]["expected_return_delta"] == pytest.approx(-0.05)
    assert changes[0]["confidence_delta"] == pytest.approx(0.2)
    assert changes[0]["distribution_js_divergence"] > 0
    summary = service._prediction_stage_metrics(post)
    assert summary["consensus_rate"] == 1.0
    assert summary["direction_counts"] == {"up": 0, "neutral": 0, "down": 1}


def test_s1_batch_uses_only_completed_graph_manifest_entries(tmp_path):
    graph_manifest = tmp_path / "zep_graphs_manifest.json"
    graph_manifest.write_text(
        json.dumps({
            "scenarios": [
                {
                    "scenario_id": "SCN_008",
                    "project_id": "proj_test",
                    "graph_id": "mirofish_graph_test",
                    "status": "completed",
                },
                {
                    "scenario_id": "SCN_009",
                    "graph_id": "mirofish_graph_incomplete",
                    "status": "failed",
                },
            ]
        }),
        encoding="utf-8",
    )
    runner = S1BatchRunner(storage_dir=tmp_path / "finance")

    manifest = runner.prepare(
        social_rounds=6, graph_manifest_path=graph_manifest
    )

    assert manifest["status"] == "prepared"
    assert manifest["random_seed"] == 4004
    assert manifest["scenario_count"] == 1
    assert manifest["runs"][0]["scenario_id"] == "SCN_008"
    assert manifest["runs"][0]["graph_id"] == "mirofish_graph_test"
    assert (
        tmp_path / "finance" / manifest["batch_id"] / "scenario_summary.csv"
    ).exists()


def test_s1_default_graph_manifest_covers_all_eighteen_scenarios(tmp_path):
    manifest = S1BatchRunner(storage_dir=tmp_path / "finance").prepare()

    assert manifest["scenario_count"] == 18
    assert [item["scenario_id"] for item in manifest["runs"]] == [
        f"SCN_{index:03d}" for index in range(1, 19)
    ]


def test_s1_exports_complete_oasis_trace_with_social_rounds(monkeypatch, tmp_path):
    simulation_root = tmp_path / "simulations"
    monkeypatch.setattr(SimulationManager, "SIMULATION_DATA_DIR", str(simulation_root))
    service = S1ExperimentService(storage_dir=tmp_path / "finance")
    run_id = "s1_reddit_trace_test"
    run_dir = service._run_dir(run_id)
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({
            "run_id": run_id,
            "simulation_id": "finance_trace_test",
            "scenario_id": "SCN_001",
            "social_rounds": 2,
        }),
        encoding="utf-8",
    )
    sim_dir = simulation_root / "finance_trace_test"
    (sim_dir / "reddit").mkdir(parents=True)
    (sim_dir / "reddit" / "actions.jsonl").write_text(
        "\n".join([
            json.dumps({"event_type": "round_start", "round": 1, "timestamp": "2026-01-01T00:00:01"}),
            json.dumps({"event_type": "round_end", "round": 1, "timestamp": "2026-01-01T00:00:02", "trace_start_rowid": 1, "trace_end_rowid": 2}),
            json.dumps({"event_type": "round_start", "round": 2, "timestamp": "2026-01-01T00:00:03"}),
            json.dumps({"event_type": "round_end", "round": 2, "timestamp": "2026-01-01T00:00:04", "trace_start_rowid": 2, "trace_end_rowid": 4}),
        ]),
        encoding="utf-8",
    )
    with sqlite3.connect(sim_dir / "reddit_simulation.db") as connection:
        connection.execute(
            "CREATE TABLE trace (user_id INTEGER, created_at DATETIME, action TEXT, info TEXT)"
        )
        connection.execute(
            "CREATE TABLE user (user_id INTEGER PRIMARY KEY, agent_id INTEGER)"
        )
        connection.execute(
            "CREATE TABLE post (post_id INTEGER PRIMARY KEY, user_id INTEGER)"
        )
        connection.execute(
            "CREATE TABLE comment (comment_id INTEGER PRIMARY KEY, post_id INTEGER, user_id INTEGER)"
        )
        connection.executemany(
            "INSERT INTO user VALUES (?, ?)", [(20, 20), (1, 0)]
        )
        connection.execute("INSERT INTO post VALUES (?, ?)", (1, 20))
        connection.execute("INSERT INTO comment VALUES (?, ?, ?)", (1, 1, 1))
        connection.executemany(
            "INSERT INTO trace VALUES (?, ?, ?, ?)",
            [
                (20, "2026-01-01 00:00:00.500000", "create_post", '{"content":"current","post_id":1}'),
                (0, "2026-01-01 00:00:01.500000", "refresh", '{"posts":[]}'),
                (0, "2026-01-01 00:00:03.400000", "create_comment", '{"content":"reply","comment_id":1}'),
                (0, "2026-01-01 00:00:03.500000", "like_post", '{"post_id":1}'),
                (0, "2026-01-01 00:00:04.500000", "interview", '{"prompt":"private"}'),
            ],
        )

    records = service._export_social_actions(run_dir)

    assert [(item["agent_id"], item["round"], item["action_type"]) for item in records] == [
        (20, 0, "create_post"),
        (0, 1, "refresh"),
        (0, 2, "create_comment"),
        (0, 2, "like_post"),
    ]
    assert records[2]["action_args"]["content"] == "reply"
    assert records[0]["round_source"] == "source_initialization"
    assert records[1]["round_source"] == "oasis_trace_rowid_range"
    assert records[-1]["target_agent_id"] == 20
    assert records[2]["target_comment_id"] is None
    assert records[2]["target_post_id"] == 1
    assert records[2]["target_agent_id"] == 20
    counts = service._social_counts(run_dir)
    assert counts[0]["total"] == 3
    assert counts[0]["comment"] == 1
    assert counts[0]["like"] == 1


def test_s1_run_id_rejects_path_traversal(tmp_path):
    service = S1ExperimentService(storage_dir=tmp_path)

    with pytest.raises(ValueError, match="invalid S1"):
        service.get_status("../s1_reddit_escape")


def test_s1_round_belief_snapshots_keep_round_alignment_and_missing_rows(
    monkeypatch, tmp_path
):
    simulation_dir = tmp_path / "simulations"
    monkeypatch.setattr(SimulationManager, "SIMULATION_DATA_DIR", str(simulation_dir))
    service = S1ExperimentService(storage_dir=tmp_path / "finance")
    manifest = service.prepare(
        scenario_id="SCN_001", source_mode="scenario", social_rounds=2
    )
    run_dir = tmp_path / "finance" / manifest["run_id"]
    profiles = json.loads((run_dir / "profiles.json").read_text(encoding="utf-8"))
    from app.finance.dataset import FinancialDatasetLoader

    loaded_scenario = FinancialDatasetLoader().load(scenario_ids=["SCN_001"])[0]
    pre = [
        {
            "scenario_id": "SCN_001",
            "agent_id": int(profile["user_id"]),
            "status": "ok",
            "direction": "neutral",
            "up_probability": 0.2,
            "neutral_probability": 0.6,
            "down_probability": 0.2,
            "expected_return": 0.0,
            "confidence": 0.5,
            "evidence_event_ids": [],
        }
        for profile in profiles
    ]
    raw = json.dumps(
        {
            "direction": "up",
            "up_probability": 0.6,
            "neutral_probability": 0.25,
            "down_probability": 0.15,
            "expected_return": 0.02,
            "confidence": 0.6,
            "evidence_event_ids": ["EVT_0001"],
            "reason": "round snapshot",
        },
        ensure_ascii=False,
    )
    simulation_run_dir = simulation_dir / manifest["simulation_id"]
    simulation_run_dir.mkdir(parents=True, exist_ok=True)
    (simulation_run_dir / "round_belief_interviews.jsonl").write_text(
        json.dumps(
            {
                "round": 1,
                "success": True,
                "attempt_count": 1,
                "results": {
                    str(profile["user_id"]): {"response": raw}
                    for profile in profiles
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    snapshots = service._parse_round_belief_snapshots(
        run_dir, manifest, loaded_scenario, profiles, pre
    )

    assert len(snapshots) == 10 * 3
    assert {item["round"] for item in snapshots} == {0, 1, 2}
    assert sum(item["status"] == "ok" for item in snapshots if item["round"] == 0) == 10
    assert sum(item["status"] == "ok" for item in snapshots if item["round"] == 1) == 10
    assert sum(item["status"] == "missing" for item in snapshots if item["round"] == 2) == 10
    assert {item["full_population_agent_id"] for item in snapshots} == {
        1, 3, 4, 5, 9, 11, 12, 13, 14, 17
    }
    assert all(item["snapshot_source"] == "private_round_interview" for item in snapshots if item["round"] == 1)

    post = service._post_predictions_from_final_snapshot(
        run_dir, manifest, profiles, snapshots
    )
    assert len(post) == 10
    assert all(item["prediction_stage"] == "post_social" for item in post)
    assert all(
        item["prediction_source"] == "final_round_belief_snapshot"
        for item in post
    )
    assert all(item["measurement_round"] == 2 for item in post)
    assert all(item["status"] == "missing" for item in post)


def test_reddit_runner_applies_and_audits_local_random_seed(tmp_path):
    config_path = tmp_path / "simulation_config.json"
    config_path.write_text(
        json.dumps({"random_seed": 12345, "agent_configs": []}),
        encoding="utf-8",
    )

    runner = RedditSimulationRunner(str(config_path), wait_for_commands=False)
    first_value = random.random()
    runner._apply_random_seed(12345)

    assert random.random() == first_value
    state = json.loads(
        (tmp_path / "random_seed_state.json").read_text(encoding="utf-8")
    )
    assert state["random_seed"] == 12345
    assert state["python_random_seeded"] is True
    assert state["numpy_seeded"] is True
    assert state["llm_provider_seeded"] is False


def test_reddit_finance_forecast_precheck_rejects_zero_sum_probabilities():
    assert RedditSimulationRunner._is_valid_finance_forecast_response(
        {
            "response": json.dumps(
                {
                    "up_probability": 0.0,
                    "neutral_probability": 0.0,
                    "down_probability": 0.0,
                }
            )
        }
    ) is False
    assert RedditSimulationRunner._is_valid_finance_forecast_response(
        {
            "response": json.dumps(
                {
                    "up_probability": 0.2,
                    "neutral_probability": 0.3,
                    "down_probability": 0.5,
                }
            )
        }
    ) is True


def test_s1_exposure_edges_preserve_content_and_auditable_stance(tmp_path):
    service = S1ExperimentService(storage_dir=tmp_path / "finance")
    run_dir = tmp_path / "finance" / "s1_reddit_exposure_test"
    run_dir.mkdir(parents=True)
    manifest = {
        "run_id": "s1_reddit_exposure_test",
        "replicate_id": "rep-1",
        "agent_set_version": "n20_full",
        "sampling_method": "full",
        "data_split": "calibration",
        "input_snapshot_hash": "hash",
        "prompt_version": "v1",
        "prompt_hash": "prompt",
        "random_seed": 1,
        "scenario_id": "SCN_001",
    }
    actions = [
        {
            "trace_id": 1,
            "agent_class": "source",
            "agent_id": 20,
            "round": 0,
            "timestamp": "2026-01-01T00:00:00",
            "action_type": "create_post",
            "post_id": 1,
            "action_args": {"content": "这是一条明确的利好消息"},
        },
        {
            "trace_id": 2,
            "agent_class": "investor",
            "agent_id": 0,
            "round": 1,
            "timestamp": "2026-01-01T00:01:00",
            "action_type": "refresh",
            "visible_post_ids": [1],
        },
        {
            "trace_id": 3,
            "agent_class": "investor",
            "agent_id": 0,
            "round": 1,
            "timestamp": "2026-01-01T00:02:00",
            "action_type": "like_post",
            "target_post_id": 1,
            "post_id": 1,
        },
    ]

    edges = service._build_exposure_edges(run_dir, manifest, actions)

    assert len(edges) == 2
    assert {edge["exposure_type"] for edge in edges} == {"feed_visible", "direct_action"}
    assert all(edge["viewer_agent_id"] == 0 for edge in edges)
    assert all(edge["author_agent_id"] == 20 for edge in edges)
    assert all(edge["content_stance"] == "informational" for edge in edges)
    assert all(edge["stance_source"] == "source_event" for edge in edges)
    assert all(edge["first_seen_round"] == 1 for edge in edges)
    feed_edge = next(edge for edge in edges if edge["exposure_type"] == "feed_visible")
    direct_edge = next(edge for edge in edges if edge["exposure_type"] == "direct_action")
    assert feed_edge["edge_layer"] == "exposure_opportunity"
    assert feed_edge["interaction_sign"] is None
    assert feed_edge["is_first_exposure"] is True
    assert feed_edge["is_self_authored"] is False
    assert direct_edge["edge_layer"] == "direct_interaction"
    assert direct_edge["interaction_sign"] == 1


def test_s1_interaction_edges_keep_typed_signed_relations(tmp_path):
    service = S1ExperimentService(storage_dir=tmp_path / "finance")
    run_dir = tmp_path / "finance" / "s1_reddit_interaction_test"
    run_dir.mkdir(parents=True)
    manifest = {
        "run_id": "s1_reddit_interaction_test",
        "replicate_id": "rep-1",
        "agent_set_version": "n10",
        "sampling_method": "full",
        "data_split": "calibration",
        "input_snapshot_hash": "hash",
        "prompt_version": "v1",
        "prompt_hash": "prompt",
        "random_seed": 1,
        "scenario_id": "SCN_001",
    }
    actions = [
        {
            "trace_id": 3,
            "agent_class": "investor",
            "agent_id": 0,
            "round": 1,
            "timestamp": "2026-01-01T00:02:00",
            "action_type": "dislike_comment",
            "target_agent_id": 2,
            "target_comment_id": 7,
            "action_args": {"comment_id": 7},
        },
        {
            "trace_id": 4,
            "agent_class": "investor",
            "agent_id": 0,
            "round": 1,
            "timestamp": "2026-01-01T00:03:00",
            "action_type": "follow",
            "target_agent_id": None,
            "action_args": {"follow_id": 3},
        },
        {
            "trace_id": 5,
            "agent_class": "investor",
            "agent_id": 0,
            "round": 1,
            "timestamp": "2026-01-01T00:04:00",
            "action_type": "create_comment",
            "target_agent_id": None,
            "action_args": {"content": "top-level"},
        },
    ]
    exposure_edges = [
        {
            "trace_id": 3,
            "exposure_type": "direct_action",
            "author_agent_id": 2,
        }
    ]

    rows = service._build_interaction_edges(
        run_dir, manifest, actions, exposure_edges
    )

    assert len(rows) == 2
    assert [(row["action_type"], row["interaction_sign"]) for row in rows] == [
        ("dislike_comment", -1),
        ("follow", 1),
    ]
    assert rows[0]["target_agent_id"] == 2
    assert rows[0]["content_type"] == "comment"
    assert rows[1]["target_agent_id"] == 3
    assert rows[1]["interaction_kind"] == "agent_relation"
    assert service.get_interaction_edges("s1_reddit_interaction_test") == rows
