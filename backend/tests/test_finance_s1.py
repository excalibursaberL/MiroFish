import json
import sqlite3
from pathlib import Path

import pytest

from app.finance.s1 import S1ExperimentService
from app.finance.s1_batch import S1BatchRunner
from app.finance.source_resolver import FinanceEventSourceResolver
from app.services.simulation_manager import SimulationManager
from app.services.zep_entity_reader import EntityNode


def test_s1_reddit_prepare_uses_scenario_attributed_publishers_and_safe_events(
    monkeypatch, tmp_path
):
    simulation_dir = tmp_path / "simulations"
    monkeypatch.setattr(SimulationManager, "SIMULATION_DATA_DIR", str(simulation_dir))

    service = S1ExperimentService(storage_dir=tmp_path / "finance")
    manifest = service.prepare(scenario_id="SCN_009")

    assert manifest["status"] == "prepared"
    assert manifest["platform"] == "reddit"
    assert manifest["investor_agent_count"] == 20
    assert manifest["source_mode"] == "scenario"
    assert manifest["source_agent_count"] == 1
    assert manifest["agent_count_total"] == 21
    assert manifest["graph_entity_count"] == 0
    assert manifest["graph_resolved_event_count"] == 0
    assert manifest["public_feed_event_count"] == 0
    assert "social_start_round" not in manifest
    assert "minutes_per_round" not in manifest
    assert manifest["social_rounds"] == 6
    assert manifest["total_rounds"] == 6
    assert manifest["history_memory_event_count"] == 5
    assert manifest["current_public_event_count"] == 1
    assert manifest["expected_prediction_count"] == 40

    run_dir = tmp_path / "finance" / manifest["run_id"]
    investors = json.loads((run_dir / "profiles.json").read_text(encoding="utf-8"))
    sources = json.loads((run_dir / "source_profiles.json").read_text(encoding="utf-8"))
    history = [
        json.loads(line)
        for line in (run_dir / "history_memory.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    current = json.loads((run_dir / "current_event.json").read_text(encoding="utf-8"))

    assert [profile["user_id"] for profile in investors] == list(range(20))
    assert all(profile["agent_class"] == "investor" for profile in investors)
    assert all("S1 社会互动实验" in profile["persona"] for profile in investors)
    assert all("只读历史记忆" in profile["persona"] for profile in investors)
    assert all(history[0]["event_id"] in profile["persona"] for profile in investors)
    assert [profile["user_id"] for profile in sources] == [20]
    assert all(profile["agent_class"] == "source" for profile in sources)
    assert sources[0]["name"] == "COMPANY_004"
    assert sources[0]["source_type"] == "company"
    assert sources[0]["source_origin"] == "scenario_text"
    assert len(history) == 5
    assert all("publisher_agent_id" not in event for event in history)
    assert current["phase"] == "current"
    assert current["round"] == 0
    assert current["publisher_agent_id"] == 20
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
    assert len(sim_config["agent_configs"]) == 21
    assert sim_config["finance_s1"]["tracked_agent_ids"] == list(range(20))
    assert sim_config["finance_s1"]["source_agent_ids"] == [20]
    assert sim_config["finance_s1"]["source_mode"] == "scenario"
    assert "event_schedule" not in sim_config["finance_s1"]
    assert "social_start_round" not in sim_config["finance_s1"]
    assert len(sim_config["finance_s1"]["pre_social_interviews"]) == 20
    assert len(sim_config["event_config"]["initial_posts"]) == 1
    assert sim_config["event_config"]["initial_posts"][0]["finance_event"]["phase"] == "current"
    reddit_profiles = json.loads(
        (simulation_dir / manifest["simulation_id"] / "reddit_profiles.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(reddit_profiles) == 21
    assert [profile["user_id"] for profile in reddit_profiles] == list(range(21))
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
    assert [profile["user_id"] for profile in sources] == [20, 21]
    assert {profile["source_entity_uuid"] for profile in sources} == {
        "zep-company-003",
        "zep-media-b",
    }
    assert {profile["source_origin"] for profile in sources} == {"zep_graph"}
    current = json.loads((run_dir / "current_event.json").read_text(encoding="utf-8"))
    assert current["publisher_agent_id"] in {20, 21}
    assert "+graph_exact" in current["publisher_resolution"]


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
    assert "还没有看到其他投资者的观点" in pre_prompt
    assert "社会互动开始前" in pre_prompt


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
            json.dumps({"event_type": "round_end", "round": 1, "timestamp": "2026-01-01T00:00:02"}),
            json.dumps({"event_type": "round_start", "round": 2, "timestamp": "2026-01-01T00:00:03"}),
            json.dumps({"event_type": "round_end", "round": 2, "timestamp": "2026-01-01T00:00:04"}),
        ]),
        encoding="utf-8",
    )
    with sqlite3.connect(sim_dir / "reddit_simulation.db") as connection:
        connection.execute(
            "CREATE TABLE trace (user_id INTEGER, created_at DATETIME, action TEXT, info TEXT)"
        )
        connection.executemany(
            "INSERT INTO trace VALUES (?, ?, ?, ?)",
            [
                (20, "2026-01-01 00:00:00.500000", "create_post", '{"content":"current"}'),
                (0, "2026-01-01 00:00:01.500000", "refresh", '{"posts":[]}'),
                (0, "2026-01-01 00:00:03.500000", "create_comment", '{"content":"reply"}'),
                (0, "2026-01-01 00:00:04.500000", "interview", '{"prompt":"private"}'),
            ],
        )

    records = service._export_social_actions(run_dir)

    assert [(item["agent_id"], item["round"], item["action_type"]) for item in records] == [
        (20, 0, "create_post"),
        (0, 1, "refresh"),
        (0, 2, "create_comment"),
    ]
    assert records[-1]["action_args"]["content"] == "reply"
    counts = service._social_counts(run_dir)
    assert counts[0]["total"] == 2
    assert counts[0]["comment"] == 1


def test_s1_run_id_rejects_path_traversal(tmp_path):
    service = S1ExperimentService(storage_dir=tmp_path)

    with pytest.raises(ValueError, match="invalid S1"):
        service.get_status("../s1_reddit_escape")
