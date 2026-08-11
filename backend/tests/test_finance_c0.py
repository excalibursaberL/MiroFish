import csv
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
from flask import Flask

from app.api import finance_bp
from app.api import finance as finance_api
from app.finance.c0 import C0ExperimentService
from app.finance.dataset import DatasetValidationError, FinancialDatasetLoader
from app.finance.evaluator import (
    FIVE_DAY_NEUTRAL_THRESHOLD,
    FinancialOutcomeEvaluator,
)
from app.finance.models import C0Forecast
from app.finance.roles import (
    SELECTED_AGENT_IDS,
    build_c0_profiles,
    build_full_c0_profiles,
    iter_c0_roles,
)


def test_c0_role_mix_is_fixed():
    roles = iter_c0_roles()
    assert len(roles) == 20
    counts = {}
    for role in roles:
        counts[role["role_category"]] = counts.get(role["role_category"], 0) + 1
    assert counts == {
        "institution": 3,
        "retail_mature": 6,
        "retail_basic": 8,
        "retail_novice": 3,
    }
    full_profiles = build_full_c0_profiles()
    assert len(full_profiles) == 20

    retail = [
        profile
        for profile in full_profiles
        if profile["role_category"] != "institution"
    ]
    assert Counter(profile["knowledge_level"] for profile in retail) == {
        "experienced": 6,
        "basic": 8,
        "novice": 3,
    }
    assert Counter(profile["risk_attitude"] for profile in retail) == {
        "low": 3,
        "medium": 10,
        "high": 3,
        "very_high": 1,
    }
    assert Counter(profile["investment_horizon"] for profile in retail) == {
        "long": 9,
        "mixed": 6,
        "short": 2,
    }
    assert Counter(profile["analysis_style"] for profile in retail) == {
        "fundamental": 7,
        "technical": 10,
    }
    assert all(
        profile["profile_version"] == "survey2019_twinmarket_minimal_v1"
        for profile in full_profiles
    )
    assert full_profiles[0]["profile_sources"]["risk_attitude"] == "institutional_role_design"
    assert retail[0]["profile_sources"]["risk_attitude"] == "SIPF_2019_natural_person_survey"
    assert retail[0]["profile_sources"]["analysis_style"].startswith("TwinMarket_")

    profiles = build_c0_profiles()
    assert len(profiles) == 10
    assert [profile["user_id"] for profile in profiles] == list(range(10))
    assert [
        profile["full_population_agent_id"] for profile in profiles
    ] == list(SELECTED_AGENT_IDS)
    assert Counter(profile["role_category"] for profile in profiles) == {
        "institution": 1,
        "retail_mature": 3,
        "retail_basic": 5,
        "retail_novice": 1,
    }


def test_default_blind_dataset_has_five_seeds():
    scenario = FinancialDatasetLoader().load(limit=1)[0]
    assert scenario.scenario_id.startswith("SCN_")
    assert scenario.symbol.startswith("ASSET_")
    assert scenario.name.startswith("COMPANY_")
    assert len(scenario.seed_events) == 5
    assert scenario.prediction_cutoff == "T+0d"


def test_expected_return_is_normalized_to_decimal_units():
    assert C0ExperimentService._expected_return(3.5) == pytest.approx(0.035)
    assert C0ExperimentService._expected_return(-2.5) == pytest.approx(-0.025)
    assert C0ExperimentService._expected_return(0.03) == pytest.approx(0.03)
    assert C0ExperimentService._expected_return("3.5%") == pytest.approx(0.035)
    assert C0ExperimentService._expected_return(
        0.5, direction="neutral"
    ) == pytest.approx(0.005)
    assert C0ExperimentService._expected_return(
        0.5, direction="up"
    ) == pytest.approx(0.5)


def test_original_or_nested_evaluator_fields_are_rejected(tmp_path):
    path = tmp_path / "unsafe.jsonl"
    payload = {
        "scenario_id": "SCN_TEST",
        "symbol": "ASSET_TEST",
        "name": "COMPANY_TEST",
        "prediction_cutoff": "T+0d",
        "horizon": "next_5_trading_days",
        "seed_events": [
            {
                "event_id": "EVT_TEST",
                "event_time": "T-1d",
                "text": "公开事件",
                "metadata": {"future_return": 0.1},
            }
        ],
        "current_event": {
            "event_id": "EVT_TEST2",
            "event_time": "T+0d",
            "text": "当前事件",
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="future_return"):
        FinancialDatasetLoader(path).load()


def test_original_snapshot_is_rejected(tmp_path):
    path = tmp_path / "original.jsonl"
    payload = {
        "scenario_id": "seed5_001",
        "symbol": "000001",
        "name": "真实公司",
        "prediction_cutoff": "2021-04-12 19:24:00",
        "horizon": "next_5_trading_days",
        "seed_events": [
            {"event_id": "EVT_TEST", "event_time": "T-1d", "text": "事件"}
        ],
        "current_event": {
            "event_id": "EVT_TEST2",
            "event_time": "T+0d",
            "text": "当前事件",
        },
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="anonymous"):
        FinancialDatasetLoader(path).load()


def test_seed_after_cutoff_is_rejected(tmp_path):
    path = tmp_path / "future_seed.jsonl"
    payload = {
        "scenario_id": "SCN_TEST",
        "symbol": "ASSET_TEST",
        "name": "COMPANY_TEST",
        "prediction_cutoff": "T+0d",
        "horizon": "next_5_trading_days",
        "seed_events": [
            {"event_id": "EVT_TEST", "event_time": "T+1d", "text": "未来事件"}
        ],
        "current_event": {
            "event_id": "EVT_TEST2",
            "event_time": "T+0d",
            "text": "当前事件",
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="precede"):
        FinancialDatasetLoader(path).load()


def test_prepare_and_dry_run_do_not_call_llm(tmp_path):
    service = C0ExperimentService(storage_dir=tmp_path)
    manifest = service.prepare(limit=1)
    assert manifest["status"] == "prepared"
    assert manifest["agent_count"] == 10
    assert manifest["scenario_count"] == 1
    assert manifest["expected_prediction_count"] == 10
    assert manifest["prediction_target"]["neutral_threshold"] == pytest.approx(0.017)
    assert manifest["prediction_target"]["expected_return_unit"] == "decimal"
    assert "-1.7%" in manifest["prediction_target"]["direction_definition"]
    assert manifest["files"]["llm_responses"] == "llm_responses.jsonl"
    assert manifest["replicate_id"] == manifest["run_id"]
    assert manifest["agent_set_version"] == "n10_k10_exact_v1"
    assert manifest["sampling_method"] == "offline_exact_enumeration_k10"
    assert manifest["selected_full_population_agent_ids"] == list(SELECTED_AGENT_IDS)
    assert len(manifest["input_snapshot_hash"]) == 64
    assert len(manifest["prompt_hash"]) == 64
    run_id = manifest["run_id"]
    prompt_lines = (tmp_path / run_id / "prompts.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(prompt_lines) == 10
    assert all("evaluator_targets" not in line for line in prompt_lines)
    assert all("stock_factors" not in line for line in prompt_lines)

    scenario = FinancialDatasetLoader().load(limit=1)[0]
    first_prompt = json.loads(prompt_lines[0])
    assert scenario.seed_events[0]["text"] in first_prompt["user"]
    assert scenario.current_event["text"] in first_prompt["user"]
    scenario_snapshot = (tmp_path / run_id / "scenarios.jsonl").read_text(
        encoding="utf-8"
    )
    assert "stock_factors" in scenario_snapshot

    dry_manifest = service.run(run_id, dry_run=True)
    assert dry_manifest["status"] == "dry_run"
    assert (tmp_path / run_id / "dry_run_prompts.jsonl").exists()
    preview = service.get_preview(run_id)
    assert preview["scenario"]["scenario_id"] == "SCN_001"
    assert preview["prompt"]["agent_id"] == 0


def test_prepare_records_institutional_skill_assignment_without_changing_baseline(
    tmp_path,
):
    service = C0ExperimentService(storage_dir=tmp_path)
    manifest = service.prepare(
        limit=1,
        enabled_finance_skills=["ashare-institutional-analyst"],
    )

    assert manifest["enabled_finance_skills"] == ["ashare-institutional-analyst"]
    assert manifest["finance_skills"][0]["name"] == "ashare-institutional-analyst"
    assignments = manifest["profile_skill_assignments"]
    assert [item["agent_id"] for item in assignments if item["skill_names"]] == [0]
    prompt_lines = (
        tmp_path / manifest["run_id"] / "prompts.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    assert "## Enabled heterogeneous finance Skill" in prompt_lines[0]
    assert all(
        "## Enabled heterogeneous finance Skill" not in line
        for line in prompt_lines[1:]
    )


def test_c0_prototype_rejects_multiple_scenarios(tmp_path):
    service = C0ExperimentService(storage_dir=tmp_path)
    with pytest.raises(ValueError, match="一个场景"):
        service.prepare(limit=2)


def test_all_mode_prepares_every_scenario_and_rejects_filters(tmp_path):
    service = C0ExperimentService(storage_dir=tmp_path)
    manifest = service.prepare(run_mode="all")

    assert manifest["run_mode"] == "all"
    assert manifest["scenario_count"] == 18
    assert manifest["agent_count"] == 10
    assert manifest["expected_prediction_count"] == 180
    prompt_lines = (tmp_path / manifest["run_id"] / "prompts.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(prompt_lines) == 180

    with pytest.raises(ValueError, match="cannot use"):
        service.prepare(run_mode="all", limit=1)
    with pytest.raises(ValueError, match="cannot use"):
        service.prepare(run_mode="all", scenario_ids=["SCN_001"])


def test_all_mode_writes_prediction_and_evaluation_csv_without_network(
    monkeypatch, tmp_path
):
    service = C0ExperimentService(storage_dir=tmp_path)
    manifest = service.prepare(run_mode="all")
    run_id = manifest["run_id"]
    run_dir = tmp_path / run_id
    call_count = 0

    def fake_request_forecast(*, scenario, profile, **_kwargs):
        nonlocal call_count
        if call_count == 1:
            with (run_dir / "predictions.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                assert len(list(csv.DictReader(handle))) == 1
            assert not (run_dir / "evaluation.csv").exists()
        call_count += 1
        return C0Forecast(
            scenario_id=scenario.scenario_id,
            agent_id=profile["user_id"],
            agent_role=profile["role_id"],
            agent_role_category=profile["role_category"],
            agent_role_label=profile["role_label"],
            as_of=scenario.prediction_cutoff,
            horizon=scenario.horizon,
            direction="up",
            up_probability=0.7,
            neutral_probability=0.2,
            down_probability=0.1,
            expected_return=0.03,
            confidence=0.8,
            evidence_event_ids=[scenario.current_event["event_id"]],
            reason="离线测试结果",
            raw_response='{"direction":"up"}',
            status="ok",
            finish_reason="stop",
            response_content_length=18,
        )

    monkeypatch.setattr("app.finance.c0.Config.LLM_API_KEY", "test-key")
    monkeypatch.setattr("app.finance.c0.OpenAI", lambda **_kwargs: object())
    monkeypatch.setattr(service, "_request_forecast", fake_request_forecast)

    completed = service.run(run_id)

    assert completed["status"] == "completed"
    assert completed["completed_prediction_count"] == 180
    assert completed["successful_prediction_count"] == 180
    assert call_count == 180

    with (run_dir / "predictions.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        prediction_rows = list(csv.DictReader(handle))
    with (run_dir / "evaluation.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        evaluation_rows = list(csv.DictReader(handle))

    assert len(prediction_rows) == 180
    assert len(evaluation_rows) == 180
    assert prediction_rows[0]["full_population_agent_id"] == "1"
    assert "agent_analysis_style" in prediction_rows[0]
    assert "agent_risk_attitude" in prediction_rows[0]
    assert prediction_rows[0]["run_id"] == run_id
    assert prediction_rows[0]["expected_return_unit"] == "decimal"
    assert float(prediction_rows[0]["expected_return"]) == pytest.approx(0.03)
    assert Counter(row["scenario_id"] for row in prediction_rows) == {
        f"SCN_{index:03d}": 10 for index in range(1, 19)
    }
    assert "actual_five_day_close_direction" not in prediction_rows[0]
    assert evaluation_rows[0]["actual_five_day_close_direction"] in {
        "up",
        "neutral",
        "down",
    }
    assert evaluation_rows[0]["five_day_direction_correct"] in {"true", "false"}
    assert float(evaluation_rows[0]["five_day_neutral_threshold"]) == pytest.approx(
        0.017
    )
    assert "neutral" in evaluation_rows[0]["five_day_direction_definition"]
    assert service.get_csv_path(run_id, "predictions").name == "predictions.csv"
    assert service.get_csv_path(run_id, "evaluation").name == "evaluation.csv"


def test_batch_api_requires_background_and_downloads_csv(monkeypatch, tmp_path):
    service = C0ExperimentService(storage_dir=tmp_path)
    manifest = service.prepare(run_mode="all")
    run_id = manifest["run_id"]
    service._write_csv(
        tmp_path / run_id / "predictions.csv",
        [{"scenario_id": "SCN_001", "agent_id": 0, "status": "ok"}],
        service.PREDICTION_CSV_FIELDS,
    )

    monkeypatch.setattr(finance_api, "_service", lambda: service)
    monkeypatch.setattr(
        finance_api.C0BackgroundRunner,
        "start",
        lambda value: {**service.get_status(value), "status": "queued"},
    )
    app = Flask(__name__)
    app.register_blueprint(finance_bp, url_prefix="/api/finance")
    client = app.test_client()

    synchronous = client.post("/api/finance/c0/run", json={"run_id": run_id})
    assert synchronous.status_code == 400
    assert "background=true" in synchronous.json["error"]

    started = client.post(
        "/api/finance/c0/run",
        json={"run_id": run_id, "background": True},
    )
    assert started.status_code == 202
    assert started.json["data"]["status"] == "queued"

    downloaded = client.get(f"/api/finance/c0/{run_id}/csv/predictions")
    assert downloaded.status_code == 200
    assert "scenario_id" in downloaded.get_data(as_text=True)

    hidden = client.get(f"/api/finance/c0/{run_id}/csv/evaluation")
    assert hidden.status_code == 404
    assert "only after" in hidden.json["error"]


def test_all_mode_cannot_be_filtered_or_rerun(monkeypatch, tmp_path):
    service = C0ExperimentService(storage_dir=tmp_path)
    manifest = service.prepare(run_mode="all")
    with pytest.raises(ValueError, match="cannot be filtered"):
        service.run(manifest["run_id"], scenario_ids=["SCN_001"], dry_run=True)

    single = service.prepare(limit=1)
    predictions = [
        {"scenario_id": "SCN_001", "agent_id": index, "status": "ok"}
        for index in range(single["expected_prediction_count"])
    ]
    service._write_jsonl(
        tmp_path / single["run_id"] / "predictions.jsonl", predictions
    )
    assert service.get_status(single["run_id"])["status"] == "completed"
    monkeypatch.setattr("app.finance.c0.Config.LLM_API_KEY", "test-key")
    with pytest.raises(ValueError, match="cannot be started again"):
        service.run(single["run_id"])


def test_atomic_replace_retries_transient_windows_lock(monkeypatch, tmp_path):
    service = C0ExperimentService(storage_dir=tmp_path)
    target = tmp_path / "manifest.json"
    original_replace = Path.replace
    replace_attempts = 0

    def flaky_replace(source, destination):
        nonlocal replace_attempts
        if source.name.startswith(".manifest.json.") and replace_attempts < 2:
            replace_attempts += 1
            raise PermissionError(5, "sharing violation")
        replace_attempts += 1
        return original_replace(source, destination)

    monkeypatch.setattr(Path, "replace", flaky_replace)
    monkeypatch.setattr("app.finance.c0.time.sleep", lambda _seconds: None)

    service._write_json(target, {"status": "ok"})

    assert json.loads(target.read_text(encoding="utf-8")) == {"status": "ok"}
    assert replace_attempts == 3
    assert not list(tmp_path.glob(".manifest.json.*.tmp"))


def test_failed_batch_resumes_existing_predictions(monkeypatch, tmp_path):
    source_loader = FinancialDatasetLoader()
    dataset_path = tmp_path / "two_scenarios.jsonl"
    dataset_path.write_text(
        "\n".join(
            json.dumps(scenario.to_safe_dict(), ensure_ascii=False)
            for scenario in source_loader.load()[:2]
        )
        + "\n",
        encoding="utf-8",
    )
    service = C0ExperimentService(
        storage_dir=tmp_path,
        dataset_path=dataset_path,
    )
    manifest = service.prepare(run_mode="all")
    run_id = manifest["run_id"]
    calls = 0

    def make_forecast(scenario, profile):
        return C0Forecast(
            scenario_id=scenario.scenario_id,
            agent_id=profile["user_id"],
            agent_role=profile["role_id"],
            agent_role_category=profile["role_category"],
            agent_role_label=profile["role_label"],
            as_of=scenario.prediction_cutoff,
            horizon=scenario.horizon,
            direction="neutral",
            up_probability=0.2,
            neutral_probability=0.6,
            down_probability=0.2,
            expected_return=0.0,
            confidence=0.5,
            evidence_event_ids=[],
            reason="resume test",
            raw_response="{}",
            status="ok",
        )

    def failing_request(*, scenario, profile, **_kwargs):
        nonlocal calls
        if calls >= 3:
            raise RuntimeError("simulated worker stop")
        calls += 1
        return make_forecast(scenario, profile)

    monkeypatch.setattr("app.finance.c0.Config.LLM_API_KEY", "test-key")
    monkeypatch.setattr("app.finance.c0.OpenAI", lambda **_kwargs: object())
    monkeypatch.setattr(service, "_request_forecast", failing_request)
    with pytest.raises(RuntimeError, match="simulated worker stop"):
        service.run(run_id)

    failed = service.mark_failed(run_id, "simulated worker stop")
    assert failed["completed_prediction_count"] == 3

    resumed_calls = 0

    def successful_request(*, scenario, profile, **_kwargs):
        nonlocal resumed_calls
        resumed_calls += 1
        return make_forecast(scenario, profile)

    monkeypatch.setattr(service, "_request_forecast", successful_request)
    completed = service.run(run_id)

    assert completed["status"] == "completed"
    assert completed["resumed_prediction_count"] == 3
    assert resumed_calls == 17
    assert len(service.get_predictions(run_id)) == 20
    with (tmp_path / run_id / "predictions.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        assert len(list(csv.DictReader(handle))) == 20


def test_scenario_selector_returns_safe_summaries(tmp_path):
    service = C0ExperimentService(storage_dir=tmp_path)
    scenarios = service.list_scenarios()
    assert len(scenarios) == 18
    assert scenarios[0]["scenario_id"] == "SCN_001"
    assert "label" not in scenarios[0]
    assert "future_return" not in scenarios[0]


def test_forecast_parser_normalizes_percent_probabilities():
    service = C0ExperimentService()
    scenario = FinancialDatasetLoader().load(limit=1)[0]
    profile = build_c0_profiles()[0]
    forecast = service.parse_forecast(
        scenario=scenario,
        profile=profile,
        raw_response=json.dumps(
            {
                "direction": "上涨",
                "probabilities": {"up": 70, "neutral": 20, "down": 10},
                "confidence": 80,
                "evidence_event_ids": [scenario.seed_events[0]["event_id"], "EVT_UNKNOWN"],
                "reason": "依据历史事件",
            },
            ensure_ascii=False,
        ),
    )
    assert forecast.status == "ok"
    assert forecast.direction == "up"
    assert forecast.agent_role_category == "institution"
    assert forecast.agent_analysis_style == profile["analysis_style"]
    assert forecast.agent_risk_attitude == profile["risk_attitude"]
    assert forecast.up_probability == pytest.approx(0.7)
    assert forecast.confidence == pytest.approx(0.8)
    assert forecast.evidence_event_ids == [scenario.seed_events[0]["event_id"]]


def test_scn_004_evaluator_exposes_both_result_definitions():
    outcome = FinancialOutcomeEvaluator().get_outcome("SCN_004")

    assert outcome["astock_direction"] == "neutral"
    assert outcome["astock_change_return"] == pytest.approx(0.0011668611435238)
    assert outcome["five_day_close_direction"] == "up"
    assert outcome["five_day_close_return"] == pytest.approx(0.04195804)
    assert outcome["five_day_neutral_threshold"] == pytest.approx(0.017)


@pytest.mark.parametrize(
    ("five_day_return", "expected_direction"),
    [
        (-0.02, "down"),
        (-FIVE_DAY_NEUTRAL_THRESHOLD, "neutral"),
        (-0.001, "neutral"),
        (0.0, "neutral"),
        (0.001, "neutral"),
        (FIVE_DAY_NEUTRAL_THRESHOLD, "neutral"),
        (0.02, "up"),
    ],
)
def test_five_day_direction_uses_closed_neutral_band(
    five_day_return, expected_direction
):
    assert (
        FinancialOutcomeEvaluator._sign_direction(five_day_return)
        == expected_direction
    )


@pytest.mark.parametrize("scenario_id", ["SCN_002", "SCN_009", "SCN_015"])
def test_small_five_day_moves_are_neutral(scenario_id):
    outcome = FinancialOutcomeEvaluator().get_outcome(scenario_id)

    assert abs(outcome["five_day_close_return"]) <= FIVE_DAY_NEUTRAL_THRESHOLD
    assert outcome["five_day_close_direction"] == "neutral"


@pytest.mark.parametrize("invalid_response", ["", '{"direction":"up"'])
def test_invalid_json_is_retried_once_with_thinking_disabled(
    monkeypatch, invalid_response, tmp_path
):
    service = C0ExperimentService()
    scenario = FinancialDatasetLoader().load(limit=1)[0]
    profile = build_c0_profiles()[0]
    system_prompt, user_prompt = service.build_prompt(scenario, profile)
    valid_payload = json.dumps(
        {
            "direction": "up",
            "up_probability": 0.7,
            "neutral_probability": 0.2,
            "down_probability": 0.1,
            "expected_return": 0.03,
            "confidence": 0.8,
            "evidence_event_ids": [scenario.current_event["event_id"]],
            "reason": "当前事件可能带来正向影响。",
        },
        ensure_ascii=False,
    )
    responses = [invalid_response, valid_payload]
    calls = []

    def fake_completion(_client, **kwargs):
        calls.append(kwargs)
        content = responses[len(calls) - 1]
        return SimpleNamespace(
            id=f"response-{len(calls)}",
            model="deepseek-v4-flash",
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        role="assistant",
                        content=content,
                        reasoning_content=None,
                    ),
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=20,
                total_tokens=120,
            ),
        )

    monkeypatch.setattr("app.finance.c0.create_chat_completion", fake_completion)
    trace_path = tmp_path / "llm_responses.jsonl"
    forecast = service._request_forecast(
        client=object(),
        scenario=scenario,
        profile=profile,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_trace_path=trace_path,
    )

    assert forecast.status == "ok"
    assert forecast.attempt_count == 2
    assert forecast.response_content_length == len(valid_payload)
    assert forecast.reasoning_content_present is False
    assert len(calls) == 2
    assert all(call["thinking_mode"] == "disabled" for call in calls)
    assert all(call["max_tokens"] == service.FORECAST_MAX_TOKENS for call in calls)
    assert all(call["temperature"] == 0.2 for call in calls)
    assert "上一次输出为空" in calls[1]["messages"][-1]["content"]

    traces = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert len(traces) == 2
    assert traces[0]["response"]["id"] == "response-1"
    assert traces[0]["parse_result"]["status"] == "parse_error"
    assert traces[1]["parse_result"]["status"] == "ok"
    assert traces[1]["response"]["usage"]["total_tokens"] == 120
    assert traces[1]["request"]["messages"][0]["role"] == "system"
    assert "api_key" not in json.dumps(traces, ensure_ascii=False).lower()

    token_traces = [
        json.loads(line)
        for line in (tmp_path / "llm_token_usage.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(token_traces) == 2
    assert all(record["agent_id"] == profile["user_id"] for record in token_traces)
    assert all(record["usage_available"] is True for record in token_traces)
    assert sum(record["prompt_tokens"] for record in token_traces) == 200
    assert sum(record["completion_tokens"] for record in token_traces) == 40
    assert sum(record["total_tokens"] for record in token_traces) == 240

    summary = service._write_token_usage_artifacts(
        tmp_path,
        [profile],
        run_id="c0_one_agent_token_test",
        scenario_id=scenario.scenario_id,
    )
    with (tmp_path / "agent_token_usage.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        agent_rows = list(csv.DictReader(handle))
    assert len(agent_rows) == 1
    assert int(agent_rows[0]["api_call_count"]) == 2
    assert int(agent_rows[0]["total_tokens"]) == 240
    assert int(agent_rows[0]["independent_forecast_total_tokens"]) == 240
    assert summary["agent_count"] == 1
    assert summary["total_tokens"] == 240


def test_prompt_meets_deepseek_json_mode_contract():
    service = C0ExperimentService()
    scenario = FinancialDatasetLoader().load(limit=1)[0]
    profile = build_c0_profiles()[0]
    system_prompt, user_prompt = service.build_prompt(scenario, profile)

    assert "json" in system_prompt
    assert "json.loads" in system_prompt
    assert '"direction":"neutral"' in system_prompt
    assert "-1.7% <= R5 <= +1.7%：neutral" in system_prompt
    assert "不表示“信息矛盾”或“无法判断”" in system_prompt
    assert "未来 5 个交易日的累计收盘收益" in user_prompt
    assert "±1.7%" in user_prompt
    assert "合法的 json 对象" in user_prompt


def test_prompt_uses_minimal_profile_without_directional_prior():
    service = C0ExperimentService()
    scenario = FinancialDatasetLoader().load(limit=1)[0]
    profile = build_c0_profiles()[4]
    system_prompt, user_prompt = service.build_prompt(scenario, profile)

    assert "固定行为画像" in system_prompt
    assert "知识水平" in system_prompt
    assert "风险态度" in system_prompt
    assert "投资期限" in system_prompt
    assert "不能预先决定股票涨跌" in system_prompt
    assert "不得编造均线、MACD、成交量" in system_prompt
    assert "必须回答题目指定的预测 horizon" in system_prompt
    assert "self_analysis" not in system_prompt
    assert profile["profile_version"] in user_prompt


def test_finish_reason_length_is_retried(monkeypatch):
    service = C0ExperimentService()
    scenario = FinancialDatasetLoader().load(limit=1)[0]
    profile = build_c0_profiles()[0]
    system_prompt, user_prompt = service.build_prompt(scenario, profile)
    valid_payload = json.dumps(
        {
            "direction": "neutral",
            "up_probability": 0.2,
            "neutral_probability": 0.6,
            "down_probability": 0.2,
            "expected_return": 0,
            "confidence": 0.5,
            "evidence_event_ids": [],
            "reason": "信息有限，暂判断为中性。",
        },
        ensure_ascii=False,
    )
    calls = []

    def fake_completion(_client, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="length",
                        message=SimpleNamespace(content='{"direction":"neutral"'),
                    )
                ]
            )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content=valid_payload),
                )
            ]
        )

    monkeypatch.setattr("app.finance.c0.create_chat_completion", fake_completion)
    forecast = service._request_forecast(
        client=object(),
        scenario=scenario,
        profile=profile,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    assert forecast.status == "ok"
    assert forecast.attempt_count == 2
    assert forecast.finish_reason == "stop"
    assert len(calls) == 2


def test_completed_status_is_repaired_and_dry_run_cannot_overwrite_it(tmp_path):
    service = C0ExperimentService(storage_dir=tmp_path)
    manifest = service.prepare(limit=1)
    run_id = manifest["run_id"]
    predictions = [
        {"scenario_id": "SCN_001", "agent_id": index, "status": "ok"}
        for index in range(manifest["expected_prediction_count"])
    ]
    service._write_jsonl(tmp_path / run_id / "predictions.jsonl", predictions)

    repaired = service.get_status(run_id)
    assert repaired["status"] == "completed"
    assert repaired["completed_prediction_count"] == 10

    after_prompt_check = service.run(run_id, dry_run=True)
    assert after_prompt_check["status"] == "completed"
    assert after_prompt_check["prediction_count"] == 10
    assert "last_prompt_check_at" in after_prompt_check


def test_ground_truth_is_hidden_until_completion(tmp_path):
    service = C0ExperimentService(storage_dir=tmp_path)
    manifest = service.prepare(scenario_ids=["SCN_004"])

    with pytest.raises(ValueError, match="only after"):
        service.get_outcome(manifest["run_id"])

    predictions = [
        {"scenario_id": "SCN_004", "agent_id": index, "status": "ok"}
        for index in range(manifest["expected_prediction_count"])
    ]
    service._write_jsonl(
        tmp_path / manifest["run_id"] / "predictions.jsonl", predictions
    )
    outcome = service.get_outcome(manifest["run_id"])
    assert outcome["scenario_id"] == "SCN_004"
    assert outcome["five_day_close_direction"] == "up"

    prompt_text = (tmp_path / manifest["run_id"] / "prompts.jsonl").read_text(
        encoding="utf-8"
    )
    assert "five_day_close_return" not in prompt_text
    assert "astock_change_return" not in prompt_text
