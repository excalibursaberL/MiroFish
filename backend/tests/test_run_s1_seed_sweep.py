import json

import pytest

from scripts import run_s1_seed_sweep as seed_sweep


def graph_manifest(tmp_path):
    path = tmp_path / "zep_graphs_manifest.json"
    path.write_text(
        json.dumps(
            {
                "scenarios": [
                    {
                        "scenario_id": f"SCN_{index:03d}",
                        "graph_id": f"graph-{index:03d}",
                        "status": "completed",
                    }
                    for index in range(1, 19)
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


def test_parse_seed_tokens_accepts_spaces_and_commas():
    assert seed_sweep.parse_seed_tokens(["4005", "4006,4007"]) == [
        4005,
        4006,
        4007,
    ]


def test_parse_agent_id_tokens_accepts_k8_candidate():
    assert seed_sweep.parse_agent_id_tokens(["1,3,4", "5", "9,13,14,17"]) == [
        1, 3, 4, 5, 9, 13, 14, 17
    ]


@pytest.mark.parametrize(
    "tokens, message",
    [
        (["4005", "4005"], "duplicate"),
        (["-1"], "between"),
        (["not-a-seed"], "invalid"),
    ],
)
def test_parse_seed_tokens_rejects_invalid_values(tokens, message):
    with pytest.raises(ValueError, match=message):
        seed_sweep.parse_seed_tokens(tokens)


def test_build_plan_fixes_k10_six_round_contract(tmp_path):
    plan = seed_sweep.build_plan([4005, 4006], graph_manifest(tmp_path))

    assert plan["social_rounds"] == 6
    assert plan["agent_count"] == 10
    assert plan["scenario_count_per_seed"] == 18
    assert plan["total_scenario_runs"] == 36
    assert plan["calls_external_llm"] is False


def test_profile_id_permutation_is_reproducible_derangement():
    first = seed_sweep.build_profile_id_permutation(42)
    second = seed_sweep.build_profile_id_permutation(42)

    assert first == second
    assert sorted(first) == list(range(10))
    assert all(profile_id != runtime_id for runtime_id, profile_id in enumerate(first))
    assert first != seed_sweep.build_profile_id_permutation(999)


def test_build_plan_records_seed_specific_profile_permutations(tmp_path):
    plan = seed_sweep.build_plan(
        [42, 999],
        graph_manifest(tmp_path),
        permute_profile_ids=True,
    )

    assert plan["profile_id_permutation_enabled"] is True
    assert plan["sampling_method"] == "paired_profile_runtime_derangement_v1"
    assert plan["profile_id_permutations"]["42"] == (
        seed_sweep.build_profile_id_permutation(42)
    )


def test_build_plan_supports_preferred_k8_candidate(tmp_path):
    selected_ids = [1, 3, 4, 5, 9, 13, 14, 17]
    plan = seed_sweep.build_plan(
        [42, 999],
        graph_manifest(tmp_path),
        selected_full_population_agent_ids=selected_ids,
    )

    assert plan["agent_count"] == 8
    assert plan["selected_full_population_agent_ids"] == selected_ids
    assert plan["agent_set_version"] == "n10_k8_enum_best_v1"
    assert plan["sampling_method"] == "offline_exact_enumeration_k10_v2_candidate"
    assert plan["data_split"] == "agent_subset_rerun_validation"
    assert plan["profile_id_permutations"]["42"] == list(range(8))


def test_build_plan_records_explicit_finance_skill(tmp_path):
    plan = seed_sweep.build_plan(
        [42],
        graph_manifest(tmp_path),
        enabled_finance_skills=["ashare-institutional-analyst"],
    )

    assert plan["enabled_finance_skills"] == ["ashare-institutional-analyst"]
    assert plan["finance_skills"][0]["name"] == "ashare-institutional-analyst"


def test_build_plan_records_all_agent_finance_skill_scope(tmp_path):
    plan = seed_sweep.build_plan(
        [42, 999, 4004],
        graph_manifest(tmp_path),
        enabled_finance_skills=["ashare-institutional-analyst"],
        finance_skill_scope="all_agents",
    )

    assert plan["finance_skill_scope"] == "all_agents"
    assert plan["agent_count"] == 10
    assert plan["total_scenario_runs"] == 54


def test_build_plan_records_pre_social_only_finance_skill_stage(tmp_path):
    plan = seed_sweep.build_plan(
        [42],
        graph_manifest(tmp_path),
        enabled_finance_skills=["ashare-institutional-analyst"],
        finance_skill_scope="eligible_roles",
        finance_skill_stage="pre_social_only",
    )

    assert plan["finance_skill_scope"] == "eligible_roles"
    assert plan["finance_skill_stage"] == "pre_social_only"


def test_run_sweep_creates_one_serial_batch_per_seed(monkeypatch, tmp_path):
    prepared = []

    class FakeRunner:
        def __init__(self, *, storage_dir):
            self.storage_dir = storage_dir

        def prepare(self, **kwargs):
            batch_id = f"s1_batch_seed{kwargs['random_seed']}"
            prepared.append(kwargs)
            batch_dir = self.storage_dir / batch_id
            batch_dir.mkdir()
            (batch_dir / "manifest.json").write_text("{}", encoding="utf-8")
            return {"batch_id": batch_id}

        def run_sync(self, batch_id):
            return {
                "batch_id": batch_id,
                "status": "completed",
                "completed_scenario_count": 18,
                "failed_scenario_count": 0,
            }

    monkeypatch.setattr(seed_sweep, "S1BatchRunner", FakeRunner)
    result = seed_sweep.run_sweep(
        seeds=[4005, 4006],
        graph_manifest=graph_manifest(tmp_path),
        storage_dir=tmp_path / "finance",
    )

    assert result["status"] == "completed"
    assert [item["random_seed"] for item in prepared] == [4005, 4006]
    assert all(item["social_rounds"] == 6 for item in prepared)
    assert all(item["agent_set_version"] == "n10_k10_exact_v1" for item in prepared)
    assert all(item["profile_id_permutation"] == list(range(10)) for item in prepared)
    assert [item["completed_scenario_count"] for item in result["batches"]] == [
        18,
        18,
    ]
    assert (tmp_path / "finance" / result["sweep_id"] / "manifest.json").exists()


def test_run_sweep_propagates_profile_permutation(monkeypatch, tmp_path):
    prepared = []

    class FakeRunner:
        def __init__(self, *, storage_dir):
            self.storage_dir = storage_dir

        def prepare(self, **kwargs):
            prepared.append(kwargs)
            batch_id = "s1_batch_profileperm"
            batch_dir = self.storage_dir / batch_id
            batch_dir.mkdir()
            (batch_dir / "manifest.json").write_text("{}", encoding="utf-8")
            return {"batch_id": batch_id}

        def run_sync(self, batch_id):
            return {
                "batch_id": batch_id,
                "status": "completed",
                "completed_scenario_count": 18,
                "failed_scenario_count": 0,
            }

    monkeypatch.setattr(seed_sweep, "S1BatchRunner", FakeRunner)
    result = seed_sweep.run_sweep(
        seeds=[42],
        graph_manifest=graph_manifest(tmp_path),
        storage_dir=tmp_path / "finance",
        permute_profile_ids=True,
    )

    assert result["status"] == "completed"
    assert prepared[0]["data_split"] == "profile_id_permutation"
    assert prepared[0]["sampling_method"] == "paired_profile_runtime_derangement_v1"
    assert prepared[0]["profile_id_permutation"] == (
        seed_sweep.build_profile_id_permutation(42)
    )


def test_run_sweep_propagates_k8_candidate(monkeypatch, tmp_path):
    prepared = []
    selected_ids = [1, 3, 4, 5, 9, 13, 14, 17]

    class FakeRunner:
        def __init__(self, *, storage_dir):
            self.storage_dir = storage_dir

        def prepare(self, **kwargs):
            prepared.append(kwargs)
            batch_id = "s1_batch_k8candidate"
            batch_dir = self.storage_dir / batch_id
            batch_dir.mkdir()
            (batch_dir / "manifest.json").write_text("{}", encoding="utf-8")
            return {"batch_id": batch_id}

        def run_sync(self, batch_id):
            return {
                "batch_id": batch_id,
                "status": "completed",
                "completed_scenario_count": 18,
                "failed_scenario_count": 0,
            }

    monkeypatch.setattr(seed_sweep, "S1BatchRunner", FakeRunner)
    result = seed_sweep.run_sweep(
        seeds=[42],
        graph_manifest=graph_manifest(tmp_path),
        storage_dir=tmp_path / "finance",
        selected_full_population_agent_ids=selected_ids,
    )

    assert result["status"] == "completed"
    assert prepared[0]["selected_full_population_agent_ids"] == selected_ids
    assert prepared[0]["profile_id_permutation"] == list(range(8))
    assert prepared[0]["agent_set_version"] == "n10_k8_enum_best_v1"
    assert prepared[0]["data_split"] == "agent_subset_rerun_validation"


def test_run_sweep_propagates_finance_skill(monkeypatch, tmp_path):
    prepared = []

    class FakeRunner:
        def __init__(self, *, storage_dir):
            self.storage_dir = storage_dir

        def prepare(self, **kwargs):
            prepared.append(kwargs)
            batch_id = "s1_batch_skilltest"
            batch_dir = self.storage_dir / batch_id
            batch_dir.mkdir()
            (batch_dir / "manifest.json").write_text("{}", encoding="utf-8")
            return {"batch_id": batch_id}

        def run_sync(self, batch_id):
            return {
                "batch_id": batch_id,
                "status": "completed",
                "completed_scenario_count": 18,
                "failed_scenario_count": 0,
            }

    monkeypatch.setattr(seed_sweep, "S1BatchRunner", FakeRunner)
    result = seed_sweep.run_sweep(
        seeds=[42],
        graph_manifest=graph_manifest(tmp_path),
        storage_dir=tmp_path / "finance",
        enabled_finance_skills=["ashare-institutional-analyst"],
    )

    assert result["status"] == "completed"
    assert prepared[0]["enabled_finance_skills"] == [
        "ashare-institutional-analyst"
    ]
