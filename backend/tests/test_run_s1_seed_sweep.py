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
    assert [item["completed_scenario_count"] for item in result["batches"]] == [
        18,
        18,
    ]
    assert (tmp_path / "finance" / result["sweep_id"] / "manifest.json").exists()
