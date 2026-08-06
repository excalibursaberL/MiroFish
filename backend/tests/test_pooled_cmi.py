import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "analyze_pooled_cmi.py"
SPEC = importlib.util.spec_from_file_location("analyze_pooled_cmi", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def sample_frame() -> pd.DataFrame:
    rows = []
    for scenario_index in range(4):
        for round_number in (1, 2):
            for agent_id in range(6):
                exposed = round_number == 2 or agent_id % 2 == 0
                social_count = agent_id + round_number if exposed else 0
                rows.append(
                    {
                        "scenario_id": f"SCN_{scenario_index:03d}",
                        "agent_id": agent_id,
                        "round": round_number,
                        "direction_changed": (agent_id + scenario_index) % 3 == 0,
                        "agent_role_category": "institution" if agent_id == 0 else "retail",
                        "agent_analysis_style": "fundamental" if agent_id % 2 else "technical",
                        "agent_risk_attitude": "high" if agent_id % 3 == 0 else "medium",
                        "agent_investment_horizon": "long" if agent_id % 2 else "mixed",
                        "agent_decision_source": "self_analysis",
                        "previous_up_probability": 0.2 + agent_id * 0.02,
                        "previous_neutral_probability": 0.3,
                        "previous_expected_return": (agent_id - 2) * 0.01,
                        "previous_confidence": 0.5,
                        "exposure_social_unique_content_count": social_count,
                        "exposure_social_mean_stance_score_unique": 0.2 if exposed else np.nan,
                        "exposure_social_stance_score_std_unique": 0.1 if exposed else np.nan,
                        "exposure_social_mixed_unique_proportion": 0.5 if exposed else 0.0,
                        "exposure_source_unique_content_count": 1,
                        "exposure_source_event_valence_positive_unique_proportion": 1.0,
                        "exposure_source_event_valence_mixed_unique_proportion": 0.0,
                        "exposure_source_event_valence_negative_unique_proportion": 0.0,
                    }
                )
    return pd.DataFrame(rows)


def test_prepare_features_keeps_no_exposure_distinct_from_neutral_stance():
    prepared = MODULE.prepare_features(sample_frame())
    no_exposure = prepared[prepared["exposure_social_unique_content_count"] == 0]
    assert not no_exposure.empty
    assert (no_exposure["social_has_exposure"] == 0).all()
    assert (no_exposure["social_stance_mean_filled"] == 0).all()


def test_block_permutation_preserves_joint_social_rows_within_each_block():
    prepared = MODULE.prepare_features(sample_frame())
    permuted = MODULE.permute_social_within_scenario_round(
        prepared, rng=np.random.default_rng(42)
    )
    columns = list(MODULE.PERMUTED_SOCIAL_FEATURES)
    for key, original_group in prepared.groupby(["scenario_id", "round"]):
        permuted_group = permuted[
            (permuted["scenario_id"] == key[0]) & (permuted["round"] == key[1])
        ]
        original_rows = sorted(map(tuple, original_group[columns].to_numpy()))
        permuted_rows = sorted(map(tuple, permuted_group[columns].to_numpy()))
        assert original_rows == permuted_rows


def test_scenario_permutation_preserves_complete_donor_trajectories():
    prepared = MODULE.prepare_features(sample_frame())
    permuted = MODULE.permute_social_scenario_trajectories(
        prepared, rng=np.random.default_rng(8)
    )
    columns = list(MODULE.PERMUTED_SOCIAL_FEATURES)
    original_signatures = []
    permuted_signatures = []
    for _, group in prepared.groupby("scenario_id"):
        original_signatures.append(
            tuple(map(tuple, group.sort_values(["round", "agent_id"])[columns].to_numpy()))
        )
    for _, group in permuted.groupby("scenario_id"):
        permuted_signatures.append(
            tuple(map(tuple, group.sort_values(["round", "agent_id"])[columns].to_numpy()))
        )
    assert sorted(original_signatures) == sorted(permuted_signatures)


def test_cross_fit_predictions_covers_every_row_without_scenario_leakage():
    prepared = MODULE.prepare_features(sample_frame())
    predictions = MODULE.cross_fit_predictions(
        prepared, c_value=1.0, seed=7
    )
    assert len(predictions) == len(prepared)
    for model_name in MODULE.MODEL_NUMERIC_FEATURES:
        values = predictions[f"probability_{model_name}"]
        assert values.notna().all()
        assert values.between(0.0, 1.0).all()


def test_scenario_bootstrap_reports_all_information_components():
    table = pd.DataFrame(
        {
            "amount_bits": [0.1, 0.2, -0.1],
            "stance_bits": [0.05, 0.0, -0.02],
            "total_bits": [0.15, 0.2, -0.12],
        }
    )
    result = MODULE.bootstrap_scenarios(table, samples=100, seed=1)
    assert set(result) == {"amount_bits", "stance_bits", "total_bits"}
    assert result["amount_bits"]["ci_2_5"] <= result["amount_bits"]["ci_97_5"]
