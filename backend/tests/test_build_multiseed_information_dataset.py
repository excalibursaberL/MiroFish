from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_multiseed_information_dataset.py"
)
SPEC = importlib.util.spec_from_file_location("multiseed_information", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_mutual_information_known_values() -> None:
    assert MODULE.mutual_information_bits([0, 0, 1, 1], [0, 0, 1, 1]) == pytest.approx(1.0)
    assert MODULE.mutual_information_bits([0, 0, 1, 1], [0, 1, 0, 1]) == pytest.approx(0.0)
    assert MODULE.normalized_mutual_information([0, 0, 1, 1], [0, 0, 1, 1]) == pytest.approx(1.0)


def test_conditional_mutual_information_removes_condition() -> None:
    condition = [0, 0, 1, 1]
    left = condition
    right = condition
    assert MODULE.mutual_information_bits(left, right) == pytest.approx(1.0)
    assert MODULE.conditional_mutual_information_bits(left, right, condition) == pytest.approx(0.0)


def test_bins_have_stable_boundaries() -> None:
    assert MODULE.stance_bin(None) == "none"
    assert MODULE.stance_bin(-0.1001) == "negative"
    assert MODULE.stance_bin(-0.1) == "mixed_neutral"
    assert MODULE.stance_bin(0.1) == "mixed_neutral"
    assert MODULE.stance_bin(0.1001) == "positive"
    assert MODULE.amount_bin(0) == "none"
    assert MODULE.amount_bin(9) == "low_1_9"
    assert MODULE.amount_bin(10) == "medium_10_17"
    assert MODULE.amount_bin(18) == "high_18_plus"
    assert MODULE.return_change_bin(-0.005) == "stable"
    assert MODULE.return_change_bin(0.005) == "stable"


def test_bh_qvalues_are_monotonic_in_p_order() -> None:
    p_values = [0.01, 0.04, 0.03, 0.20]
    q_values = MODULE.bh_qvalues(p_values)
    ordered = sorted(zip(p_values, q_values))
    assert all(left[1] <= right[1] for left, right in zip(ordered, ordered[1:]))
    assert all(0 <= value <= 1 for value in q_values)


def test_permutation_q_column_name_drops_one_sided_suffix() -> None:
    column = "predictive_incremental_cmi_one_sided_p_value"
    assert column.replace("_one_sided_p_value", "_bh_q_value") == (
        "predictive_incremental_cmi_bh_q_value"
    )


def test_prepare_panel_derives_analysis_categories() -> None:
    frame = pd.DataFrame(
        {
            "round": [1, 2],
            "agent_id": [0, 0],
            "exposure_social_mean_stance_score": [-0.2, 0.2],
            "exposure_social_unique_count": [3, 19],
            "expected_return_delta": [-0.01, 0.01],
            "direction_flip": [False, True],
        }
    )
    result = MODULE.prepare_panel(frame)
    assert result["stance_bin"].tolist() == ["negative", "positive"]
    assert result["amount_bin"].tolist() == ["low_1_9", "high_18_plus"]
    assert result["return_change_bin"].tolist() == ["down", "up"]
    assert result["direction_flip_bin"].tolist() == ["stable", "changed"]
