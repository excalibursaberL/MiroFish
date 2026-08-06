import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "enumerate_agent_subsets.py"
)
SPEC = importlib.util.spec_from_file_location("enumerate_agent_subsets", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_torch_js_divergence_has_expected_endpoints():
    same = torch.tensor([[0.5, 0.3, 0.2]], dtype=torch.float64)
    opposite_left = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    opposite_right = torch.tensor([[0.0, 1.0]], dtype=torch.float64)

    assert torch.allclose(
        MODULE.torch_js_divergence(same, same), torch.tensor([0.0], dtype=torch.float64)
    )
    assert torch.allclose(
        MODULE.torch_js_divergence(opposite_left, opposite_right),
        torch.tensor([1.0], dtype=torch.float64),
    )


def test_counts_to_distribution_represents_no_activity_explicitly():
    counts = torch.tensor([[0.0, 0.0], [1.0, 3.0]])
    distribution = MODULE.counts_to_distribution(counts)

    assert torch.allclose(distribution[0], torch.tensor([0.0, 0.0, 1.0]))
    assert torch.allclose(distribution[1], torch.tensor([0.25, 0.75, 0.0]))


def test_weighted_mean_error_handles_absence_on_both_sides():
    numerator = torch.tensor([[0.0, 0.8, 0.0]])
    denominator = torch.tensor([[0.0, 2.0, 0.0]])
    target_numerator = torch.tensor([0.0, 1.0, 1.0])
    target_denominator = torch.tensor([0.0, 2.0, 2.0])

    error = MODULE.normalized_weighted_mean_error(
        numerator, denominator, target_numerator, target_denominator
    )

    assert torch.allclose(error, torch.tensor([[0.0, 0.05, 1.0]]))


def test_combination_array_is_complete_and_lexicographic():
    combinations = MODULE.combination_array(5, 3)

    assert combinations.shape == (10, 3)
    assert combinations[0].tolist() == [0, 1, 2]
    assert combinations[-1].tolist() == [2, 3, 4]
    assert len({tuple(row) for row in combinations.tolist()}) == 10


def test_pareto_front_3d_marks_only_nondominated_points():
    values = np.asarray(
        [
            [0.1, 0.5, 0.5],
            [0.2, 0.6, 0.6],
            [0.5, 0.1, 0.5],
            [0.5, 0.5, 0.1],
            [0.4, 0.4, 0.4],
        ]
    )

    front = MODULE.pareto_front_3d(values)

    assert front.tolist() == [True, False, True, True, True]


def test_subset_masks_use_actual_agent_ids():
    combinations = np.asarray([[0, 2], [1, 3]], dtype=np.int16)
    agent_ids = np.asarray([2, 4, 6, 8], dtype=np.int16)

    masks, labels = MODULE.subset_masks(combinations, agent_ids)

    assert masks.tolist() == [(1 << 2) | (1 << 6), (1 << 4) | (1 << 8)]
    assert labels == ["2|6", "4|8"]
