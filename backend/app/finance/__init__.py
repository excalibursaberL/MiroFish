"""A-share financial adaptation layer for MiroFish.

The first supported condition is C0: independent investor forecasts without
exposure to other investors' posts or predictions.
"""

from .c0 import C0ExperimentService
from .background import C0BackgroundRunner
from .dataset import FinancialDatasetLoader, FinancialScenario
from .evaluator import FinancialOutcomeEvaluator
from .roles import C0_AGENT_COUNT, build_c0_profiles

__all__ = [
    "C0ExperimentService",
    "C0BackgroundRunner",
    "FinancialDatasetLoader",
    "FinancialScenario",
    "FinancialOutcomeEvaluator",
    "C0_AGENT_COUNT",
    "build_c0_profiles",
]
