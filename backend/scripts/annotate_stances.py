"""Annotate a completed S1 Reddit run with an independent LLM.

Usage from ``MiroFish/backend``::

    python scripts/annotate_stances.py --run-id s1_reddit_...

Set STANCE_LLM_API_KEY, STANCE_LLM_BASE_URL and STANCE_LLM_MODEL_NAME in
``MiroFish/.env`` for a genuinely independent annotator. For local debugging
only, ``--allow-primary-fallback`` explicitly permits using the main model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND_DIR))

from app.finance.stance_annotator import OfflineStanceAnnotator  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Annotate S1 social content offline")
    parser.add_argument("--run-id", required=True, help="completed s1_reddit run id")
    parser.add_argument(
        "--storage-dir",
        default=None,
        help="finance artifact directory; defaults to Config.FINANCE_ADAPTER_DATA_DIR",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-annotate content even when a successful cached label exists",
    )
    parser.add_argument(
        "--allow-primary-fallback",
        action="store_true",
        help="local debugging only: use LLM_* when STANCE_LLM_* is not configured",
    )
    args = parser.parse_args()
    result = OfflineStanceAnnotator(
        storage_dir=args.storage_dir,
        allow_primary_fallback=args.allow_primary_fallback,
    ).annotate_run(
        args.run_id, force=args.force
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
