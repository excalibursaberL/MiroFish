"""Run the fixed K=10, six-round S1 experiment for one or more seeds.

Examples from ``MiroFish/backend``::

    .venv/Scripts/python.exe scripts/run_s1_seed_sweep.py --seeds 4005 4006 4007 --dry-run
    .venv/Scripts/python.exe scripts/run_s1_seed_sweep.py --seeds 4005 4006 4007

The 18 pre-built Zep graphs are reused. Seeds run serially, and each seed gets
its own ``s1_batch_*`` directory. A separate ``s1_seed_sweep_*`` manifest
links the batches so an interrupted or partial experiment remains auditable.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


_BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND_DIR))

from app.config import Config  # noqa: E402
from app.finance.dataset import FinancialDatasetLoader  # noqa: E402
from app.finance.roles import (  # noqa: E402
    C0_AGENT_COUNT,
    DEFAULT_AGENT_SET_VERSION,
    DEFAULT_SAMPLING_METHOD,
    SELECTED_AGENT_IDS,
)
from app.finance.s1_batch import S1BatchRunner  # noqa: E402


SOCIAL_ROUNDS = 6
EXPECTED_SCENARIO_IDS = tuple(f"SCN_{index:03d}" for index in range(1, 19))
DATA_SPLIT = "seed_stability"
SEED_MIN = 0
SEED_MAX = 0xFFFFFFFF
SWEEP_ID_PATTERN = re.compile(r"s1_seed_sweep_[a-f0-9]{12}")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_seed_tokens(tokens: Sequence[str]) -> list[int]:
    values: list[int] = []
    for token in tokens:
        for part in str(token).split(","):
            part = part.strip()
            if not part:
                continue
            try:
                seed = int(part)
            except ValueError as exc:
                raise ValueError(f"invalid random seed: {part}") from exc
            if not SEED_MIN <= seed <= SEED_MAX:
                raise ValueError(
                    f"random seed must be between {SEED_MIN} and {SEED_MAX}: {seed}"
                )
            if seed in values:
                raise ValueError(f"duplicate random seed: {seed}")
            values.append(seed)
    if not values:
        raise ValueError("at least one random seed is required")
    return values


def read_completed_graphs(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    completed = [
        item
        for item in payload.get("scenarios", [])
        if item.get("status") == "completed"
        and item.get("scenario_id")
        and item.get("graph_id")
    ]
    scenario_ids = tuple(str(item["scenario_id"]) for item in completed)
    if scenario_ids != EXPECTED_SCENARIO_IDS:
        raise ValueError(
            "the graph manifest must contain completed SCN_001 through SCN_018 "
            f"in order; found {list(scenario_ids)}"
        )
    if len({str(item["graph_id"]) for item in completed}) != len(completed):
        raise ValueError("the graph manifest contains duplicate graph IDs")
    FinancialDatasetLoader().load(scenario_ids=list(scenario_ids))
    return completed


def build_plan(seeds: Sequence[int], graph_manifest: Path) -> dict[str, Any]:
    graphs = read_completed_graphs(graph_manifest)
    if C0_AGENT_COUNT != 10 or len(SELECTED_AGENT_IDS) != 10:
        raise RuntimeError(
            "the active finance role configuration is no longer the fixed K=10 set"
        )
    return {
        "mode": "dry_run_plan",
        "social_rounds": SOCIAL_ROUNDS,
        "agent_count": C0_AGENT_COUNT,
        "selected_full_population_agent_ids": list(SELECTED_AGENT_IDS),
        "agent_set_version": DEFAULT_AGENT_SET_VERSION,
        "sampling_method": DEFAULT_SAMPLING_METHOD,
        "graph_manifest_path": str(graph_manifest),
        "scenario_ids": [str(item["scenario_id"]) for item in graphs],
        "scenario_count_per_seed": len(graphs),
        "seeds": list(seeds),
        "seed_count": len(seeds),
        "total_scenario_runs": len(graphs) * len(seeds),
        "execution": "serial",
        "calls_external_llm": False,
        "note": (
            "A real run calls the configured LLM. Local random sources are seeded, "
            "but the DeepSeek service does not guarantee token-identical responses."
        ),
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def create_sweep_manifest(
    *,
    storage_dir: Path,
    seeds: Sequence[int],
    graph_manifest: Path,
) -> tuple[Path, dict[str, Any]]:
    sweep_id = f"s1_seed_sweep_{uuid.uuid4().hex[:12]}"
    if not SWEEP_ID_PATTERN.fullmatch(sweep_id):
        raise RuntimeError("failed to generate a valid seed sweep ID")
    sweep_dir = storage_dir / sweep_id
    sweep_dir.mkdir(parents=True, exist_ok=False)
    timestamp = now()
    manifest = {
        "sweep_id": sweep_id,
        "status": "prepared",
        "experiment": "S1 K=10 six-round random-seed stability",
        "social_rounds": SOCIAL_ROUNDS,
        "agent_count": C0_AGENT_COUNT,
        "selected_full_population_agent_ids": list(SELECTED_AGENT_IDS),
        "agent_set_version": DEFAULT_AGENT_SET_VERSION,
        "sampling_method": DEFAULT_SAMPLING_METHOD,
        "data_split": DATA_SPLIT,
        "graph_manifest_path": str(graph_manifest),
        "scenario_ids": list(EXPECTED_SCENARIO_IDS),
        "scenario_count_per_seed": len(EXPECTED_SCENARIO_IDS),
        "seeds": list(seeds),
        "created_at": timestamp,
        "updated_at": timestamp,
        "batches": [
            {
                "seed": seed,
                "status": "pending",
                "batch_id": None,
                "batch_manifest_path": None,
                "completed_scenario_count": 0,
                "failed_scenario_count": 0,
                "error": None,
            }
            for seed in seeds
        ],
    }
    write_json(sweep_dir / "manifest.json", manifest)
    return sweep_dir, manifest


def run_sweep(
    *,
    seeds: Sequence[int],
    graph_manifest: Path,
    storage_dir: Path,
    continue_on_error: bool = False,
) -> dict[str, Any]:
    build_plan(seeds, graph_manifest)
    sweep_dir, sweep = create_sweep_manifest(
        storage_dir=storage_dir,
        seeds=seeds,
        graph_manifest=graph_manifest,
    )
    runner = S1BatchRunner(storage_dir=storage_dir)
    sweep["status"] = "running"
    sweep["updated_at"] = now()
    write_json(sweep_dir / "manifest.json", sweep)

    for index, entry in enumerate(sweep["batches"], start=1):
        seed = int(entry["seed"])
        print(
            f"[{index}/{len(seeds)}] preparing seed={seed}, "
            f"rounds={SOCIAL_ROUNDS}, scenarios=18, agents=10",
            flush=True,
        )
        try:
            prepared = runner.prepare(
                social_rounds=SOCIAL_ROUNDS,
                graph_manifest_path=graph_manifest,
                data_split=DATA_SPLIT,
                replicate_id=f"rounds6_all18_k10_seed{seed}_{sweep['sweep_id']}",
                agent_set_version=DEFAULT_AGENT_SET_VERSION,
                sampling_method=DEFAULT_SAMPLING_METHOD,
                random_seed=seed,
            )
            batch_id = str(prepared["batch_id"])
            entry.update(
                {
                    "status": "running",
                    "batch_id": batch_id,
                    "batch_manifest_path": str(
                        storage_dir / batch_id / "manifest.json"
                    ),
                    "error": None,
                }
            )
            sweep["updated_at"] = now()
            write_json(sweep_dir / "manifest.json", sweep)
            result = runner.run_sync(batch_id)
            entry.update(
                {
                    "status": result.get("status"),
                    "completed_scenario_count": result.get(
                        "completed_scenario_count", 0
                    ),
                    "failed_scenario_count": result.get(
                        "failed_scenario_count", 0
                    ),
                    "error": result.get("error"),
                }
            )
            print(
                f"[{index}/{len(seeds)}] seed={seed} finished: "
                f"status={entry['status']}, batch_id={batch_id}",
                flush=True,
            )
        except KeyboardInterrupt:
            entry.update({"status": "interrupted", "error": "KeyboardInterrupt"})
            sweep.update({"status": "interrupted", "updated_at": now()})
            write_json(sweep_dir / "manifest.json", sweep)
            raise
        except Exception as exc:
            entry.update({"status": "failed", "error": str(exc)})
            print(f"seed={seed} failed: {exc}", file=sys.stderr, flush=True)
        finally:
            sweep["updated_at"] = now()
            write_json(sweep_dir / "manifest.json", sweep)

        if entry["status"] != "completed" and not continue_on_error:
            sweep["status"] = "stopped_on_error"
            break

    else:
        sweep["status"] = (
            "completed"
            if all(entry["status"] == "completed" for entry in sweep["batches"])
            else "partial_failed"
        )
    sweep["updated_at"] = now()
    write_json(sweep_dir / "manifest.json", sweep)
    return {**sweep, "sweep_manifest_path": str(sweep_dir / "manifest.json")}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        nargs="+",
        required=True,
        metavar="SEED",
        help="distinct integer seeds; spaces and comma-separated values are accepted",
    )
    parser.add_argument(
        "--graph-manifest",
        type=Path,
        default=S1BatchRunner.DEFAULT_GRAPH_MANIFEST,
        help="completed Zep graph manifest; must cover SCN_001 through SCN_018",
    )
    parser.add_argument(
        "--storage-dir",
        type=Path,
        default=Path(getattr(Config, "FINANCE_ADAPTER_DATA_DIR")),
        help="directory containing s1_batch_* and s1_reddit_* artifacts",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the plan without creating files or calling an LLM",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="continue with later seeds when a seed batch is partial or failed",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        seeds = parse_seed_tokens(args.seeds)
        graph_manifest = args.graph_manifest.resolve()
        storage_dir = args.storage_dir.resolve()
        if args.dry_run:
            result = build_plan(seeds, graph_manifest)
        else:
            result = run_sweep(
                seeds=seeds,
                graph_manifest=graph_manifest,
                storage_dir=storage_dir,
                continue_on_error=args.continue_on_error,
            )
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0 if result.get("status") in {None, "completed"} else 1
    except KeyboardInterrupt:
        print("seed sweep interrupted", file=sys.stderr, flush=True)
        return 130
    except Exception as exc:
        print(f"seed sweep failed: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
