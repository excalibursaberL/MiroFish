"""Run a six-round S1 experiment for one or more random seeds.

Examples from ``MiroFish/backend``::

    .venv/Scripts/python.exe scripts/run_s1_seed_sweep.py --seeds 4005 4006 4007 --dry-run
    .venv/Scripts/python.exe scripts/run_s1_seed_sweep.py --seeds 4005 4006 4007
    .venv/Scripts/python.exe scripts/run_s1_seed_sweep.py --seeds 42 --full-population-agent-ids 1 3 4 5 9 13 14 17

The 18 pre-built Zep graphs are reused. Seeds run serially within one process,
and each seed gets
its own ``s1_batch_*`` directory. A separate ``s1_seed_sweep_*`` manifest
links the batches so an interrupted or partial experiment remains auditable.
"""

from __future__ import annotations

import argparse
import json
import random
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
    normalize_selected_agent_ids,
)
from app.finance.s1_batch import S1BatchRunner  # noqa: E402


SOCIAL_ROUNDS = 6
EXPECTED_SCENARIO_IDS = tuple(f"SCN_{index:03d}" for index in range(1, 19))
DATA_SPLIT = "seed_stability"
SUBSET_DATA_SPLIT = "agent_subset_rerun_validation"
SUBSET_AGENT_SET_VERSION = "n10_k8_enum_best_v1"
SUBSET_SAMPLING_METHOD = "offline_exact_enumeration_k10_v2_candidate"
PROFILE_PERMUTATION_DATA_SPLIT = "profile_id_permutation"
PROFILE_PERMUTATION_METHOD = "paired_profile_runtime_derangement_v1"
PROFILE_PERMUTATION_SALT = 0x50524F46
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


def parse_agent_id_tokens(tokens: Sequence[str]) -> list[int]:
    values: list[int] = []
    for token in tokens:
        for part in str(token).split(","):
            part = part.strip()
            if not part:
                continue
            try:
                agent_id = int(part)
            except ValueError as exc:
                raise ValueError(f"invalid full-population Agent ID: {part}") from exc
            values.append(agent_id)
    return list(normalize_selected_agent_ids(values))


def build_profile_id_permutation(seed: int, agent_count: int = C0_AGENT_COUNT) -> list[int]:
    """Return a reproducible derangement indexed by runtime Agent ID."""
    if agent_count < 2:
        raise ValueError("Profile-ID permutation requires at least two Agents")
    rng = random.Random(int(seed) ^ PROFILE_PERMUTATION_SALT)
    identity = list(range(agent_count))
    permutation = identity.copy()
    for _ in range(10_000):
        rng.shuffle(permutation)
        if all(profile_id != runtime_id for runtime_id, profile_id in enumerate(permutation)):
            return permutation.copy()
    raise RuntimeError("failed to generate a Profile-ID derangement")


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


def build_plan(
    seeds: Sequence[int],
    graph_manifest: Path,
    *,
    permute_profile_ids: bool = False,
    selected_full_population_agent_ids: Sequence[int] | None = None,
    agent_set_version: str | None = None,
    sampling_method: str | None = None,
    data_split: str | None = None,
) -> dict[str, Any]:
    graphs = read_completed_graphs(graph_manifest)
    if C0_AGENT_COUNT != 10 or len(SELECTED_AGENT_IDS) != 10:
        raise RuntimeError(
            "the active finance role configuration is no longer the fixed K=10 set"
        )
    selected_ids = normalize_selected_agent_ids(selected_full_population_agent_ids)
    agent_count = len(selected_ids)
    is_default_set = selected_ids == tuple(SELECTED_AGENT_IDS)
    resolved_agent_set_version = str(
        agent_set_version
        or (DEFAULT_AGENT_SET_VERSION if is_default_set else SUBSET_AGENT_SET_VERSION)
    )
    resolved_sampling_method = str(
        sampling_method
        or (
            PROFILE_PERMUTATION_METHOD
            if permute_profile_ids
            else DEFAULT_SAMPLING_METHOD
            if is_default_set
            else SUBSET_SAMPLING_METHOD
        )
    )
    resolved_data_split = str(
        data_split
        or (
            PROFILE_PERMUTATION_DATA_SPLIT
            if permute_profile_ids
            else DATA_SPLIT
            if is_default_set
            else SUBSET_DATA_SPLIT
        )
    )
    return {
        "mode": "dry_run_plan",
        "social_rounds": SOCIAL_ROUNDS,
        "agent_count": agent_count,
        "selected_full_population_agent_ids": list(selected_ids),
        "agent_set_version": resolved_agent_set_version,
        "sampling_method": resolved_sampling_method,
        "data_split": resolved_data_split,
        "graph_manifest_path": str(graph_manifest),
        "scenario_ids": [str(item["scenario_id"]) for item in graphs],
        "scenario_count_per_seed": len(graphs),
        "seeds": list(seeds),
        "profile_id_permutation_enabled": permute_profile_ids,
        "profile_id_permutations": {
            str(seed): build_profile_id_permutation(seed, agent_count)
            for seed in seeds
        } if permute_profile_ids else {
            str(seed): list(range(agent_count))
            for seed in seeds
        },
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
    permute_profile_ids: bool = False,
    selected_full_population_agent_ids: Sequence[int] | None = None,
    agent_set_version: str | None = None,
    sampling_method: str | None = None,
    data_split: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    plan = build_plan(
        seeds,
        graph_manifest,
        permute_profile_ids=permute_profile_ids,
        selected_full_population_agent_ids=selected_full_population_agent_ids,
        agent_set_version=agent_set_version,
        sampling_method=sampling_method,
        data_split=data_split,
    )
    sweep_id = f"s1_seed_sweep_{uuid.uuid4().hex[:12]}"
    if not SWEEP_ID_PATTERN.fullmatch(sweep_id):
        raise RuntimeError("failed to generate a valid seed sweep ID")
    sweep_dir = storage_dir / sweep_id
    sweep_dir.mkdir(parents=True, exist_ok=False)
    timestamp = now()
    manifest = {
        "sweep_id": sweep_id,
        "status": "prepared",
        "experiment": (
            f"S1 K={plan['agent_count']} six-round paired Profile-ID permutation"
            if permute_profile_ids
            else f"S1 K={plan['agent_count']} six-round random-seed validation"
        ),
        "social_rounds": SOCIAL_ROUNDS,
        "agent_count": plan["agent_count"],
        "selected_full_population_agent_ids": plan[
            "selected_full_population_agent_ids"
        ],
        "agent_set_version": plan["agent_set_version"],
        "sampling_method": plan["sampling_method"],
        "data_split": plan["data_split"],
        "profile_id_permutation_enabled": permute_profile_ids,
        "graph_manifest_path": str(graph_manifest),
        "scenario_ids": list(EXPECTED_SCENARIO_IDS),
        "scenario_count_per_seed": len(EXPECTED_SCENARIO_IDS),
        "seeds": list(seeds),
        "created_at": timestamp,
        "updated_at": timestamp,
        "batches": [
            {
                "seed": seed,
                "profile_id_permutation": (
                    build_profile_id_permutation(seed, plan["agent_count"])
                    if permute_profile_ids
                    else list(range(plan["agent_count"]))
                ),
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
    permute_profile_ids: bool = False,
    selected_full_population_agent_ids: Sequence[int] | None = None,
    agent_set_version: str | None = None,
    sampling_method: str | None = None,
    data_split: str | None = None,
) -> dict[str, Any]:
    build_plan(
        seeds,
        graph_manifest,
        permute_profile_ids=permute_profile_ids,
        selected_full_population_agent_ids=selected_full_population_agent_ids,
        agent_set_version=agent_set_version,
        sampling_method=sampling_method,
        data_split=data_split,
    )
    sweep_dir, sweep = create_sweep_manifest(
        storage_dir=storage_dir,
        seeds=seeds,
        graph_manifest=graph_manifest,
        permute_profile_ids=permute_profile_ids,
        selected_full_population_agent_ids=selected_full_population_agent_ids,
        agent_set_version=agent_set_version,
        sampling_method=sampling_method,
        data_split=data_split,
    )
    runner = S1BatchRunner(storage_dir=storage_dir)
    sweep["status"] = "running"
    sweep["updated_at"] = now()
    write_json(sweep_dir / "manifest.json", sweep)

    for index, entry in enumerate(sweep["batches"], start=1):
        seed = int(entry["seed"])
        profile_id_permutation = list(entry["profile_id_permutation"])
        print(
            f"[{index}/{len(seeds)}] preparing seed={seed}, "
            f"rounds={SOCIAL_ROUNDS}, scenarios=18, agents={sweep['agent_count']}",
            flush=True,
        )
        try:
            prepared = runner.prepare(
                social_rounds=SOCIAL_ROUNDS,
                graph_manifest_path=graph_manifest,
                data_split=sweep["data_split"],
                replicate_id=(
                    f"rounds6_all18_k{sweep['agent_count']}_profileperm_seed{seed}_{sweep['sweep_id']}"
                    if permute_profile_ids
                    else f"rounds6_all18_k{sweep['agent_count']}_seed{seed}_{sweep['sweep_id']}"
                ),
                agent_set_version=sweep["agent_set_version"],
                sampling_method=sweep["sampling_method"],
                random_seed=seed,
                profile_id_permutation=profile_id_permutation,
                selected_full_population_agent_ids=sweep[
                    "selected_full_population_agent_ids"
                ],
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
        "--full-population-agent-ids",
        nargs="+",
        metavar="AGENT_ID",
        help=(
            "subset of the configured K=10 full-population Agent IDs; "
            "spaces and comma-separated values are accepted"
        ),
    )
    parser.add_argument(
        "--agent-set-version",
        help="auditable Agent-set version override",
    )
    parser.add_argument(
        "--sampling-method",
        help="auditable sampling-method override",
    )
    parser.add_argument(
        "--data-split",
        help="auditable data-split override",
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
    parser.add_argument(
        "--permute-profile-ids",
        action="store_true",
        help=(
            "assign each canonical Profile to a different runtime Agent ID "
            "using a reproducible seed-specific derangement"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        seeds = parse_seed_tokens(args.seeds)
        selected_agent_ids = (
            parse_agent_id_tokens(args.full_population_agent_ids)
            if args.full_population_agent_ids
            else None
        )
        graph_manifest = args.graph_manifest.resolve()
        storage_dir = args.storage_dir.resolve()
        if args.dry_run:
            result = build_plan(
                seeds,
                graph_manifest,
                permute_profile_ids=args.permute_profile_ids,
                selected_full_population_agent_ids=selected_agent_ids,
                agent_set_version=args.agent_set_version,
                sampling_method=args.sampling_method,
                data_split=args.data_split,
            )
        else:
            result = run_sweep(
                seeds=seeds,
                graph_manifest=graph_manifest,
                storage_dir=storage_dir,
                continue_on_error=args.continue_on_error,
                permute_profile_ids=args.permute_profile_ids,
                selected_full_population_agent_ids=selected_agent_ids,
                agent_set_version=args.agent_set_version,
                sampling_method=args.sampling_method,
                data_split=args.data_split,
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
