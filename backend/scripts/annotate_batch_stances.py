"""Annotate every completed S1 run referenced by a batch manifest.

The per-run annotator is resumable: successful labels are reused when their
content hash is unchanged. This wrapper adds batch-level progress and quality
artifacts without modifying the raw simulation traces.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND_DIR))

from app.finance.stance_annotator import OfflineStanceAnnotator  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def annotate_batch(
    batch_dir: Path,
    *,
    force: bool = False,
    allow_primary_fallback: bool = False,
) -> dict[str, Any]:
    manifest_path = batch_dir / "manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("status") != "completed":
        raise ValueError("stance annotation requires a completed S1 batch")

    annotator = OfflineStanceAnnotator(
        storage_dir=batch_dir.parent,
        allow_primary_fallback=allow_primary_fallback,
    )
    scenario_rows: list[dict[str, Any]] = []
    for index, run in enumerate(manifest.get("runs", []), start=1):
        if run.get("status") != "completed" or not run.get("run_id"):
            continue
        scenario_id = str(run.get("scenario_id", ""))
        run_id = str(run["run_id"])
        print(
            f"[{index}/{len(manifest['runs'])}] annotating {scenario_id} ({run_id})",
            flush=True,
        )
        try:
            result = annotator.annotate_run(run_id, force=force)
            row = {
                "scenario_id": scenario_id,
                "run_id": run_id,
                "status": result["status"],
                "content_count": result["content_count"],
                "success_count": result["success_count"],
                "failed_count": result["failed_count"],
                "annotator_model": result["annotator_model"],
                "prompt_version": result["prompt_version"],
                "config_source": result["config_source"],
                "error": "",
            }
        except Exception as exc:  # Keep the remaining scenarios resumable.
            row = {
                "scenario_id": scenario_id,
                "run_id": run_id,
                "status": "failed",
                "content_count": 0,
                "success_count": 0,
                "failed_count": 0,
                "annotator_model": annotator.model or "",
                "prompt_version": annotator.PROMPT_VERSION,
                "config_source": annotator.config_source,
                "error": str(exc),
            }
        scenario_rows.append(row)
        write_csv(batch_dir / "stance_annotation_scenarios.csv", scenario_rows)

    annotations: list[dict[str, Any]] = []
    for row in scenario_rows:
        annotations.extend(
            read_jsonl(batch_dir.parent / row["run_id"] / "stance_annotations.jsonl")
        )
    status_counts = Counter(str(row.get("status")) for row in annotations)
    stance_counts = Counter(
        str(row.get("stance"))
        for row in annotations
        if row.get("status") == "ok"
    )
    summary = {
        "batch_id": manifest.get("batch_id"),
        "status": (
            "completed"
            if scenario_rows and all(row["status"] == "completed" for row in scenario_rows)
            else "completed_with_errors"
        ),
        "scenario_count": len(scenario_rows),
        "content_count": len(annotations),
        "success_count": status_counts.get("ok", 0),
        "failed_count": len(annotations) - status_counts.get("ok", 0),
        "stance_counts": dict(sorted(stance_counts.items())),
        "annotator_model": annotator.model,
        "annotator_base_url": annotator.base_url,
        "config_source": annotator.config_source,
        "prompt_version": annotator.PROMPT_VERSION,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    (batch_dir / "stance_annotation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest.setdefault("files", {}).update(
        {
            "stance_annotation_scenarios": "stance_annotation_scenarios.csv",
            "stance_annotation_summary": "stance_annotation_summary.json",
        }
    )
    manifest["stance_annotation"] = summary
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_dir", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-primary-fallback", action="store_true")
    args = parser.parse_args()
    result = annotate_batch(
        args.batch_dir.resolve(),
        force=args.force,
        allow_primary_fallback=args.allow_primary_fallback,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
