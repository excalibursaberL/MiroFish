"""Build isolated Zep Cloud graphs for anonymous benchmark scenarios.

The script drives the same HTTP endpoints used by the MiroFish frontend.  It
reconstructs the upload text from structured seed events instead of trusting
the legacy ``mirofish_seed_text`` field, which may be stale.  The private
mapping is read only for a local leakage check and is never written to the
graph, logs, or manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "Dataset" / "seed5_small_blind" / "mirofish_inputs.jsonl"
PRIVATE_MAPPING = ROOT / "Dataset" / "reports" / "seed5_small_blind_mapping_private.tsv"
OUTPUT_DIR = ROOT / "Dataset" / "seed5_small_blind" / "zep_inputs"
MANIFEST = ROOT / "Dataset" / "seed5_small_blind" / "zep_graphs_manifest.json"


# Names that the previous blind-copy pass failed to replace in a few event
# sentences.  These are replaced by role-preserving aliases, not a generic
# token, so the graph still captures the event semantics.
RESIDUAL_ALIASES = {
    "东北制药": "医药企业B",
    "吉林化纤": "化纤企业A",
    "永福股份": "工程企业A",
    "重庆太衡": "地产企业B",
    "宜宾鲁能": "地产企业C",
    "林中漫步": "物业项目A",
    "鑫领寓": "物业项目B",
    "方威": "主要股东A",
    "张建军": "公司负责人A",
    "高毅邓晓峰": "投资管理人A",
    "小财": "财经媒体",
    "上海": "某地区",
    "重庆": "某地区",
    "成都": "某地区",
    "福州": "某地区",
    "合肥": "某地区",
    "内蒙古赤峰市": "某地区",
    "双流": "某地区",
}

ABSOLUTE_DATE_RE = re.compile(
    r"(?<![A-Za-z])20\d{2}(?:[-/.]\d{1,2}(?:[-/.]\d{1,2})?|年\d{1,2}月(?:\d{1,2}日)?)"
)


def load_scenarios() -> dict[str, dict[str, Any]]:
    scenarios: dict[str, dict[str, Any]] = {}
    with DATASET.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            scenarios[payload["scenario_id"]] = payload
    return scenarios


def load_private_names() -> dict[str, str]:
    """Return scenario -> original target name for local redaction checks."""

    names: dict[str, str] = {}
    if not PRIVATE_MAPPING.exists():
        return names
    with PRIVATE_MAPPING.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            names[row["scenario_id_blind"]] = row["NAME_original"]
    return names


def target_variants(name: str) -> set[str]:
    value = name.strip()
    if not value:
        return set()
    variants = {value}
    short = re.sub(r"^[*STst]+", "", value).strip()
    if short:
        variants.update({short, f"ST{short}", f"*ST{short}"})
    # A few source NAME fields include a legal suffix while the event uses the
    # trading-name stem (for example "岩石股份" -> "ST岩石").
    for suffix in ("股份", "有限公司", "集团"):
        if short.endswith(suffix) and len(short) > len(suffix) + 1:
            stem = short[: -len(suffix)]
            variants.update({stem, f"ST{stem}", f"*ST{stem}"})
    return {v for v in variants if len(v) >= 2}


def redact_text(text: str, scenario: dict[str, Any], private_name: str) -> str:
    """Remove residual identities while retaining role and event meaning."""

    result = str(text)
    for value in sorted(target_variants(private_name), key=len, reverse=True):
        result = result.replace(value, scenario["name"])
    for value, replacement in sorted(RESIDUAL_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        result = result.replace(value, replacement)

    # Keep the issuer anonymous even if a source sentence uses a trading name
    # directly before “公告”.  Existing COMPANY_### tokens are left intact.
    result = re.sub(
        r"(?<![A-Z0-9_])(?:\*?ST)?[\u4e00-\u9fff]{2,12}(?=公告)",
        scenario["name"],
        result,
    )
    result = ABSOLUTE_DATE_RE.sub("DATE_REL", result)
    return result


def build_upload_text(scenario: dict[str, Any], private_name: str) -> str:
    events = sorted(scenario["seed_events"], key=lambda item: int(item.get("seed_rank", 0)))
    sections: list[str] = []
    for index, event in enumerate(events, start=1):
        event_text = redact_text(event["text"], scenario, private_name)
        sections.append(
            f"[Historical event {index} | {event['event_id']} | {event['event_time']}]\n{event_text}"
        )
    current = scenario["current_event"]
    current_text = redact_text(current["text"], scenario, private_name)
    sections.append(
        f"[Current public event | {current['event_id']} | {current['event_time']}]\n{current_text}"
    )
    return "\n\n".join(sections)


def assert_safe(payload: Any, forbidden_names: set[str], label: str) -> None:
    text = json.dumps(payload, ensure_ascii=False)
    leaked = [name for name in forbidden_names if name and name in text]
    if leaked:
        raise RuntimeError(f"{label} contains a private identity; refusing Cloud write")
    if ABSOLUTE_DATE_RE.search(text):
        raise RuntimeError(f"{label} contains an absolute date; refusing Cloud write")


def load_manifest() -> dict[str, dict[str, Any]]:
    if not MANIFEST.exists():
        return {}
    try:
        payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {row["scenario_id"]: row for row in payload.get("scenarios", []) if row.get("scenario_id")}


def save_manifest(rows: dict[str, dict[str, Any]]) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "Dataset/seed5_small_blind/mirofish_inputs.jsonl",
        "input_source": "structured seed_events + current_event (mirofish_seed_text intentionally bypassed)",
        "scenarios": [rows[key] for key in sorted(rows)],
    }
    temp = MANIFEST.with_suffix(".json.tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(MANIFEST)


def api_json(client: httpx.Client, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
    response = client.request(method, path, **kwargs)
    try:
        body = response.json()
    except ValueError:
        raise RuntimeError(f"{method} {path} returned HTTP {response.status_code} with non-JSON body")
    if response.status_code >= 400 or not body.get("success", False):
        raise RuntimeError(f"{method} {path} failed (HTTP {response.status_code}): {body.get('error', body)}")
    return body


def build_one(
    client: httpx.Client,
    scenario: dict[str, Any],
    private_name: str,
    existing: dict[str, dict[str, Any]],
    poll_seconds: float,
) -> dict[str, Any]:
    scenario_id = scenario["scenario_id"]
    if existing.get(scenario_id, {}).get("status") == "completed":
        return existing[scenario_id]

    upload_text = build_upload_text(scenario, private_name)
    forbidden = set(target_variants(private_name))
    assert_safe(upload_text, forbidden, f"{scenario_id} input")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    input_path = OUTPUT_DIR / f"{scenario_id}.txt"
    input_path.write_text(upload_text, encoding="utf-8")

    requirement = (
        "Build a knowledge graph for an anonymous A-share social-interaction "
        "prediction benchmark. Extract only actors and relations explicitly "
        "present in the supplied historical and current public events. Keep "
        "ASSET/COMPANY/EVT identifiers unchanged. Do not infer future prices, "
        "labels, dates, or identities."
    )
    files = {"files": (f"{scenario_id}.txt", upload_text.encode("utf-8"), "text/plain")}
    data = {
        "simulation_requirement": requirement,
        "project_name": f"Finance {scenario_id} Anonymous Graph",
        "additional_context": "Use only the six point-in-time events in this file. Relative T-n dates are intentional.",
    }
    ontology_response = api_json(client, "POST", "/api/graph/ontology/generate", files=files, data=data)
    ontology_data = ontology_response["data"]
    project_id = ontology_data["project_id"]
    assert_safe(ontology_data.get("ontology", {}), forbidden, f"{scenario_id} ontology")

    build_response = api_json(
        client,
        "POST",
        "/api/graph/build",
        json={"project_id": project_id, "graph_name": f"Finance {scenario_id} Anonymous Graph"},
    )
    task_id = build_response["data"]["task_id"]
    started = time.monotonic()
    task: dict[str, Any] = {}
    while time.monotonic() - started < 1800:
        task = api_json(client, "GET", f"/api/graph/task/{task_id}")["data"]
        status = task.get("status")
        if status in {"completed", "failed"}:
            break
        time.sleep(poll_seconds)
    if task.get("status") != "completed":
        raise RuntimeError(f"{scenario_id} graph build did not complete: {task.get('error') or task}")

    result = task.get("result") or {}
    graph_id = result.get("graph_id")
    graph_data = api_json(client, "GET", f"/api/graph/data/{graph_id}")["data"]
    row = {
        "scenario_id": scenario_id,
        "project_id": project_id,
        "graph_id": graph_id,
        "task_id": task_id,
        "zep_batch_id": result.get("zep_batch_id"),
        "chunk_count": result.get("chunk_count"),
        "node_count": graph_data.get("node_count", 0),
        "edge_count": graph_data.get("edge_count", 0),
        "status": "completed",
        "input_file": str(input_path.relative_to(ROOT)).replace("\\", "/"),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", action="append", help="Scenario ID; repeat for multiple IDs")
    parser.add_argument("--base-url", default="http://127.0.0.1:5001")
    parser.add_argument("--poll-seconds", type=float, default=3.0)
    args = parser.parse_args()

    scenarios = load_scenarios()
    selected = args.scenario or [f"SCN_{i:03d}" for i in range(8, 19)]
    missing = [scenario_id for scenario_id in selected if scenario_id not in scenarios]
    if missing:
        raise SystemExit(f"Scenario IDs not found: {', '.join(missing)}")
    private_names = load_private_names()
    rows = load_manifest()
    with httpx.Client(base_url=args.base_url, timeout=240) as client:
        for scenario_id in selected:
            print(f"[{scenario_id}] starting", flush=True)
            try:
                row = build_one(client, scenarios[scenario_id], private_names.get(scenario_id, ""), rows, args.poll_seconds)
                rows[scenario_id] = row
                save_manifest(rows)
                print(f"[{scenario_id}] completed graph={row['graph_id']} nodes={row['node_count']} edges={row['edge_count']}", flush=True)
            except Exception as error:
                rows[scenario_id] = {
                    "scenario_id": scenario_id,
                    "status": "failed",
                    "error": str(error),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                save_manifest(rows)
                print(f"[{scenario_id}] failed: {error}", file=sys.stderr, flush=True)
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
