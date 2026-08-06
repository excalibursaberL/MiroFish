"""Serial batch execution for S1 scenarios with pre-built Zep graphs."""

from __future__ import annotations

import json
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from ..config import Config
from ..utils.logger import get_logger
from .dataset import FinancialDatasetLoader, PROJECT_ROOT
from .s1 import S1ExperimentService


logger = get_logger("mirofish.finance.s1_batch")


class S1BatchRunner:
    """Run every completed graph-manifest scenario, one OASIS run at a time."""

    DEFAULT_GRAPH_MANIFEST = (
        PROJECT_ROOT / "Dataset" / "seed5_small_blind" / "zep_graphs_manifest.json"
    )
    BATCH_ID_PATTERN = re.compile(r"s1_batch_[A-Za-z0-9_-]{6,64}")
    _lock = threading.Lock()
    _threads: Dict[str, threading.Thread] = {}
    SUMMARY_FIELDS = (
        "replicate_id",
        "agent_set_version",
        "sampling_method",
        "data_split",
        "random_seed",
        "scenario_id",
        "run_id",
        "status",
        "direction_flip_rate",
        "mean_distribution_js_divergence",
        "mean_expected_return_delta",
        "mean_confidence_delta",
        "pre_consensus_rate",
        "post_consensus_rate",
        "pre_direction_entropy_bits",
        "post_direction_entropy_bits",
        "pre_polarization",
        "post_polarization",
        "investor_action_count",
        "belief_snapshot_valid_rate",
        "exposure_edge_count",
        "error",
    )

    def __init__(self, *, storage_dir: Optional[str | Path] = None) -> None:
        self.storage_dir = Path(
            storage_dir or getattr(Config, "FINANCE_ADAPTER_DATA_DIR")
        ).resolve()
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _batch_dir(self, batch_id: str) -> Path:
        if not isinstance(batch_id, str) or not self.BATCH_ID_PATTERN.fullmatch(batch_id):
            raise ValueError("invalid S1 batch_id")
        return self.storage_dir / batch_id

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        S1ExperimentService._write_json(path, value)

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(path)
        return json.loads(path.read_text(encoding="utf-8"))

    def prepare(
        self,
        *,
        social_rounds: int = S1ExperimentService.DEFAULT_SOCIAL_ROUNDS,
        graph_manifest_path: Optional[str | Path] = None,
        data_split: str = S1ExperimentService.DEFAULT_DATA_SPLIT,
        replicate_id: Optional[str] = None,
        agent_set_version: Optional[str] = None,
        sampling_method: str = S1ExperimentService.DEFAULT_SAMPLING_METHOD,
        random_seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        if isinstance(social_rounds, bool) or not isinstance(social_rounds, int):
            raise ValueError("social_rounds must be an integer")
        if not (
            S1ExperimentService.MIN_SOCIAL_ROUNDS
            <= social_rounds
            <= S1ExperimentService.MAX_SOCIAL_ROUNDS
        ):
            raise ValueError(
                f"social_rounds must be between {S1ExperimentService.MIN_SOCIAL_ROUNDS} "
                f"and {S1ExperimentService.MAX_SOCIAL_ROUNDS}"
            )
        graph_manifest = Path(
            graph_manifest_path or self.DEFAULT_GRAPH_MANIFEST
        ).resolve()
        payload = self._read_json(graph_manifest)
        completed = [
            item
            for item in payload.get("scenarios", [])
            if item.get("status") == "completed"
            and item.get("scenario_id")
            and item.get("graph_id")
        ]
        if not completed:
            raise ValueError("graph manifest has no completed scenarios")
        scenario_ids = [str(item["scenario_id"]) for item in completed]
        if len(scenario_ids) != len(set(scenario_ids)):
            raise ValueError("graph manifest contains duplicate scenario IDs")
        FinancialDatasetLoader().load(scenario_ids=scenario_ids)

        batch_id = f"s1_batch_{uuid.uuid4().hex[:12]}"
        batch_dir = self._batch_dir(batch_id)
        batch_dir.mkdir(parents=True, exist_ok=False)
        manifest = {
            "batch_id": batch_id,
            "group": "S1",
            "platform": "reddit",
            "run_mode": "all_prebuilt_graphs",
            "graph_manifest_path": str(graph_manifest),
            "social_rounds": social_rounds,
            "data_split": str(data_split or S1ExperimentService.DEFAULT_DATA_SPLIT),
            "replicate_id": replicate_id,
            "agent_set_version": str(
                agent_set_version or S1ExperimentService.DEFAULT_AGENT_SET_VERSION
            ),
            "sampling_method": str(
                sampling_method or S1ExperimentService.DEFAULT_SAMPLING_METHOD
            ),
            "random_seed": random_seed,
            "scenario_count": len(completed),
            "completed_scenario_count": 0,
            "failed_scenario_count": 0,
            "current_scenario_id": None,
            "status": "prepared",
            "created_at": self._now(),
            "updated_at": self._now(),
            "files": {"scenario_summary": "scenario_summary.csv"},
            "runs": [
                {
                    "scenario_id": str(item["scenario_id"]),
                    "project_id": str(item.get("project_id", "")),
                    "graph_id": str(item["graph_id"]),
                    "status": "pending",
                    "run_id": None,
                    "error": None,
                }
                for item in completed
            ],
        }
        self._write_json(batch_dir / "manifest.json", manifest)
        self._write_summary(batch_dir, manifest)
        return manifest

    def _write_summary(self, batch_dir: Path, manifest: Dict[str, Any]) -> None:
        records = []
        for item in manifest.get("runs", []):
            metrics = item.get("metrics") or {}
            change = metrics.get("group_change") or {}
            pre = metrics.get("pre_social") or {}
            post = metrics.get("post_social") or {}
            behavior = metrics.get("social_behavior") or {}
            snapshots = metrics.get("belief_snapshots") or {}
            expected_snapshots = int(snapshots.get("expected_count", 0) or 0)
            records.append(
                {
                    "replicate_id": (metrics.get("replicate_id") or manifest.get("replicate_id")),
                    "agent_set_version": (
                        metrics.get("agent_set_version")
                        or manifest.get("agent_set_version")
                    ),
                    "sampling_method": (
                        metrics.get("sampling_method")
                        or manifest.get("sampling_method")
                    ),
                    "data_split": metrics.get("data_split") or manifest.get("data_split"),
                    "random_seed": metrics.get("random_seed", manifest.get("random_seed")),
                    "scenario_id": item.get("scenario_id"),
                    "run_id": item.get("run_id"),
                    "status": item.get("status"),
                    "direction_flip_rate": change.get("direction_flip_rate"),
                    "mean_distribution_js_divergence": change.get(
                        "mean_distribution_js_divergence"
                    ),
                    "mean_expected_return_delta": change.get(
                        "mean_expected_return_delta"
                    ),
                    "mean_confidence_delta": change.get("mean_confidence_delta"),
                    "pre_consensus_rate": pre.get("consensus_rate"),
                    "post_consensus_rate": post.get("consensus_rate"),
                    "pre_direction_entropy_bits": pre.get("direction_entropy_bits"),
                    "post_direction_entropy_bits": post.get("direction_entropy_bits"),
                    "pre_polarization": pre.get("mean_pairwise_js_divergence"),
                    "post_polarization": post.get("mean_pairwise_js_divergence"),
                    "investor_action_count": behavior.get("investor_action_count"),
                    "belief_snapshot_valid_rate": (
                        float(snapshots.get("valid_count", 0) or 0) / expected_snapshots
                        if expected_snapshots else None
                    ),
                    "exposure_edge_count": behavior.get("exposure_edge_count"),
                    "error": item.get("error"),
                }
            )
        S1ExperimentService._write_csv(
            batch_dir / "scenario_summary.csv", records, self.SUMMARY_FIELDS
        )

    def _try_write_summary(
        self, batch_dir: Path, manifest: Dict[str, Any]
    ) -> bool:
        """Keep the batch alive when Windows temporarily locks the summary."""
        try:
            self._write_summary(batch_dir, manifest)
            manifest.pop("summary_write_error", None)
            return True
        except PermissionError as error:
            manifest["summary_write_error"] = str(error)
            logger.warning(
                "S1 batch summary is locked; scenario execution will continue: %s",
                batch_dir / "scenario_summary.csv",
            )
            return False

    @classmethod
    def is_active(cls, batch_id: str) -> bool:
        with cls._lock:
            thread = cls._threads.get(batch_id)
            return bool(thread and thread.is_alive())

    def start(self, batch_id: str) -> Dict[str, Any]:
        manifest = self.get_status(batch_id, reconcile=False)
        if manifest.get("status") != "prepared":
            raise ValueError("S1 batch must be prepared before it can run")
        with self._lock:
            if any(thread.is_alive() for thread in self._threads.values()):
                raise ValueError("another S1 batch is already active")
            manifest.update({"status": "queued", "updated_at": self._now()})
            self._write_json(self._batch_dir(batch_id) / "manifest.json", manifest)
            thread = threading.Thread(
                target=self._execute,
                args=(batch_id,),
                name=f"finance-{batch_id}",
                daemon=True,
            )
            self._threads[batch_id] = thread
            thread.start()
        return manifest

    def _execute(self, batch_id: str) -> None:
        path = self._batch_dir(batch_id) / "manifest.json"
        manifest = self._read_json(path)
        manifest.update({"status": "running", "updated_at": self._now()})
        self._write_json(path, manifest)
        try:
            for item in manifest["runs"]:
                if item.get("status") == "completed":
                    continue
                scenario_id = item["scenario_id"]
                manifest["current_scenario_id"] = scenario_id
                item.update({"status": "preparing", "error": None})
                manifest["updated_at"] = self._now()
                self._write_json(path, manifest)
                try:
                    service = S1ExperimentService()
                    run_manifest = service.prepare(
                        scenario_id=scenario_id,
                        graph_id=item["graph_id"],
                        source_mode="graph",
                        social_rounds=int(manifest["social_rounds"]),
                        data_split=manifest.get("data_split", "unspecified"),
                        replicate_id=(
                            f"{manifest.get('replicate_id') or batch_id}:{scenario_id}"
                        ),
                        agent_set_version=manifest.get("agent_set_version"),
                        sampling_method=manifest.get("sampling_method", "full"),
                        random_seed=manifest.get("random_seed"),
                    )
                    item.update(
                        {"status": "running", "run_id": run_manifest["run_id"]}
                    )
                    self._write_json(path, manifest)
                    service.run_sync(run_manifest["run_id"])
                    item["metrics"] = service.get_metrics(run_manifest["run_id"])
                    item["status"] = "completed"
                except Exception as error:
                    logger.exception(
                        "S1 batch scenario failed: batch_id=%s, scenario_id=%s",
                        batch_id,
                        scenario_id,
                    )
                    item.update({"status": "failed", "error": str(error)})
                manifest["completed_scenario_count"] = sum(
                    run.get("status") == "completed" for run in manifest["runs"]
                )
                manifest["failed_scenario_count"] = sum(
                    run.get("status") == "failed" for run in manifest["runs"]
                )
                manifest["updated_at"] = self._now()
                self._write_json(path, manifest)
                self._try_write_summary(path.parent, manifest)
            manifest["current_scenario_id"] = None
            manifest["status"] = (
                "completed"
                if manifest["failed_scenario_count"] == 0 else "partial_failed"
            )
            manifest["updated_at"] = self._now()
            self._write_json(path, manifest)
            self._try_write_summary(path.parent, manifest)
            self._write_json(path, manifest)
        finally:
            with self._lock:
                current = self._threads.get(batch_id)
                if current is threading.current_thread():
                    self._threads.pop(batch_id, None)

    def get_status(
        self, batch_id: str, *, reconcile: bool = True
    ) -> Dict[str, Any]:
        path = self._batch_dir(batch_id) / "manifest.json"
        manifest = self._read_json(path)
        if (
            reconcile
            and manifest.get("status") in {"queued", "running"}
            and not self.is_active(batch_id)
        ):
            manifest.update(
                {
                    "status": "failed",
                    "error": "batch worker is not active; the backend may have restarted",
                    "updated_at": self._now(),
                }
            )
            self._write_json(path, manifest)
        return manifest

    def get_summary_path(self, batch_id: str) -> Path:
        path = self._batch_dir(batch_id) / "scenario_summary.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        return path
