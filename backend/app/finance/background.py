"""In-process background execution for long C0 batch runs."""

from __future__ import annotations

import threading
from typing import Any, Dict

from ..utils.logger import get_logger
from .c0 import C0ExperimentService


logger = get_logger("mirofish.finance.background")


class C0BackgroundRunner:
    """Run one C0 job per run ID without holding the HTTP request open."""

    _lock = threading.Lock()
    _threads: Dict[str, threading.Thread] = {}

    @classmethod
    def start(cls, run_id: str) -> Dict[str, Any]:
        service = C0ExperimentService()
        manifest = service.get_status(run_id)
        if manifest.get("run_mode") != "all":
            raise ValueError("background execution is reserved for all-scenario runs")

        with cls._lock:
            existing = cls._threads.get(run_id)
            if existing is not None and existing.is_alive():
                raise ValueError(f"C0 background run is already active: {run_id}")
            queued = service.mark_queued(run_id)
            thread = threading.Thread(
                target=cls._execute,
                args=(run_id,),
                name=f"finance-{run_id}",
                daemon=True,
            )
            cls._threads[run_id] = thread
            thread.start()
        return queued

    @classmethod
    def is_active(cls, run_id: str) -> bool:
        with cls._lock:
            thread = cls._threads.get(run_id)
            return bool(thread and thread.is_alive())

    @classmethod
    def reconcile(cls, run_id: str) -> Dict[str, Any]:
        """Turn stale running state into an explicit recoverable failure."""
        service = C0ExperimentService()
        manifest = service.get_status(run_id)
        if (
            manifest.get("execution_mode") == "background"
            and manifest.get("status") in {"queued", "running"}
            and not cls.is_active(run_id)
        ):
            return service.mark_failed(
                run_id,
                "background worker is not active; the backend may have restarted",
            )
        return manifest

    @classmethod
    def _execute(cls, run_id: str) -> None:
        try:
            C0ExperimentService().run(run_id)
        except Exception as error:
            logger.exception("C0 background run failed: run_id=%s", run_id)
            try:
                C0ExperimentService().mark_failed(run_id, str(error))
            except Exception:
                logger.exception(
                    "Could not persist C0 background failure: run_id=%s", run_id
                )
        finally:
            with cls._lock:
                current = cls._threads.get(run_id)
                if current is threading.current_thread():
                    cls._threads.pop(run_id, None)
