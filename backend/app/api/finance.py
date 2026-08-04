"""Financial adaptation API.

Only the C0 control group is exposed at this stage. Preparation is deliberately
separate from execution so a researcher can inspect the frozen prompts and
roles before any LLM call is made.
"""

from __future__ import annotations

import traceback

from flask import jsonify, request, send_file

from ..finance import C0BackgroundRunner, C0ExperimentService
from ..finance.dataset import DatasetValidationError
from ..utils.logger import get_logger
from . import finance_bp


logger = get_logger("mirofish.api.finance")


def _service() -> C0ExperimentService:
    return C0ExperimentService()


@finance_bp.route("/c0/scenarios", methods=["GET"])
def list_c0_scenarios():
    """List safe anonymous scenarios for the C0 workbench selector."""
    try:
        return jsonify({"success": True, "data": _service().list_scenarios()})
    except (DatasetValidationError, FileNotFoundError, ValueError) as error:
        return jsonify({"success": False, "error": str(error)}), 400


@finance_bp.route("/c0/prepare", methods=["POST"])
def prepare_c0():
    """Freeze an anonymous C0 run without contacting the LLM."""
    try:
        data = request.get_json(silent=True) or {}
        run_mode = data.get("run_mode", "single")
        if run_mode not in {"single", "all"}:
            return jsonify({
                "success": False,
                "error": "run_mode must be 'single' or 'all'",
            }), 400
        scenario_ids = data.get("scenario_ids")
        if scenario_ids is not None and not isinstance(scenario_ids, list):
            return jsonify({"success": False, "error": "scenario_ids must be a list"}), 400
        if run_mode == "single" and scenario_ids is not None and len(scenario_ids) > 1:
            return jsonify({
                "success": False,
                "error": "C0 prototype accepts exactly one scenario per run",
            }), 400
        if run_mode == "all" and (scenario_ids or data.get("limit") is not None):
            return jsonify({
                "success": False,
                "error": "all-scenario mode cannot use scenario_ids or limit",
            }), 400
        limit = data.get("limit")
        if limit is not None:
            try:
                limit = int(limit)
            except (TypeError, ValueError):
                return jsonify({"success": False, "error": "limit must be an integer"}), 400
        elif run_mode == "single" and not scenario_ids:
            limit = 1
        result = _service().prepare(
            run_id=data.get("run_id"),
            dataset_path=data.get("dataset_path"),
            scenario_ids=scenario_ids,
            limit=limit,
            run_mode=run_mode,
        )
        return jsonify({"success": True, "data": result})
    except (DatasetValidationError, FileNotFoundError, ValueError) as error:
        return jsonify({"success": False, "error": str(error)}), 400
    except Exception as error:
        logger.error("C0 preparation failed: %s", error)
        return jsonify({
            "success": False,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }), 500


@finance_bp.route("/c0/run", methods=["POST"])
def run_c0():
    """Run independent C0 forecasts, or use dry_run to only validate prompts."""
    try:
        data = request.get_json(silent=True) or {}
        run_id = data.get("run_id")
        if not run_id:
            return jsonify({"success": False, "error": "run_id is required"}), 400
        scenario_ids = data.get("scenario_ids")
        if scenario_ids is not None and not isinstance(scenario_ids, list):
            return jsonify({"success": False, "error": "scenario_ids must be a list"}), 400
        dry_run = data.get("dry_run", False)
        if not isinstance(dry_run, bool):
            return jsonify({"success": False, "error": "dry_run must be boolean"}), 400
        background = data.get("background", False)
        if not isinstance(background, bool):
            return jsonify({"success": False, "error": "background must be boolean"}), 400
        if background and dry_run:
            return jsonify({
                "success": False,
                "error": "dry_run cannot be started as a background job",
            }), 400
        if background and scenario_ids:
            return jsonify({
                "success": False,
                "error": "background runs cannot filter scenario_ids",
            }), 400
        if background:
            result = C0BackgroundRunner.start(run_id)
            return jsonify({"success": True, "data": result}), 202
        service = _service()
        if not dry_run and service.get_status(run_id).get("run_mode") == "all":
            return jsonify({
                "success": False,
                "error": "all-scenario runs must set background=true",
            }), 400
        result = service.run(
            run_id,
            scenario_ids=scenario_ids,
            dry_run=dry_run,
        )
        return jsonify({"success": True, "data": result})
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        return jsonify({"success": False, "error": str(error)}), 400
    except Exception as error:
        logger.error("C0 run failed: %s", error)
        return jsonify({
            "success": False,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }), 500


@finance_bp.route("/c0/<run_id>", methods=["GET"])
def get_c0_status(run_id: str):
    """Read C0 artifact status and counts."""
    try:
        return jsonify({
            "success": True,
            "data": C0BackgroundRunner.reconcile(run_id),
        })
    except (FileNotFoundError, ValueError) as error:
        return jsonify({"success": False, "error": str(error)}), 404


@finance_bp.route("/c0/<run_id>/preview", methods=["GET"])
def get_c0_preview(run_id: str):
    """Read the first frozen prompt and safe scenario snapshot."""
    try:
        return jsonify({"success": True, "data": _service().get_preview(run_id)})
    except (FileNotFoundError, ValueError) as error:
        return jsonify({"success": False, "error": str(error)}), 404


@finance_bp.route("/c0/<run_id>/predictions", methods=["GET"])
def get_c0_predictions(run_id: str):
    """Read partial or complete C0 predictions for progress display."""
    try:
        return jsonify({"success": True, "data": _service().get_predictions(run_id)})
    except (FileNotFoundError, ValueError) as error:
        return jsonify({"success": False, "error": str(error)}), 404


@finance_bp.route("/c0/<run_id>/outcome", methods=["GET"])
def get_c0_outcome(run_id: str):
    """Expose evaluator-only ground truth after all Agent calls have finished."""
    try:
        return jsonify({"success": True, "data": _service().get_outcome(run_id)})
    except FileNotFoundError as error:
        return jsonify({"success": False, "error": str(error)}), 404
    except ValueError as error:
        return jsonify({"success": False, "error": str(error)}), 400


@finance_bp.route("/c0/<run_id>/csv/<kind>", methods=["GET"])
def download_c0_csv(run_id: str, kind: str):
    """Download prediction-only or researcher evaluation CSV artifacts."""
    try:
        path = _service().get_csv_path(run_id, kind)
        return send_file(
            path,
            as_attachment=True,
            download_name=f"{run_id}_{path.name}",
            mimetype="text/csv; charset=utf-8",
        )
    except FileNotFoundError as error:
        return jsonify({"success": False, "error": str(error)}), 404
    except ValueError as error:
        return jsonify({"success": False, "error": str(error)}), 400
