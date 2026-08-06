"""Financial adaptation API.

C0 remains the independent control group.  S1 adds a Reddit-only OASIS social
interaction prototype while keeping preparation separate from execution.
"""

from __future__ import annotations

import traceback

from flask import jsonify, request, send_file

from ..finance import (
    C0BackgroundRunner,
    C0ExperimentService,
    OfflineStanceAnnotator,
    S1BatchRunner,
    S1ExperimentService,
)
from ..finance.dataset import DatasetValidationError
from ..utils.logger import get_logger
from . import finance_bp


logger = get_logger("mirofish.api.finance")


def _service() -> C0ExperimentService:
    return C0ExperimentService()


def _s1_service() -> S1ExperimentService:
    return S1ExperimentService()


@finance_bp.route("/s1/reddit/prepare", methods=["POST"])
def prepare_s1_reddit():
    """Freeze one anonymous Reddit S1 scenario without contacting the LLM."""
    try:
        data = request.get_json(silent=True) or {}
        result = _s1_service().prepare(
            run_id=data.get("run_id"),
            dataset_path=data.get("dataset_path"),
            scenario_id=data.get("scenario_id"),
            graph_id=data.get("graph_id"),
            project_id=data.get("project_id"),
            source_mode=data.get("source_mode", "auto"),
            social_rounds=data.get("social_rounds", S1ExperimentService.DEFAULT_SOCIAL_ROUNDS),
            replicate_id=data.get("replicate_id"),
            data_split=data.get("data_split", S1ExperimentService.DEFAULT_DATA_SPLIT),
            agent_set_version=data.get("agent_set_version"),
            sampling_method=data.get(
                "sampling_method", S1ExperimentService.DEFAULT_SAMPLING_METHOD
            ),
            random_seed=data.get("random_seed"),
        )
        return jsonify({"success": True, "data": result})
    except (DatasetValidationError, FileNotFoundError, ValueError) as error:
        return jsonify({"success": False, "error": str(error)}), 400
    except Exception as error:
        logger.error("S1 Reddit preparation failed: %s", error)
        return jsonify({"success": False, "error": str(error)}), 500


@finance_bp.route("/s1/reddit/scenarios/<scenario_id>/seed", methods=["GET"])
def get_s1_reddit_scenario_seed(scenario_id: str):
    """Return only safe, pre-cutoff events for in-page graph construction."""
    try:
        return jsonify({
            "success": True,
            "data": _s1_service().get_scenario_seed_document(scenario_id),
        })
    except (DatasetValidationError, FileNotFoundError, ValueError) as error:
        return jsonify({"success": False, "error": str(error)}), 404


@finance_bp.route("/s1/reddit/batch/prepare", methods=["POST"])
def prepare_s1_reddit_batch():
    """Freeze all scenarios that have completed entries in the Zep manifest."""
    try:
        data = request.get_json(silent=True) or {}
        result = S1BatchRunner().prepare(
            social_rounds=data.get(
                "social_rounds", S1ExperimentService.DEFAULT_SOCIAL_ROUNDS
            ),
            graph_manifest_path=data.get("graph_manifest_path"),
            data_split=data.get("data_split", S1ExperimentService.DEFAULT_DATA_SPLIT),
            replicate_id=data.get("replicate_id"),
            agent_set_version=data.get("agent_set_version"),
            sampling_method=data.get(
                "sampling_method", S1ExperimentService.DEFAULT_SAMPLING_METHOD
            ),
            random_seed=data.get("random_seed"),
        )
        return jsonify({"success": True, "data": result})
    except (DatasetValidationError, FileNotFoundError, ValueError) as error:
        return jsonify({"success": False, "error": str(error)}), 400


@finance_bp.route("/s1/reddit/batch/run", methods=["POST"])
def run_s1_reddit_batch():
    """Start a prepared all-graph batch in one serial background worker."""
    try:
        data = request.get_json(silent=True) or {}
        batch_id = data.get("batch_id")
        if not batch_id:
            return jsonify({"success": False, "error": "batch_id is required"}), 400
        result = S1BatchRunner().start(batch_id)
        return jsonify({"success": True, "data": result}), 202
    except (FileNotFoundError, ValueError) as error:
        return jsonify({"success": False, "error": str(error)}), 400


@finance_bp.route("/s1/reddit/batch/<batch_id>", methods=["GET"])
def get_s1_reddit_batch_status(batch_id: str):
    try:
        return jsonify({
            "success": True,
            "data": S1BatchRunner().get_status(batch_id),
        })
    except (FileNotFoundError, ValueError) as error:
        return jsonify({"success": False, "error": str(error)}), 404


@finance_bp.route("/s1/reddit/batch/<batch_id>/csv", methods=["GET"])
def download_s1_reddit_batch_summary(batch_id: str):
    try:
        path = S1BatchRunner().get_summary_path(batch_id)
        return send_file(
            path,
            as_attachment=True,
            download_name=f"{batch_id}_scenario_summary.csv",
            mimetype="text/csv; charset=utf-8",
        )
    except (FileNotFoundError, ValueError) as error:
        return jsonify({"success": False, "error": str(error)}), 404


@finance_bp.route("/s1/reddit/run", methods=["POST"])
def run_s1_reddit():
    """Start one prepared S1 run in a background thread."""
    try:
        data = request.get_json(silent=True) or {}
        run_id = data.get("run_id")
        if not run_id:
            return jsonify({"success": False, "error": "run_id is required"}), 400
        result = _s1_service().run_background(run_id)
        return jsonify({"success": True, "data": result}), 202
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        return jsonify({"success": False, "error": str(error)}), 400


@finance_bp.route("/s1/reddit/<run_id>", methods=["GET"])
def get_s1_reddit_status(run_id: str):
    try:
        return jsonify({"success": True, "data": _s1_service().get_status(run_id)})
    except (FileNotFoundError, ValueError) as error:
        return jsonify({"success": False, "error": str(error)}), 404


@finance_bp.route("/s1/reddit/<run_id>/settings", methods=["PATCH"])
def update_s1_reddit_settings(run_id: str):
    """Change interaction rounds after preparation, before running."""
    try:
        data = request.get_json(silent=True) or {}
        result = _s1_service().update_settings(
            run_id,
            social_rounds=data.get("social_rounds"),
        )
        return jsonify({"success": True, "data": result})
    except (FileNotFoundError, ValueError) as error:
        return jsonify({"success": False, "error": str(error)}), 400


@finance_bp.route("/s1/reddit/<run_id>/predictions", methods=["GET"])
def get_s1_reddit_predictions(run_id: str):
    try:
        stage = request.args.get("stage", "all")
        return jsonify({
            "success": True,
            "data": _s1_service().get_predictions(run_id, stage=stage),
        })
    except (FileNotFoundError, ValueError) as error:
        return jsonify({"success": False, "error": str(error)}), 404


@finance_bp.route("/s1/reddit/<run_id>/metrics", methods=["GET"])
def get_s1_reddit_metrics(run_id: str):
    try:
        return jsonify({"success": True, "data": _s1_service().get_metrics(run_id)})
    except (FileNotFoundError, ValueError) as error:
        return jsonify({"success": False, "error": str(error)}), 404


@finance_bp.route("/s1/reddit/<run_id>/actions", methods=["GET"])
def get_s1_reddit_actions(run_id: str):
    try:
        raw_limit = request.args.get("limit")
        limit = int(raw_limit) if raw_limit else None
        return jsonify({
            "success": True,
            "data": _s1_service().get_actions(run_id, limit=limit),
        })
    except (FileNotFoundError, ValueError) as error:
        return jsonify({"success": False, "error": str(error)}), 400


@finance_bp.route("/s1/reddit/<run_id>/agent-round-states", methods=["GET"])
def get_s1_reddit_agent_round_states(run_id: str):
    """Return observed per-agent/per-round exposure and action aggregates."""
    try:
        return jsonify({
            "success": True,
            "data": _s1_service().get_agent_round_states(run_id),
        })
    except (FileNotFoundError, ValueError) as error:
        return jsonify({"success": False, "error": str(error)}), 404


@finance_bp.route("/s1/reddit/<run_id>/belief-snapshots", methods=["GET"])
def get_s1_reddit_belief_snapshots(run_id: str):
    """Return private round-level belief measurements."""
    try:
        return jsonify({
            "success": True,
            "data": _s1_service().get_belief_snapshots(run_id),
        })
    except (FileNotFoundError, ValueError) as error:
        return jsonify({"success": False, "error": str(error)}), 404


@finance_bp.route("/s1/reddit/<run_id>/exposure-edges", methods=["GET"])
def get_s1_reddit_exposure_edges(run_id: str):
    """Return one row per observed viewer/content exposure or interaction."""
    try:
        return jsonify({
            "success": True,
            "data": _s1_service().get_exposure_edges(run_id),
        })
    except (FileNotFoundError, ValueError) as error:
        return jsonify({"success": False, "error": str(error)}), 404


@finance_bp.route("/s1/reddit/<run_id>/stance-annotate", methods=["POST"])
def annotate_s1_reddit_stances(run_id: str):
    """Annotate saved posts/comments after an S1 run using an independent LLM."""
    try:
        data = request.get_json(silent=True) or {}
        force = data.get("force", False)
        if not isinstance(force, bool):
            return jsonify({"success": False, "error": "force must be a boolean"}), 400
        result = OfflineStanceAnnotator().annotate_run(run_id, force=force)
        return jsonify({"success": True, "data": result})
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        return jsonify({"success": False, "error": str(error)}), 400
    except Exception as error:
        logger.error("S1 stance annotation failed: %s", error)
        return jsonify({"success": False, "error": str(error)}), 500


@finance_bp.route("/s1/reddit/<run_id>/stance-annotations", methods=["GET"])
def get_s1_reddit_stance_annotations(run_id: str):
    try:
        return jsonify({
            "success": True,
            "data": OfflineStanceAnnotator().get_annotations(run_id),
        })
    except (FileNotFoundError, ValueError) as error:
        return jsonify({"success": False, "error": str(error)}), 404


@finance_bp.route("/s1/reddit/<run_id>/exposure-edges-annotated", methods=["GET"])
def get_s1_reddit_annotated_exposure_edges(run_id: str):
    try:
        return jsonify({
            "success": True,
            "data": OfflineStanceAnnotator().get_annotated_exposure_edges(run_id),
        })
    except (FileNotFoundError, ValueError) as error:
        return jsonify({"success": False, "error": str(error)}), 404


@finance_bp.route("/s1/reddit/<run_id>/mapping", methods=["GET"])
def get_s1_reddit_mapping(run_id: str):
    """Return the frozen publisher/entity mapping without evaluation data."""
    try:
        return jsonify({"success": True, "data": _s1_service().get_mapping(run_id)})
    except (FileNotFoundError, ValueError) as error:
        return jsonify({"success": False, "error": str(error)}), 404


@finance_bp.route("/s1/reddit/<run_id>/csv/<kind>", methods=["GET"])
def download_s1_reddit_csv(run_id: str, kind: str):
    try:
        path = _s1_service().get_csv_path(run_id, kind)
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
            replicate_id=data.get("replicate_id"),
            data_split=data.get("data_split", C0ExperimentService.DEFAULT_DATA_SPLIT),
            agent_set_version=data.get("agent_set_version"),
            sampling_method=data.get(
                "sampling_method", C0ExperimentService.DEFAULT_SAMPLING_METHOD
            ),
            random_seed=data.get("random_seed"),
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
