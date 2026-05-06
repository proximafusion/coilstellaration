"""Dapper tasks for batch scoring of trained ML models on the public test set."""

import logging

import dapper

from coilstellaration.benchmark import (
    _batch_types,
    _diagnostics,
    _types,
    _utils,
)

logger = logging.getLogger(__name__)


@dapper.task
def score_model_on_batch(
    batch_spec: _batch_types.ScoreModelBatchSpec,
    settings: _types.ScoringSettings,
) -> _batch_types.ScoreModelBatchResult:
    """Score one slice of the public test set on one VM.

    Reads the launcher-materialized `EvalSet` from dapper, slices to the rows
    in `batch_spec.indices`, runs the model in a single batched call, and
    scores each predicted coilset against `EvalData.requirement_metrics`.
    See `scoring/README.md` for the full design. Per-step GCS diagnostic
    markers are emitted via `_diagnostics` so progress is visible without
    log-read permissions.
    """
    _diagnostics.write(
        "01_task_started",
        f"indices={len(batch_spec.indices)} "
        f"cache={batch_spec.eval_cache_uri} "
        f"model={batch_spec.model_checkpoint_id} type={batch_spec.model_type}",
    )
    try:
        import pickle

        import gcsfs

        logger.info("Reading eval cache from %s...", batch_spec.eval_cache_uri)
        fs = gcsfs.GCSFileSystem()
        raw = fs.cat(batch_spec.eval_cache_uri)
        assert isinstance(raw, bytes)
        eval_cache = pickle.loads(raw)
        _diagnostics.write("02_eval_cache_loaded", f"total={len(eval_cache)}")

        eval_dataset = [eval_cache[i] for i in batch_spec.indices]
        logger.info("Sliced %d / %d rows.", len(eval_dataset), len(batch_spec.indices))
        _diagnostics.write("03_eval_sliced", f"selected={len(eval_dataset)}")

        logger.info(
            "Reading checkpoint %s (%s)...",
            batch_spec.model_checkpoint_id,
            batch_spec.model_type,
        )
        checkpoint = _utils.read_checkpoint(
            batch_spec.model_type, batch_spec.model_checkpoint_id
        )
        _diagnostics.write("04_checkpoint_read")
        model = _utils.read_model_from_checkpoint(checkpoint)
        _diagnostics.write("05_model_loaded")

        logger.info("Predicting %d coilsets...", len(eval_dataset))
        predictions = _utils.predict_coilsets(model, eval_dataset)
        _diagnostics.write("06_predicted", f"n={len(predictions)}")

        logger.info("Scoring predictions (heavy DESC eval per instance)...")
        instance_scores, failed_boundary_ids = _utils.score_predictions(
            predictions, settings
        )
        _diagnostics.write(
            "07_scoring_done",
            f"ok={len(instance_scores)} failed={len(failed_boundary_ids)}",
        )

        result = _batch_types.ScoreModelBatchResult(
            instance_scores=instance_scores,
            n_processed=len(instance_scores),
            n_failed=len(failed_boundary_ids),
            failed_boundary_ids=failed_boundary_ids,
        )
        _diagnostics.write("08_result_built")
        return result
    except Exception:
        _diagnostics.write_traceback("99_task_failed")
        raise
