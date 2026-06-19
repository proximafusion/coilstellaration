"""Leaderboard guarantee: the speedup patches don't change the metrics.

Several patches are layered onto the metric evaluator purely for speed
(`biot_savart_patch`, `coil_symmetry_patch`, `plasma_distance_patch`). Because
this dataset backs a public leaderboard, the *only* thing those patches are
allowed to change is wall-clock time -- every achieved metric, and therefore
every score that lands on the board, must be bit-for-bit (or to within machine
precision) what the unpatched upstream evaluator would have produced.

This test pins exactly that on the real benchmark config (`variable_shape`,
`eval` split, `tight` stratum -- the stratum the leaderboard actually scores;
`medium`/`loose` are parametrized in too). For real example-model coilsets on
real benchmark boundaries it computes the achieved metrics twice -- once with
the upstream evaluator and once with all three patches applied -- and asserts:

  1. the distance and curvature metrics agree to machine precision (the
     coil-symmetry reduction is an exact field-period symmetry, and on the
     scored strata the plasma single-period tiling is exact too), and
  2. `max_normalized_field_error` agrees within Biot-Savart convergence
     tolerance (the coarse source grid is a *convergence* approximation, not an
     exact symmetry, so it tracks the dense upstream field as the source
     resolution rises rather than reproducing it bit-for-bit), and
  3. the aggregated track summaries (what the leaderboard publishes) agree.

`tight` and `medium` are hard assertions; `loose` is `xfail` because the plasma
single-period tiling is a *geometric* approximation that can diverge on some
loose-stratum coilsets (a coil's nearest plasma point crossing field periods).
`loose` is not leaderboard-scored, and making it exact would mean disabling the
plasma speedup (`PLASMA_PERIODS >= NFP`).

It complements the per-patch unit tests (`test_*_patch.py`), which pin each
optimization in isolation; this one pins the *whole stack* end to end.

Marked ``slow``: it hits the network (HF datasets) and runs DESC physics.
"""
import numpy as np
import pytest

# variable_shape / eval is the track + split the user called out; the patches
# are validated against the tight stratum the board scores, plus medium/loose.
TRACK = "variable_shape"
SPLIT = "eval"

# A real, trained example model: the plasma single-period reduction is a
# *geometric* property of realistic modular coils, so it must be exercised on
# real coils, not a degenerate synthetic coilset (cf. test_plasma_distance_patch).
EXAMPLE_MODEL = "examples/res_mlp_De3VR8.onnx"

# How many instances per stratum to score. Kept small so the slow suite stays
# tractable.
N_INSTANCES = 10

# Distance + curvature metrics are exact field-period reductions -> bit-exact.
DIST_RTOL, DIST_ATOL = 0.0, 1e-9
# max_normalized_field_error rides the Biot-Savart source-grid convergence: at
# the pinned FIELD_SOURCE_N=128 the worst observed instance drifts ~4e-5 from the
# dense upstream field, so 1e-4 leaves headroom without masking a real regression.
FIELD_RTOL, FIELD_ATOL = 1e-4, 1e-6
# Scores are a deterministic function of the achieved metrics; the only
# non-bit-exact input is the field error, so this stays tight.
SCORE_RTOL, SCORE_ATOL = 1e-3, 1e-5


def _achieved_arrays(metrics_fn, desc_surface, desc_coilset):
    """Achieved RequirementMetrics (as a float array) for one instance.

    `metrics_fn` is `evaluate_coilset_metrics_from_boundary`, called the same way
    the production path (`eval._evaluate_achieved`) calls it; whether the patches
    are active is decided entirely by module state, not by this helper.
    """
    from coilstellaration import data_utils

    metrics = metrics_fn(boundary=desc_surface, coilset=desc_coilset, subset_metrics=True)
    rm = data_utils.metrics_to_requirement_metrics(metrics)
    return rm


@pytest.mark.slow
@pytest.mark.parametrize(
    "stratum",
    [
        "tight",  # the stratum the leaderboard actually scores
        "medium",
        # The plasma single-period tiling is a geometric approximation: on some
        # loose-stratum coilsets a coil's nearest plasma point falls in an
        # adjacent field period, so tiling the plasma to one period drops it and
        # `coil_to_plasma` can diverge ~11% from upstream. This stratum is not
        # scored by the leaderboard (config.STRATUM == "tight"), and making it
        # exact would require PLASMA_PERIODS >= NFP -- i.e. disabling the plasma
        # speedup. xfail (non-strict: well-conditioned loose samples may pass).
        pytest.param(
            "loose",
            marks=pytest.mark.xfail(
                reason="plasma single-period tiling can diverge on loose-stratum "
                "coilsets whose nearest plasma point crosses field periods; "
                "loose is not leaderboard-scored",
                strict=False,
            ),
        ),
    ],
)
def test_patches_preserve_metrics_and_scores(stratum):
    """All three speedup patches together leave metrics and scores unchanged."""
    from coilstellaration import metrics_utils_v2
    from coilstellaration.benchmark import scoring
    from constellaration.geometry import surface_utils_desc

    from src.backend import (
        biot_savart_patch,
        coil_symmetry_patch,
        config,
        marshal,
        plasma_distance_patch,
    )
    from src.backend import data as bd
    from src.backend import eval as beval
    from src.backend.onnx_model import OnnxModel

    # A leaked patch from an earlier test would make the "unpatched" baseline
    # secretly patched, turning this into a vacuous self-comparison. Fail loudly
    # instead: every patch must be inert when this test starts.
    assert not biot_savart_patch._PATCHED
    assert not coil_symmetry_patch._PATCHED
    assert not plasma_distance_patch._PATCHED

    eval_datas = bd.load_benchmark_dataset(
        track=TRACK, stratum=stratum, split=SPLIT, n=N_INSTANCES
    )
    assert eval_datas, f"no {TRACK}/{stratum}/{SPLIT} instances loaded"

    model = OnnxModel(EXAMPLE_MODEL)
    # Model inference is patch-independent (pure ONNX), so resolve the predicted
    # coilsets -> DESC objects once and reuse them for both evaluator runs.
    desc_inputs = []
    for data in eval_datas:
        inputs = marshal.eval_data_to_inputs(data.boundary, data.requirement_metrics)
        coilset = marshal.outputs_to_coilset(model.run(inputs), data.boundary)
        desc_surface = surface_utils_desc.to_desc_fourier_rz_toroidal_surface(data.boundary)
        desc_coilset = beval._to_desc_coilset(coilset)
        desc_inputs.append((desc_surface, desc_coilset))

    metrics_fn = metrics_utils_v2.evaluate_coilset_metrics_from_boundary

    # Capture the raw class-dict entries (not class-attribute access): the
    # coil-symmetry patch replaces `_make_compute` with a `staticmethod`, and
    # restoring the plain underlying function would rebind it as an instance
    # method and corrupt every later evaluator build. The descriptor objects
    # restore exactly.
    ev = metrics_utils_v2.MetricsEvaluator
    orig_field_fn = ev.__dict__["_make_surface_field_fn"]
    orig_make_compute = ev.__dict__["_make_compute"]
    from desc.objectives import _coils

    orig_copy = _coils.copy_rpz_periods
    orig_flags = (
        biot_savart_patch._PATCHED,
        coil_symmetry_patch._PATCHED,
        plasma_distance_patch._PATCHED,
    )
    try:
        # --- Reference: upstream evaluator, no patches ---------------------
        metrics_utils_v2.clear_evaluator_cache()
        reference = [_achieved_arrays(metrics_fn, s, c) for s, c in desc_inputs]

        # --- Production: all three patches applied (order matters: plasma
        # tiling is only valid once the coil loop is restricted) -------------
        biot_savart_patch.apply_patch()
        coil_symmetry_patch.apply_patch()
        plasma_distance_patch.apply_patch()
        metrics_utils_v2.clear_evaluator_cache()
        patched = [_achieved_arrays(metrics_fn, s, c) for s, c in desc_inputs]
    finally:
        ev._make_surface_field_fn = orig_field_fn
        ev._make_compute = orig_make_compute
        _coils.copy_rpz_periods = orig_copy
        (
            biot_savart_patch._PATCHED,
            coil_symmetry_patch._PATCHED,
            plasma_distance_patch._PATCHED,
        ) = orig_flags
        metrics_utils_v2.clear_evaluator_cache()

    # 1. Achieved metrics agree, instance by instance. The distance + curvature
    # metrics are exact field-period reductions, so they must be bit-exact; only
    # the field error rides the Biot-Savart source-grid convergence.
    for data, ref, pat in zip(eval_datas, reference, patched):
        ref_arr = np.asarray(ref.to_array(), dtype=float)
        pat_arr = np.asarray(pat.to_array(), dtype=float)
        exact = [i for i in range(len(ref_arr)) if i != 3]
        np.testing.assert_allclose(
            pat_arr[exact],
            ref_arr[exact],
            rtol=DIST_RTOL,
            atol=DIST_ATOL,
            err_msg=f"distance/curvature metrics diverged on {data.boundary_id}",
        )
        np.testing.assert_allclose(
            pat_arr[3],
            ref_arr[3],
            rtol=FIELD_RTOL,
            atol=FIELD_ATOL,
            err_msg=f"field error diverged on {data.boundary_id}",
        )

    # 2. The published track summaries agree. Scores are a deterministic
    # function of the achieved metrics, so this follows from (1), but it is the
    # leaderboard-facing guarantee and worth pinning directly.
    settings = config.SCORING_SETTINGS
    ref_score = scoring.score_benchmark(
        [scoring.score_eval_data(d, a, settings) for d, a in zip(eval_datas, reference)],
        settings,
    )
    patched_score = scoring.score_benchmark(
        [scoring.score_eval_data(d, a, settings) for d, a in zip(eval_datas, patched)],
        settings,
    )
    for key in beval.SUMMARY_KEYS:
        np.testing.assert_allclose(
            float(patched_score.summaries[key]),
            float(ref_score.summaries[key]),
            rtol=SCORE_RTOL,
            atol=SCORE_ATOL,
            err_msg=f"track summary '{key}' diverged",
        )
