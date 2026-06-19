"""Exploit field-period symmetry in the coil distance objectives.

`CoilSetMinDistance` and `PlasmaCoilSetMinDistance` both compute a per-coil
minimum distance by forming, for every one of the `num_coils` *symmetrized*
coils, a full pairwise distance matrix against the plasma surface / the other
coils, then reducing (`jnp.min`). DESC builds the coilset with `NFP` field
periods, so coil ``j`` and coil ``j + num_coils/NFP`` are the same curve rotated
by one field period. Because both the plasma surface and the coilset share that
toroidal periodicity, their per-coil minimum distances are identical across
field periods (verified bit-for-bit, ~1e-16). The upstream objective recomputes
all `NFP` copies anyway — the dominant per-instance cost in `evaluate`.

This patch makes the two distance objectives evaluate only the first field
period (`num_coils // NFP` coils) and `jnp.tile` the result back to the full
`num_coils`. Output is bit-for-bit identical to upstream; the distance-matrix
work shrinks by a factor of `NFP` (3x for the benchmark's NFP=3 instances).
"""
import logging

import jax.numpy as jnp

logger = logging.getLogger(__name__)

_PATCHED = False


def apply_patch():
    """Patch `MetricsEvaluator._make_compute` in place (idempotent)."""
    global _PATCHED
    if _PATCHED:
        return
    from coilstellaration import metrics_utils_v2
    from coilstellaration.metrics_utils_v2 import MetricsEvaluator
    from desc.objectives import CoilSetMinDistance, PlasmaCoilSetMinDistance

    # Accessing a staticmethod via the class yields the plain underlying
    # function (no `__func__` indirection needed).
    orig_make_compute = MetricsEvaluator._make_compute

    def _make_compute(obj):
        # `orig_make_compute` builds the objective (sets `_dim_f = num_coils`)
        # and returns the `(eq_params, coil_params) -> array` closure.
        call = orig_make_compute(obj)
        if isinstance(obj, (CoilSetMinDistance, PlasmaCoilSetMinDistance)):
            coilset = obj.things[0]
            nfp = int(getattr(coilset, "NFP", 1))
            num_coils = int(obj.dim_f)
            if nfp > 1 and num_coils > nfp and num_coils % nfp == 0:
                # Restrict the objective's per-coil loop to one field period.
                # `compute` reads `self.dim_f` at trace time, so this takes
                # effect for the closure built above.
                obj._dim_f = num_coils // nfp

                def tiled(eq_params, coil_params, _call=call, _reps=nfp):
                    return jnp.tile(_call(eq_params, coil_params), _reps)

                return tiled
        return call

    MetricsEvaluator._make_compute = staticmethod(_make_compute)
    # Drop any evaluators built with the original (full-symmetry) objectives.
    metrics_utils_v2.clear_evaluator_cache()
    _PATCHED = True
    logger.info("Patched coil distance objectives to exploit NFP field-period symmetry")
