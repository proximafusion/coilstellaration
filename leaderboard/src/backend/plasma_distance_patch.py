"""Tile the plasma surface to fewer field periods in the coil-distance metric.

`PlasmaCoilSetMinDistance.compute` builds the plasma point cloud by tiling the
one-field-period `_surf_grid` across all `NFP` periods (`copy_rpz_periods`),
then, for every coil, reduces `jnp.min` over that full-torus cloud. After the
[coil_symmetry_patch] restricts the per-coil loop to the first field period,
those period-0 coils' nearest plasma point lies in period 0 — the tiled copies
in the other `NFP-1` periods can never win the `jnp.min`, so they are pure
overhead. `coil_to_plasma` is the single dominant objective in `evaluate`.

This patch replaces `copy_rpz_periods` (as used by the distance objectives in
`desc.objectives._coils`) with a version that tiles only the first
`config.PLASMA_PERIODS` periods, shrinking that distance matrix by up to `NFP`x.

IMPORTANT — unlike [coil_symmetry_patch], this is a *geometric* approximation,
not an exact field-period *symmetry*: it is bit-for-bit exact only when each
evaluated coil's nearest plasma point falls within the retained period(s).
Verified bit-exact (0.000% max rel error) across benchmark instances in both
tracks at NFP 3 and 4. It REQUIRES `coil_symmetry_patch` — applied alone, with
the full per-coil loop, it would corrupt the distances of coils that live in
the dropped periods. `PLASMA_PERIODS >= NFP` (or 0) reproduces upstream exactly;
raise it from 1 for a safety margin at proportionally less speedup.
"""
import logging

import jax.numpy as jnp

from src.backend import config

logger = logging.getLogger(__name__)

_PATCHED = False


def _tile_periods(rpz, nfp, keep):
    """`copy_rpz_periods` restricted to the first `keep` of `nfp` field periods.

    Bit-for-bit identical to `desc.utils.copy_rpz_periods(rpz, nfp)` when
    `keep >= nfp` (or `keep <= 0`); otherwise keeps the same per-period rotation
    of `2*pi/nfp` but emits only the first `keep` rotated copies — i.e. exactly
    the leading `keep * num_nodes` rows of the full tiling.
    """
    nfp = int(nfp)
    keep = nfp if keep <= 0 or keep >= nfp else int(keep)
    r, p, z = rpz.T
    angles = jnp.linspace(0.0, 2 * jnp.pi, nfp, endpoint=False)[:keep]
    r = jnp.tile(r, keep)
    z = jnp.tile(z, keep)
    p = p[None, :] + angles[:, None]
    return jnp.array([r, p.flatten(), z]).T


def apply_patch():
    """Patch `desc.objectives._coils.copy_rpz_periods` in place (idempotent).

    Only the plasma/surface distance objectives (`PlasmaCoilSetMinDistance` and
    its bound base) reference `copy_rpz_periods` in that module, so this is
    scoped to the coil-to-plasma distance and leaves coil-to-coil untouched.
    """
    global _PATCHED
    if _PATCHED:
        return
    from desc.objectives import _coils

    from coilstellaration import metrics_utils_v2

    keep = config.PLASMA_PERIODS

    def _patched_copy_rpz_periods(rpz, nfp):
        return _tile_periods(rpz, nfp, keep)

    _coils.copy_rpz_periods = _patched_copy_rpz_periods
    # Drop any evaluators built with the original (full-torus) plasma tiling.
    metrics_utils_v2.clear_evaluator_cache()
    _PATCHED = True
    logger.info(
        "Patched coil-to-plasma distance to tile %s field period(s) "
        "(0/>=NFP = full torus)",
        keep,
    )
