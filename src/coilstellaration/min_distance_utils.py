"""Exact minimum distances between point clouds, pruned with bounding spheres.

Brute-force minimum distances (every source point against every target point)
dominate the cost of the coil-to-plasma and coil-to-coil metrics. Both point clouds
are split into small patches; the triangle inequality bounds every patch-pair distance
from below, and only the `num_pairs` patch pairs with the smallest lower bounds are
evaluated point by point. When no unevaluated patch pair could beat the minimum found,
that minimum is the brute-force minimum: a minimum is independent of evaluation order
and each candidate distance is computed with the same expression as the brute force,
so the result is bit-for-bit identical. `pruned_min_distance` reports per source
whether this certificate holds so callers can fall back to brute force when it does
not.
"""

from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from coilstellaration import types
from desc.utils import safenorm
from jaxtyping import Bool, Float, Int

# Relative slack on the lower bounds. It covers the rounding error of the
# bounding-sphere arithmetic (a few ulp of the coordinate scale); without it a
# rounded-up bound could certify a pruned patch pair that in fact holds the minimum.
_BOUND_RTOL = 1e-9


class PrunedMinDistance(NamedTuple):
    min_distance: Float[jax.Array, " n_src"]
    # False where an unevaluated patch pair might hold a smaller distance.
    certified: Bool[jax.Array, " n_src"]


@types.runtime_check_array_sizes
def segment_patches(
    n_points: int, patch_size: int
) -> Int[np.ndarray, "n_patches patch_size"]:
    """Split `n_points` consecutive points into patches of `patch_size` indices.

    The last patch is padded by repeating the last index; a repeated point never changes
    a minimum.
    """
    n_patches = -(-n_points // patch_size)
    index = np.minimum(np.arange(n_patches * patch_size), n_points - 1)
    return index.reshape(n_patches, patch_size)


@types.runtime_check_array_sizes
def grid_patches(
    n_rows: int, n_cols: int, tile_rows: int, tile_cols: int
) -> Int[np.ndarray, "n_patches patch_size"]:
    """Tile a row-major `(n_rows, n_cols)` point grid into rectangular patches.

    Edge tiles are padded by repeating the last row/column index.
    """
    rows = segment_patches(n_rows, tile_rows)
    cols = segment_patches(n_cols, tile_cols)
    index = rows[:, None, :, None] * n_cols + cols[None, :, None, :]
    return index.reshape(rows.shape[0] * cols.shape[0], tile_rows * tile_cols)


@types.runtime_check_array_sizes
def pruned_min_distance(
    src_pts: Float[jax.Array, "n_src n_src_pts 3"],
    src_patches: Int[np.ndarray, "n_src_patches src_patch_size"],
    tgt_pts: Float[jax.Array, "n_tgt_pts 3"],
    tgt_patches: Int[np.ndarray, "n_tgt_patches tgt_patch_size"],
    allowed: Bool[np.ndarray, "n_src n_tgt_patches"] | None,
    num_pairs: int,
) -> PrunedMinDistance:
    """Minimum distance from each source point set to the allowed target patches.

    Args:
        src_pts: One point set per source (e.g. one coil each).
        src_patches: Indices into the second axis of `src_pts`, shared by all sources.
        tgt_pts: Target points shared by all sources.
        tgt_patches: Indices into `tgt_pts`.
        allowed: Which target patches count for each source; `None` allows all. Used
            to exclude a coil's own points from its coil-to-coil distance.
        num_pairs: Patch pairs evaluated point by point per source. Larger values
            certify more often at a higher cost.

    Returns:
        The minimum `safenorm(src - tgt)` per source and whether it is certified equal
        to the brute-force minimum over all allowed target points.
    """
    n_src = src_pts.shape[0]
    n_src_patches = src_patches.shape[0]
    n_tgt_patches = tgt_patches.shape[0]
    num_pairs = min(num_pairs, n_src_patches * n_tgt_patches)

    src = src_pts[:, src_patches]
    tgt = tgt_pts[tgt_patches]
    src_center = jnp.mean(src, axis=-2)
    tgt_center = jnp.mean(tgt, axis=-2)
    src_radius = jnp.max(_norm(src - src_center[..., None, :]), axis=-1)
    tgt_radius = jnp.max(_norm(tgt - tgt_center[..., None, :]), axis=-1)

    center_dist = _norm(src_center[:, :, None, :] - tgt_center[None, None, :, :])
    slack = _BOUND_RTOL * (
        center_dist + src_radius[:, :, None] + tgt_radius[None, None, :]
    )
    lower = center_dist - src_radius[:, :, None] - tgt_radius[None, None, :] - slack
    if allowed is not None:
        lower = jnp.where(jnp.asarray(allowed)[:, None, :], lower, jnp.inf)
    lower = lower.reshape(n_src, n_src_patches * n_tgt_patches)

    has_rest = num_pairs < n_src_patches * n_tgt_patches
    neg_lower, pair = jax.lax.top_k(-lower, num_pairs + 1 if has_rest else num_pairs)
    chosen = pair[:, :num_pairs]
    src_sel = jnp.take_along_axis(
        src, (chosen // n_tgt_patches)[:, :, None, None], axis=1
    )
    tgt_sel = tgt[chosen % n_tgt_patches]
    dist = _norm(src_sel[:, :, :, None, :] - tgt_sel[:, :, None, :, :])
    # Disallowed pairs only get picked when fewer than `num_pairs` pairs are allowed.
    dist = jnp.where(
        jnp.isfinite(-neg_lower[:, :num_pairs])[:, :, None, None], dist, jnp.inf
    )
    min_distance = jnp.min(dist, axis=(1, 2, 3))

    # Non-finite points poison the bounds (NaN compares false, so pruning could drop
    # the very pair the brute-force NaN comes from); leave those to the brute force.
    finite = jnp.all(jnp.isfinite(src_pts)) & jnp.all(jnp.isfinite(tgt_pts))
    if has_rest:
        certified = finite & (min_distance <= -neg_lower[:, num_pairs])
    else:
        certified = jnp.broadcast_to(finite, (n_src,))
    return PrunedMinDistance(min_distance=min_distance, certified=certified)


def _norm(x: jax.Array) -> jax.Array:
    return cast(jax.Array, safenorm(x, axis=-1))
