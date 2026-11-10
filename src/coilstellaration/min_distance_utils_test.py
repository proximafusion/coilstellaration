import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Int
from coilstellaration import min_distance_utils
from desc.utils import safenorm

jax.config.update("jax_enable_x64", True)

_N_COILS = 4
_N_COIL_PTS = 101
_N_ZETA = 60
_N_THETA = 30


@pytest.fixture(params=list(range(3)))
def seed(request) -> int:
    return int(request.param)


@pytest.fixture
def coils(seed: int) -> jax.Array:
    """Wobbly poloidal loops around a torus, one per toroidal angle.

    Loop centres sit outboard of the plasma axis so each loop has a localized closest
    approach (concentric loops would be equidistant everywhere, defeating pruning).
    """
    rng = np.random.default_rng(seed)
    s = np.linspace(0, 2 * np.pi, _N_COIL_PTS, endpoint=False)
    coils = []
    for k in range(_N_COILS):
        phi = 2 * np.pi * k / _N_COILS + 0.05 * rng.standard_normal()
        r = 0.5 + 0.05 * rng.standard_normal() * np.cos(2 * s)
        big_r = 1.2 + r * np.cos(s)
        coils.append(
            np.stack([big_r * np.cos(phi), big_r * np.sin(phi), r * np.sin(s)], axis=-1)
        )
    return jnp.asarray(np.stack(coils))


@pytest.fixture
def plasma(seed: int) -> jax.Array:
    rng = np.random.default_rng(seed + 100)
    zeta, theta = np.meshgrid(
        np.linspace(0, 2 * np.pi, _N_ZETA, endpoint=False),
        np.linspace(0, 2 * np.pi, _N_THETA, endpoint=False),
        indexing="ij",
    )
    a = 0.2 + 0.03 * rng.standard_normal() * np.cos(theta + zeta)
    big_r = 1.0 + a * np.cos(theta)
    pts = np.stack(
        [big_r * np.cos(zeta), big_r * np.sin(zeta), a * np.sin(theta)], axis=-1
    )
    return jnp.asarray(pts.reshape(-1, 3))


@pytest.fixture
def coil_segments() -> Int[np.ndarray, "n_patches patch_size"]:
    return min_distance_utils.segment_patches(_N_COIL_PTS, 8)


@pytest.fixture
def plasma_patches() -> Int[np.ndarray, "n_patches patch_size"]:
    return min_distance_utils.grid_patches(_N_ZETA, _N_THETA, 4, 5)


def _brute_force(src: jax.Array, tgt: jax.Array) -> jax.Array:
    dist = jnp.asarray(safenorm(src[:, :, None, :] - tgt[None, None, :, :], axis=-1))
    return jnp.min(dist, axis=(1, 2))


def test_pruned_equals_brute_force_coil_to_plasma(
    coils: jax.Array,
    plasma: jax.Array,
    coil_segments: np.ndarray,
    plasma_patches: np.ndarray,
) -> None:
    result = min_distance_utils.pruned_min_distance(
        src_pts=coils,
        src_patches=coil_segments,
        tgt_pts=plasma,
        tgt_patches=plasma_patches,
        allowed=None,
        num_pairs=64,
    )
    assert np.all(np.asarray(result.certified))
    # Minima are order independent, so a certified result is bit-identical.
    np.testing.assert_array_equal(
        np.asarray(result.min_distance), np.asarray(_brute_force(coils, plasma))
    )


def test_pruned_excludes_disallowed_patches(
    coils: jax.Array, coil_segments: np.ndarray
) -> None:
    segments = coil_segments
    tgt_patches = (np.arange(_N_COILS)[:, None, None] * _N_COIL_PTS + segments).reshape(
        -1, segments.shape[1]
    )
    patch_coil = np.repeat(np.arange(_N_COILS), segments.shape[0])
    result = min_distance_utils.pruned_min_distance(
        src_pts=coils,
        src_patches=segments,
        tgt_pts=coils.reshape(-1, 3),
        tgt_patches=tgt_patches,
        allowed=patch_coil[None, :] != np.arange(_N_COILS)[:, None],
        num_pairs=64,
    )
    expected = [
        _brute_force(coils[k : k + 1], jnp.delete(coils, k, axis=0).reshape(-1, 3))[0]
        for k in range(_N_COILS)
    ]
    assert np.all(np.asarray(result.certified))
    np.testing.assert_array_equal(np.asarray(result.min_distance), np.asarray(expected))


def test_too_few_pairs_is_not_certified_but_never_too_small(
    coils: jax.Array,
    plasma: jax.Array,
    coil_segments: np.ndarray,
    plasma_patches: np.ndarray,
) -> None:
    result = min_distance_utils.pruned_min_distance(
        src_pts=coils,
        src_patches=coil_segments,
        tgt_pts=plasma,
        tgt_patches=plasma_patches,
        allowed=None,
        num_pairs=1,
    )
    brute = np.asarray(_brute_force(coils, plasma))
    pruned = np.asarray(result.min_distance)
    # An uncertified minimum is over a subset of pairs, so it can only overshoot.
    assert np.all(pruned >= brute)
    certified = np.asarray(result.certified)
    np.testing.assert_array_equal(pruned[certified], brute[certified])


def test_non_finite_points_are_not_certified(
    coils: jax.Array,
    plasma: jax.Array,
    coil_segments: np.ndarray,
    plasma_patches: np.ndarray,
) -> None:
    coils = coils.at[1, 3, 0].set(jnp.nan)
    result = min_distance_utils.pruned_min_distance(
        src_pts=coils,
        src_patches=coil_segments,
        tgt_pts=plasma,
        tgt_patches=plasma_patches,
        allowed=None,
        num_pairs=10_000,
    )
    assert not np.any(np.asarray(result.certified))


def test_grid_patches_cover_every_point_once_up_to_padding() -> None:
    patches = min_distance_utils.grid_patches(7, 5, 3, 2)
    assert patches.shape == (3 * 3, 6)
    np.testing.assert_array_equal(np.unique(patches), np.arange(35))
