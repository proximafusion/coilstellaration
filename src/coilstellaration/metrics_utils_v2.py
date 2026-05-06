"""Cached, jit-friendly entry points for ConStellaration metrics.

`MetricsEvaluator` binds DESC objectives and per-grid transforms once to a
template `(eq, coilset)` shape and exposes `jax.jit`-wrapped callables that
dispatch via `params_dict`. Repeated evaluations on shape-identical inputs
reuse the compiled traces.
"""

import collections
import hashlib
import logging
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from desc.coils import CoilSet as DescCoilSet
from desc.coils import _Coil as DescCoil
from desc.compute.utils import _compute as compute_fun
from desc.compute.utils import get_profiles, get_transforms
from desc.equilibrium import Equilibrium as DescEquilibrium
from desc.geometry import FourierRZToroidalSurface as DescFourierRZToroidalSurface
from desc.grid import LinearGrid, QuadratureGrid
from desc.objectives import (
    CoilArclengthVariance,
    CoilCurrentLength,
    CoilCurvature,
    CoilIntegratedCurvature,
    CoilLength,
    CoilSetLinkingNumber,
    CoilSetMinDistance,
    CoilTorsion,
    LinkingCurrentConsistency,
    PlasmaCoilSetMinDistance,
    ToroidalFlux,
)

from coilstellaration import (
    metrics_utils,
    types,
)

logger = logging.getLogger(__name__)

METRICS_FIELD_ORDER = metrics_utils.METRICS_FIELD_ORDER
METRICS_LOG1P_MASK = metrics_utils.METRICS_LOG1P_MASK


class MetricsEvaluator:
    """Reusable, jit-cached metrics evaluator.

    Bind to a template (eq, coilset) shape: same eq basis (L, M, N, NFP,
    sym), same coilset structure (num_coils, coil basis). Call `.evaluate`
    repeatedly with structurally-identical inputs for cache-hit speedup.
    """

    def __init__(
        self,
        template_eq: DescEquilibrium,
        template_coilset: DescCoilSet,
        surf_eval_m: int = 80,
        surf_eval_n: int = 80,
        coil_eval_n: int = 200,
        minor_radius_m: int = 16,
        minor_radius_n: int = 16,
        minor_radius_l: int = 16,
        subset_metrics: bool = False,
    ) -> None:
        self._template_eq = template_eq
        self._template_surface = template_eq.surface
        self._template_coilset = template_coilset
        self._surf_grid = LinearGrid(
            M=surf_eval_m, N=surf_eval_n, NFP=template_eq.NFP, sym=template_eq.sym
        )
        self._coil_grid = LinearGrid(N=coil_eval_n)
        self._toroidal_flux_eval_grid = LinearGrid(
            L=surf_eval_m, M=surf_eval_m, zeta=0.0, NFP=template_eq.NFP
        )
        self._minor_radius_quadrature_grid = QuadratureGrid(
            L=minor_radius_l, M=minor_radius_m, N=minor_radius_n, NFP=template_eq.NFP
        )
        self._axis_grid = LinearGrid(
            rho=jnp.array([0.0]),
            theta=jnp.array([0.0]),
            zeta=jnp.linspace(
                0, 2 * jnp.pi / template_eq.NFP, coil_eval_n, endpoint=False
            ),
            NFP=template_eq.NFP,
            sym=template_eq.sym,
        )
        self._zeta_n = coil_eval_n
        self._template_NFP = template_eq.NFP
        self._template_sym = template_eq.sym
        self._template_num_coils = template_coilset.num_coils
        # DESC per-coil objectives (CoilLength/CoilCurvature/CoilTorsion) emit
        # results over unique (primary) coils only, not the symmetrized total.
        self._template_num_unique_coils = len(template_coilset)

        self._subset_metrics = subset_metrics
        self._num_zeta_per_coil = self._coil_grid.num_zeta

        # Build un-jitted per-objective compute callables. They are folded
        # into a single fused `jax.jit` below (`self._fused_evaluate`) so we
        # pay one trace, one compile, one dispatch instead of one per
        # objective.
        coil_to_coil_fn = self._make_compute(
            CoilSetMinDistance(coil=template_coilset, grid=self._coil_grid)
        )
        coil_length_fn = self._make_compute(
            CoilLength(coil=template_coilset, grid=self._coil_grid)
        )
        coil_curvature_fn = self._make_compute(
            CoilCurvature(coil=template_coilset, grid=self._coil_grid)
        )
        coil_to_plasma_fn = self._make_compute(
            PlasmaCoilSetMinDistance(
                eq=template_eq,
                coil=template_coilset,
                coil_grid=self._coil_grid,
                plasma_grid=self._surf_grid,
                eq_fixed=False,
            )
        )
        minor_radius_inner = self._make_minor_radius_fn()
        surface_field_inner = self._make_surface_field_fn()

        if subset_metrics:
            linking_num_fn = None
            arclen_var_fn = None
            coil_current_len_fn = None
            coil_int_curv_fn = None
            coil_torsion_fn = None
            toroidal_flux_fn = None
            linking_consist_fn = None
            on_axis_b_inner = None
        else:
            linking_num_fn = self._make_compute(
                CoilSetLinkingNumber(coil=template_coilset, grid=self._coil_grid)
            )
            arclen_var_fn = self._make_compute(
                CoilArclengthVariance(template_coilset, grid=self._coil_grid)
            )
            coil_current_len_fn = self._make_compute(
                CoilCurrentLength(coil=template_coilset, grid=self._coil_grid)
            )
            coil_int_curv_fn = self._make_compute(
                CoilIntegratedCurvature(coil=template_coilset, grid=self._coil_grid)
            )
            coil_torsion_fn = self._make_compute(
                CoilTorsion(coil=template_coilset, grid=self._coil_grid)
            )
            toroidal_flux_fn = self._make_compute(
                ToroidalFlux(
                    eq=template_eq,
                    field=template_coilset,
                    eq_fixed=False,
                    field_grid=self._coil_grid,
                    eval_grid=self._toroidal_flux_eval_grid,
                )
            )
            linking_consist_fn = self._make_compute(
                LinkingCurrentConsistency(
                    coil=template_coilset,
                    eq=template_eq,
                    grid=self._surf_grid,
                    eq_fixed=False,
                )
            )
            on_axis_b_inner = self._make_on_axis_b_fn()

        num_unique_coils = self._template_num_unique_coils
        num_zeta_per_coil = self._num_zeta_per_coil

        @jax.jit
        def _fused_evaluate(eq_params, coil_params, surface_params):
            out = {
                "coil_length": coil_length_fn(eq_params, coil_params),
                "coil_curvature": coil_curvature_fn(eq_params, coil_params).reshape(
                    num_unique_coils, num_zeta_per_coil
                ),
                "coil_to_coil_min": coil_to_coil_fn(eq_params, coil_params),
                "coil_to_plasma_min": coil_to_plasma_fn(eq_params, coil_params),
                "minor_radius": minor_radius_inner(eq_params),
            }
            (
                out["normalized_field_error"],
                out["local_quadratic_flux"],
                out["quadratic_flux"],
            ) = surface_field_inner(surface_params, coil_params)
            if not subset_metrics:
                assert coil_torsion_fn is not None
                assert linking_num_fn is not None
                assert arclen_var_fn is not None
                assert coil_current_len_fn is not None
                assert coil_int_curv_fn is not None
                assert toroidal_flux_fn is not None
                assert linking_consist_fn is not None
                assert on_axis_b_inner is not None
                out["coil_torsion"] = coil_torsion_fn(eq_params, coil_params).reshape(
                    num_unique_coils, num_zeta_per_coil
                )
                out["linking_num"] = linking_num_fn(eq_params, coil_params)
                out["arclen_var"] = arclen_var_fn(eq_params, coil_params)
                out["coil_current_len"] = coil_current_len_fn(eq_params, coil_params)
                out["coil_int_curv"] = coil_int_curv_fn(eq_params, coil_params)
                out["toroidal_flux"] = toroidal_flux_fn(eq_params, coil_params)
                out["linking_consist"] = linking_consist_fn(eq_params, coil_params)
                out["B_avg"] = on_axis_b_inner(eq_params, coil_params)
            return out

        self._fused_evaluate = _fused_evaluate

    def _make_minor_radius_fn(self):
        a_transforms = get_transforms(
            ["a"], obj=self._template_eq, grid=self._minor_radius_quadrature_grid
        )
        a_profiles = get_profiles(
            ["a"], obj=self._template_eq, grid=self._minor_radius_quadrature_grid
        )
        template_eq = self._template_eq

        def _minor_radius_fn(eq_params):
            data = compute_fun(
                template_eq,
                ["a"],
                params=eq_params,
                transforms=a_transforms,
                profiles=a_profiles,
            )
            return data["a"]

        return _minor_radius_fn

    def _make_on_axis_b_fn(self):
        axis_transforms = get_transforms(
            ["x"], obj=self._template_eq, grid=self._axis_grid
        )
        axis_profiles = get_profiles(["x"], obj=self._template_eq, grid=self._axis_grid)
        template_eq = self._template_eq
        template_coilset = self._template_coilset

        def _on_axis_b_fn(eq_params, coil_params):
            data = compute_fun(
                template_eq,
                ["x"],
                params=eq_params,
                transforms=axis_transforms,
                profiles=axis_profiles,
            )
            B_vec = template_coilset.compute_magnetic_field(
                data["x"], basis="xyz", params=coil_params
            )
            return jnp.mean(jnp.linalg.norm(B_vec, axis=-1))

        return _on_axis_b_fn

    def _make_surface_field_fn(self):
        surf_keys = ["x", "n_rho", "|e_theta x e_zeta|"]
        surf_transforms = get_transforms(
            surf_keys, obj=self._template_surface, grid=self._surf_grid
        )
        surf_profiles = get_profiles(
            surf_keys, obj=self._template_surface, grid=self._surf_grid
        )
        surf_shape = (self._surf_grid.num_theta, self._surf_grid.num_zeta)
        template_surface = self._template_surface
        template_coilset = self._template_coilset
        coil_grid = self._coil_grid

        def _surface_field_fn(surface_params, coil_params):
            data = compute_fun(
                template_surface,
                surf_keys,
                params=surface_params,
                transforms=surf_transforms,
                profiles=surf_profiles,
            )
            B_vec = template_coilset.compute_magnetic_field(
                data["x"],
                basis="rpz",
                source_grid=coil_grid,
                params=coil_params,
            )
            Bn = jnp.sum(B_vec * data["n_rho"], axis=-1)
            B_mag = jnp.linalg.norm(B_vec, axis=-1)
            dS = data["|e_theta x e_zeta|"]
            normalized_field_error = (jnp.abs(Bn) / B_mag).reshape(surf_shape)
            local_quadratic_flux = (Bn**2 / B_mag**2 * dS).reshape(surf_shape)
            # `QuadraticFlux.compute` body in vacuum mode (B_plasma=0):
            # `f = Bn * sqrt(dS)`. Folded in here so the eq side comes from
            # live `surface_params`, not the template baked into
            # `QuadraticFlux.constants["eval_data"]` at build time.
            quadratic_flux = (Bn * jnp.sqrt(dS)).reshape(surf_shape)
            return normalized_field_error, local_quadratic_flux, quadratic_flux

        return _surface_field_fn

    @staticmethod
    def _make_compute(obj):
        """Build a DESC objective and return an un-jitted compute callable.

        Returned callable has signature `(eq_params, coil_params) -> array` and
        dispatches `obj.compute(...)` honouring `obj.things` order. Folded
        into the single fused `self._fused_evaluate` jit at evaluator
        construction time.
        """
        obj.build(verbose=0)
        unexpected = [
            type(t).__name__
            for t in obj.things
            if not isinstance(t, (DescEquilibrium, DescCoilSet))
        ]
        assert (
            not unexpected
        ), f"Unexpected `things` for {type(obj).__name__}: {unexpected}"
        things_is_eq = tuple(isinstance(t, DescEquilibrium) for t in obj.things)

        if not any(things_is_eq):

            def call_coil_only(eq_params, coil_params):
                return obj.compute(coil_params)

            return call_coil_only

        def call(eq_params, coil_params):
            args = [eq_params if is_eq else coil_params for is_eq in things_is_eq]
            return obj.compute(*args)

        return call

    def evaluate(self, eq: DescEquilibrium, coilset: DescCoilSet) -> types.Metrics:
        assert eq.NFP == self._template_NFP and eq.sym == self._template_sym
        assert coilset.num_coils == self._template_num_coils
        assert len(coilset) == self._template_num_unique_coils

        coil_currents = coilset.current

        eq_params = eq.params_dict
        coil_params = coilset.params_dict
        surface_params = eq.surface.params_dict

        out = self._fused_evaluate(eq_params, coil_params, surface_params)

        coil_lengths = out["coil_length"]
        coil_curvatures = out["coil_curvature"]
        coil_to_coil_min = out["coil_to_coil_min"]
        coil_to_plasma_min = out["coil_to_plasma_min"]
        minor_radius = out["minor_radius"]
        normalized_field_error = out["normalized_field_error"]
        local_quadratic_flux = out["local_quadratic_flux"]
        quadratic_flux = out["quadratic_flux"]
        if self._subset_metrics:
            coil_torsions = None
            linking_numbers = None
            arclen_var = None
            coil_current_lengths = None
            coil_int_curvatures = None
            toroidal_flux = None
            linking_consist = None
            B_avg = None
        else:
            coil_torsions = out["coil_torsion"]
            linking_numbers = out["linking_num"]
            arclen_var = out["arclen_var"]
            coil_current_lengths = out["coil_current_len"]
            coil_int_curvatures = out["coil_int_curv"]
            toroidal_flux = out["toroidal_flux"]
            linking_consist = out["linking_consist"]
            B_avg = out["B_avg"]

        n_symmetrized = 2 * eq.NFP if eq.sym else eq.NFP
        n_coils_per_half_period = coilset.num_coils // n_symmetrized

        return types.Metrics(
            surf_eval_coords=self._surf_grid.nodes[:, [1, 2]]
            .reshape(self._surf_grid.num_theta, self._surf_grid.num_zeta, 2)
            .transpose(2, 0, 1),
            coil_eval_params=self._coil_grid.nodes[:, 2],
            coil_currents=jnp.asarray(coil_currents),
            n_coils_per_half_period=n_coils_per_half_period,
            on_axis_average_magnetic_field=B_avg,
            normalized_field_error=normalized_field_error,
            coil_linking_numbers=linking_numbers,
            coil_to_coil_min_distances=coil_to_coil_min,
            coil_to_plasma_min_distances=coil_to_plasma_min,
            coil_lengths=coil_lengths,
            coil_curvatures=coil_curvatures,
            coil_current_lengths=coil_current_lengths,
            coil_integrated_curvatures=coil_int_curvatures,
            coil_torsions=coil_torsions,
            coil_arclength_variances=arclen_var,
            quadratic_flux=quadratic_flux,
            local_quadratic_flux=local_quadratic_flux,
            toroidal_flux=toroidal_flux,
            linking_current_consistency=linking_consist,
            minor_radius=minor_radius,
        )


def _surface_content_key(
    surface: DescFourierRZToroidalSurface,
) -> tuple[int | bool | str, ...]:
    """Hashable content descriptor for a surface — keys equilibrium reuse.

    Surfaces with equal basis (NFP, sym, L, M, N) and equal coefficients share
    a cached `DescEquilibrium`. The basis dims are kept explicit so collisions
    are impossible if `R_lmn`/`Z_lmn` happen to share a byte hash across bases.
    """
    return (
        int(surface.NFP),
        bool(surface.sym),
        int(surface.L),
        int(surface.M),
        int(surface.N),
        hashlib.sha256(np.asarray(surface.R_lmn).tobytes()).hexdigest()[:32],
        hashlib.sha256(np.asarray(surface.Z_lmn).tobytes()).hexdigest()[:32],
    )


_EQUILIBRIUM_CACHE_MAX = 512
_EQUILIBRIUM_CACHE: collections.OrderedDict[tuple, DescEquilibrium] = (
    collections.OrderedDict()
)


def get_cached_equilibrium(surface: DescFourierRZToroidalSurface) -> DescEquilibrium:
    """Return an LRU-cached `DescEquilibrium` constructed from `surface`.

    `DescEquilibrium(surface=...)` is dominated by `ensure_positive_jacobian`
    (~30s+ at high resolution from JIT-compiling `eq.compute("sqrt(g)")`) and
    by the `set_initial_guess` projection of surface coefficients. We disable
    both — orientation can be enforced upstream and nesting checks aren't
    relevant for surface-only metrics — and memoize the result.
    """
    key = _surface_content_key(surface)
    cached = _EQUILIBRIUM_CACHE.get(key)
    if cached is not None:
        _EQUILIBRIUM_CACHE.move_to_end(key)
        return cached
    logger.info(f"Equilibrium cache miss for key {key}. Building new equilibrium...")
    equilibrium = DescEquilibrium(
        surface=surface, check_orientation=False, ensure_nested=False
    )
    _EQUILIBRIUM_CACHE[key] = equilibrium
    if len(_EQUILIBRIUM_CACHE) > _EQUILIBRIUM_CACHE_MAX:
        popped_key, _ = _EQUILIBRIUM_CACHE.popitem(last=False)
        logger.info(
            f"Equilibrium cache full. Evicting oldest equilibrium for key {popped_key}."
        )
    return equilibrium


def clear_equilibrium_cache() -> None:
    """Drop all cached `DescEquilibrium` instances.

    Useful in tests.
    """
    _EQUILIBRIUM_CACHE.clear()


def _evaluator_shape_key(
    eq: DescEquilibrium,
    coilset: DescCoilSet,
    surf_eval_m: int,
    surf_eval_n: int,
    coil_eval_n: int,
    subset_metrics: bool = False,
) -> tuple[str | int | bool, ...]:
    """Hashable shape descriptor — anything that affects DESC trace shapes.

    Excludes parameter values: identical shapes (e.g. two equilibria with
    the same basis but different coefficients) share an evaluator.
    """
    coil_0 = cast(DescCoil, coilset[0])

    return (
        int(eq.NFP),
        bool(eq.sym),
        int(eq.L),
        int(eq.M),
        int(eq.N),
        int(eq.surface.L),
        int(eq.surface.M),
        int(eq.surface.N),
        int(coilset.num_coils),
        int(len(coilset)),
        # Coil basis resolution: every coil shares basis structure in our
        # use cases, so the first coil's params shape is representative.
        *tuple(sorted(coil_0.params_dict.keys())),
        *tuple(
            int(v.size) if hasattr(v, "size") else 0
            for _, v in sorted(coil_0.params_dict.items())
        ),
        int(surf_eval_m),
        int(surf_eval_n),
        int(coil_eval_n),
        subset_metrics,
    )


_EVALUATOR_CACHE_MAX = 8
_EVALUATOR_CACHE: collections.OrderedDict[tuple, MetricsEvaluator] = (
    collections.OrderedDict()
)


def get_cached_evaluator(
    eq: DescEquilibrium,
    coilset: DescCoilSet,
    surf_eval_m: int = 80,
    surf_eval_n: int = 80,
    coil_eval_n: int = 200,
    subset_metrics: bool = False,
) -> MetricsEvaluator:
    """Return an LRU-cached `MetricsEvaluator` for this shape.

    Misses build a new evaluator with `(eq, coilset)` as templates. Cache is
    bounded at `_EVALUATOR_CACHE_MAX` entries; oldest is evicted on overflow.
    """
    key = _evaluator_shape_key(
        eq, coilset, surf_eval_m, surf_eval_n, coil_eval_n, subset_metrics
    )
    cached = _EVALUATOR_CACHE.get(key)
    if cached is not None:
        _EVALUATOR_CACHE.move_to_end(key)
        return cached
    logger.info(f"MetricsEvaluator cache miss for key {key}. Building new evaluator...")
    evaluator = MetricsEvaluator(
        template_eq=eq,
        template_coilset=coilset,
        surf_eval_m=surf_eval_m,
        surf_eval_n=surf_eval_n,
        coil_eval_n=coil_eval_n,
        subset_metrics=subset_metrics,
    )
    _EVALUATOR_CACHE[key] = evaluator
    if len(_EVALUATOR_CACHE) > _EVALUATOR_CACHE_MAX:
        popped_key, _ = _EVALUATOR_CACHE.popitem(last=False)
        logger.info(
            "MetricsEvaluator cache full. Evicting oldest evaluator for key "
            f"{popped_key}."
        )
    return evaluator


def clear_evaluator_cache() -> None:
    """Drop all cached `MetricsEvaluator` instances.

    Useful in tests.
    """
    _EVALUATOR_CACHE.clear()


@types.runtime_check_array_sizes
def evaluate_coilset_metrics(
    eq: DescEquilibrium,
    coilset: DescCoilSet,
    surf_eval_m: int = 80,
    surf_eval_n: int = 80,
    coil_eval_n: int = 200,
    subset_metrics: bool = False,
) -> types.Metrics:
    """One-shot evaluation. Mirrors `metrics_utils.evaluate_coilset_metrics`.

    Repeated calls with shape-identical inputs reuse a cached
    `MetricsEvaluator` — first call pays the ~0.7s build, subsequent calls
    skip it and reuse the JIT-compiled objectives.
    """
    return get_cached_evaluator(
        eq=eq,
        coilset=coilset,
        surf_eval_m=surf_eval_m,
        surf_eval_n=surf_eval_n,
        coil_eval_n=coil_eval_n,
        subset_metrics=subset_metrics,
    ).evaluate(eq, coilset)


@types.runtime_check_array_sizes
def evaluate_coilset_metrics_from_boundary(
    boundary: DescFourierRZToroidalSurface,
    coilset: DescCoilSet,
    surf_eval_m: int = 80,
    surf_eval_n: int = 80,
    coil_eval_n: int = 200,
    subset_metrics: bool = False,
) -> types.Metrics:
    return evaluate_coilset_metrics(
        eq=get_cached_equilibrium(boundary),
        coilset=coilset,
        surf_eval_m=surf_eval_m,
        surf_eval_n=surf_eval_n,
        coil_eval_n=coil_eval_n,
        subset_metrics=subset_metrics,
    )
