from typing import Annotated, Generic, Literal, TypeVar

import pydantic
from constellaration.utils.pytree import register_pydantic_data
from jaxtyping import Float

from constellaration_update.utils.types import NpOrJaxArray, ScalarFloat, ScalarInt

ScalarT = str | ScalarInt | ScalarFloat
RangeT = tuple[ScalarT, ScalarT]
ChoicesT = frozenset[ScalarT]

_Numeric = TypeVar("_Numeric")
_Categorical = TypeVar("_Categorical")


class ConStellarationUpdateRequirementsTemplate(
    pydantic.BaseModel,
    Generic[_Numeric, _Categorical],
):
    """Template for sampled or bounds-valued requirements.

    Parametrize with ``[ScalarT, ScalarT]`` for sampled instances or
    ``[RangeT, ChoicesT]`` for bounds instances.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    normalized_min_coil_to_coil_distance: _Numeric

    normalized_min_coil_plasma_distance: _Numeric

    regcoil_winding_surface_plasma_distance: _Numeric
    """The normalized distance from the plasma surface at which the REGCOIL winding
    surface is constructed."""

    normalized_max_coil_curvature: _Numeric

    n_coils_per_half_period: _Numeric
    """Number of unique coils per half field period."""

    coil_fourier_order: _Numeric

    desc_objective_lambda_log10: _Numeric
    """Base-10 logarithm of the DESC objective tradeoff weight."""

    desc_x_scale_mode: _Categorical
    """The DESC x scaling mode.

    If ``"ess"`` (exponential spectral scaling), it must encode what order to use.
    """

    regcoil_target_option: _Categorical
    """High-level REGCOIL target used to choose the lambda selection rule.

    The allowed choices are ``"normalized_coil_to_coil_distance"`` and
    ``"normalized_field_error"``.
    """

    regcoil_maximum_normalized_field_error: _Numeric | None = None
    """Maximum normalized field error target for REGCOIL.

    Warning:
        This value is only used when the REGCOIL target option is
        ``"normalized_field_error"``.
    """


class ConStellarationUpdateRequirements(
    ConStellarationUpdateRequirementsTemplate[ScalarT, ScalarT]
):
    pass


class ConStellarationUpdateRequirementsBounds(
    ConStellarationUpdateRequirementsTemplate[RangeT, ChoicesT]
):
    pass


class ConStellarationUpdateConfigSamplingSettings(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    seed: str


class ConStellarationUpdateRegcoilSettings(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    n_surface_poloidal_grid_points: int = 64
    """Number of grid points in the poloidal angle used to evaluate surface integrals on
    the plasma surface."""

    n_surface_toroidal_grid_points: int = 64
    """Number of grid points in the toroidal angle used to evaluate surface integrals on
    the plasma surface."""

    n_coil_poloidal_grid_points: int = 64
    """Number of grid points in the poloidal angle used to evaluate surface integrals on
    the coil winding surface."""

    n_coil_toroidal_grid_points: int = 64
    """Number of grid points in the toroidal angle used to evaluate surface integrals on
    the coil winding surface."""

    current_potential_max_poloidal_mode_number: int = 16
    """Maximum poloidal mode number to describe the current potential on the coil
    winding surface."""

    current_potential_max_toroidal_mode_number: int = 16
    """Maximum toroidal mode number to describe the current potential on the coil
    winding surface."""

    n_fourier_modes: int = 10
    """Number of Fourier modes used when cutting filamentary coils from REGCOIL
    output."""

    n_discretization_points: int = 300
    """Number of discretization points used when cutting filamentary coils from REGCOIL
    output."""

    verbose: bool = False
    """Whether to print REGCOIL output to console."""

    target_option: Literal[
        "normalized_coil_to_coil_distance",
        "normalized_field_error",
        "minimum_chi2_k_times_chi2_b",
    ] = "normalized_coil_to_coil_distance"
    """High-level REGCOIL target used to choose the lambda selection rule."""


class ConStellarationUpdateDescOptimizerSettings(pydantic.BaseModel):
    """Settings for the DESC augmented Lagrangian coilset optimizer."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    eval_grid_m: int = 32
    """Poloidal resolution of the evaluation grid."""

    eval_grid_n: int = 32
    """Toroidal resolution of the evaluation grid."""

    coil_grid_n: int = 128
    """Tangential resolution of the coil grid."""

    maxiter: int = 500
    """Maximum number of optimizer iterations."""

    ftol: float = 1e-15
    """Objective function tolerance."""

    xtol: float = 1e-15
    """Parameter change tolerance."""

    ctol: float = 1e-15
    """Constraint violation tolerance."""

    verbose: bool = False
    """If True, print optimization diagnostics at each iteration."""

    optimizer: Literal["fmin-auglag-bfgs", "scipy-SLSQP"] = "fmin-auglag-bfgs"
    """Optimizer to use for the DESC coilset optimization."""

    desc_ess_alpha: float = 1.0
    """The DESC exponential spectral scaling alpha parameter."""

    optimizer_options: dict | None = None
    """Additional options to pass to the DESC optimizer."""


@register_pydantic_data
class ConStellarationUpdateCoilset(pydantic.BaseModel):
    r"""A coilset stored in DESC's FourierXYZCoil representation.

    Each coil curve is parameterized as:

    .. math::
        x(\theta) = \sum_{n=-N}^{N} X_n \, f_n(\theta)

    where :math:`f_n = \cos(n\theta)` for :math:`n \geq 0` and
    :math:`f_n = \sin(|n|\theta)` for :math:`n < 0`. The coefficient array layout
    is ``[sin(N), ..., sin(1), const, cos(1), ..., cos(N)]`` with length
    ``2*N + 1``.

    Only unique (pre-symmetry-expansion) coils are stored.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    coil_x_n: Float[NpOrJaxArray, "n_unique_coils n_modes"]
    """X Fourier coefficients for each unique coil, in DESC layout."""

    coil_y_n: Float[NpOrJaxArray, "n_unique_coils n_modes"]
    """Y Fourier coefficients for each unique coil, in DESC layout."""

    coil_z_n: Float[NpOrJaxArray, "n_unique_coils n_modes"]
    """Z Fourier coefficients for each unique coil, in DESC layout."""

    currents: Float[NpOrJaxArray, " n_unique_coils"]
    """Current in Amperes for each unique coil."""

    n_field_periods: ScalarInt
    """Number of toroidal field periods."""

    is_stellarator_symmetric: bool
    """Whether the coilset has stellarator symmetry."""

    @property
    def n_unique_coils(self) -> ScalarInt:
        """Number of unique coils."""
        return self.coil_x_n.shape[0]

    @property
    def fourier_order(self) -> ScalarInt:
        """Fourier order N (number of modes is 2*N + 1)."""
        return (self.coil_x_n.shape[1] - 1) // 2


@register_pydantic_data
class MetricsTargets(pydantic.BaseModel):
    """Six scalar target metrics that condition `CoilPredictor`.

    Field order is the public contract; see `metrics_utils.METRICS_FIELD_ORDER`.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    min_normalized_coil_to_coil_distance: ScalarFloat
    min_normalized_coil_to_plasma_distance: ScalarFloat
    max_normalized_coil_curvature: ScalarFloat
    max_normalized_field_error: ScalarFloat
    on_axis_average_magnetic_field: ScalarFloat
    toroidal_flux: ScalarFloat

    def to_array(self) -> Float[NpOrJaxArray, " 6"]:
        """Return the six fields as a length-6 array in canonical order."""
        import jax.numpy as jnp

        return jnp.asarray(
            [
                self.min_normalized_coil_to_coil_distance,
                self.min_normalized_coil_to_plasma_distance,
                self.max_normalized_coil_curvature,
                self.max_normalized_field_error,
                self.on_axis_average_magnetic_field,
                self.toroidal_flux,
            ]
        )


@register_pydantic_data
class RequirementMetrics(pydantic.BaseModel):
    """The 4 `MetricsTargets` fields with a counterpart in
    `ConStellarationUpdateRequirements`.

    Used to condition `CoilPredictor` on a small subset of achieved metrics.
    The 2 `MetricsTargets` fields with no requirement counterpart
    (`on_axis_average_magnetic_field`, `toroidal_flux`) are intentionally absent.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    min_normalized_coil_to_coil_distance: ScalarFloat
    min_normalized_coil_to_plasma_distance: ScalarFloat
    max_normalized_coil_curvature: ScalarFloat
    max_normalized_field_error: ScalarFloat

    def to_array(self) -> Float[NpOrJaxArray, " 4"]:
        """Return the four fields as a length-4 array in declaration order."""
        import jax.numpy as jnp

        return jnp.asarray(
            [
                self.min_normalized_coil_to_coil_distance,
                self.min_normalized_coil_to_plasma_distance,
                self.max_normalized_coil_curvature,
                self.max_normalized_field_error,
            ]
        )


class ConStellarationUpdateDescSolveResults(pydantic.BaseModel):
    """Convergence diagnostics from a DESC coilset optimization.

    Fields are the common subset available from both scipy-slsqp and fmin-auglag-bfgs
    optimizers.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    success: bool
    """Whether the optimizer converged."""

    message: str
    """Termination reason from the optimizer."""

    n_iterations: int
    """Total number of optimizer iterations."""

    n_function_evals: int
    """Total number of objective function evaluations."""

    objective_value: ScalarFloat
    """Final objective function value."""

    optimality: ScalarFloat | None = None
    """Gradient norm (L-inf) at the solution.

    Only available for some optimizers.
    """

    constraint_violation: ScalarFloat | None = None
    """Infinity norm of the constraint violation.

    Only available for some optimizers.
    """


class ConStellarationUpdateDescOutput(pydantic.BaseModel):
    """Combined output of a DESC coilset optimization run."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    coilset: ConStellarationUpdateCoilset
    """The optimized coilset in Fourier representation."""

    desc_solve_results: ConStellarationUpdateDescSolveResults
    """Convergence diagnostics from the DESC optimization."""


@register_pydantic_data
class ConStellarationUpdateMetrics(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    surf_eval_coords: Annotated[Float[NpOrJaxArray, "2 n_theta n_zeta"], ...]
    """Meshgrid of (theta, zeta) at which surface-field quantities are evaluated."""

    coil_eval_params: Annotated[Float[NpOrJaxArray, " n_coil_eval_points"], ...]
    """Toroidal coordinates at which coil-wise quantities are evaluated."""

    coil_currents: Float[NpOrJaxArray, " n_unique_coils"]

    n_coils_per_half_period: ScalarInt | None = None
    """Number of unique coils per half field period."""

    on_axis_average_magnetic_field: ScalarFloat
    """The on-axis average magnetic field."""

    normalized_field_error: Annotated[Float[NpOrJaxArray, "n_theta n_zeta"], ...]
    """|Bn| / |B| on the plasma surface."""

    coil_linking_numbers: Float[NpOrJaxArray, " n_coils"]
    """Per-coil linking number (sum of abs linking numbers with other coils)."""

    coil_to_coil_min_distances: Float[NpOrJaxArray, " n_coils"]
    """Minimum distance from each coil to any other coil (m)."""

    @pydantic.computed_field
    @property
    def normalized_coil_to_coil_min_distances(self) -> Float[NpOrJaxArray, " n_coils"]:
        """Minimum coil-to-coil distances normalized by minor radius."""
        return self.coil_to_coil_min_distances / self.minor_radius

    coil_to_plasma_min_distances: Float[NpOrJaxArray, " n_coils"]
    """Minimum distance from each coil to plasma surface (m)."""

    @pydantic.computed_field
    @property
    def normalized_coil_to_plasma_min_distances(
        self,
    ) -> Float[NpOrJaxArray, " n_coils"]:
        """Minimum coil-to-plasma distances normalized by minor radius."""
        return self.coil_to_plasma_min_distances / self.minor_radius

    coil_lengths: Float[NpOrJaxArray, " n_coil"]
    """Arc length of each unique coil (m)."""

    @pydantic.computed_field
    @property
    def normalized_coil_lengths(self) -> Float[NpOrJaxArray, " n_coil"]:
        """Coil lengths normalized by minor radius."""
        return self.coil_lengths / self.minor_radius

    coil_curvatures: Annotated[Float[NpOrJaxArray, "n_coil n_coil_eval_points"], ...]
    """Curvature along each unique coil at each evaluation point (1/m)."""

    @pydantic.computed_field
    @property
    def normalized_coil_curvatures(
        self,
    ) -> Float[NpOrJaxArray, "n_coil n_coil_eval_points"]:
        """Curvatures normalized by minor radius."""
        return self.coil_curvatures * self.minor_radius

    coil_current_lengths: Float[NpOrJaxArray, " n_coil"]
    """Current times length for each unique coil (A*m)."""

    coil_integrated_curvatures: Float[NpOrJaxArray, " n_coil"]
    """Integrated curvature for each unique coil; equals 2*pi for convex."""

    coil_torsions: Annotated[Float[NpOrJaxArray, "n_coil n_coil_eval_points"], ...]
    """Torsion along each unique coil at each evaluation point."""

    @pydantic.computed_field
    @property
    def normalized_coil_torsions(
        self,
    ) -> Float[NpOrJaxArray, "n_coil n_coil_eval_points"]:
        """Torsions normalized by minor radius."""
        return self.coil_torsions * self.minor_radius

    coil_arclength_variances: Float[NpOrJaxArray, " n_coil"]
    """Arclength variance for each unique coil."""

    quadratic_flux: Annotated[Float[NpOrJaxArray, "n_theta n_zeta"], ...]
    """Quadratic flux on the plasma surface (T^2*m^2)."""

    local_quadratic_flux: Annotated[Float[NpOrJaxArray, "n_theta n_zeta"], ...]
    """Local quadratic flux on the plasma surface (T^2*m^2)."""

    toroidal_flux: ScalarFloat
    """Toroidal flux through the plasma surface (Wb)."""

    linking_current_consistency: ScalarFloat
    """Linking-current consistency error between coils and equilibrium (A)."""

    minor_radius: ScalarFloat
    """Minor radius of the equilibrium (m)."""


class ConStellarationUpdatePipelineInput(pydantic.BaseModel):
    """A single work unit for the Dataproc coilset generation pipeline.

    Stores dapper IDs referencing the equilibrium, sampled requirements, and shared
    settings needed to generate one coilset.  Each instance maps to exactly one coilset
    solve on the Dataproc cluster.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    equilibrium_id: str
    """Dapper storage ID of the DescEquilibrium to use."""

    requirements_bounds_id: str
    """Dapper storage ID of the ConStellarationUpdateRequirementsBounds."""

    regcoil_settings_id: str
    """Dapper storage ID of the ConStellarationUpdateRegcoilSettings."""

    desc_optimizer_settings_id: str
    """Dapper storage ID of the ConStellarationUpdateDescOptimizerSettings."""

    sampling_settings_id: str
    """Dapper storage ID of the ConStellarationUpdateSamplingSettings."""

    optimize_coilset_with_desc: bool = True
    """If True, run DESC augmented-Lagrangian optimization after REGCOIL."""

    debug: bool = False
    """If True, execute the debug pipeline."""


class ConStellarationUpdatePoincarePlotSettings(pydantic.BaseModel):
    """Settings for Poincaré plot generation via DESC field line tracing.

    DESC traces field lines directly from a coilset without an intermediate
    magnetic field grid (Makegrid).  Starting points are computed from the
    equilibrium flux surfaces at the given normalised-flux coordinates.

    Upstream equivalences
    ---------------------
    * ``MakegridSettings.coil_discretization_settings.n_points_tangential``
      → ``source_grid_points``
    * ``FieldLinesSettings.n_field_lines`` → ``n_field_lines``
    * ``FieldLinesSettings.n_toroidal_transits`` → ``n_toroidal_transits``
    * ``FieldLinesSettings.start_field_lines_offset`` /
      ``end_field_lines_offset`` → ``rho_min`` / ``rho_max``
    * ``FieldLinesSettings.integration_tolerance`` → ``rtol`` / ``atol``
    * ``PoincarePlotSettings.toroidal_angles`` → ``toroidal_angles``
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    n_field_lines: int = 7
    """Number of field lines to trace, spaced along the midplane."""

    n_toroidal_transits: int = 100
    """Number of toroidal transits per field line."""

    toroidal_angles: int = 3
    """Number of toroidal cross-sections to plot.

    Equivalent to ``PoincarePlotSettings.toroidal_angles`` when given as an
    integer (DESC interprets this as the number of evenly-spaced sections
    within one field period).
    """

    rho_min: float = 0.2
    """Minimum normalised flux coordinate for field line starting points."""

    rho_max: float = 0.7
    """Maximum normalised flux coordinate for field line starting points."""

    source_grid_points: int = 300
    """Number of equally-spaced quadrature points per coil for the Biot-Savart
    evaluation.

    Passed as ``source_grid`` (int) to DESC's ``poincare_plot`` / ``CoilSet``.
    The Makegrid equivalent is
    ``MakegridSettings.coil_discretization_settings.n_points_tangential``
    (default 300).  DESC's own default (``2 * fourier_order + 5`` nodes) is
    much lower and may under-resolve high-order coils.
    """

    rtol: float = 1e-8
    """Relative tolerance for the field-line ODE integration.

    Passed through to ``desc.magnetic_fields.field_line_integrate``.
    The FIELDLINES pipeline equivalent is ``FieldLinesSettings.integration_tolerance``.
    """

    atol: float = 1e-8
    """Absolute tolerance for the field-line ODE integration.

    Passed through to ``desc.magnetic_fields.field_line_integrate``.
    """

    max_steps: int | None = None
    """Maximum number of ODE solver steps for field-line integration.

    ``None`` uses the DESC default of ``abs((phis[-1] - phis[0]) * 1000)``.
    Increase if field lines are being truncated; decrease to limit runtime on
    divergent configurations.
    """
