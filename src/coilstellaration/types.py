import base64
import time
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import beartype
import numpy as np
import pydantic
from constellaration.geometry import surface_rz_fourier
from constellaration.utils import pytree
from jaxtyping import Array, Float, Int, jaxtyped
from vmecpp import _pydantic_numpy as pydantic_numpy

NpOrJaxArray = Array | np.ndarray
ScalarFloat = Float[NpOrJaxArray, " "] | float
ScalarInt = Int[NpOrJaxArray, " "] | int
ScalarT = str | ScalarInt | ScalarFloat
RangeT = tuple[ScalarT, ScalarT]
ChoicesT = frozenset[ScalarT]
Binary = object()


def runtime_check_array_sizes(f):
    return jaxtyped(typechecker=beartype.beartype)(f)


class BaseModel(pydantic_numpy.BaseModelWithNumpy):
    model_config = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
    )


class Blob(pydantic.BaseModel):
    """Opaque bytes payload with an optional file-suffix hint."""

    content: bytes
    """Raw binary content. Prefer constructing via `Blob.from_bytes(...)` and
    reading via `as_bytes()` to keep call sites stable if the storage
    representation evolves."""

    file_suffix: str | None = None
    """Optional file suffix (with leading dot, e.g. `.tar.gz`) used when
    materializing the blob to disk. Carries no semantics for in-memory use."""

    @pydantic.computed_field
    @property
    def content_length(self) -> int:
        """Byte size of `content`, surfaced in JSON for self-description."""
        return len(self.content)

    @classmethod
    def from_bytes(cls, content: bytes, file_suffix: str | None = None) -> Self:
        if file_suffix is not None and not file_suffix.startswith("."):
            raise ValueError("file_suffix must start with a dot")
        return cls(content=content, file_suffix=file_suffix)

    @classmethod
    def from_path(cls, path: str | Path) -> Self:
        p = Path(path)
        return cls(content=p.read_bytes(), file_suffix=p.suffix or None)

    def as_bytes(self) -> bytes:
        return self.content

    def write_to(self, path: str | Path) -> Path:
        out = Path(path)
        out.write_bytes(self.content)
        return out

    @pydantic.field_serializer("content", when_used="json")
    def _serialize_content(self, value: bytes) -> str:
        return base64.b64encode(value).decode("ascii")

    @pydantic.field_validator("content", mode="before")
    @classmethod
    def _validate_content(cls, value: Any) -> bytes:
        if isinstance(value, bytes):
            return value
        if isinstance(value, str):
            return base64.b64decode(value, validate=True)
        raise TypeError(
            f"Expected bytes or base64-encoded str, got {type(value).__name__}"
        )

    def __repr__(self) -> str:
        return f"Blob({len(self.content)} bytes, file_suffix={self.file_suffix!r})"


class RequirementsTemplate[Numeric, Categorical](
    BaseModel,
):
    """Template for sampled or bounds-valued requirements.

    Parametrize with ``[ScalarT, ScalarT]`` for sampled instances or
    ``[RangeT, ChoicesT]`` for bounds instances.
    """

    normalized_min_coil_to_coil_distance: Numeric

    normalized_min_coil_plasma_distance: Numeric

    regcoil_winding_surface_plasma_distance: Numeric
    """The normalized distance from the plasma surface at which the REGCOIL winding
    surface is constructed."""

    normalized_max_coil_curvature: Numeric

    n_coils_per_half_period: Numeric
    """Number of unique coils per half field period."""

    coil_fourier_order: Numeric

    desc_objective_lambda_log10: Numeric
    """Base-10 logarithm of the DESC objective tradeoff weight."""

    desc_x_scale_mode: Categorical
    """The DESC x scaling mode.

    If ``"ess"`` (exponential spectral scaling), it must encode what order to use.
    """

    regcoil_target_option: Categorical
    """High-level REGCOIL target used to choose the lambda selection rule.

    The allowed choices are ``"normalized_coil_to_coil_distance"`` and
    ``"normalized_field_error"``.
    """

    regcoil_maximum_normalized_field_error: Numeric | None = None
    """Maximum normalized field error target for REGCOIL.

    Warning:
        This value is only used when the REGCOIL target option is
        ``"normalized_field_error"``.
    """


class Requirements(RequirementsTemplate[ScalarT, ScalarT]):
    pass


class RequirementsBounds(RequirementsTemplate[RangeT, ChoicesT]):
    pass


class ConfigSamplingSettings(
    BaseModel,
):
    seed: str


class RegcoilSettings(
    BaseModel,
):
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


class DescOptimizerSettings(
    BaseModel,
):
    """Settings for the DESC augmented Lagrangian coilset optimizer."""

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


@pytree.register_pydantic_data
class Coilset(
    BaseModel,
):
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


@pytree.register_pydantic_data
class RequirementMetrics(BaseModel):
    """Achieved metrics that condition `CoilPredictor`.

    The first four mirror requirement-side counterparts in
    `CoilStellarationRequirements`; `on_axis_average_magnetic_field`
    additionally serves as a magnitude prior for coil currents.
    """

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


class DescSolveResults(
    BaseModel,
):
    """Convergence diagnostics from a DESC coilset optimization.

    Fields are the common subset available from both scipy-slsqp and fmin-auglag-bfgs
    optimizers.
    """

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


class DescOutput(
    BaseModel,
):
    """Combined output of a DESC coilset optimization run."""

    coilset: Coilset
    """The optimized coilset in Fourier representation."""

    desc_solve_results: DescSolveResults
    """Convergence diagnostics from the DESC optimization."""


@pytree.register_pydantic_data
class Metrics(
    BaseModel,
):
    surf_eval_coords: Annotated[Float[NpOrJaxArray, "2 n_theta n_zeta"], Binary]
    """Meshgrid of (theta, zeta) at which surface-field quantities are evaluated."""

    coil_eval_params: Annotated[Float[NpOrJaxArray, " n_coil_eval_points"], Binary]
    """Toroidal coordinates at which coil-wise quantities are evaluated."""

    coil_currents: Float[NpOrJaxArray, " n_unique_coils"]

    n_coils_per_half_period: ScalarInt | None = None
    """Number of unique coils per half field period."""

    on_axis_average_magnetic_field: ScalarFloat | None = None
    """The on-axis average magnetic field."""

    normalized_field_error: Annotated[Float[NpOrJaxArray, "n_theta n_zeta"], Binary]
    """|Bn| / |B| on the plasma surface."""

    coil_linking_numbers: Float[NpOrJaxArray, " n_coils"] | None = None
    """Per-coil linking number (sum of abs linking numbers with other coils)."""

    coil_to_coil_min_distances: Float[NpOrJaxArray, " n_coils"]
    """Minimum distance from each coil to any other coil (m)."""

    @pydantic.computed_field
    @property
    def normalized_coil_to_coil_min_distances(
        self,
    ) -> Float[NpOrJaxArray, " n_coils"]:
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

    coil_curvatures: Annotated[Float[NpOrJaxArray, "n_coil n_coil_eval_points"], Binary]
    """Curvature along each unique coil at each evaluation point (1/m)."""

    @pydantic.computed_field
    @property
    def normalized_coil_curvatures(
        self,
    ) -> Float[NpOrJaxArray, "n_coil n_coil_eval_points"]:
        """Curvatures normalized by minor radius."""
        return self.coil_curvatures * self.minor_radius

    coil_current_lengths: Float[NpOrJaxArray, " n_coil"] | None = None
    """Current times length for each unique coil (A*m)."""

    coil_integrated_curvatures: Float[NpOrJaxArray, " n_coil"] | None = None
    """Integrated curvature for each unique coil; equals 2*pi for convex."""

    coil_torsions: Annotated[
        Float[NpOrJaxArray, "n_coil n_coil_eval_points"] | None, Binary
    ] = None
    """Torsion along each unique coil at each evaluation point."""

    @pydantic.computed_field
    @property
    def normalized_coil_torsions(
        self,
    ) -> Float[NpOrJaxArray, "n_coil n_coil_eval_points"] | None:
        """Torsions normalized by minor radius."""
        return (
            self.coil_torsions * self.minor_radius
            if self.coil_torsions is not None
            else None
        )

    coil_arclength_variances: Float[NpOrJaxArray, " n_coil"] | None = None
    """Arclength variance for each unique coil."""

    quadratic_flux: Annotated[Float[NpOrJaxArray, "n_theta n_zeta"], Binary]
    """Quadratic flux on the plasma surface (T^2*m^2)."""

    local_quadratic_flux: Annotated[Float[NpOrJaxArray, "n_theta n_zeta"], Binary]
    """Local quadratic flux on the plasma surface (T^2*m^2)."""

    toroidal_flux: ScalarFloat | None = None
    """Toroidal flux through the plasma surface (Wb)."""

    linking_current_consistency: ScalarFloat | None = None
    """Linking-current consistency error between coils and equilibrium (A)."""

    minor_radius: ScalarFloat
    """Minor radius of the equilibrium (m)."""


class PipelineInput(
    BaseModel,
):
    """A single work unit for the Dataproc coilset generation pipeline.

    Stores dapper IDs referencing the equilibrium, sampled requirements, and shared
    settings needed to generate one coilset.  Each instance maps to exactly one coilset
    solve on the Dataproc cluster.
    """

    equilibrium_id: str
    """Dapper storage ID of the DescEquilibrium to use."""

    requirements_bounds_id: str
    """Dapper storage ID of the CoilStellarationRequirementsBounds."""

    regcoil_settings_id: str
    """Dapper storage ID of the CoilStellarationRegcoilSettings."""

    desc_optimizer_settings_id: str
    """Dapper storage ID of the CoilStellarationDescOptimizerSettings."""

    sampling_settings_id: str
    """Dapper storage ID of the CoilStellarationSamplingSettings."""

    optimize_coilset_with_desc: bool = True
    """If True, run DESC augmented-Lagrangian optimization after REGCOIL."""

    debug: bool = False
    """If True, execute the debug pipeline."""


class PoincarePlotSettings(
    BaseModel,
):
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
    * ``PoincarePlotSettings.toroidal_angles`` → ``normalized_toroidal_angles``
    """

    n_field_lines: int = 7
    """Number of field lines to trace, spaced along the midplane."""

    n_toroidal_transits: int = 100
    """Number of toroidal transits per field line."""

    normalized_toroidal_angles: int | list[float] = 3
    """Toroidal cross-sections to plot, normalized to one field period.

    If an int, the number of evenly-spaced sections within one field period. If a list
    of floats, explicit angles normalized to [0, 1] where 0 corresponds to phi=0 and 1
    corresponds to phi=2*pi/NFP.
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

    @classmethod
    def default_fast(cls) -> Self:
        """Reduced settings for fast visual-quality poincaré plots."""
        return cls(
            n_field_lines=7,
            n_toroidal_transits=60,
            normalized_toroidal_angles=[1.0],
            source_grid_points=128,
            rtol=1e-6,
            atol=1e-6,
            max_steps=10000,
        )


class FlaxNnxCheckpoint[ConfigT](BaseModel):
    """Serialized state of a Flax NNX module.

    Parameterize with a concrete pydantic config type and bind the alias at
    module level (e.g. `MyCheckpoint = FlaxNnxCheckpoint[MyConfig]`) so the
    consumer sees a fully resolved class. The associated module class must
    accept the config as a single positional argument alongside keyword-only
    `rngs`; `from_checkpoint` calls `module_cls(config, rngs=...)` to rebuild
    the abstract state.
    """

    archive: Blob
    """tar.gz archive of the directory written by `ocp.StandardCheckpointer`."""

    config: ConfigT
    """Typed init config used to reconstruct the module's abstract template."""


class CoilPredictorConfig(BaseModel):
    """Init config for `CoilPredictor`, stored alongside its checkpoint."""

    is_stellarator_symmetric: bool
    n_field_periods: int
    n_poloidal_modes: int
    n_toroidal_modes: int
    n_unique_coils: int
    n_modes_coils_max: int
    """Global max fourier order over the training dataset.

    Inferred via `eval_data_utils.infer_max_fourier_order(train_df)` and frozen
    into the resolved config. The MLP readout dimensions and the spectral-scaling
    helper signatures all use this value.
    """
    coil_currents_mean: float
    coil_currents_std: float

    hidden_dim: int = 512
    num_layers: int = 4
    requirement_metrics_means: list[float]
    """Per-feature train-set means used to standardize `RequirementMetrics` inputs.

    Populated from the train split when the resolved config is built. Length is always
    4, matching `RequirementMetrics.to_array()`.
    """
    requirement_metrics_stds: list[float]
    """Per-feature train-set stds (clamped at `1e-8`) used for standardization."""
    spectral_scaling_alpha: float = 1.2
    spectral_scaling_min: float = 1e-7
    spectral_scaling_order: float = float("inf")


CoilPredictorCheckpoint = FlaxNnxCheckpoint[CoilPredictorConfig]


class ResMlpCoilPredictorConfig(BaseModel):
    """Init config for `ResMlpCoilPredictor`, stored alongside its checkpoint.

    Sibling of `CoilPredictorConfig` for the residual-block MLP variant.
    Dataset/scaling fields are duplicated because `BaseModel` does not
    support subclassing.
    """

    is_stellarator_symmetric: bool
    n_field_periods: int
    n_poloidal_modes: int
    n_toroidal_modes: int
    n_unique_coils: int
    n_modes_coils_max: int
    """Global max fourier order over the training dataset.

    Inferred via `eval_data_utils.infer_max_fourier_order(train_df)` and frozen
    into the resolved config. The MLP readout dimensions and the spectral-scaling
    helper signatures all use this value.
    """
    coil_currents_mean: float
    coil_currents_std: float

    hidden_dim: int = 512
    num_layers: int = 4
    dropout: float = 0.2
    """Dropout rate applied inside each residual block.

    0.0 disables dropout.
    """
    requirement_metrics_means: list[float]
    """Per-feature train-set means used to standardize `RequirementMetrics` inputs."""
    requirement_metrics_stds: list[float]
    """Per-feature train-set stds (clamped at `1e-8`) used for standardization."""
    spectral_scaling_alpha: float = 1.2
    spectral_scaling_min: float = 1e-7
    spectral_scaling_order: float = float("inf")


ResMlpCoilPredictorCheckpoint = FlaxNnxCheckpoint[ResMlpCoilPredictorConfig]


class MlpEnsembleCoilPredictorConfig(BaseModel):
    """Init config for `MlpEnsembleCoilPredictor`, stored alongside its checkpoint.

    Sibling of `CoilPredictorConfig`. Dataset/scaling fields are duplicated
    because `BaseModel` does not support subclassing.
    """

    is_stellarator_symmetric: bool
    n_field_periods: int
    n_poloidal_modes: int
    n_toroidal_modes: int
    n_unique_coils: int
    n_modes_coils_max: int
    """Global max fourier order over the training dataset.

    Inferred via `eval_data_utils.infer_max_fourier_order(train_df)` and frozen
    into the resolved config. The MLP readout dimensions and the spectral-scaling
    helper signatures all use this value.
    """
    coil_currents_mean: float
    coil_currents_std: float

    hidden_dim: int = 512
    num_layers: int = 4
    n_ensemble_members: int = 5
    """Number of independent MLPs whose predictions are averaged at inference."""
    requirement_metrics_means: list[float]
    """Per-feature train-set means used to standardize `RequirementMetrics` inputs."""
    requirement_metrics_stds: list[float]
    """Per-feature train-set stds (clamped at `1e-8`) used for standardization."""
    spectral_scaling_alpha: float = 1.2
    spectral_scaling_min: float = 1e-7
    spectral_scaling_order: float = float("inf")


MlpEnsembleCoilPredictorCheckpoint = FlaxNnxCheckpoint[MlpEnsembleCoilPredictorConfig]


class ResMlpEnsembleCoilPredictorConfig(BaseModel):
    """Init config for `ResMlpEnsembleCoilPredictor`, stored alongside its checkpoint.

    Sibling of `ResMlpCoilPredictorConfig` for ensembles of residual-block MLPs.
    """

    is_stellarator_symmetric: bool
    n_field_periods: int
    n_poloidal_modes: int
    n_toroidal_modes: int
    n_unique_coils: int
    n_modes_coils_max: int
    """Global max fourier order over the training dataset.

    Inferred via `eval_data_utils.infer_max_fourier_order(train_df)` and frozen
    into the resolved config. The MLP readout dimensions and the spectral-scaling
    helper signatures all use this value.
    """
    coil_currents_mean: float
    coil_currents_std: float

    hidden_dim: int = 512
    num_layers: int = 4
    dropout: float = 0.2
    """Dropout rate applied inside each residual block.

    0.0 disables dropout.
    """
    n_ensemble_members: int = 5
    """Number of independent MLPs whose predictions are averaged at inference."""
    requirement_metrics_means: list[float]
    """Per-feature train-set means used to standardize `RequirementMetrics` inputs."""
    requirement_metrics_stds: list[float]
    """Per-feature train-set stds (clamped at `1e-8`) used for standardization."""
    spectral_scaling_alpha: float = 1.2
    spectral_scaling_min: float = 1e-7
    spectral_scaling_order: float = float("inf")


ResMlpEnsembleCoilPredictorCheckpoint = FlaxNnxCheckpoint[
    ResMlpEnsembleCoilPredictorConfig
]


@pytree.register_pydantic_data
class Batch(BaseModel):
    boundaries: surface_rz_fourier.SurfaceRZFourier
    """Batched surface; leaves have leading dim ``batch_size``."""
    coilsets: Coilset
    """Batched coilset (padded to ``2*batch_max_fourier_order + 1`` modes); leaves have
    leading dim ``batch_size``."""
    requirement_metrics: RequirementMetrics
    """Batched conditioning; each scalar field has leading dim ``batch_size``."""
    fourier_order_mask: Float[NpOrJaxArray, "batch_size n_modes"]
    """1.0 at real coefficient indices, 0.0 at zero-padded indices.

    `n_modes = 2 * batch_max_fourier_order + 1`. Created at collate time so
    that loss can mask per-sample padded entries.
    """
    batch_max_fourier_order: int
    """Per-batch fourier order N.

    Every coilset in the batch is padded to `(2*N + 1)` modes.
    """


AnyModelLiteral = Literal["mlp", "res_mlp", "mlp_ensemble", "res_mlp_ensemble"]
AnyModelConfig = (
    CoilPredictorConfig
    | ResMlpCoilPredictorConfig
    | MlpEnsembleCoilPredictorConfig
    | ResMlpEnsembleCoilPredictorConfig
)
AnyModelCheckpoint = (
    CoilPredictorCheckpoint
    | ResMlpCoilPredictorCheckpoint
    | MlpEnsembleCoilPredictorCheckpoint
    | ResMlpEnsembleCoilPredictorCheckpoint
)


class TrainConfig(BaseModel):
    """Everything that controls a training run."""

    seed: int = pydantic.Field(default_factory=lambda: int(time.time_ns() % (2**32)))
    learning_rate: float = 1e-3
    end_lr: float = 1e-5
    warmup_steps: int = 50
    steps: int = 1_000
    max_wall_time_s: int = 30 * 60
    batch_size: int = 512
    eval_batch_size: int = 1024
    eval_every: int = 50
    early_stopping_patience: int | None = None
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    model_type: AnyModelLiteral = "mlp"
    model_config_overrides: AnyModelConfig | None = None

    @pydantic.model_validator(mode="after")
    def _check_overrides_match_model_type(self) -> Self:
        if self.model_config_overrides is None:
            return self
        expected_config_types: dict[str, type] = {
            "mlp": CoilPredictorConfig,
            "res_mlp": ResMlpCoilPredictorConfig,
            "mlp_ensemble": MlpEnsembleCoilPredictorConfig,
            "res_mlp_ensemble": ResMlpEnsembleCoilPredictorConfig,
        }
        expected = expected_config_types[self.model_type]
        if not isinstance(self.model_config_overrides, expected):
            raise ValueError(
                f"model_type={self.model_type!r} requires {expected.__name__} "
                f"overrides; got {type(self.model_config_overrides).__name__}."
            )
        return self


class EvalSettings(BaseModel):
    """Inputs that fully determine an evaluation run."""

    relative_min_coil_to_plasma_distance_error_threshold: float = -0.4
    n_eval: int = 128


class EvalData(BaseModel):
    boundary: surface_rz_fourier.SurfaceRZFourier
    boundary_id: str
    true_coilset: Coilset
    true_metrics: str | Metrics
    requirement_metrics: RequirementMetrics
    predicted_coilset: Coilset | None = None
