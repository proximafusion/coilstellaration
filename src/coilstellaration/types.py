import base64
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Any, Final, Literal, Self

import beartype
import jax
import jaxtyping as jt
import numpy as np
import pydantic
from coilstellaration import pydantic_numpy
from constellaration.geometry import surface_rz_fourier
from jaxtyping import Array, Float, Int, jaxtyped

NpOrJaxArray = Array | np.ndarray
ScalarFloat = Float[NpOrJaxArray, " "] | float
ScalarInt = Int[NpOrJaxArray, " "] | int
ScalarT = str | ScalarInt | ScalarFloat
RangeT = tuple[ScalarT, ScalarT]
ChoicesT = frozenset[ScalarT]
Binary = object()

DAPPER_TYPE_FIELD: Final = "dapper_type"
"""Type tag that `constellaration` models carry when `dapper` is installed."""


def runtime_check_array_sizes(f):
    return jaxtyped(typechecker=beartype.beartype)(f)


def _jax_to_python(value: Any) -> Any:
    if isinstance(value, jt.Array):
        np_value = np.asarray(value)
        return np_value.item() if np_value.ndim == 0 else np_value.tolist()
    if isinstance(value, dict):
        return {k: _jax_to_python(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return type(value)(_jax_to_python(v) for v in value)
    return value


class BaseModel(pydantic_numpy.BaseModelWithNumpy):
    model_config = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
    )

    @pydantic.field_serializer("*", mode="wrap", when_used="json")
    def _serialize_field(
        self,
        value: Any,
        default_handler: pydantic.SerializerFunctionWrapHandler,
        info: pydantic.FieldSerializationInfo,
    ):
        value = pydantic_numpy.serialize_special_field(
            type(self), info.field_name, value
        )
        value = _jax_to_python(value)
        return default_handler(value)

    @pydantic.model_validator(mode="wrap")
    @classmethod
    def _validate_model(
        cls,
        value: Any,
        default_handler: pydantic.ValidatorFunctionWrapHandler,
    ) -> Any:
        """Checks array axis names in a scope of their own, so models nested in one
        parent, like a true and a predicted `Coilset` of different Fourier orders, don't
        have to agree on them."""
        # `jt.jaxtyped` only decorates dataclasses and functions; "context" opens the
        # scope as a context manager instead.
        with jt.jaxtyped("context"):
            return default_handler(value)


def pydantic_flatten[T: pydantic.BaseModel](
    something: T,
    meta_fields: list[str] | None = None,
) -> tuple[
    tuple,
    tuple[type[T], tuple[str, ...], tuple[str, ...], tuple[Any, ...]],
]:
    """A jax pytree compatible implementation of flattening pydantic objects.

    A general pydantic.BaseModel is flattened into a tuple of children and aux_data.
    aux_data is used to reconstruct the same type of Pydantic in unflatten. meta_fields
    are used to specify fields that should not be visible to pytree operations. See
    https://jax.readthedocs.io/en/latest/_autosummary/jax.tree_util.register_dataclass.html
    for details.
    """

    meta_fields = list(meta_fields or [])
    # A string type tag is not array data; as a child it would break tree_map/stack.
    if (
        DAPPER_TYPE_FIELD in type(something).model_fields
        and DAPPER_TYPE_FIELD not in meta_fields
    ):
        meta_fields.append(DAPPER_TYPE_FIELD)

    data_fields = [
        field for field in type(something).model_fields if field not in meta_fields
    ]

    aux_data = (
        type(something),
        tuple(data_fields),
        tuple(meta_fields),
        tuple(getattr(something, field) for field in meta_fields),
    )
    children = tuple(
        (jax.tree_util.GetAttrKey(field), getattr(something, field))
        for field in data_fields
    )
    return children, aux_data


def pydantic_unflatten[T: pydantic.BaseModel](
    aux_data: tuple[type[T], tuple[str, ...], tuple[str, ...], tuple[Any, ...]],
    children: Iterable[Any],
) -> T:
    """A jax pytree compatible implementation of un-flattening Pydantic objects.

    pydantic_unflatten is the inverse of pydantic_flatten. Using the type annotation
    and children-meta_fields split in aux_data, it constructs a new Pydantic object.

    Note that this object will typically not validate against the Pydantic
    specification. Pytrees are used to manipulate the data stored in leafs and thus can
    contain anything. Pytrees only make statements about the structure of the data, not
    the content. An example of data manipulation might be to filter data in a pytree by
    type:

    ```
    my_data = MyModel(...)
    strings = jax.tree_map(lambda x: x if isinstance(x, str) else None, my_data)
    ```

    See
    https://jax.readthedocs.io/en/latest/pytrees.html#custom-pytrees-and-initialization
    for details.
    """
    cls, data_fields, meta_fields, metas = aux_data
    kwargs = {
        **dict(zip(data_fields, children)),
        **dict(zip(meta_fields, metas)),
    }
    return cls.model_construct(**kwargs)


def register_pydantic_data[T: pydantic.BaseModel](
    cls: type[T], meta_fields: list[str] | None = None
) -> type[T]:
    """Register a pydantic.BaseModel class for jax pytree compatibility.

    Args:
        cls: The pydantic.BaseModel class to register.
        meta_fields: Fields that should be part of aux_data rather becoming children.
            Defaults to None.

    Returns:
        The registered class, to enable the function to be used as a decorator.
    """
    jax.tree_util.register_pytree_with_keys(
        cls,
        lambda x: pydantic_flatten(x, meta_fields),
        pydantic_unflatten,
    )

    return cls


register_pydantic_data(surface_rz_fourier.SurfaceRZFourier)


class Blob(pydantic.BaseModel):
    """Opaque bytes payload with an optional file-suffix hint."""

    content: bytes
    """Raw binary content.

    Prefer constructing via `Blob.from_bytes(...)` and
    reading via `as_bytes()` to keep call sites stable if the storage
    representation evolves.
    """

    file_suffix: str | None = None
    """Optional file suffix (with leading dot, e.g. `.tar.gz`) used when materializing
    the blob to disk.

    Carries no semantics for in-memory use.
    """

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


@register_pydantic_data
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


@register_pydantic_data
class RequirementMetrics(BaseModel):
    """Metrics achieved by a coilset; the coil predictors are conditioned on them.

    The four continuous metrics are the achieved counterparts of the distance,
    curvature and field-error targets in `Requirements`. `n_coils_per_half_period` is
    the target number of unique coils.
    """

    min_normalized_coil_to_coil_distance: ScalarFloat
    min_normalized_coil_to_plasma_distance: ScalarFloat
    max_normalized_coil_curvature: ScalarFloat
    """Largest curvature magnitude over all coils, convex or concave, normalized by the
    minor radius."""
    max_normalized_field_error: ScalarFloat
    n_coils_per_half_period: ScalarInt

    def to_array(self) -> Float[NpOrJaxArray, " 4"]:  # noqa: ARR002
        """Return the four continuous metrics in declaration order.

        The coil count is excluded.
        """
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


class DescSplineProfile(BaseModel):
    """A DESC `SplineProfile`."""

    method: str = "cubic2"
    """Spline interpolation method (e.g. `linear`, `cubic`, `cubic2`)."""

    name: str = ""
    """Name of the profile (e.g. `pressure`, `current`, `iota`)."""

    knots: Float[NpOrJaxArray, " n_knots"]

    params: Float[NpOrJaxArray, " n_knots"]
    """Profile values at the knots."""


class DescEquilibrium(BaseModel):
    """A solved DESC `Equilibrium`.

    The dataset's DESC coilsets and their metrics were computed on these equilibria, so
    scoring a coilset on the same equilibrium reproduces the dataset's metrics.
    """

    r_lmn: Float[NpOrJaxArray, " n_modes_r"]
    """Fourier-Zernike coefficients of R."""

    z_lmn: Float[NpOrJaxArray, " n_modes_z"]
    """Fourier-Zernike coefficients of Z."""

    lambda_lmn: Float[NpOrJaxArray, " n_modes_lambda"]
    """Fourier-Zernike coefficients of the stream function lambda."""

    radial_resolution: int
    poloidal_resolution: int
    toroidal_resolution: int
    spectral_indexing: str = "ansi"
    n_field_periods: int
    is_stellarator_symmetric: bool = True

    total_toroidal_flux: float = 1.0
    """Total toroidal flux within the last closed flux surface (Wb)."""

    pressure_profile: DescSplineProfile
    current_profile: DescSplineProfile | None = None
    iota_profile: DescSplineProfile | None = None

    r_axis: Float[NpOrJaxArray, " n_modes_r_axis"] | None = None
    z_axis: Float[NpOrJaxArray, " n_modes_z_axis"] | None = None
    r_boundary_lmn: Float[NpOrJaxArray, " n_boundary_modes_r"] | None = None
    z_boundary_lmn: Float[NpOrJaxArray, " n_boundary_modes_z"] | None = None


@register_pydantic_data
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
    def normalized_coil_to_coil_min_distances(  # noqa: ARR002
        self,
    ) -> Float[NpOrJaxArray, " n_coils"]:
        """Minimum coil-to-coil distances normalized by minor radius."""
        return self.coil_to_coil_min_distances / self.minor_radius

    coil_to_plasma_min_distances: Float[NpOrJaxArray, " n_coils"]
    """Minimum distance from each coil to plasma surface (m)."""

    @pydantic.computed_field
    @property
    def normalized_coil_to_plasma_min_distances(  # noqa: ARR002
        self,
    ) -> Float[NpOrJaxArray, " n_coils"]:
        """Minimum coil-to-plasma distances normalized by minor radius."""
        return self.coil_to_plasma_min_distances / self.minor_radius

    coil_lengths: Float[NpOrJaxArray, " n_coil"]
    """Arc length of each unique coil (m)."""

    @pydantic.computed_field
    @property
    def normalized_coil_lengths(self) -> Float[NpOrJaxArray, " n_coil"]:  # noqa: ARR002
        """Coil lengths normalized by minor radius."""
        return self.coil_lengths / self.minor_radius

    coil_curvatures: Annotated[Float[NpOrJaxArray, "n_coil n_coil_eval_points"], Binary]
    """Signed curvature along each unique coil at each evaluation point (1/m).

    Following DESC, it is negative where the coil is concave relative to its centre.
    """

    @pydantic.computed_field
    @property
    def normalized_coil_curvatures(  # noqa: ARR002
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
    def normalized_coil_torsions(  # noqa: ARR002
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


class PoincarePlotSettings(
    BaseModel,
):
    n_field_lines: int = 7
    n_toroidal_transits: int = 100
    normalized_toroidal_angles: int | list[float] = 3
    rho_min: float = 0.2
    rho_max: float = 0.7
    source_grid_points: int = 300
    rtol: float = 1e-8
    atol: float = 1e-8
    max_steps: int | None = None

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


class _FixedShapeCoilPredictorConfig(BaseModel):
    """Fields shared by the fixed-shape coil predictors.

    The number of field periods and unique coils are architecture constants, so a model
    only supports the single value of each it was trained on.
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
    """Mean used to standardize the predicted currents, in units of the major radius.

    `train` sets it to the median over the training coils.
    """
    coil_currents_std: float
    """Std used to standardize the predicted currents and to scale the currents loss.

    `train` sets it to the normal-consistent interquartile range over the training
    coils, in units of the major radius.
    """

    hidden_dim: int = 512
    num_layers: int = 4
    requirement_metrics_means: list[float]
    """Means used to standardize the four `RequirementMetrics` inputs.

    Ordered like `model_definition.requirement_metrics_vector`. `train` sets them to the
    medians over the training rows.
    """
    requirement_metrics_stds: list[float]
    """Stds used to standardize the four `RequirementMetrics` inputs.

    Ordered like `requirement_metrics_means`. `train` sets them to the
    normal-consistent interquartile ranges over the training rows, floored at `1e-8`.
    """
    spectral_scaling_alpha: float = 1.2
    spectral_scaling_min: float = 1e-7
    spectral_scaling_order: float = float("inf")


class CoilPredictorConfig(_FixedShapeCoilPredictorConfig):
    """Init config for `CoilPredictor`, stored alongside its checkpoint."""

    model_kind: Literal["mlp"] = "mlp"
    """Tag that disambiguates this config in `AnyModelConfig` unions."""


CoilPredictorCheckpoint = FlaxNnxCheckpoint[CoilPredictorConfig]


class ResMlpCoilPredictorConfig(_FixedShapeCoilPredictorConfig):
    """Init config for `ResMlpCoilPredictor`, stored alongside its checkpoint."""

    model_kind: Literal["res_mlp"] = "res_mlp"
    """Tag that disambiguates this config in `AnyModelConfig` unions."""
    dropout: float = 0.2
    """Dropout rate applied inside each residual block.

    0.0 disables dropout.
    """


ResMlpCoilPredictorCheckpoint = FlaxNnxCheckpoint[ResMlpCoilPredictorConfig]


class MlpEnsembleCoilPredictorConfig(_FixedShapeCoilPredictorConfig):
    """Init config for `MlpEnsembleCoilPredictor`, stored alongside its checkpoint."""

    model_kind: Literal["mlp_ensemble"] = "mlp_ensemble"
    """Tag that disambiguates this config in `AnyModelConfig` unions."""
    n_ensemble_members: int = 5
    """Number of independent MLPs whose predictions are averaged at inference."""


MlpEnsembleCoilPredictorCheckpoint = FlaxNnxCheckpoint[MlpEnsembleCoilPredictorConfig]


class ResMlpEnsembleCoilPredictorConfig(_FixedShapeCoilPredictorConfig):
    """Init config for `ResMlpEnsembleCoilPredictor`, stored alongside its
    checkpoint."""

    model_kind: Literal["res_mlp_ensemble"] = "res_mlp_ensemble"
    """Tag that disambiguates this config in `AnyModelConfig` unions."""
    dropout: float = 0.2
    """Dropout rate applied inside each residual block.

    0.0 disables dropout.
    """
    n_ensemble_members: int = 5
    """Number of independent MLPs whose predictions are averaged at inference."""


ResMlpEnsembleCoilPredictorCheckpoint = FlaxNnxCheckpoint[
    ResMlpEnsembleCoilPredictorConfig
]


class _VariableShapeCoilPredictorConfig(BaseModel):
    """Fields shared by the variable-shape coil predictors.

    The number of field periods and unique coils are inputs of the model rather than
    architecture constants; the value sets below fix the one-hot widths.
    """

    is_stellarator_symmetric: bool
    n_poloidal_modes: int
    n_toroidal_modes: int
    n_modes_coils_max: int
    coil_currents_mean: float
    """See `CoilPredictorConfig.coil_currents_mean`."""
    coil_currents_std: float
    """See `CoilPredictorConfig.coil_currents_std`."""
    n_field_periods_values: list[int]
    """Supported numbers of field periods (one-hot encoding vocabulary)."""
    n_coils_values: list[int]
    """Supported numbers of unique coils (one-hot encoding vocabulary)."""
    max_coils: int
    """Number of coil slots in the padded output."""

    hidden_dim: int = 512
    num_layers: int = 4
    requirement_metrics_means: list[float]
    """See `CoilPredictorConfig.requirement_metrics_means`."""
    requirement_metrics_stds: list[float]
    """See `CoilPredictorConfig.requirement_metrics_stds`."""
    spectral_scaling_alpha: float = 1.2
    spectral_scaling_min: float = 1e-7
    spectral_scaling_order: float = float("inf")


class PerCoilCoilPredictorConfig(_VariableShapeCoilPredictorConfig):
    """Init config for `PerCoilCoilPredictor`, stored alongside its checkpoint."""

    model_kind: Literal["per_coil"] = "per_coil"
    """Tag that disambiguates this config in `AnyModelConfig` unions."""


PerCoilCoilPredictorCheckpoint = FlaxNnxCheckpoint[PerCoilCoilPredictorConfig]


class MaskedCoilPredictorConfig(_VariableShapeCoilPredictorConfig):
    """Init config for `MaskedCoilPredictor`, stored alongside its checkpoint."""

    model_kind: Literal["masked"] = "masked"
    """Tag that disambiguates this config in `AnyModelConfig` unions."""


MaskedCoilPredictorCheckpoint = FlaxNnxCheckpoint[MaskedCoilPredictorConfig]


class PerCoilEnsembleCoilPredictorConfig(_VariableShapeCoilPredictorConfig):
    """Init config for `PerCoilEnsembleCoilPredictor`, stored alongside its
    checkpoint."""

    model_kind: Literal["per_coil_ensemble"] = "per_coil_ensemble"
    """Tag that disambiguates this config in `AnyModelConfig` unions."""
    n_ensemble_members: int = 5
    """Number of independent members whose predictions are averaged at inference."""


PerCoilEnsembleCoilPredictorCheckpoint = FlaxNnxCheckpoint[
    PerCoilEnsembleCoilPredictorConfig
]


class MaskedEnsembleCoilPredictorConfig(_VariableShapeCoilPredictorConfig):
    """Init config for `MaskedEnsembleCoilPredictor`, stored alongside its
    checkpoint."""

    model_kind: Literal["masked_ensemble"] = "masked_ensemble"
    """Tag that disambiguates this config in `AnyModelConfig` unions."""
    n_ensemble_members: int = 5
    """Number of independent members whose predictions are averaged at inference."""


MaskedEnsembleCoilPredictorCheckpoint = FlaxNnxCheckpoint[
    MaskedEnsembleCoilPredictorConfig
]


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
    coil_mask: Float[NpOrJaxArray, "batch_size max_coils"]
    """1.0 at real coil slots, 0.0 at zero-padded coil slots.

    All ones when coilsets are not padded to a common coil count.
    """
    batch_max_fourier_order: int
    """Per-batch fourier order N.

    Every coilset in the batch is padded to `(2*N + 1)` modes.
    """


register_pydantic_data(Batch, meta_fields=["batch_max_fourier_order"])

AnyModelLiteral = Literal[
    "mlp",
    "res_mlp",
    "mlp_ensemble",
    "res_mlp_ensemble",
    "per_coil",
    "masked",
    "per_coil_ensemble",
    "masked_ensemble",
]
VARIABLE_SHAPE_MODEL_KIND = frozenset(
    {"per_coil", "masked", "per_coil_ensemble", "masked_ensemble"}
)
AnyModelConfig = (
    CoilPredictorConfig
    | ResMlpCoilPredictorConfig
    | MlpEnsembleCoilPredictorConfig
    | ResMlpEnsembleCoilPredictorConfig
    | PerCoilCoilPredictorConfig
    | MaskedCoilPredictorConfig
    | PerCoilEnsembleCoilPredictorConfig
    | MaskedEnsembleCoilPredictorConfig
)
AnyModelCheckpoint = (
    CoilPredictorCheckpoint
    | ResMlpCoilPredictorCheckpoint
    | MlpEnsembleCoilPredictorCheckpoint
    | ResMlpEnsembleCoilPredictorCheckpoint
    | PerCoilCoilPredictorCheckpoint
    | MaskedCoilPredictorCheckpoint
    | PerCoilEnsembleCoilPredictorCheckpoint
    | MaskedEnsembleCoilPredictorCheckpoint
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
    model_kind: AnyModelLiteral = "mlp"
    model_config_overrides: AnyModelConfig | None = None
    validation: Literal["eval_split", "train_holdout"] = "eval_split"
    """Validate on the dataset's eval split, or on a boundary-disjoint holdout carved
    out of the train split."""
    holdout_fraction: float = pydantic.Field(default=0.05, gt=0.0, lt=1.0)
    """Approximate fraction of train-split boundaries held out when `validation` is
    `"train_holdout"`."""
    checkpoint_best: bool = False
    """Checkpoint the weights at the best evaluation instead of the final weights."""

    @pydantic.model_validator(mode="after")
    def _check_overrides_match_model_kind(self) -> Self:
        if self.model_config_overrides is None:
            return self
        expected_config_kind: dict[str, type] = {
            "mlp": CoilPredictorConfig,
            "res_mlp": ResMlpCoilPredictorConfig,
            "mlp_ensemble": MlpEnsembleCoilPredictorConfig,
            "res_mlp_ensemble": ResMlpEnsembleCoilPredictorConfig,
            "per_coil": PerCoilCoilPredictorConfig,
            "masked": MaskedCoilPredictorConfig,
            "per_coil_ensemble": PerCoilEnsembleCoilPredictorConfig,
            "masked_ensemble": MaskedEnsembleCoilPredictorConfig,
        }
        expected = expected_config_kind[self.model_kind]
        if not isinstance(self.model_config_overrides, expected):
            raise TypeError(
                f"model_kind={self.model_kind!r} requires {expected.__name__} "
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
    desc_equilibrium: DescEquilibrium | None = None
    """The equilibrium `requirement_metrics` were computed on."""
