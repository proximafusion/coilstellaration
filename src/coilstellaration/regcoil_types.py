import dataclasses
import pathlib
from typing import Annotated, Literal, Self

import jaxtyping as jt
import numpy as np
import pydantic
from scipy import constants

from coilstellaration import types

RegcoilTargetOption = Literal[
    "normalized_coil_to_coil_distance",
    "normalized_field_error",
    "minimum_chi2_k_times_chi2_b",
]


class RegcoilSettings(types.BaseModel):
    coils_surface_distance_over_minor_radius: float
    """The target distance between the coils and the plasma in units of the minor
    radius."""

    n_coils_per_half_period: int
    """The number of coils per half period."""

    normalized_coil_to_coil_distance: float | None = None
    r"""Normalized coil-to-coil distance used for the target option
    ``"normalized_coil_to_coil_distance"``.

    The maximum k target is calculated as:

    .. math::

        \text{max\_k\_target} = \frac{\text{external\_current}}{n_c \tilde{d}_{cc} a},

    where :math:`\text{external\_current}` is the external poloidal current,
    :math:`n_c` is the number of coils, :math:`\tilde{d}_{cc}` is the normalized
    coil-to-coil distance, and :math:`a` is the minor radius.
    """

    target_option: RegcoilTargetOption = "normalized_coil_to_coil_distance"
    """High-level objective used to choose lambda."""

    maximum_normalized_field_error: float | None = None
    """Maximum normalized field error target.

    This value is only used when ``target_option`` is ``"normalized_field_error"``.
    """

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

    n_fourier_modes: int = 15
    """Number of Fourier modes used when cutting filamentary coils from REGCOIL
    output."""

    n_discretization_points: int = 300
    """Number of discretization points used when cutting filamentary coils from REGCOIL
    output."""

    verbose: bool = True
    """Whether to print REGCOIL output to console."""

    include_plasma_contribution: bool = False
    """Whether to include the contribution of plasma currents to compute the normal
    magnetic field on the plasma surface."""

    @classmethod
    def construct_high_fidelity(
        cls,
        coils_surface_distance_over_minor_radius: float,
        n_coils_per_half_period: int,
        normalized_coil_to_coil_distance: float | None = None,
        target_option: RegcoilTargetOption = "normalized_coil_to_coil_distance",
        maximum_normalized_field_error: float | None = None,
    ) -> "RegcoilSettings":
        return cls(
            coils_surface_distance_over_minor_radius=coils_surface_distance_over_minor_radius,
            n_coils_per_half_period=n_coils_per_half_period,
            normalized_coil_to_coil_distance=normalized_coil_to_coil_distance,
            target_option=target_option,
            maximum_normalized_field_error=maximum_normalized_field_error,
            n_surface_poloidal_grid_points=96,
            n_surface_toroidal_grid_points=96,
            n_coil_poloidal_grid_points=96,
            n_coil_toroidal_grid_points=96,
            current_potential_max_poloidal_mode_number=32,
            current_potential_max_toroidal_mode_number=32,
        )

    @classmethod
    def construct_low_fidelity(
        cls,
        coils_surface_distance_over_minor_radius: float,
        n_coils_per_half_period: int,
        normalized_coil_to_coil_distance: float | None = None,
        target_option: RegcoilTargetOption = "normalized_coil_to_coil_distance",
        maximum_normalized_field_error: float | None = None,
    ) -> "RegcoilSettings":
        return cls(
            coils_surface_distance_over_minor_radius=coils_surface_distance_over_minor_radius,
            n_coils_per_half_period=n_coils_per_half_period,
            normalized_coil_to_coil_distance=normalized_coil_to_coil_distance,
            target_option=target_option,
            maximum_normalized_field_error=maximum_normalized_field_error,
            n_surface_poloidal_grid_points=96,
            n_surface_toroidal_grid_points=96,
            current_potential_max_poloidal_mode_number=16,
            current_potential_max_toroidal_mode_number=16,
            n_coil_poloidal_grid_points=16 * 2 + 1,
            n_coil_toroidal_grid_points=16 * 2 + 1,
            n_fourier_modes=10,
        )

    @pydantic.model_validator(mode="after")
    def _validate_target_settings(self) -> Self:
        if (
            self.target_option == "normalized_field_error"
            and self.maximum_normalized_field_error is None
        ):
            raise ValueError(
                "maximum_normalized_field_error must be provided when "
                "target_option='normalized_field_error'."
            )
        if (
            self.target_option == "normalized_coil_to_coil_distance"
            and self.normalized_coil_to_coil_distance is None
        ):
            raise ValueError(
                "normalized_coil_to_coil_distance must be provided when "
                "target_option='normalized_coil_to_coil_distance'."
            )
        return self


class RegcoilOutput(types.BaseModel):
    chi2_B: jt.Float[np.ndarray, " n_lambda"]
    """Values of chi^2_B (the area integral over the plasma surface of |B_normal|^2)
    that resulted for each value of lambda.

    @stardash-unit: T^2 * m^2
    """

    chi2_K: jt.Float[np.ndarray, " n_lambda"]
    """Values of chi^2_K (the area integral over the coil winding surface of current
    density squared) that resulted for each value of lambda.

    @stardash-unit: A^2
    """

    lambda_: jt.Float[np.ndarray, " n_lambda"]
    """Values of the regularization parameter that were used.

    @stardash-unit: T^2 * m^2 / A^2
    """

    maximum_K: jt.Float[np.ndarray, " n_lambda"]
    """Maximum (over the coil surface) current density that resulted for each value of
    lambda.

    `max_K` in the REGCOIL output file.

    @stardash-unit: A/m
    """

    plasma_major_radius: float
    """Major radius of the plasma surface.

    @stardash-unit: m
    """

    coil_winding_surface_major_radius: float
    """Major radius of the coil winding surface.

    @stardash-unit: m
    """

    plasma_area: float
    """Area of the plasma surface.

    @stardash-unit: m^2
    """

    coil_winding_surface_area: float
    """Area of the coil winding surface.

    @stardash-unit: m^2
    """

    plasma_volume: float
    """Volume of the plasma surface.

    @stardash-unit: m^3
    """

    coil_winding_surface_volume: float
    """Volume of the coil winding surface.

    @stardash-unit: m^3
    """

    net_poloidal_current: float
    """Net current that flows on the coil winding surface in the poloidal direction.

    @stardash-unit: A
    """

    net_toroidal_current: float
    """Net current that flows on the coil winding surface in the toroidal direction.

    @stardash-unit: A
    """

    orthogonal_magnetic_field_component: Annotated[
        jt.Float[np.ndarray, "n_lambda n_zeta_plasma n_theta_plasma"], types.Binary
    ]
    """Residual magnetic field normal to the plasma surface, for each value of the
    regularization parameter lambda considered.

    @stardash-unit: T
    """

    plasma_normal_vector_norm: Annotated[
        jt.Float[np.ndarray, "n_zeta_plasma n_theta_plasma"], types.Binary
    ]
    """|N|, where N = (d r / d zeta) cross (d r / d theta) is a non-unit-length normal
    vector and r is the position vector, for the plasma surface.

    This quantity is the Jacobian appearing in area integrals:
    int d^2a = int dtheta int dzeta |N|.
    """

    @property
    def n_lambda(self) -> int:
        """Number of values of the regularization parameter lambda examined.

        `nlambda` in the REGCOIL output file.
        """
        return self.lambda_.shape[0]

    @pydantic.computed_field
    @property
    def on_axis_averaged_magnetic_field_strength(self) -> float:
        """On-axis averaged magnetic field strength.

        @stardash-unit: T
        """
        return (
            constants.mu_0
            * self.net_poloidal_current
            / (2 * np.pi * self.plasma_major_radius)
        )

    @pydantic.computed_field
    @property
    def normalized_field_error(
        self,
    ) -> Annotated[
        jt.Float[np.ndarray, "n_lambda n_zeta_plasma n_theta_plasma"], types.Binary
    ]:
        """Normalized field error, defined as the absolute value of the orthogonal
        magnetic field component divided by the on-axis averaged magnetic field
        strength."""
        return (
            np.abs(self.orthogonal_magnetic_field_component)
            / self.on_axis_averaged_magnetic_field_strength
        )

    @pydantic.computed_field
    @property
    def average_normalized_field_error(self) -> jt.Float[np.ndarray, " n_lambda"]:
        """Average normalized field error, defined as the area integral of the
        normalized field error divided by the plasma surface area."""
        return np.sum(
            self.normalized_field_error * self.plasma_normal_vector_norm, axis=(1, 2)
        ) / np.sum(self.plasma_normal_vector_norm)

    @pydantic.computed_field
    @property
    def normalized_chi2_B(self) -> jt.Float[np.ndarray, " n_lambda"]:
        """Normalized chi^2_B, defined as chi^2_B divided by the product of the plasma
        surface area and the square of the on-axis averaged magnetic field strength."""
        return self.chi2_B / (
            self.plasma_area * self.on_axis_averaged_magnetic_field_strength**2
        )

    @pydantic.computed_field
    @property
    def normalized_chi2_K(self) -> jt.Float[np.ndarray, " n_lambda"]:
        """Normalized chi^2_K, defined as chi^2_K divided by the square of the net
        poloidal current."""
        return self.chi2_K / self.net_poloidal_current**2

    @pydantic.computed_field
    @property
    def maximum_normalized_field_error(self) -> jt.Float[np.ndarray, " n_lambda"]:
        """Maximum normalized field error on the plasma surface for each lambda."""
        return np.max(self.normalized_field_error, axis=(1, 2))

    @pydantic.computed_field
    @property
    def normalized_maximum_K(self) -> jt.Float[np.ndarray, " n_lambda"]:
        """Maximum current density normalized by ``G / (2 * pi * R0_plasma)``."""
        return self.maximum_K / (
            self.net_poloidal_current / (2 * np.pi * self.plasma_major_radius)
        )

    @pydantic.computed_field
    @property
    def normalized_chi2_K_times_chi2_B(self) -> jt.Float[np.ndarray, " n_lambda"]:
        """Normalized chi^2_K times chi^2_B, defined as the product of the normalized
        chi^2_K and normalized chi^2_B."""
        return self.normalized_chi2_K * self.normalized_chi2_B

    @pydantic.computed_field
    @property
    def success(self) -> bool:
        """Whether REGCOIL has converged based on the sampled lambda sequence.

        REGCOIL uses early-stop patterns when the requested target is not achievable
        because it is too low. We treat the following sequences as non-converged:

        - one sampled value: ``[1e200]``. This is usually the case for
            `target_option="normalized_coil_to_coil_distance"` and the target is too
            low.
        - two sampled values: ``[1e200, 0.0]``. This is usually the case for
            `target_option="normalized_field_error"` and the target is too low.

        For other cases, convergence is approximated by whether REGCOIL sampled more
        than one lambda value.
        """
        if self.n_lambda == 1 and self.lambda_[0] > 1.0e100:
            return False
        if (
            self.n_lambda == 2
            and self.lambda_[0] > 1.0e100
            and np.isclose(self.lambda_[1], 0.0)
        ):
            return False
        return self.n_lambda > 1


REGCOIL_INPUT_FILENAME_PREFIX = "regcoil_in."
REGCOIL_INPUT_TEMPLATE = """\
! For documentation of these input parameters and their default values, see the manual.

&regcoil_nml
  general_option = {general_option}

{target_block}

  nlambda = {nlambda}
  lambda_min = 1.0e-23
  lambda_max = 1.0e-13

  ntheta_plasma = {ntheta_plasma}
  ntheta_coil   = {ntheta_coil}
  nzeta_plasma  = {nzeta_plasma}
  nzeta_coil    = {nzeta_coil}

  mpol_potential = {mpol_potential}
  ntor_potential = {ntor_potential}

  geometry_option_plasma = 2
  wout_filename = '{wout_filename}'

  geometry_option_coil = 4
  separation = {coils_plasma_distance}

  load_bnorm=.f.
  bnorm_filename=
/
"""


@dataclasses.dataclass
class RegcoilConfig:
    wout_filepath: pathlib.Path
    coils_plasma_distance: float
    target_value: float | None
    target_option: str | None = "max_K"
    general_option: int = 5
    nlambda: int = 31
    ntheta_plasma: int = 96
    ntheta_coil: int = 96
    nzeta_plasma: int = 96
    nzeta_coil: int = 96
    mpol_potential: int = 32
    ntor_potential: int = 32

    def __post_init__(self) -> None:
        has_target_option = self.target_option is not None
        has_target_value = self.target_value is not None
        if has_target_option != has_target_value:
            raise ValueError(
                "target_option and target_value must both be provided or both be None."
            )

        if self.general_option == 5 and not (has_target_option and has_target_value):
            raise ValueError(
                "general_option=5 requires target_option and target_value to be set."
            )

    def write_input_file(self, filepath: pathlib.Path) -> None:
        target_block = ""
        if self.target_option is not None and self.target_value is not None:
            target_block = (
                f"  target_option = '{self.target_option}'\n"
                f"  target_value = {self.target_value}\n"
            )

        with open(filepath, "w") as f:
            f.write(
                REGCOIL_INPUT_TEMPLATE.format(
                    general_option=self.general_option,
                    target_block=target_block,
                    wout_filename=str(self.wout_filepath),
                    coils_plasma_distance=self.coils_plasma_distance,
                    nlambda=self.nlambda,
                    ntheta_plasma=self.ntheta_plasma,
                    ntheta_coil=self.ntheta_coil,
                    nzeta_plasma=self.nzeta_plasma,
                    nzeta_coil=self.nzeta_coil,
                    mpol_potential=self.mpol_potential,
                    ntor_potential=self.ntor_potential,
                )
            )
