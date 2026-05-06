import logging
import os
import pathlib
import subprocess
import tempfile
import textwrap

import jaxtyping as jt
import netCDF4
import numpy as np
from constellaration.mhd import vmec_utils
from desc.coils import CoilSet as DescCoilSet
from scipy import constants
from scipy.io import netcdf_file
from simsopt import field
from skimage.measure import find_contours as skimage_find_contours

from coilstellaration import coilset_utils, regcoil_types, types
from coilstellaration.types import NpOrJaxArray, runtime_check_array_sizes

logger = logging.getLogger(__name__)


_TARGET_OPTION_MINIMUM_CHI2_PRODUCT = "minimum_chi2_k_times_chi2_b"

REGCOIL_OUTPUT_FILENAME_PREFIX = "regcoil_out."
NESCIN_OUTPUT_FILENAME = "nescin.out"

logger = logging.getLogger(__name__)


def _cut_coils_from_regcoil(
    regcoil_out_filename: pathlib.Path,
    nescin_filename: pathlib.Path,
    coils_per_half_period: int,
    theta_shift: int,
    ilambda: int,
) -> None:
    if not regcoil_out_filename.name.startswith("regcoil_out."):
        raise RuntimeError(
            f"filename should be in the pattern **/regcoil_out.XXX, is {regcoil_out_filename}"
        )
    coils_output_filename = (
        regcoil_out_filename.parent / f"coils.{regcoil_out_filename.name[12:-3]}"
    )

    logger.info(textwrap.dedent(f"""
            coilsFilename: {coils_output_filename}
            coilsPerHalfPeriod: {coils_per_half_period}
            thetaShift: {theta_shift}
            ilambda: {ilambda}
            """))

    with netcdf_file(regcoil_out_filename, "r", mmap=False) as f:
        theta = f.variables["theta_coil"][()]
        zeta = f.variables["zeta_coil"][()]
        nfp = f.variables["nfp"][()]
        net_poloidal_current_Amperes = f.variables["net_poloidal_current_Amperes"][()]
        current_potential = f.variables["current_potential"][()]

    logger.info(f"{current_potential.shape=}")
    if abs(net_poloidal_current_Amperes) > np.finfo(float).eps:
        data = current_potential[ilambda, :, :] / net_poloidal_current_Amperes * nfp
    else:
        data = current_potential[ilambda, :, :] / np.max(
            current_potential[ilambda, :, :]
        )

    logger.info(f"Theta before shift: {theta}")
    theta = np.roll(theta, theta_shift)
    theta = theta[0] + np.linspace(0, 2 * np.pi, len(theta), endpoint=False)
    logger.info(f"Theta after shift: {theta}")

    data = np.roll(data, theta_shift, axis=1)

    d = 2 * np.pi / nfp
    zeta_3 = np.concatenate((zeta - d, zeta, zeta + d))
    data_3 = np.concatenate((data - 1, data, data + 1))
    logger.info(f"{data_3.shape=}")
    contours = np.linspace(0, 1, coils_per_half_period * 2, endpoint=False)
    d = contours[1] - contours[0]
    contours = contours + d / 2
    contour_collections = find_contours(zeta_3, theta, np.transpose(data_3), contours)
    if not contour_collections:
        raise RuntimeError(
            "No contours found. Check that the regcoil output file is valid and contains the expected data."
        )
    numCoilsFound = len(contour_collections)
    logger.info(f"{len(contour_collections)=}")
    if numCoilsFound != 2 * coils_per_half_period:
        raise ValueError(
            f"The expected number of coils was not the number found. Expected {2 * coils_per_half_period}, found {numCoilsFound}."
        )

    contour_zeta = []
    contour_theta = []
    numCoils = 0
    for j in range(numCoilsFound):
        v = contour_collections[j][0]
        if v[1, 1] < v[0, 1]:
            v = np.flipud(v)
        for jfp in range(nfp):
            d = 2 * np.pi / nfp * jfp
            contour_zeta.append(v[:, 0] + d)
            contour_theta.append(v[:, 1])
            numCoils += 1

    with open(nescin_filename, "r") as f:
        line = ""
        while "np     iota_edge       phip_edge       curpol" not in line:
            line = f.readline()
        line = f.readline()
        nfp_nescin = int(line.split()[0])
        logger.info(f"{nfp_nescin=}")
        if nfp != nfp_nescin:
            raise ValueError(
                f"{nfp=} from regcoil_out does not match {nfp_nescin=} from nescin!"
            )

        contour_R = []
        contour_Z = []
        for j in range(numCoils):
            contour_R.append(contour_zeta[j] * 0)
            contour_Z.append(contour_zeta[j] * 0)

        line = ""
        while "------ Current Surface" not in line:
            line = f.readline()
        line = f.readline()
        line = f.readline()
        logger.info(f"Number of Fourier modes in coil surface from nescin file: {line}")
        nmodes = int(line)
        line = f.readline()
        line = f.readline()
        for _ in range(nmodes):
            data = f.readline().split()
            m = int(data[0])
            # n = -int(data[1])*nfp
            n = -int(data[1]) * nfp
            # Sign flip in n because bnormal uses NESCOIL convention.
            # See bn_read_vmecf90.f line 89.
            crc = float(data[2])
            czs = float(data[3])
            crs = float(data[4])
            czc = float(data[5])
            # Skip remaining columns
            for j in range(numCoils):
                angle = m * contour_theta[j] - n * contour_zeta[j]
                contour_R[j] = contour_R[j] + crc * np.cos(angle) + crs * np.sin(angle)
                contour_Z[j] = contour_Z[j] + czs * np.sin(angle) + czc * np.cos(angle)

    contour_X = []
    contour_Y = []
    maxR = 0
    for j in range(numCoils):
        maxR = np.max((maxR, np.max(contour_R[j])))
        contour_X.append(contour_R[j] * np.cos(contour_zeta[j]))
        contour_Y.append(contour_R[j] * np.sin(contour_zeta[j]))
    coilCurrent = net_poloidal_current_Amperes / numCoils

    minSeparation2 = 1.0e20
    for whichCoil1 in range(numCoils):
        for whichCoil2 in range(whichCoil1):
            for whichPoint in range(len(contour_X[whichCoil1])):
                dx = contour_X[whichCoil1][whichPoint] - contour_X[whichCoil2]
                dy = contour_Y[whichCoil1][whichPoint] - contour_Y[whichCoil2]
                dz = contour_Z[whichCoil1][whichPoint] - contour_Z[whichCoil2]
                separation2 = dx * dx + dy * dy + dz * dz
                this_minSeparation2 = np.min(separation2)
                if this_minSeparation2 < minSeparation2:
                    minSeparation2 = this_minSeparation2
                    x1 = contour_X[whichCoil1][whichPoint]
                    y1 = contour_Y[whichCoil1][whichPoint]
                    z1 = contour_Z[whichCoil1][whichPoint]
                    index = np.argmin(separation2)
                    x2 = contour_X[whichCoil2][index]
                    y2 = contour_Y[whichCoil2][index]
                    z2 = contour_Z[whichCoil2][index]

    logger.info(f"Minimum coil separation: {np.sqrt(minSeparation2)}")

    with open(coils_output_filename, "w") as f:
        f.write("periods " + str(nfp) + "\n")
        f.write("begin filament\n")
        f.write("mirror NIL\n")

        for j in range(numCoils):
            N = len(contour_X[j])
            for k in range(N):
                f.write(
                    "{:14.22e} {:14.22e} {:14.22e} {:14.22e}\n".format(
                        contour_X[j][k], contour_Y[j][k], contour_Z[j][k], coilCurrent
                    )
                )
            # Close the loop
            k = 0
            f.write(
                "{:14.22e} {:14.22e} {:14.22e} {:14.22e} {:} Modular\n".format(
                    contour_X[j][k], contour_Y[j][k], contour_Z[j][k], 0, j + 1
                )
            )

        f.write("end\n")


def _filepath_to_vmec_id(filepath: str) -> str:
    filename = os.path.basename(filepath)
    if filename.endswith(".json"):
        return filename[:-5]
    elif filename.startswith("input."):
        return filename[6:]
    elif filename.startswith("wout_") and filename.endswith(".nc"):
        return filename[5:-3]
    else:
        raise ValueError(f"Invalid filename: {filename}")


def _run_binary(
    config: regcoil_types.RegcoilConfig,
    path: pathlib.Path | None = None,
    verbose: bool = True,
) -> pathlib.Path:
    """Runs REGCOIL with the given configuration and returns REGCOIL output file path.

    Args:
        config: input configuration for REGCOIL.
        path: directory where to run REGCOIL. If None, uses current directory.
        verbose: whether to print REGCOIL output to console.

    Returns:
        The path to the output file generated by REGCOIL.
    """
    if path is None:
        path = pathlib.Path()

    config.wout_filepath = path / config.wout_filepath

    vmec_id = _filepath_to_vmec_id(str(config.wout_filepath))

    input_filename = path / f"{regcoil_types.REGCOIL_INPUT_FILENAME_PREFIX}{vmec_id}"

    config.write_input_file(input_filename)

    if not verbose:
        log_file = tempfile.NamedTemporaryFile(
            mode="w+", delete=False, prefix="regcoil_log_", suffix=".txt"
        )
        stdout = log_file
        stderr = log_file
    else:
        log_file = None
        stdout = None
        stderr = None

    try:
        subprocess.run(
            ["regcoil", input_filename.name],
            check=True,
            cwd=input_filename.parent,
            stdout=stdout,
            stderr=stderr,
        )
    finally:
        if log_file is not None:
            log_file.close()
            pathlib.Path(log_file.name).unlink(missing_ok=True)

    regcoil_output_filename = path / f"{REGCOIL_OUTPUT_FILENAME_PREFIX}{vmec_id}.nc"
    return regcoil_output_filename


@runtime_check_array_sizes
def find_contours(
    x: jt.Float[NpOrJaxArray, " xn"],
    y: jt.Float[NpOrJaxArray, " yn"],
    z: jt.Float[NpOrJaxArray, " yn xn"],
    levels: jt.Float[NpOrJaxArray, " p"],
) -> list[jt.Float[NpOrJaxArray, " _n_contours _n_vertices xy=2"]]:
    """Find contours using scikit-image.

    Args:
        x: 1D array of x coordinates
        y: 1D array of y coordinates
        z: 2D array of values at (x, y) coordinates
        levels: 1D array of contour levels to find

    Returns:
        List of lists containing contour vertices arrays
    """
    all_contours = []

    for level in levels:
        # Find contours at this level using skimage
        contour_list = skimage_find_contours(z, level)

        level_contours = []
        for contour in contour_list:
            # Convert from pixel coordinates to actual coordinates
            # contour has shape (n_points, 2) with (row, col) coordinates
            # We need to map these to (x, y) coordinates

            # Map row indices to y coordinates
            y_coords = np.interp(contour[:, 0], np.arange(len(y)), y)
            # Map column indices to x coordinates
            x_coords = np.interp(contour[:, 1], np.arange(len(x)), x)

            vertices = np.column_stack([x_coords, y_coords])
            level_contours.append(vertices)

        if level_contours:
            all_contours.append(np.array(level_contours))

    return all_contours


def regcoil_output_file_to_makegrid_input_file(
    regcoil_output_filepath: pathlib.Path,
    makegrid_input_filepath: pathlib.Path,
    number_of_coils_per_half_period: int,
    theta_shift: int = 0,
    lambda_index: int = -1,
    nescin_output_path: pathlib.Path | None = None,
    n_fourier_modes: int = 15,
) -> None:
    """Writes a MAKEGRID file from a REGCOIL output file.

    Args:
        regcoil_output_filepath: path to the REGCOIL output file.
        makegrid_input_filepath: path to the MAKEGRID file to be written.
        number_of_coils_per_half_period: number of coils per half period.
        theta_shift: shift in theta index. Default to 0.
        lambda_index: index of REGCOIL lambda run to be used. Default to -1 (last run).
    """
    if nescin_output_path is None:
        nescin_output_path = pathlib.Path()
    nescin_output_filename = nescin_output_path / NESCIN_OUTPUT_FILENAME

    # Regcoil doesn't play nice with paths which include directories.
    _cut_coils_from_regcoil(
        regcoil_out_filename=regcoil_output_filepath,
        nescin_filename=nescin_output_filename,
        coils_per_half_period=number_of_coils_per_half_period,
        theta_shift=theta_shift,
        ilambda=lambda_index,
    )

    suffix = regcoil_output_filepath.stem.removeprefix(REGCOIL_OUTPUT_FILENAME_PREFIX)
    intermediate_makegrid_filepath = regcoil_output_filepath.with_name(
        f"coils.{suffix}"
    )
    coils = field.load_coils_from_makegrid_file(
        str(intermediate_makegrid_filepath), order=n_fourier_modes
    )

    # Modify MAKEGRID file to be compatible with SIMSOPT
    with netcdf_file(regcoil_output_filepath, "r", mmap=False) as f:
        nfp = int(f.variables["nfp"][()])
    independent_coil_indices = [i * nfp for i in range(number_of_coils_per_half_period)]
    independent_coils = [coils[i] for i in independent_coil_indices]

    curves = [coil.curve for coil in independent_coils]
    currents = [coil.current for coil in independent_coils]
    groups = list(range(number_of_coils_per_half_period)) * int(2 * nfp)
    field.coils_to_makegrid(
        filename=str(makegrid_input_filepath),
        curves=curves,
        currents=currents,
        groups=groups,
        nfp=nfp,
        stellsym=True,
    )


def _build_regcoil_input_config(
    equilibrium: vmec_utils.VmecppWOut,
    settings: regcoil_types.RegcoilSettings,
    temp_dir: pathlib.Path | None = None,
) -> regcoil_types.RegcoilConfig:
    minor_radius = equilibrium.Aminor_p
    n_field_periods = equilibrium.n_field_periods

    simsopt_vmec = vmec_utils.as_simsopt_vmec(equilibrium)  # , directory=temp_dir)

    coils_to_plasma_distance = (
        minor_radius * settings.coils_surface_distance_over_minor_radius
    )
    logger.info(f"Coils to plasma distance: {coils_to_plasma_distance}m.")

    n_coils = settings.n_coils_per_half_period * 2 * n_field_periods

    general_option = 5
    target_option: str | None
    target_value: float | None

    if settings.target_option == "normalized_coil_to_coil_distance":
        assert settings.normalized_coil_to_coil_distance is not None
        external_current = simsopt_vmec.external_current()
        coil_to_coil_distance = settings.normalized_coil_to_coil_distance * minor_radius
        max_k_target_value = external_current / n_coils / coil_to_coil_distance
        target_value = max_k_target_value
        target_option = "max_K"
        logger.info(f"Max k target: {1e-6 * max_k_target_value} MA/m.")
    elif settings.target_option == "normalized_field_error":
        assert settings.maximum_normalized_field_error is not None
        on_axis_averaged_b = _on_axis_averaged_magnetic_field_strength(
            net_poloidal_current=simsopt_vmec.external_current(),
            plasma_major_radius=equilibrium.Rmajor_p,
        )
        max_field_error_target_value = (
            settings.maximum_normalized_field_error * on_axis_averaged_b
        )
        target_value = max_field_error_target_value
        target_option = "max_Bnormal"
        logger.info(f"Max field error target: {max_field_error_target_value} T.")
    elif settings.target_option == _TARGET_OPTION_MINIMUM_CHI2_PRODUCT:
        general_option = 1
        target_option = None
        target_value = None
        logger.info(
            "Using lambda scan (general_option=1) and selecting the minimum "
            "chi2_K * chi2_B solution."
        )
    else:
        raise ValueError(
            f"Unsupported REGCOIL target option: {settings.target_option}."
        )

    if temp_dir is None:
        temp_dir = pathlib.Path()
    wout_file_path = temp_dir / pathlib.Path(simsopt_vmec.output_file).name

    equilibrium.save(wout_file_path)

    return regcoil_types.RegcoilConfig(
        wout_filepath=wout_file_path,
        coils_plasma_distance=coils_to_plasma_distance,
        general_option=general_option,
        target_option=target_option,
        target_value=target_value,
        ntheta_plasma=settings.n_surface_poloidal_grid_points,
        nzeta_plasma=settings.n_surface_toroidal_grid_points,
        ntheta_coil=settings.n_coil_poloidal_grid_points,
        nzeta_coil=settings.n_coil_toroidal_grid_points,
        mpol_potential=settings.current_potential_max_poloidal_mode_number,
        ntor_potential=settings.current_potential_max_toroidal_mode_number,
    )


def _on_axis_averaged_magnetic_field_strength(
    net_poloidal_current: float, plasma_major_radius: float
) -> float:
    """Compute the on-axis averaged magnetic field strength.."""
    return constants.mu_0 * net_poloidal_current / (2 * np.pi * plasma_major_radius)


def _get_unmasked_array(data: netCDF4.Dataset, key: str) -> np.ndarray:
    arr = data[key][()]
    if hasattr(arr, "mask"):
        return arr.data
    return arr


def _lambda_objective_from_chi2(
    chi2_B: np.ndarray,
    chi2_K: np.ndarray,
) -> np.ndarray:
    return np.asarray(chi2_B) * np.asarray(chi2_K)


def _minimum_chi2_product_lambda_index(
    chi2_B: np.ndarray,
    chi2_K: np.ndarray,
) -> int:
    objective = _lambda_objective_from_chi2(chi2_B=chi2_B, chi2_K=chi2_K)
    finite_objective = np.where(np.isfinite(objective), objective, np.inf)
    return int(np.argmin(finite_objective))


def _lambda_sort_indices_for_target_option(
    chi2_B: np.ndarray,
    chi2_K: np.ndarray,
    target_option: regcoil_types.RegcoilTargetOption,
) -> np.ndarray:
    if target_option != _TARGET_OPTION_MINIMUM_CHI2_PRODUCT:
        return np.arange(chi2_B.shape[0])

    objective = _lambda_objective_from_chi2(chi2_B=chi2_B, chi2_K=chi2_K)
    finite_objective = np.where(np.isfinite(objective), objective, np.inf)

    # Keep the minimum objective value at the end to match the "chosen solution is
    # last" behavior used by existing high-level target options.
    return np.argsort(finite_objective, kind="stable")[::-1]


def _select_lambda_index(
    regcoil_output_filepath: pathlib.Path,
    target_option: regcoil_types.RegcoilTargetOption,
) -> int:
    """Select a lambda index from a REGCOIL output file for a high-level target.

    Args:
        regcoil_output_filepath: Path to the REGCOIL output file.
        target_option: High-level target option.

    Returns:
        Lambda index to use when extracting coils. For native REGCOIL target options,
        this is ``-1``.
    """
    if target_option != _TARGET_OPTION_MINIMUM_CHI2_PRODUCT:
        return -1

    with netCDF4.Dataset(regcoil_output_filepath, mode="r") as regcoil_output:
        chi2_B = _get_unmasked_array(regcoil_output, "chi2_B")
        chi2_K = _get_unmasked_array(regcoil_output, "chi2_K")
    return _minimum_chi2_product_lambda_index(chi2_B=chi2_B, chi2_K=chi2_K)


def _read_regcoil_output(
    regcoil_output_filepath: pathlib.Path,
    target_option: regcoil_types.RegcoilTargetOption,
) -> regcoil_types.RegcoilOutput:
    with netCDF4.Dataset(regcoil_output_filepath, mode="r") as regcoil_output:
        chi2_B = _get_unmasked_array(regcoil_output, "chi2_B")
        chi2_K = _get_unmasked_array(regcoil_output, "chi2_K")
        lambda_ = _get_unmasked_array(regcoil_output, "lambda")
        maximum_K = _get_unmasked_array(regcoil_output, "max_K")
        orthogonal_magnetic_field_component = _get_unmasked_array(
            regcoil_output, "Bnormal_total"
        )

        sort_indices = _lambda_sort_indices_for_target_option(
            chi2_B=chi2_B,
            chi2_K=chi2_K,
            target_option=target_option,
        )

        return regcoil_types.RegcoilOutput(
            chi2_B=chi2_B[sort_indices],
            chi2_K=chi2_K[sort_indices],
            lambda_=lambda_[sort_indices],
            maximum_K=maximum_K[sort_indices],
            plasma_major_radius=regcoil_output["R0_plasma"][()],
            coil_winding_surface_major_radius=regcoil_output["R0_coil"][()],
            plasma_area=regcoil_output["area_plasma"][()],
            coil_winding_surface_area=regcoil_output["area_coil"][()],
            plasma_volume=regcoil_output["volume_plasma"][()],
            coil_winding_surface_volume=regcoil_output["volume_coil"][()],
            net_poloidal_current=regcoil_output["net_poloidal_current_Amperes"][()],
            net_toroidal_current=regcoil_output["net_toroidal_current_Amperes"][()],
            orthogonal_magnetic_field_component=orthogonal_magnetic_field_component[
                sort_indices
            ],
            plasma_normal_vector_norm=_get_unmasked_array(
                regcoil_output, "norm_normal_plasma"
            ),
        )


def generate_regcoil_coilset_from_equilibrium(
    equilibrium: vmec_utils.VmecppWOut,
    settings: regcoil_types.RegcoilSettings,
) -> types.Coilset:
    """Gets a coilset from an ideal-MHD equilibrium using REGCOIL.

    Args:
        equilibrium: The ideal-MHD equilibrium.
        settings: The REGCOIL settings.
    """

    _COIL_FILE_NAME = "coils.best"
    with tempfile.TemporaryDirectory(prefix="regcoil_") as temp_dir_str:
        temp_dir = pathlib.Path(temp_dir_str)

        fortran_regcoil_config = _build_regcoil_input_config(
            equilibrium, settings, temp_dir=temp_dir
        )

        regcoil_output_filepath = _run_binary(
            fortran_regcoil_config, path=temp_dir, verbose=settings.verbose
        )

        makegrid_input_filepath = temp_dir / _COIL_FILE_NAME
        lambda_index = _select_lambda_index(
            regcoil_output_filepath=regcoil_output_filepath,
            target_option=settings.target_option,
        )

        regcoil_output_file_to_makegrid_input_file(
            regcoil_output_filepath=regcoil_output_filepath,
            makegrid_input_filepath=makegrid_input_filepath,
            number_of_coils_per_half_period=settings.n_coils_per_half_period,
            lambda_index=lambda_index,
            nescin_output_path=temp_dir,
        )
        desc_any_coilset = DescCoilSet.from_makegrid_coilfile(makegrid_input_filepath)
        assert desc_any_coilset is not None
        desc_fourier_coilset = desc_any_coilset.to_FourierXYZ(settings.n_fourier_modes)

        return coilset_utils.coilstellaration_from_desc(
            desc_coilset=desc_fourier_coilset,
            n_field_periods=equilibrium.n_field_periods,
            is_stellarator_symmetric=not equilibrium.lasym,
        )


__all__ = ["generate_regcoil_coilset_from_equilibrium"]
