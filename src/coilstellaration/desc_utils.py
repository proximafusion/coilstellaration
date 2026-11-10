import pathlib
import tempfile

import desc.vmec
import numpy as np
from constellaration.geometry import surface_rz_fourier, surface_utils_desc
from constellaration.mhd import vmec_utils
from desc.equilibrium import Equilibrium
from desc.profiles import SplineProfile

from coilstellaration import types


def desc_equilibrium_from_vmecpp_wout(
    vmec_equilibrium: vmec_utils.VmecppWOut,
) -> Equilibrium:
    with tempfile.TemporaryDirectory() as temp_dir_name:
        temp_directory = pathlib.Path(temp_dir_name)
        vmec_input_path = temp_directory / "vmec_input.nc"
        vmec_equilibrium.save(vmec_input_path)

        return desc.vmec.VMECIO.load(vmec_input_path, profile="current")


def desc_equilibrium_from_boundary(
    boundary: surface_rz_fourier.SurfaceRZFourier,
) -> types.DescEquilibrium:
    """An unsolved equilibrium built from `boundary` alone, with zero pressure and
    current, as `metrics_utils_v2.get_cached_equilibrium` builds it.

    Its minor radius and axis can differ from those of the equilibrium solved for the
    boundary, so metrics evaluated on it may not match the dataset's.
    """
    # DESC's default profiles for a boundary-only equilibrium are zero power series,
    # which `DescEquilibrium` cannot store; zero splines evaluate identically.
    return desc_equilibrium_from_desc_object(
        Equilibrium(
            surface=surface_utils_desc.to_desc_fourier_rz_toroidal_surface(boundary),
            pressure=_zero_spline_profile("pressure"),
            current=_zero_spline_profile("current"),
            check_orientation=False,
            ensure_nested=False,
        )
    )


def desc_equilibrium_to_desc_object(
    desc_equilibrium: types.DescEquilibrium,
) -> Equilibrium:
    eq = Equilibrium(
        R_lmn=desc_equilibrium.r_lmn,
        Z_lmn=desc_equilibrium.z_lmn,
        L_lmn=desc_equilibrium.lambda_lmn,
        L=desc_equilibrium.radial_resolution,
        M=desc_equilibrium.poloidal_resolution,
        N=desc_equilibrium.toroidal_resolution,
        spectral_indexing=desc_equilibrium.spectral_indexing,
        Psi=desc_equilibrium.total_toroidal_flux,
        NFP=desc_equilibrium.n_field_periods,
        sym=desc_equilibrium.is_stellarator_symmetric,
        check_orientation=False,
        pressure=_spline_profile_to_desc_object(desc_equilibrium.pressure_profile),
        current=_spline_profile_to_desc_object(desc_equilibrium.current_profile),
        iota=_spline_profile_to_desc_object(desc_equilibrium.iota_profile),
    )
    # The constructor takes no axis or boundary coefficients, so it would derive its
    # own instead of the ones the equilibrium was solved with.
    if desc_equilibrium.r_axis is not None:
        eq.Ra_n = desc_equilibrium.r_axis
    if desc_equilibrium.z_axis is not None:
        eq.Za_n = desc_equilibrium.z_axis
    if desc_equilibrium.r_boundary_lmn is not None:
        eq.Rb_lmn = desc_equilibrium.r_boundary_lmn
    if desc_equilibrium.z_boundary_lmn is not None:
        eq.Zb_lmn = desc_equilibrium.z_boundary_lmn
    return eq


def desc_equilibrium_from_desc_object(eq: Equilibrium) -> types.DescEquilibrium:
    pressure = _spline_profile_from_desc_object(eq.pressure)
    if pressure is None:
        raise ValueError("The DESC equilibrium has no pressure profile.")
    current = _spline_profile_from_desc_object(eq.current)
    return types.DescEquilibrium(
        r_lmn=np.asarray(eq.R_lmn),
        z_lmn=np.asarray(eq.Z_lmn),
        lambda_lmn=np.asarray(eq.L_lmn),
        radial_resolution=int(eq.L),
        poloidal_resolution=int(eq.M),
        toroidal_resolution=int(eq.N),
        spectral_indexing=str(eq.spectral_indexing),
        n_field_periods=int(eq.NFP),
        is_stellarator_symmetric=bool(eq.sym),
        total_toroidal_flux=float(eq.Psi),
        pressure_profile=pressure,
        current_profile=current,
        # A current profile fixes iota, so DESC rejects an equilibrium with both.
        iota_profile=None
        if current is not None
        else _spline_profile_from_desc_object(eq.iota),
        r_axis=np.asarray(eq.Ra_n),
        z_axis=np.asarray(eq.Za_n),
        r_boundary_lmn=np.asarray(eq.Rb_lmn),
        z_boundary_lmn=np.asarray(eq.Zb_lmn),
    )


def _zero_spline_profile(name: str) -> SplineProfile:
    return SplineProfile(values=np.zeros(2), knots=np.array([0.0, 1.0]), name=name)


def _spline_profile_to_desc_object(
    profile: types.DescSplineProfile | None,
) -> SplineProfile | None:
    if profile is None:
        return None
    return SplineProfile(
        values=profile.params,
        knots=profile.knots,
        method=profile.method,
        name=profile.name,
    )


def _spline_profile_from_desc_object(profile: object) -> types.DescSplineProfile | None:
    if profile is None:
        return None
    if not isinstance(profile, SplineProfile):
        raise TypeError(
            f"Only DESC SplineProfiles can be stored, got {type(profile).__name__}."
        )
    return types.DescSplineProfile(
        method=profile._method,
        name=profile.name,
        knots=np.asarray(profile.knots),
        params=np.asarray(profile.params),
    )
