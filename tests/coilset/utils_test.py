"""Tests for coilset_utils module."""

from unittest import mock

import numpy as np
import pytest

_SKIP_REASON = (
    "requires constellaration_update.types (Task 8)"
    " and constellaration_update.metrics (Task 4)"
)

constellaration_update_types = pytest.importorskip(
    "constellaration_update.types",
    reason=_SKIP_REASON,
)
coilset_utils = pytest.importorskip(
    "constellaration_update.coilset.utils",
    reason=_SKIP_REASON,
)

pytestmark = pytest.mark.skip(reason=_SKIP_REASON)

try:
    from desc.coils import CoilSet as DescCoilSet
    from desc.equilibrium import Equilibrium as DescEquilibrium
except Exception:  # noqa: BLE001
    DescCoilSet = None  # type: ignore[assignment,misc]
    DescEquilibrium = None  # type: ignore[assignment,misc]


def _make_coilset(n_unique_coils: int, fourier_order: int):  # type: ignore[return]
    """Build a coilset with deterministic, asymmetric coefficients per coil."""
    n_modes = 2 * fourier_order + 1
    base = np.arange(1, n_modes + 1, dtype=np.float64)
    coefs = np.stack([base + 100 * c for c in range(n_unique_coils)], axis=0)
    return constellaration_update_types.ConStellarationUpdateCoilset(
        coil_x_n=coefs,
        coil_y_n=coefs + 0.5,
        coil_z_n=coefs - 0.5,
        currents=np.arange(n_unique_coils, dtype=np.float64) + 1.0,
        n_field_periods=3,
        is_stellarator_symmetric=True,
    )


@pytest.mark.parametrize(("fourier_order", "n_max"), [(1, 1), (2, 4), (3, 6)])
def test_pad_coilset_to_fourier_order_symmetric(fourier_order: int, n_max: int):
    coilset = _make_coilset(n_unique_coils=2, fourier_order=fourier_order)
    padded = coilset_utils.pad_coilset_to_fourier_order(coilset, n_max)

    assert padded.coil_x_n.shape == (2, 2 * n_max + 1)
    pad = n_max - fourier_order
    np.testing.assert_array_equal(padded.coil_x_n[:, :pad], 0.0)
    np.testing.assert_array_equal(
        padded.coil_x_n[:, padded.coil_x_n.shape[1] - pad :], 0.0
    )
    np.testing.assert_allclose(
        padded.coil_x_n[:, pad : pad + 2 * fourier_order + 1], coilset.coil_x_n
    )
    for component_name in ("coil_y_n", "coil_z_n"):
        actual = getattr(padded, component_name)
        expected_central = getattr(coilset, component_name)
        np.testing.assert_array_equal(actual[:, :pad], 0.0)
        np.testing.assert_array_equal(actual[:, actual.shape[1] - pad :], 0.0)
        np.testing.assert_allclose(
            actual[:, pad : pad + 2 * fourier_order + 1], expected_central
        )
    np.testing.assert_allclose(padded.currents, coilset.currents)
    assert padded.n_field_periods == coilset.n_field_periods
    assert padded.is_stellarator_symmetric == coilset.is_stellarator_symmetric


def test_pad_coilset_to_fourier_order_zero_when_already_at_n_max():
    coilset = _make_coilset(n_unique_coils=3, fourier_order=4)
    padded = coilset_utils.pad_coilset_to_fourier_order(coilset, 4)
    np.testing.assert_array_equal(padded.coil_x_n, coilset.coil_x_n)
    np.testing.assert_array_equal(padded.coil_y_n, coilset.coil_y_n)
    np.testing.assert_array_equal(padded.coil_z_n, coilset.coil_z_n)


def test_pad_coilset_to_fourier_order_rejects_shrinking():
    coilset = _make_coilset(n_unique_coils=1, fourier_order=4)
    with pytest.raises(ValueError, match="cannot shrink"):
        coilset_utils.pad_coilset_to_fourier_order(coilset, 2)


@pytest.mark.parametrize(("n", "n_max"), [(1, 4), (2, 2), (0, 3)])
def test_make_fourier_order_mask_shape_and_values(n: int, n_max: int):
    mask = coilset_utils.make_fourier_order_mask(n, n_max)
    assert mask.shape == (2 * n_max + 1,)
    pad = n_max - n
    np.testing.assert_array_equal(mask[:pad], 0.0)
    np.testing.assert_array_equal(mask[mask.shape[0] - pad :], 0.0)
    np.testing.assert_array_equal(mask[pad : pad + 2 * n + 1], 1.0)


def test_coilset_utils_properties():
    """CoilsetDESC exposes correct n_unique_coils and fourier_order."""
    cs = constellaration_update_types.ConStellarationUpdateCoilset(
        coil_x_n=np.zeros((3, 5)),
        coil_y_n=np.zeros((3, 5)),
        coil_z_n=np.zeros((3, 5)),
        currents=np.array([1e5, 2e5, 3e5]),
        n_field_periods=5,
        is_stellarator_symmetric=True,
    )
    assert cs.n_unique_coils == 3
    assert cs.fourier_order == 2


def test_round_trip():
    """CoilsetDESC -> DESC CoilSet -> CoilsetDESC preserves coefficients."""
    original = constellaration_update_types.ConStellarationUpdateCoilset(
        coil_x_n=np.array([[0.2, 0.0, 5.5, 0.5, -0.1], [0.1, 0.0, 5.6, 0.4, 0.0]]),
        coil_y_n=np.array([[0.5, 1.0, 0.0, 0.3, 0.1], [0.4, 0.8, 0.0, 0.2, 0.0]]),
        coil_z_n=np.array([[0.05, 0.3, 0.0, 0.0, 0.0], [0.04, 0.2, 0.0, 0.0, 0.0]]),
        currents=np.array([1e5, 1.1e5]),
        n_field_periods=5,
        is_stellarator_symmetric=True,
    )

    desc_cs = coilset_utils.constellaration_update_to_desc(original)
    result = coilset_utils.constellaration_update_from_desc(
        desc_cs, n_field_periods=5, is_stellarator_symmetric=True
    )

    np.testing.assert_allclose(result.coil_x_n, original.coil_x_n)
    np.testing.assert_allclose(result.coil_y_n, original.coil_y_n)
    np.testing.assert_allclose(result.coil_z_n, original.coil_z_n)
    np.testing.assert_allclose(result.currents, original.currents)


@pytest.mark.parametrize("B_old", [0.5, 2.5, 100])
@pytest.mark.parametrize("NFP", [1, 3])
@pytest.mark.parametrize("sym", [True, False])
def test_scale_coil_currents_to_B_0_equals_1_T(B_old, NFP, sym):
    """Currents are scaled so that the average |B| on the axis becomes 1 T."""
    eq = mock.MagicMock(spec=DescEquilibrium)
    eq.NFP = NFP
    eq.sym = sym

    original_currents = np.array([1e5, 2e5, 3e5])
    coilset = mock.MagicMock(spec=DescCoilSet)

    copied_coilset = mock.MagicMock(spec=DescCoilSet)
    copied_coilset.current = original_currents.copy()
    coilset.copy.return_value = copied_coilset

    with mock.patch.object(
        coilset_utils.metrics_utils,
        "compute_on_axis_average_magnetic_field",
        return_value=B_old,
    ):
        result = coilset_utils.scale_coil_currents_to_B_0_equals_1_T(coilset, eq)

    assert result is copied_coilset
    expected_k = 1.0 / B_old
    np.testing.assert_allclose(result.current, original_currents * expected_k)


def test_scale_coil_currents_returns_copy_not_original():
    """The function operates on a copy, not the original coilset."""
    eq = mock.MagicMock(spec=DescEquilibrium)
    eq.NFP = 1
    eq.sym = False

    coilset = mock.MagicMock(spec=DescCoilSet)

    copied_coilset = mock.MagicMock(spec=DescCoilSet)
    copied_coilset.current = np.array([1e5])
    coilset.copy.return_value = copied_coilset

    with mock.patch.object(
        coilset_utils.metrics_utils,
        "compute_on_axis_average_magnetic_field",
        return_value=1.0,
    ):
        result = coilset_utils.scale_coil_currents_to_B_0_equals_1_T(coilset, eq)

    coilset.copy.assert_called_once()
    assert result is copied_coilset
