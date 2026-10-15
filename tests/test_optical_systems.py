from jax import numpy as np, config

config.update("jax_debug_nans", True)
import jax
import pytest
from dLuxToliman import (
    TolimanOpticalSystem,
)
from dLux import Wavefront, PSF
from dLux.layers import Optic, TransmissiveLayer, BasisOptic, AberratedLayer

wavelength = 585e-9


def _centroid(psf):
    """Returns the (x, y) centroid in pixels, with x along the columns."""
    ys, xs = np.indices(psf.shape)
    return (psf * xs).sum() / psf.sum(), (psf * ys).sum() / psf.sum()


def _test_propagate_mono(optics):
    npixels = optics.psf_npixels * optics.oversample
    psf = optics.propagate_mono(wavelength)
    assert psf.shape == (npixels, npixels)
    assert 0 < psf.sum() <= 1
    assert isinstance(optics.propagate_mono(wavelength, return_wf=True), Wavefront)


def _test_propagate(optics):
    wavels = np.array([530e-9, 640e-9])
    npixels = optics.psf_npixels * optics.oversample
    assert optics.propagate(wavels).shape == (npixels, npixels)
    assert isinstance(optics.propagate(wavels, return_wf=True), Wavefront)
    assert isinstance(optics.propagate(wavels, return_psf=True), PSF)
    with pytest.raises(ValueError):
        optics.propagate(wavels, return_wf=True, return_psf=True)


def test_toliman_optical_system(create_optics):
    optics = create_optics()
    _test_propagate_mono(optics)
    _test_propagate(optics)

    assert list(optics.layers.keys()) == ["aperture", "pupil"]
    assert isinstance(optics.layers["aperture"], TransmissiveLayer)
    assert isinstance(optics.layers["pupil"], AberratedLayer)
    assert np.allclose(optics.diameter, 0.125)
    assert optics.psf_npixels == 16
    assert optics.oversample == 2
    assert np.allclose(optics.psf_pixel_scale, 0.375)


def test_default_construction():
    optics = TolimanOpticalSystem()
    assert optics.wf_npixels == 256
    assert optics.psf_npixels == 128
    assert optics.oversample == 4


def test_diffractive_pupil_is_binary(create_optics):
    # A pi phase step at the 585 nm design wavelength, as an OPD
    opd = create_optics().layers["pupil"].opd
    assert np.allclose(np.unique(opd), np.array([0.0, 585e-9 / 2]))


def test_custom_mask(create_optics):
    mask = Optic()
    optics = create_optics(mask=mask)
    assert optics.layers["pupil"] is mask
    _test_propagate_mono(optics)


@pytest.mark.parametrize(
    "radial_orders,n_terms",
    [
        ([0], 1),
        ([1], 2),
        ([2, 3], 7),
        ([1, 3], 6),
    ],
)
def test_radial_orders(create_optics, radial_orders, n_terms):
    optics = create_optics(radial_orders=radial_orders)
    aperture = optics.layers["aperture"]
    assert isinstance(aperture, BasisOptic)
    assert aperture.basis.shape == (n_terms, 64, 64)
    assert np.allclose(aperture.coefficients, 0)
    _test_propagate_mono(optics)


def test_radial_orders_match_noll_indices(create_optics):
    # Radial order 1 is tip and tilt, Noll indices 2 and 3
    from_orders = create_optics(radial_orders=[1]).layers["aperture"].basis
    from_noll = create_optics(noll_indices=[2, 3]).layers["aperture"].basis
    assert np.allclose(from_orders, from_noll)


def test_negative_radial_order(create_optics):
    with pytest.raises(ValueError):
        create_optics(radial_orders=[-1])


def test_coefficients(create_optics):
    coefficients = np.array([1e-8, -2e-8])
    optics = create_optics(noll_indices=[4, 5], coefficients=coefficients)
    assert np.allclose(optics.layers["aperture"].coefficients, coefficients)


def test_zero_coefficients_match_unaberrated(create_optics):
    aberrated = create_optics(radial_orders=[2, 3]).propagate_mono(wavelength)
    unaberrated = create_optics().propagate_mono(wavelength)
    assert np.allclose(aberrated, unaberrated, atol=1e-6)


def test_coefficient_gradients(create_optics):
    optics = create_optics(radial_orders=[2])

    def loss(coefficients):
        return optics.set("aperture.coefficients", coefficients).propagate_mono(
            wavelength
        )[0, 0]

    grads = jax.grad(loss)(np.full(3, 1e-8))
    assert grads.shape == (3,)
    assert np.isfinite(grads).all()


def test_tilt_sign_convention(create_optics):
    """
    Locks in the sign convention of dLux 0.14, where a positive tilt
    coefficient (Noll index 2) moves the PSF towards -x. dLux 0.15 reverses
    the sign of every OPD, which moves the PSF the other way and changes the
    PSF of the diffractive pupil away from 585 nm. If this test fails after a
    dLux upgrade, the change in convention is deliberate only once this test
    is updated to match.
    """
    tilt = 1e-7  # metres of OPD, a shift of roughly three pixels

    def x_centroid(coefficient):
        optics = create_optics(
            mask=Optic(),
            noll_indices=[2, 3],
            coefficients=np.array([coefficient, 0.0]),
        )
        return _centroid(optics.propagate_mono(wavelength))[0]

    assert x_centroid(tilt) < x_centroid(0.0) - 1
    assert x_centroid(-tilt) > x_centroid(0.0) + 1
