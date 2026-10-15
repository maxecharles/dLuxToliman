from jax import numpy as np, config

config.update("jax_debug_nans", True)
import pytest
import dLux.utils as dlu
from dLux import Wavefront, PSF
from dLuxToliman import AlphaCen


def _test_model(source, optics):
    npixels = optics.psf_npixels * optics.oversample
    psf = source.model(optics)
    assert psf.shape == (npixels, npixels)
    assert 0 < psf.sum() <= 10**source.log_flux
    assert isinstance(source.model(optics, return_wf=True), Wavefront)
    assert isinstance(source.model(optics, return_psf=True), PSF)
    with pytest.raises(ValueError):
        source.model(optics, return_wf=True, return_psf=True)


def test_alpha_cen(create_optics, create_source):
    _test_model(create_source(), create_optics())


def test_defaults():
    source = AlphaCen()
    assert source.wavelengths.shape == (5,)
    assert np.allclose(source.wavelengths[np.array([0, -1])], np.array([530, 640]))
    assert np.allclose(source.weights, 1 / 5)
    assert source.weights.shape == (2, 5)
    assert np.allclose(source.log_flux, 6.979)
    assert np.allclose(source.contrast, 3.37)


def test_bandpass(create_source):
    source = create_source(n_wavels=3, bandpass=(500, 600))
    assert np.allclose(source.wavelengths, np.array([500, 550, 600]))


@pytest.mark.parametrize("position_angle", [0.0, 30.0, 90.0])
def test_xy_positions(create_source, position_angle):
    source = create_source(
        separation=8.0,
        position_angle=position_angle,
        x_position=1.0,
        y_position=-2.0,
    )
    positions = source.xy_positions
    assert positions.shape == (2, 2)

    # The pair is centred on (x_position, y_position)
    assert np.allclose(positions.mean(0), dlu.arcsec2rad(np.array([1.0, -2.0])))

    # A is offset from B by the separation, at the position angle
    phi = dlu.deg2rad(position_angle)
    sep_vec = 8.0 * np.array([np.sin(phi), np.cos(phi)])
    assert np.allclose(positions[0] - positions[1], dlu.arcsec2rad(sep_vec))


def test_raw_fluxes(create_source):
    source = create_source(log_flux=6.0, contrast=3.0)
    flux_a, flux_b = source.raw_fluxes
    assert np.allclose(flux_a + flux_b, 1e6)
    assert np.allclose(flux_a / flux_b, 3.0)


def test_weights(create_source):
    weights = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    source = create_source(n_wavels=3, weights=weights)
    assert np.allclose(source.weights, weights)
    assert np.allclose(source.norm_weights.sum(1), 1)

    normalised = source.normalise()
    assert isinstance(normalised, AlphaCen)
    assert np.allclose(normalised.weights.sum(1), 1)


def test_model_is_linear_in_flux(create_optics, create_source):
    optics = create_optics()
    source = create_source()
    doubled = source.set("log_flux", source.log_flux + np.log10(2.0))
    assert np.allclose(doubled.model(optics), 2 * source.model(optics), rtol=1e-4)
