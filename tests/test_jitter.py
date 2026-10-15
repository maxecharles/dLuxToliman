"""Checks that the configured jitter reaches dLuxToliman.GaussianJitter with the right width."""

import copy

import dLux
import jax.numpy as np
import numpy as onp
import pytest

from toliman_sim import build_optics
from toliman_sim.builders import _jitter

from test_simulate import SMALL_CONFIG


def _blur_moments(layer, pixel_scale_arcsec=0.01, npix=129):
    """Second moments [arcsec^2] of a point blurred by the layer (Fourier-space blur)."""
    delta = np.zeros((npix, npix)).at[npix // 2, npix // 2].set(1.0)
    psf = dLux.PSF(delta, dLux.utils.arcsec2rad(pixel_scale_arcsec))
    image = onp.asarray(layer(psf).data)
    x = (onp.arange(npix) - npix // 2) * pixel_scale_arcsec
    X, Y = onp.meshgrid(x, x)
    total = image.sum()
    return (image * X**2).sum() / total, (image * Y**2).sum() / total, (image * X * Y).sum() / total


@pytest.mark.parametrize("shear, phi", [(0.0, 0.0), (0.5, 30.0)])
def test_jitter_width_matches_config(shear, phi):
    config = copy.deepcopy(SMALL_CONFIG)
    spec = {"name": "high_frequency_jitter", "kernel": "gaussian", "sigma_mas": 50.0,
            "shear": shear, "phi_deg": phi}
    layer = _jitter(spec, build_optics(config), config, None)

    sigma_major, sigma_minor = 0.05, 0.05 * (1 - shear)
    onp.testing.assert_allclose(onp.sqrt(onp.linalg.eigvalsh(layer.covariance_matrix))[::-1],
                                [sigma_major, sigma_minor], rtol=1e-6)

    xx, yy, xy = _blur_moments(layer)
    c, s = onp.cos(onp.radians(phi)), onp.sin(onp.radians(phi))
    R = onp.array([[c, -s], [s, c]])
    expected = R @ onp.diag([sigma_major**2, sigma_minor**2]) @ R.T
    onp.testing.assert_allclose([xx, yy, xy], [expected[0, 0], expected[1, 1], expected[0, 1]],
                                rtol=2e-2, atol=1e-6)
