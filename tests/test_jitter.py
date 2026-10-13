import numpy as onp
import pytest

import dLuxToliman as dlT


def _kernel_sigma(layer, pixel_scale=1.0):
    k = onp.asarray(layer.generate_kernel(pixel_scale))
    x = onp.arange(k.shape[0]) - k.shape[0] // 2
    X, Y = onp.meshgrid(x, x)
    return onp.sqrt((k * X**2).sum()), onp.sqrt((k * Y**2).sum())


@pytest.mark.parametrize("kernel_size", [11, 21])
def test_gaussian_kernel_width_matches_sigma(kernel_size):
    # Width must not depend on kernel_size (the old grid was stretched by K / (K - 1))
    sx, sy = _kernel_sigma(dlT.GaussianJitter.from_sigma(1.0, kernel_size=kernel_size))
    assert sx == pytest.approx(1.0, abs=1e-3) and sy == pytest.approx(1.0, abs=1e-3)


def test_r_is_determinant_of_covariance():
    layer = dlT.GaussianJitter.from_sigma(2.0, shear=0.25, phi=30.0)
    onp.testing.assert_allclose(onp.linalg.det(layer.covariance_matrix), layer.r)
    major, minor = onp.sqrt(onp.linalg.eigvalsh(layer.covariance_matrix))[::-1]
    onp.testing.assert_allclose([major, minor], [2.0, 1.5])
