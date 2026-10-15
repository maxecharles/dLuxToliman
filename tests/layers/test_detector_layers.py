from jax import numpy as np, config

config.update("jax_debug_nans", True)
import jax
import pytest
import equinox as eqx
import dLux.utils as dlu
from dLux import PSF, LayeredDetector
from dLuxToliman import GaussianJitter, SHMJitter

pixel_scale = 0.375  # arcseconds


def _point_psf(npixels=33):
    data = np.zeros((npixels, npixels)).at[npixels // 2, npixels // 2].set(1.0)
    return PSF(data, dlu.arcsec2rad(pixel_scale))


psf = _point_psf()


def _test_apply(layer):
    # dLux calls the layer directly, and keeps `apply` as an alias
    for output in (layer.apply(psf), layer(psf), eqx.filter_jit(layer)(psf)):
        assert isinstance(output, PSF)
        assert output.data.shape == psf.data.shape

    # It works inside a detector, and keeps the flux of a central source
    output = LayeredDetector([("Jitter", layer)]).model(psf)
    assert np.allclose(output.sum(), 1, atol=1e-5)


def _moments(data):
    """Returns the covariance matrix of an image in arcsec^2, x along the columns."""
    ys, xs = np.indices(data.shape) * pixel_scale
    data = data / data.sum()
    x, y = xs - (data * xs).sum(), ys - (data * ys).sum()
    xy = (data * x * y).sum()
    return np.array([[(data * x**2).sum(), xy], [xy, (data * y**2).sum()]])


class TestGaussianJitter:
    def test_constructor(self):
        layer = GaussianJitter(0.1, 0.2, 30.0, kernel_size=5, kernel_oversample=2)
        _test_apply(layer)
        assert layer.r.shape == ()
        assert layer.shear.shape == ()
        assert layer.phi.shape == ()
        assert layer.kernel_size == 5
        assert layer.kernel_oversample == 2

        with pytest.raises(ValueError):
            GaussianJitter(0.1, kernel_size=4)
        with pytest.raises(ValueError):
            GaussianJitter(0.1, shear=1.0)
        with pytest.raises(ValueError):
            GaussianJitter(0.1, shear=-0.1)

    @pytest.mark.parametrize("r", [0.1, 1.0])
    @pytest.mark.parametrize("shear", [0.0, 0.5])
    @pytest.mark.parametrize("phi", [0.0, 30.0])
    def test_covariance_matrix(self, r, shear, phi):
        cov = GaussianJitter(r, shear, phi).covariance_matrix
        assert cov.shape == (2, 2)
        assert np.allclose(cov, cov.T)

        # r is the determinant, and the shear sets the ratio of the axes
        assert np.allclose(np.linalg.det(cov), r)
        minor, major = np.linalg.eigvalsh(cov)
        assert np.allclose(np.sqrt(minor / major), 1 - shear)

    def test_covariance_orientation(self):
        # Without shear the kernel is round, and a phi of 0 puts the major axis
        # along x
        assert np.allclose(GaussianJitter(1.0).covariance_matrix, np.eye(2))
        cov = GaussianJitter(1.0, 0.5, 0.0).covariance_matrix
        assert cov[0, 0] > cov[1, 1]
        assert np.allclose(cov[0, 1], 0, atol=1e-6)
        rotated = GaussianJitter(1.0, 0.5, 90.0).covariance_matrix
        assert np.allclose(rotated, cov[::-1, ::-1], atol=1e-6)

    def test_transfer_function(self):
        layer = GaussianJitter(1.0, 0.5, 30.0)
        assert np.allclose(layer.transfer_function(0.0, 0.0), 1)
        assert layer.transfer_function(1.0, 1.0) < 1

    @pytest.mark.parametrize("shear,phi", [(0.0, 0.0), (0.3, 30.0), (0.5, -60.0)])
    def test_blur_matches_covariance(self, shear, phi):
        # Blurring a point source gives an image with the kernel's covariance
        layer = GaussianJitter(1.0, shear, phi, kernel_size=31)
        output = layer(_point_psf(65)).data
        assert np.allclose(output.sum(), 1, atol=1e-5)
        assert np.allclose(_moments(output), layer.covariance_matrix, atol=1e-3)

    def test_generate_kernel(self):
        kernel = GaussianJitter(1.0, kernel_size=9).generate_kernel(pixel_scale)
        assert kernel.shape == (9, 9)
        assert np.allclose(kernel.sum(), 1)
        assert np.allclose(kernel, kernel.T, atol=1e-6)
        assert np.allclose(kernel, kernel[::-1, ::-1], atol=1e-6)
        assert kernel[4, 4] == kernel.max()

    def test_gradient(self):
        def loss(r):
            return GaussianJitter(0.1).set("r", r)(psf).data[16, 16]

        assert np.isfinite(jax.grad(loss)(np.array(0.1)))


class TestSHMJitter:
    def test_constructor(self):
        layer = SHMJitter(0.5, 30.0, kernel_size=7)
        _test_apply(layer)
        assert layer.kernel_size == 7
        assert np.allclose(layer.A, 0.5)
        assert np.allclose(layer.phi, 30.0)

        with pytest.raises(ValueError):
            SHMJitter(-0.5, 0.0)

    @pytest.mark.parametrize("A", [0.1, 1.0, 2.5])
    def test_optimal_kernel_size(self, A):
        kernel_size = SHMJitter(A, 0.0).optimal_kernel_size(pixel_scale)
        assert kernel_size % 2 == 1
        assert kernel_size >= 2 * A / pixel_scale

    def test_machine_epsilon(self):
        assert SHMJitter.machine_epsilon(np.float32) == np.finfo(np.float32).eps

    def test_CDF(self):
        layer = SHMJitter(2.0, 0.0)
        assert np.allclose(layer.CDF(np.array([-2.0, 0.0, 2.0])), np.pi * np.array([-0.5, 0, 0.5]))

    def test_generate_kernel(self):
        kernel = SHMJitter(1.0, 0.0, kernel_size=9).generate_kernel(pixel_scale)
        assert kernel.shape == (9, 9)
        assert np.allclose(kernel.sum(), 1)

        # A phi of 0 oscillates along x, symmetric about the centre
        assert np.allclose(kernel[4].sum(), 1)
        assert np.allclose(kernel, kernel[:, ::-1], atol=1e-6)

        # A phi of 90 oscillates along y
        rotated = SHMJitter(1.0, 90.0, kernel_size=9).generate_kernel(pixel_scale)
        assert np.allclose(rotated, kernel.T, atol=1e-6)

    def test_gradient(self):
        def loss(A):
            return SHMJitter(1.0, 0.0).set("A", A)(psf).data[16, 16]

        assert np.isfinite(jax.grad(loss)(np.array(1.0)))
