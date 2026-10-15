from jax import numpy as np, config

config.update("jax_debug_nans", True)
import jax
import pytest
from dLux import PSF, LayeredDetector
from dLux.layers import Downsample, Optic
from dLuxToliman import Toliman, JitteredToliman, AlphaCen


@pytest.fixture
def telescope(create_optics, create_source):
    return Toliman(create_optics(), create_source())


def _test_model(telescope):
    npixels = telescope.psf_npixels * telescope.oversample
    assert telescope.model().shape == (npixels, npixels)
    assert isinstance(telescope.model(return_psf=True), PSF)


def _variances(psf):
    """Returns the (x, y) variances in pixels^2, with x along the columns."""
    ys, xs = np.indices(psf.shape)
    psf = psf / psf.sum()
    x_mean, y_mean = (psf * xs).sum(), (psf * ys).sum()
    return (psf * (xs - x_mean) ** 2).sum(), (psf * (ys - y_mean) ** 2).sum()


class TestToliman:
    def test_model(self, telescope):
        _test_model(telescope)

    def test_detector(self, telescope):
        detector = LayeredDetector([("Downsample", Downsample(2))])
        telescope = telescope.set("detector", detector)
        assert telescope.model().shape == (16, 16)

    def test_getattr(self, telescope):
        # Attributes are looked up on the optics and the source
        assert telescope.psf_npixels == 16
        assert np.allclose(telescope.separation, 10.0)
        with pytest.raises(AttributeError):
            telescope.not_an_attr

    def test_normalise(self, create_optics, create_source):
        weights = np.array([[1.0, 2.0], [3.0, 4.0]])
        telescope = Toliman(create_optics(), create_source(weights=weights))
        normalised = telescope.normalise()
        assert isinstance(normalised, Toliman)
        assert isinstance(normalised.source, AlphaCen)
        assert np.allclose(normalised.source.weights.sum(1), 1)

    def test_gradient(self, telescope):
        def loss(separation):
            return telescope.set("separation", separation).model().sum()

        assert np.isfinite(jax.grad(loss)(np.array(10.0)))


@pytest.fixture
def create_jittered(create_optics, create_source):
    def _create_jittered(
        jitter_mag=0.5, jitter_angle=30.0, optics=None, source=None, **kwargs
    ):
        # The test field is only 6 arcseconds across, so keep the stars close
        optics = create_optics() if optics is None else optics
        source = create_source(separation=2.0) if source is None else source
        return JitteredToliman(optics, source, jitter_mag, jitter_angle, **kwargs)

    return _create_jittered


class TestJitteredToliman:
    def test_constructor(self, create_jittered):
        telescope = create_jittered(n_psfs=4)
        assert telescope.jitter_mag.shape == ()
        assert telescope.jitter_angle.shape == ()
        assert telescope.n_psfs == 4
        assert telescope.jitter_shape == "linear"
        _test_model(telescope)

        with pytest.raises(ValueError):
            create_jittered(n_psfs=1)
        with pytest.raises(ValueError):
            create_jittered(jitter_shape="gaussian")

    @pytest.mark.parametrize("jitter_shape", ["linear", "shm"])
    def test_jitter_model(self, create_jittered, jitter_shape):
        telescope = create_jittered(n_psfs=4, jitter_shape=jitter_shape)
        psf = telescope.jitter_model()
        assert psf.shape == telescope.model().shape

        # Jitter moves the light around but keeps the flux
        assert np.allclose(psf.sum(), telescope.model().sum(), rtol=1e-2)

    @pytest.mark.parametrize("jitter_shape", ["linear", "shm"])
    def test_small_jitter_matches_model(self, create_jittered, jitter_shape):
        telescope = create_jittered(
            jitter_mag=1e-4, n_psfs=3, jitter_shape=jitter_shape
        )
        model = telescope.model()
        assert np.allclose(telescope.jitter_model(), model, atol=1e-4 * model.max())

    @pytest.mark.parametrize("jitter_angle,direction", [(0.0, "x"), (90.0, "y")])
    def test_jitter_angle(
        self, create_optics, create_source, create_jittered, jitter_angle, direction
    ):
        # A single compact PSF without the diffractive pupil, so the smear is easy
        # to see
        telescope = create_jittered(
            jitter_mag=1.0,
            jitter_angle=jitter_angle,
            optics=create_optics(mask=Optic()),
            source=create_source(separation=0.0),
            n_psfs=5,
        )
        jittered_x, jittered_y = _variances(telescope.jitter_model())
        unjittered_x, unjittered_y = _variances(telescope.model())
        growth = {"x": jittered_x - unjittered_x, "y": jittered_y - unjittered_y}

        # An angle of 0 smears along x, and 90 along y
        other = "y" if direction == "x" else "x"
        assert growth[direction] > 1
        assert np.abs(growth[other]) < 0.1 * growth[direction]

    @pytest.mark.parametrize("jitter_shape", ["linear", "shm"])
    def test_gradient(self, create_jittered, jitter_shape):
        telescope = create_jittered(n_psfs=4, jitter_shape=jitter_shape)

        def loss(jitter_mag):
            return telescope.set("jitter_mag", jitter_mag).jitter_model().sum()

        assert np.isfinite(jax.grad(loss)(np.array(0.5)))

    def test_get_bounds(self):
        bounds = JitteredToliman._get_bounds(np.array([0.0, 1.0, 2.0, 3.0]))
        assert np.allclose(bounds, np.array([0.0, 0.5, 1.5, 2.5, 3.0]))

    def test_inv_shm(self, create_jittered):
        telescope = create_jittered(jitter_mag=2.0)
        assert np.allclose(telescope.inv_shm(np.array(1.0)), np.pi / 2)
        assert np.allclose(telescope.inv_shm(np.array(-1.0)), -np.pi / 2)
        assert np.allclose(telescope.inv_shm(np.array(0.0)), 0.0)

    def test_machine_epsilon(self):
        assert JitteredToliman._machine_epsilon(np.float32) == np.finfo(np.float32).eps
