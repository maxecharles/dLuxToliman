from jax import numpy as np, config

config.update("jax_debug_nans", True)
import pytest
import dLux.utils as dlu
from dLuxToliman import get_GE, get_RGE, get_RWGE, get_radial_mask

npixels = 15
x, y = dlu.pixel_coords(npixels, npixels)  # x along columns, in pixels
radii = np.hypot(x, y)

# np.gradient is exact for quadratics away from the edges
interior = (slice(1, -1), slice(1, -1))


@pytest.mark.parametrize("fn", [get_GE, get_RGE, get_RWGE])
def test_constant_has_no_gradient_energy(fn):
    output = fn(np.ones((npixels, npixels)))
    assert output.shape == (npixels, npixels)
    assert np.allclose(output, 0)


def test_get_GE():
    # A ramp of slope 2 along x and 3 along y
    assert np.allclose(get_GE(2 * x + 3 * y), np.hypot(2, 3))


def test_get_RGE():
    # For r^2 the radial gradient is 2r, weighted by r and squared
    assert np.allclose(get_RGE(radii**2)[interior], (4 * radii**4)[interior])


def test_get_RWGE():
    # For r^2 the radial gradient is 2r, squared
    output = get_RWGE(radii**2)
    centre = npixels // 2
    output = output.at[centre, centre].set(0.0)  # undefined at the centre
    expected = (4 * radii**2).at[centre, centre].set(0.0)
    assert np.allclose(output[interior], expected[interior], rtol=1e-5)


def test_get_RGE_anisotropic():
    # For x * y, x df/dx + y df/dy = 2xy, which needs x and y the right way round
    assert np.allclose(get_RGE(x * y)[interior], (4 * x**2 * y**2)[interior])


def test_get_radial_mask():
    mask = get_radial_mask(npixels, 2.0, 5.0)
    assert mask.shape == (npixels, npixels)
    assert set(np.unique(mask).tolist()) == {0.0, 1.0}
    assert np.allclose(mask, (radii > 2.0) & (radii < 5.0))

    centre = npixels // 2
    assert mask[centre, centre] == 0  # inside rmin
    assert mask[centre, centre + 3] == 1  # between rmin and rmax
    assert mask[0, 0] == 0  # outside rmax
