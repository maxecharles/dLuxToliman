from jax import numpy as np, config

config.update("jax_debug_nans", True)
import pytest
from dLux import Wavefront, LayeredOpticalSystem
from dLuxToliman import ApplyBasisCLIMB
from dLuxToliman.layers import ApplyBasisClimb

npixels = 8
wavelength = 585e-9
wf = Wavefront(npixels, 1, wavelength)

# CLIMB works on a basis sampled three times finer than the wavefront
basis = np.ones((2, 3 * npixels, 3 * npixels))


def _test_apply(layer):
    assert isinstance(layer.apply(wf), Wavefront)


class TestApplyBasisCLIMB:
    def test_constructor(self):
        layer = ApplyBasisCLIMB(basis, wavelength)
        _test_apply(layer)
        assert np.allclose(layer.coefficients, np.zeros(2))
        assert layer.ideal_wavelength.shape == ()
        assert tuple(int(n) for n in layer.applied_shape) == (npixels, npixels)
        assert ApplyBasisClimb is ApplyBasisCLIMB

    @pytest.mark.parametrize("sign,phase", [(1.0, np.pi), (-1.0, 0.0)])
    def test_binary_phase(self, sign, phase):
        # A positive latent OPD gives a pi phase at the ideal wavelength, and a
        # negative one gives no phase
        layer = ApplyBasisCLIMB(basis, wavelength, sign * np.ones(2))
        assert np.allclose(layer.apply(wf).phase, phase, atol=1e-5)

    def test_in_optical_system(self):
        layer = ApplyBasisCLIMB(basis, wavelength, np.ones(2))
        optics = LayeredOpticalSystem(npixels, 1.0, [("climb", layer)])
        output = optics.propagate_mono(wavelength, return_wf=True)
        assert np.allclose(output.phase, np.pi, atol=1e-5)

    def test_opd_phase_conversion(self):
        layer = ApplyBasisCLIMB(basis, wavelength)
        assert np.allclose(layer.opd_to_phase(wavelength / 2, wavelength), np.pi)
        opd = np.linspace(-1e-7, 1e-7, 5)
        phase = layer.opd_to_phase(opd, wavelength)
        assert np.allclose(layer.phase_to_opd(phase, wavelength), opd)

    def test_total_opd(self):
        layer = ApplyBasisCLIMB(basis, wavelength, np.array([1.0, 2.0]))
        assert np.allclose(layer.get_total_opd(), 3.0)
        assert np.allclose(layer.get_opd(layer.basis, layer.coefficients), 3.0)

    def test_lsq(self):
        # Recovers the coefficients of a plane, a * x + b * y + c
        layer = ApplyBasisCLIMB(basis, wavelength)
        xs, ys = np.meshgrid(np.linspace(0, 1, 3), np.linspace(0, 1, 3))
        plane = 2 * xs + 3 * ys + 1
        assert np.allclose(layer.lsq(plane), np.array([2.0, 3.0, 1.0]), atol=1e-5)

    def test_area(self):
        layer = ApplyBasisCLIMB(basis, wavelength)
        assert np.allclose(layer.area(np.ones((3, 3))), 1)
        assert np.allclose(layer.area(-np.ones((3, 3))), 0)

    def test_CLIMB(self):
        layer = ApplyBasisCLIMB(basis, wavelength)
        latent = np.linspace(-1, 1, (3 * npixels) ** 2).reshape(3 * npixels, -1)
        output = layer.CLIMB(latent, ppsz=npixels)
        assert output.shape == (npixels, npixels)
        assert (output >= 0).all() and (output <= 1).all()

    def test_binary_phase_full_size(self):
        # get_binary_phase is fixed to a 256 pixel output
        layer = ApplyBasisCLIMB(np.ones((1, 768, 768)), wavelength, np.ones(1))
        binary_phase = layer.get_binary_phase()
        assert binary_phase.shape == (256, 256)
        assert np.allclose(binary_phase, np.pi)
