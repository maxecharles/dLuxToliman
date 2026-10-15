import pytest
import dLuxToliman
import dLuxToliman.layers


@pytest.mark.parametrize("module", [dLuxToliman, dLuxToliman.layers])
def test_all_is_flat(module):
    assert len(module.__all__) > 0
    assert all(isinstance(name, str) for name in module.__all__)
    assert len(set(module.__all__)) == len(module.__all__)
    for name in module.__all__:
        assert hasattr(module, name)


def test_star_import():
    namespace = {}
    exec("from dLuxToliman import *", namespace)
    for name in [
        "TolimanOpticalSystem",
        "AlphaCen",
        "Toliman",
        "JitteredToliman",
        "GaussianJitter",
        "SHMJitter",
        "ApplyBasisCLIMB",
        "get_GE",
        "get_RGE",
        "get_RWGE",
        "get_radial_mask",
    ]:
        assert name in namespace


def test_star_import_layers():
    namespace = {}
    exec("from dLuxToliman.layers import *", namespace)
    for name in ["GaussianJitter", "SHMJitter", "ApplyBasisCLIMB"]:
        assert name in namespace
