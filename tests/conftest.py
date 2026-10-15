from __future__ import annotations

import pytest
from dLuxToliman import TolimanOpticalSystem, AlphaCen


@pytest.fixture
def create_optics():
    def _create_optics(
        wf_npixels: int = 64,
        psf_npixels: int = 16,
        oversample: int = 2,
        **kwargs,
    ) -> TolimanOpticalSystem:
        return TolimanOpticalSystem(
            wf_npixels=wf_npixels,
            psf_npixels=psf_npixels,
            oversample=oversample,
            **kwargs,
        )

    return _create_optics


@pytest.fixture
def create_source():
    def _create_source(n_wavels: int = 2, **kwargs) -> AlphaCen:
        return AlphaCen(n_wavels=n_wavels, **kwargs)

    return _create_source
