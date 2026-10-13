"""
Throughput-weighted spectra with optional Hermite wavelength sparsity, re-exported from
dluxtras (https://github.com/milolangker/dluxtras) so that dLuxToliman sources can use them.

Conventions: wavelengths in metres; spec_weights() = INCIDENT photons (or photon rate)
per bin; filt_weights = mean system throughput per bin; proper_flux() = DETECTED total.
n_coarse = None propagates every wavelength; an integer propagates only n_coarse nodes
and recovers the rest by cubic Hermite interpolation (n_coarse ~ 2K).
"""

from dluxtras.spectra import CombinedBasisSpectrum, CombinedSpectrum, propagate_spectrum

__all__ = ["CombinedSpectrum", "CombinedBasisSpectrum", "propagate_spectrum"]
