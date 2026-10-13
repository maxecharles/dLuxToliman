"""
Throughput-weighted spectra with optional Hermite wavelength sparsity.

Adapted from milolangker/spectrum (spectrum/spectra.py). Conventions:
- Wavelengths are in metres.
- `spec_weights()` is the INCIDENT photon count (or rate) per wavelength bin,
  before throughput. `filt_weights` is the mean system throughput per bin.
- `proper_flux()` is the DETECTED total, sum(spec_weights * filt_weights).
- `n_coarse=None` propagates every wavelength. An integer propagates only
  n_coarse evenly spread nodes (value and d/dlambda) and recovers the full grid
  by cubic Hermite interpolation. Rule of thumb: n_coarse ~ 2K, where K is the
  number of constrainable spectral modes.
"""

from __future__ import annotations
import dLux
import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp
from jax import Array
from dLux.spectra import SimpleSpectrum

__all__ = ["CombinedSpectrum", "CombinedBasisSpectrum", "propagate_spectrum"]


def coarse_indices(n_wavels, n_coarse):
    """Indices of n_coarse evenly spread nodes in a fine grid, both ends included."""
    if int(n_coarse) != n_coarse or not 2 <= n_coarse <= n_wavels:
        raise ValueError(
            "n_coarse must be an integer between 2 and the number of wavelengths"
        )
    return onp.unique(
        onp.round(onp.linspace(0, n_wavels - 1, int(n_coarse))).astype(int)
    )


def hermite_node_coeffs(wavelengths_fine, weights, wavelengths_coarse):
    """
    Coefficients of coarse-node images and d/dlambda images under cubic Hermite:

    sum_j weights[j] O(lambda_fine[j]) ~= sum_k coeff_val[k] O(lambda_k)
                                          + coeff_der[k] dO/dlambda(lambda_k).

    Fine wavelengths must lie within the coarse range.
    """
    wavelengths_coarse = np.asarray(wavelengths_coarse, dtype=float)
    n_coarse = wavelengths_coarse.size

    idx = np.searchsorted(wavelengths_coarse, wavelengths_fine, side="right") - 1
    idx = np.clip(idx, 0, n_coarse - 2)

    lam0 = wavelengths_coarse[idx]
    dlam = wavelengths_coarse[idx + 1] - lam0
    t = (wavelengths_fine - lam0) / dlam

    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2

    coeff_val = np.zeros(n_coarse, dtype=weights.dtype)
    coeff_val = coeff_val.at[idx].add(weights * h00).at[idx + 1].add(weights * h01)
    coeff_der = np.zeros(n_coarse, dtype=weights.dtype)
    coeff_der = (
        coeff_der.at[idx].add(weights * dlam * h10).at[idx + 1].add(weights * dlam * h11)
    )
    return coeff_val, coeff_der


def propagate_spectrum(optics, spectrum, offset, flux=1.0, return_psf=False):
    """
    Propagates `spectrum` through `optics` at `offset` (radians), scaled so the
    weights sum to `flux`. Honours spectrum.n_coarse (Hermite sparsity); without
    it this is exactly optics.propagate(wavelengths, offset, weights * flux).
    """
    weights = spectrum.weights * flux
    n_coarse = getattr(spectrum, "n_coarse", None)
    if n_coarse is None:
        return optics.propagate(
            spectrum.wavelengths, offset, weights, return_psf=return_psf
        )

    def mono(wavelength):
        wf = optics.propagate_mono(wavelength, offset, return_wf=True)
        return wf.psf, wf.pixel_scale

    def node(wavelength):
        # PSF and its wavelength derivative in one forward-mode pass
        (psf, pixel_scale), (dpsf, _) = jax.jvp(
            mono, (wavelength,), (np.ones_like(wavelength),)
        )
        return psf, dpsf, pixel_scale

    nodes = spectrum.coarse_wavelengths
    coeff_val, coeff_der = hermite_node_coeffs(spectrum.wavelengths, weights, nodes)
    psfs, dpsfs, pixel_scales = jax.vmap(node)(nodes)
    data = np.tensordot(coeff_val, psfs, axes=1) + np.tensordot(coeff_der, dpsfs, axes=1)
    return dLux.PSF(data, pixel_scales.mean()) if return_psf else data


class CombinedSpectrum(SimpleSpectrum):
    """Base class: a source spectrum (spec_weights) times a system throughput."""

    filt_weights: Array
    basis_weights: Array
    n_coarse: int | None = eqx.field(static=True)

    def __init__(self, wavels, filt_weights, basis_weights, n_coarse=None):
        super().__init__(wavels)
        self.filt_weights = np.asarray(filt_weights, float)
        self.basis_weights = np.asarray(basis_weights, dtype=float)
        if self.filt_weights.shape != self.wavelengths.shape:
            raise ValueError("Filter weights and wavelengths must have matching shapes")
        if n_coarse is not None:
            coarse_indices(self.wavelengths.size, n_coarse)
            n_coarse = int(n_coarse)
        self.n_coarse = n_coarse

    @property
    def coarse_wavelengths(self):
        """Wavelengths actually propagated (all of them when n_coarse is None)."""
        if self.n_coarse is None:
            return self.wavelengths
        return self.wavelengths[coarse_indices(self.wavelengths.size, self.n_coarse)]

    def spec_weights(self):
        raise NotImplementedError("Use a concrete CombinedSpectrum subclass")

    @property
    def weights(self):
        weights = self.filt_weights * self.spec_weights()
        return weights / weights.sum()

    @property
    def flux(self):
        """Total incident photons (or photon rate)."""
        return np.sum(self.spec_weights())

    def proper_flux(self):
        """Total detected photons (or photon rate), after throughput."""
        return np.sum(self.spec_weights() * self.filt_weights)

    def normalise(self):
        return self


class CombinedBasisSpectrum(CombinedSpectrum):
    """spec_weights = basis @ (proj @ basis_weights), clipped to be positive."""

    basis_vects: Array
    proj: Array

    def __init__(
        self, wavels, filt_weights, basis_weights, basis, proj=None, n_coarse=None
    ):
        self.basis_vects = np.asarray(basis)
        super().__init__(wavels, filt_weights, basis_weights, n_coarse)
        n = self.basis_weights.shape[0]
        self.proj = np.eye(n) if proj is None else np.asarray(proj)
        if self.basis_vects.shape != (len(self.wavelengths), n):
            raise ValueError("basis must have shape (n_wavelengths, n_coefficients)")
        if self.proj.shape != (n, n):
            raise ValueError("proj must be square with one row per basis coefficient")

    def spec_weights(self):
        return np.maximum(self.basis_vects @ (self.proj @ self.basis_weights), 1e-08)
