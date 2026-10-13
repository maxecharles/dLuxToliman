"""
Building the photon-rate spectra of Alpha Cen A and B from a config.

    stellar SED [photons / s / m^2 / nm]  x  collecting area  ->  spec_weights
    system throughput (filter x QE x glass)                   ->  filt_weights

Both are binned onto the same n_wavels fine bins across the band. The result is
a dLuxToliman.CombinedBasisSpectrum per star (identity basis: one coefficient
per bin), so detected photons per frame = exposure_time * proper_flux().
"""

from __future__ import annotations
from pathlib import Path
import numpy as onp
import dLuxToliman as dlT

from .config import strip_comments

H = 6.62607015e-34  # J s
C = 2.99792458e8  # m / s
K_B = 1.380649e-23  # J / K

# Vega (V = 0) flux density at 550 nm, Bessell et al. (1998): 3.631e-11 W / m^2 / nm,
# converted to photons / s / m^2 / nm
V_ZERO_POINT = 3.631e-11 / (H * C / 550e-9)

_UNITS = {"angstrom": 0.1, "nm": 1.0, "um": 1e3, "m": 1e9}


def blackbody_sed(wavelengths_nm, teff_k, v_mag):
    """
    Simple SED for tests: a blackbody photon spectrum [photons / s / m^2 / nm],
    normalised so its flux density at 550 nm matches the V magnitude.
    """
    lam = onp.asarray(wavelengths_nm, float) * 1e-9
    photon_planck = (2 * C / lam**4) / onp.expm1(H * C / (lam * K_B * teff_k))
    lam_v = 550e-9
    planck_v = (2 * C / lam_v**4) / onp.expm1(H * C / (lam_v * K_B * teff_k))
    return V_ZERO_POINT * 10 ** (-0.4 * v_mag) * photon_planck / planck_v


def load_table(path, wavelength_unit="nm"):
    """Two-column table (comma or whitespace separated, # comments) -> (nm, values)."""
    with open(path) as f:
        delimiter = "," if "," in f.read() else None
    table = onp.loadtxt(path, comments="#", delimiter=delimiter, ndmin=2)
    if table.shape[1] != 2:
        raise ValueError(f"{path}: expected two columns (wavelength, value)")
    return table[:, 0] * _UNITS[wavelength_unit], table[:, 1]


def bin_mean(x, y, edges, n_sub=32):
    """Mean of the linearly interpolated y(x) within each bin."""
    samples = onp.linspace(edges[:-1], edges[1:], n_sub, axis=-1)
    return onp.interp(samples, x, y, left=0.0, right=0.0).mean(-1)


def resolve(path, base=None) -> Path:
    """Relative input paths are resolved against `base` (the config file's folder)."""
    path = Path(path)
    return path if path.is_absolute() or base is None else Path(base) / path


def system_throughput(spec: dict, base=None):
    """
    Product of the throughput components [(nm, fraction)] on the union of their
    wavelength grids. Components: {"path": table}, {"tophat_nm": [lo, hi]} or
    {"constant": value}. Outside a table's range its throughput is zero.
    """
    tables, constant = [], 1.0
    for component in spec["components"]:
        component = strip_comments(component)
        if "path" in component:
            path = resolve(component["path"], base)
            tables.append(load_table(path, component.get("wavelength_unit", "nm")))
        elif "tophat_nm" in component:
            tables.append((onp.asarray(component["tophat_nm"], float), onp.ones(2)))
        elif "constant" in component:
            constant *= component["constant"]
        else:
            raise ValueError(f"Unknown throughput component {component}")
    if not tables:
        raise ValueError("Throughput needs at least one table or top-hat component")

    grid = onp.unique(onp.concatenate([x for x, _ in tables]))
    throughput = constant * onp.ones_like(grid)
    for x, y in tables:
        throughput *= onp.interp(grid, x, y, left=0.0, right=0.0)
    return grid, throughput


def band_edges(spec: dict, grid, throughput, n_wavels: int):
    """
    n_wavels + 1 evenly spaced bin edges [nm] across the band: `band_nm` if given,
    else where the system throughput exceeds `threshold` x peak.
    """
    if spec.get("band_nm") is not None:
        lo, hi = spec["band_nm"]
    else:
        keep = onp.flatnonzero(throughput > spec.get("threshold", 0.03) * throughput.max())
        lo, hi = grid[keep[0]], grid[keep[-1]]
    return onp.linspace(lo, hi, n_wavels + 1)


def sed_per_bin(spec: dict, edges, base=None):
    """Incident photon flux per bin [photons / s / m^2] for one star."""
    if spec["sed"] == "blackbody":
        fine = onp.linspace(edges[0], edges[-1], 32 * (len(edges) - 1))
        x, y = fine, blackbody_sed(fine, spec["teff_k"], spec["v_mag"])
    elif spec["sed"] == "table":
        x, y = load_table(resolve(spec["path"], base), spec.get("wavelength_unit", "nm"))
    else:
        raise ValueError(f"Unknown sed '{spec['sed']}'. Known: blackbody, table")
    return bin_mean(x, y, edges) * onp.diff(edges)


def collecting_area(optics: dlT.TolimanOpticalSystem) -> float:
    """Collecting area [m^2] of the modelled aperture (struts included)."""
    pixel_area = (optics.diameter / optics.wf_npixels) ** 2
    return float(optics.aperture.transmission.sum() * pixel_area)


def build_spectra(
    spectra: dict, optics: dlT.TolimanOpticalSystem, base=None
) -> dict:
    """
    Returns {"A": CombinedBasisSpectrum, "B": CombinedBasisSpectrum} (photon rates).
    Relative paths are resolved against `base`.
    """
    spectra = strip_comments(spectra)
    n_wavels = spectra.get("n_wavels", 300)
    grid, system = system_throughput(spectra["throughput"], base)
    edges = band_edges(spectra["throughput"], grid, system, n_wavels)
    throughput = bin_mean(grid, system, edges)
    wavelengths = 1e-9 * (edges[:-1] + edges[1:]) / 2  # metres
    area = collecting_area(optics)

    out = {}
    for star in ("A", "B"):
        spec = strip_comments(spectra[star])
        rates = sed_per_bin(spec, edges, base) * area  # incident photons / s per bin
        out[star] = dlT.CombinedBasisSpectrum(
            wavels=wavelengths,
            filt_weights=throughput,
            basis_weights=rates,
            basis=onp.eye(n_wavels),
            n_coarse=spec.get("n_coarse", spectra.get("n_coarse")),
        )
    return out
