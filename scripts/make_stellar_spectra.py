"""
Generates PHOENIX spectra of Alpha Cen A and B as photon flux densities at the telescope,
renormalised to their V magnitudes (the method of photon_count.md, but WITHOUT the system
throughput or collecting area: those are applied by the simulator).

    python scripts/make_stellar_spectra.py --cdbs external/cdbs --out data/spectra

Writes alpha_cen_{A,B}.csv: wavelength [nm], photon flux density [photons / s / m^2 / nm].
Use them with {"sed": "table", "path": ...} in a config.

Interpolation. The CDBS PHOENIX grid is high resolution (~1.8 A) at [M/H] = 0.0 but only
~50 A at [M/H] = +0.3. Interpolating in metallicity directly (as pysynphot's Icat does)
would make A and B mostly low resolution. Instead:
    F = F_0.0(Teff, logg)  x  smooth[F_Z(Teff, logg)] / smooth[F_0.0(Teff, logg)]
i.e. the high-resolution solar-metallicity spectrum (bilinear in Teff, logg) times a
low-resolution metallicity correction (linear in [M/H], evaluated on the +0.3 grid).

V normalisation is photon-counting synthetic photometry in the Vega system:
    V = -2.5 log10( int F T lambda dlambda / int F_Vega T lambda dlambda )
with Johnson V (comp/nonhst/johnson_v_004_syn.fits) and Vega (calspec/alpha_lyr_stis_011.fits).

CDBS files needed (https://archive.stsci.edu/hlsps/reference-atlases/cdbs/, Johnson V from
https://ssb.stsci.edu/trds/comp/nonhst/):
    grid/phoenix/phoenix{m00,p03}/phoenix{m00,p03}_{5200,5300,5700,5800}.fits
    comp/nonhst/johnson_v_004_syn.fits
    calspec/alpha_lyr_stis_011.fits
"""

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits

# Stellar parameters from photon_count.md (SIMBAD)
STARS = {
    "A": dict(teff=5790.0, metallicity=0.20, log_g=4.00, v_mag=0.01),
    "B": dict(teff=5260.0, metallicity=0.23, log_g=4.37, v_mag=1.33),
}
TEFF_STEP, LOGG_STEP = 100, 0.5
METALLICITIES = {0.0: "m00", 0.3: "p03"}

H, C = 6.62607015e-34, 2.99792458e8


def read_table(path, value_column):
    data = fits.open(path)[1].data
    return np.asarray(data.field(0), float), np.asarray(data[value_column], float)


def bracket(x, step):
    """Grid points either side of x and the interpolation weight of the upper one."""
    lo = np.floor(x / step) * step
    return lo, lo + step, (x - lo) / step


def phoenix(cdbs, metallicity, teff, log_g, wavelengths=None):
    """Bilinear (Teff, log g) interpolation of one metallicity grid, in FLAM."""
    grid = METALLICITIES[metallicity]
    t_lo, t_hi, wt = bracket(teff, TEFF_STEP)
    g_lo, g_hi, wg = bracket(log_g, LOGG_STEP)
    spectrum = 0.0
    for t, a in ((t_lo, 1 - wt), (t_hi, wt)):
        path = Path(cdbs) / f"grid/phoenix/phoenix{grid}/phoenix{grid}_{int(t)}.fits"
        for g, b in ((g_lo, 1 - wg), (g_hi, wg)):
            if a * b == 0:
                continue
            wave, flux = read_table(path, f"g{int(round(10 * g)):02d}")
            if wavelengths is None:
                wavelengths = wave
            spectrum = spectrum + a * b * np.interp(wavelengths, wave, flux)
    return wavelengths, spectrum


def bin_to(wave, flux, coarse):
    """Averages a high-resolution spectrum onto the bins centred on `coarse`."""
    edges = np.concatenate([[coarse[0]], (coarse[1:] + coarse[:-1]) / 2, [coarse[-1]]])
    idx = np.clip(np.searchsorted(edges, wave) - 1, 0, len(coarse) - 1)
    sums = np.bincount(idx, weights=flux, minlength=len(coarse))
    counts = np.bincount(idx, minlength=len(coarse))
    return np.where(counts > 0, sums / np.maximum(counts, 1), np.interp(coarse, wave, flux))


def stellar_flam(cdbs, teff, metallicity, log_g):
    """High-resolution solar spectrum times a low-resolution metallicity correction (A, FLAM)."""
    wave, solar = phoenix(cdbs, 0.0, teff, log_g)
    coarse, rich = phoenix(cdbs, 0.3, teff, log_g)
    solar_coarse = bin_to(wave, solar, coarse)
    w = metallicity / 0.3
    correction = ((1 - w) * solar_coarse + w * rich) / solar_coarse
    return wave, solar * np.interp(wave, coarse, correction)


def photon_integral(wave, flam, band_wave, band):
    """Photon-counting integral int F T lambda dlambda on the band's support."""
    grid = np.linspace(band_wave[0], band_wave[-1], 20000)
    integrand = np.interp(grid, wave, flam) * np.interp(grid, band_wave, band) * grid
    return np.trapezoid(integrand, grid)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cdbs", default="external/cdbs")
    parser.add_argument("--out", default="data/spectra")
    parser.add_argument("--range-nm", type=float, nargs=2, default=(300.0, 1100.0))
    args = parser.parse_args()

    v_wave, v_band = read_table(Path(args.cdbs) / "comp/nonhst/johnson_v_004_syn.fits", "THROUGHPUT")
    vega_wave, vega = read_table(Path(args.cdbs) / "calspec/alpha_lyr_stis_011.fits", "FLUX")
    vega_v = photon_integral(vega_wave, vega, v_wave, v_band)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lo, hi = args.range_nm
    for name, p in STARS.items():
        wave, flam = stellar_flam(args.cdbs, p["teff"], p["metallicity"], p["log_g"])
        flam *= 10 ** (-0.4 * p["v_mag"]) * vega_v / photon_integral(wave, flam, v_wave, v_band)

        # FLAM [erg/s/cm^2/A] -> photons/s/m^2/nm: x 1e-7 J/erg x 1e4 cm^2/m^2 x 10 A/nm / (hc/lambda)
        wave_nm = wave / 10
        photons = flam * 1e-2 * (wave_nm * 1e-9) / (H * C)

        keep = (wave_nm >= lo) & (wave_nm <= hi)
        path = out / f"alpha_cen_{name}.csv"
        header = (
            f"PHOENIX (CDBS grid) Teff={p['teff']} [M/H]={p['metallicity']} logg={p['log_g']}, "
            f"renormalised to V={p['v_mag']} (Vega, johnson_v_004_syn, alpha_lyr_stis_011).\n"
            "High-resolution [M/H]=0.0 spectrum x low-resolution metallicity correction; "
            "see scripts/make_stellar_spectra.py\n"
            "wavelength_nm,photon_flux_density_photons_per_s_per_m2_per_nm"
        )
        np.savetxt(path, np.column_stack([wave_nm[keep], photons[keep]]),
                   delimiter=",", header=header, fmt=["%.4f", "%.6e"])
        print(f"Wrote {path} ({keep.sum()} samples)")


if __name__ == "__main__":
    main()
