"""
Fraction of each star's detected photons that falls outside the simulated band.

    python validation/out_of_band_leak.py configs/example.json [--thresholds 0.01 0.03 0.1]

The system throughput and stellar SEDs are taken from the config. For each threshold
(the band is where throughput > threshold x peak) it prints the band and the fraction
of detected photons outside it, then the fractions blueward of 520 nm and redward of
650 nm.
"""

import argparse

import numpy as onp

from toliman_sim import load_config
from toliman_sim.config import config_dir, strip_comments
from toliman_sim.spectra import load_table, resolve, system_throughput


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config")
    parser.add_argument("--thresholds", type=float, nargs="*", default=[0.01, 0.03, 0.1])
    args = parser.parse_args()

    config = load_config(args.config)
    base = config_dir(config)
    spectra = config["source"]["spectra"]
    grid, throughput = system_throughput(spectra["throughput"], base)
    wavelengths = onp.arange(grid[0], grid[-1], 0.1)  # nm
    throughput = onp.interp(wavelengths, grid, throughput)

    for star in ("A", "B"):
        spec = strip_comments(spectra[star])
        if spec["sed"] != "table":
            raise ValueError("Needs SED tables (sed: table) to cover the out-of-band region")
        x, sed = load_table(resolve(spec["path"], base), spec.get("wavelength_unit", "nm"))
        detected = throughput * onp.interp(wavelengths, x, sed, left=0, right=0)
        total = detected.sum()

        print(f"Alpha Cen {star}:")
        for threshold in args.thresholds:
            keep = onp.flatnonzero(throughput > threshold * throughput.max())
            lo, hi = wavelengths[keep[0]], wavelengths[keep[-1]]
            inside = detected[(wavelengths >= lo) & (wavelengths <= hi)].sum()
            print(f"  threshold {threshold:5.2f}: band {lo:5.0f}-{hi:4.0f} nm, "
                  f"outside {1 - inside / total:.4f}")
        blue = detected[wavelengths < 520].sum() / total
        red = detected[wavelengths > 650].sum() / total
        print(f"  < 520 nm: {blue:.4f}   > 650 nm: {red:.4f}")


if __name__ == "__main__":
    main()
