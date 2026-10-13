"""
Compares data/spectra/alpha_cen_{A,B}.csv with pysynphot's Icat PHOENIX spectra, the
method of photon_count.md, through the same throughput (bandpass x QE x 0.91^2 glass).

    PYSYN_CDBS=external/cdbs python validation/spectra_vs_pysynphot.py

Needs an environment with pysynphot (it does not import toliman_sim), and the CDBS
files downloaded by scripts/make_stellar_spectra.py (plus grid/phoenix/catalog.fits).
Prints, per star: the band-integrated detected photon flux [photons/s/m^2] from both,
their ratio, the range of the 5 nm-binned shape ratio over 520-650 nm, and the in-band
contrast A/B. pysynphot's A value should reproduce photon_count.md (6.342e5 ph/s/cm^2).
"""

import argparse
import os
from pathlib import Path

import numpy as np

STARS = {"A": (5790, 0.20, 4.00, 0.01), "B": (5260, 0.23, 4.37, 1.33)}
GLASS = 0.91**2


def load(path):
    return np.loadtxt(path, delimiter=",", comments="#")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo", default=Path(__file__).parents[1], type=Path)
    args = parser.parse_args()
    if "PYSYN_CDBS" not in os.environ:
        raise SystemExit("Set PYSYN_CDBS to the CDBS folder (e.g. external/cdbs)")
    import pysynphot as S

    cdbs = Path(os.environ["PYSYN_CDBS"])
    v_band = S.FileBandpass(str(cdbs / "comp/nonhst/johnson_v_004_syn.fits"))
    bandpass = load(args.repo / "data/throughput/bandpass.csv")
    qe = load(args.repo / "data/throughput/qe_mono.csv")
    wl = bandpass[:, 0]
    throughput = bandpass[:, 1] * np.interp(wl, qe[:, 0], qe[:, 1], left=0, right=0) * GLASS
    system = S.ArrayBandpass(wl * 10, throughput, waveunits="angstrom")

    detected = {}
    for name, (teff, metallicity, log_g, v_mag) in STARS.items():
        sp = S.Icat("phoenix", teff, metallicity, log_g).renorm(v_mag, "vegamag", v_band)
        sp.convert("photlam")
        reference = S.Observation(sp, system, binset=wl * 10).integrate() * 1e4  # per m^2

        ours = load(args.repo / f"data/spectra/alpha_cen_{name}.csv")
        x = ours[:, 0]
        inside = (x >= wl[0]) & (x <= wl[-1])
        integrand = ours[inside, 1] * np.interp(x[inside], wl, throughput)
        mine = np.sum(np.diff(x[inside]) * (integrand[1:] + integrand[:-1]) / 2)
        detected[name] = (reference, mine)

        edges = np.arange(520, 651, 5.0)
        ratios = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            samples = np.linspace(lo, hi, 50)
            ratios.append(np.interp(samples, x, ours[:, 1]).mean()
                          / (sp.sample(samples * 10) * 1e5).mean())  # photlam -> /m^2/nm
        print(f"{name}: detected ph/s/m^2  pysynphot {reference:.4e}  ours {mine:.4e}  "
              f"ratio {mine / reference:.4f};  5 nm shape ratio {min(ratios):.3f}-{max(ratios):.3f}")

    print(f"contrast A/B in band: pysynphot {detected['A'][0] / detected['B'][0]:.3f}, "
          f"ours {detected['A'][1] / detected['B'][1]:.3f}")
    print(f"pysynphot A in photon_count.md units: {detected['A'][0] / 1e4:.4e} ph/s/cm^2")


if __name__ == "__main__":
    main()
