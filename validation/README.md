# Validation

Scripts that reproduce the numbers quoted in [DEVELOPMENT.md](../DEVELOPMENT.md). Run them from the repository root in the `toliman_data` env, unless a row says otherwise. Each one prints the numbers it reproduces and writes nothing.

| Result in DEVELOPMENT.md | Command | Runtime |
| --- | --- | --- |
| PHOENIX spectra vs pysynphot (band flux, 5 nm shape, contrast, `photon_count.md` check) | `PYSYN_CDBS=external/cdbs python validation/spectra_vs_pysynphot.py` (in an env with **pysynphot**, e.g. `pysynphotEnv`) | seconds |
| Out-of-band leak fractions | `python validation/out_of_band_leak.py configs/example.json` | seconds |
| K (constrainable spectral modes) table | `python validation/constrainable_modes.py configs/example.json --n-frames 1` (and `10`, `1000`) | ~10 s each |
| Hermite accuracy and speed (Δχ², node scan) | `python validation/hermite_accuracy.py configs/example.json` | ~10 s |
| Reference point (`centre_weight`) Fisher table | `python validation/reference_point_fisher.py configs/example.json --weights 0 0.2 0.237 0.5 1` | ~40 s |
| Jitter width (config → `GaussianJitter`) | `pytest tests/test_jitter.py` | seconds |

Requirements:
- **CDBS files** (PHOENIX grid, Johnson V, Vega) go in `external/cdbs/` (gitignored). The URLs are in the docstring of [`scripts/make_stellar_spectra.py`](../scripts/make_stellar_spectra.py). `spectra_vs_pysynphot.py` also needs `grid/phoenix/catalog.fits`.
- **`constrainable_modes.py`** uses the spectral Fisher tools of [dluxtras](https://github.com/milolangker/dluxtras), which is already a dependency.

## Expected output

The numbers are for `configs/example.json`: seed 42, 10 observations, read noise 5 e⁻, 0.1 s exposures. The scripts model the *nominal* telescope, without the parameter draws.

**`spectra_vs_pysynphot.py`**
```
A: detected ph/s/m^2  pysynphot 6.3418e+09  ours 6.3450e+09  ratio 1.0005;  5 nm shape ratio 0.989-1.010
B: detected ph/s/m^2  pysynphot 1.9670e+09  ours 1.9666e+09  ratio 0.9998;  5 nm shape ratio 0.990-1.017
contrast A/B in band: pysynphot 3.224, ours 3.226
pysynphot A in photon_count.md units: 6.3418e+05 ph/s/cm^2
```

**`out_of_band_leak.py`**
```
Alpha Cen A:
  threshold  0.01: band   342- 737 nm, outside 0.0067
  threshold  0.03: band   512- 647 nm, outside 0.0250
  threshold  0.10: band   523- 644 nm, outside 0.0316
  < 520 nm: 0.0151   > 650 nm: 0.0123
Alpha Cen B:
  threshold  0.01: band   342- 737 nm, outside 0.0075
  threshold  0.03: band   512- 647 nm, outside 0.0235
  threshold  0.10: band   523- 644 nm, outside 0.0296
  < 520 nm: 0.0119   > 650 nm: 0.0135
```

**`constrainable_modes.py --n-frames N`** (the `Alpha Cen` lines)
```
N = 1:     A: K = 13 at 6.92e+06 detected photons    B: K = 9  at 2.15e+06
N = 10:    A: K = 16 at 6.92e+07                     B: K = 13 at 2.15e+07
N = 1000:  A: K = 20 at 6.92e+09                     B: K = 20 at 2.15e+09
```
K sits on a hard threshold, so it can move by ±1 with small changes to the model, such as a different random flat-field realisation.

**`hermite_accuracy.py`**
```
exact: 300 wavelengths per star, ~1.2 s; delta chi2 over 10 observations
      configured A/B = 32/26  delta chi2 =   0.000711     ~0.3 s  (~4x faster)
                n_coarse = 8  delta chi2 =   1.12e+03
               n_coarse = 12  delta chi2 =       4.67
               n_coarse = 16  delta chi2 =      0.112
               n_coarse = 24  delta chi2 =    0.00332
               n_coarse = 32  delta chi2 =   0.000453
```
Timings depend on the machine.

**`reference_point_fisher.py`**
```
photocentre weight F_B / (F_A + F_B) = 0.237
  weight  sigma_sep [uas]  corr(sep,x)  corr(sep,y)  max|corr| with sep  cond(corr)
   0.000          501.833       -0.435       -0.009               0.435         2.6
   0.200          501.833        0.024       -0.011               0.024         1.6
   0.237          501.833        0.117       -0.011               0.117         1.6
   0.500          501.833        0.617       -0.010               0.617         4.5
   1.000          501.833        0.899       -0.007               0.899        19.6
```

## Scripts vs notebooks

These are plain scripts rather than notebooks. They are easy to diff and review, run headless, and print numbers that can be compared with the text above. A notebook is still the better place for figures. If one is added, it should import from these scripts rather than duplicate them.
