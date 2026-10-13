# toliman_data

Simulated TOLIMAN data at varying levels of fidelity, built on [dLuxToliman](https://github.com/maxecharles/dLuxToliman).

You describe a simulation in a JSON config: the optics, the source, which detector effects to include, the noise and the output format. The simulator writes out a stack of simulated frames, the noiseless "truth" image and a copy of the config that produced them.

This repository is a fork of [`maxecharles/dLuxToliman`](https://github.com/maxecharles/dLuxToliman). The `dLuxToliman/` package keeps its upstream models, plus the additions listed under [Changes to dLuxToliman](#changes-to-dluxtoliman). The simulator itself lives in `toliman_sim/`.

## Quickstart

```bash
conda activate toliman_data
pip install -e ".[fits,scripts,test]"   # dLux 0.15.x, JAX, astropy, openpyxl, pytest
python -m toliman_sim configs/example.json
pytest
```

This writes `simulated_data/sim_run_0001/` (the folder is gitignored), containing:

- `sim_run_0001.fits`
  - Primary HDU `FRAMES`: the noisy frames, shape `(n_frames, ny, nx)`, in electrons.
  - `EXPECTED`: the noiseless expected image.
- `config.json`: a copy of the config, kept for provenance.

From Python:

```python
from toliman_sim import load_config, simulate

result = simulate(load_config("configs/example.json"))
result["frames"], result["expected"], result["telescope"]  # telescope is a dLuxToliman.Toliman
```

## How it works

```
config.json
   │
   ├─ optics  ──► dLuxToliman.TolimanOpticalSystem ─┐
   ├─ source  ──► dLuxToliman.SpectralAlphaCen ─────┼─► dLuxToliman.Toliman ──► .model()
   └─ detector.layers ──► dLux.LayeredDetector ─────┘          │
         [oversampled layers] → Downsample → [pixel layers]     ▼
                                                     expected image (e⁻, deterministic)
                                                                │
   noise_sampling.sequence ──► shot noise → read noise → ...   ▼
                                                     n_frames noisy frames ──► FITS / npz
```

| File | Role |
| --- | --- |
| `toliman_sim/config.py` | Loads JSON. Keys starting with `_` are comments. Unit suffixes (`_arcsec`, `_deg`, `_nm`, `_m`, `_mas`) are stripped to match constructor arguments, so `separation_arcsec` becomes `SpectralAlphaCen(separation=...)`. |
| `toliman_sim/spectra.py` | Builds the photon-rate spectra of A and B: SED × collecting area, binned, with the throughput kept separate (see [Spectra](#spectra)). |
| `toliman_sim/builders.py` | Builds the optics, source, detector and telescope. `DETECTOR_LAYERS` maps each layer `name` to a builder and says which grid the layer acts on: the oversampled grid or detector pixels. |
| `toliman_sim/noise.py` | Stochastic sampling steps, applied in config order (`NOISE_STEPS`). |
| `toliman_sim/io.py` | Output writers (`WRITERS`: `fits`, `npz`). |
| `toliman_sim/simulate.py` | Orchestration and the command-line interface. |
| `dLuxToliman/spectra.py` | `CombinedBasisSpectrum` and Hermite-sparse `propagate_spectrum`, adapted from the `spectrum` repo. |
| `data/spectra/alpha_cen_{A,B}.csv` | PHOENIX SEDs at the telescope [photons / s / m² / nm], 300–1100 nm. |
| `data/throughput/{bandpass,qe_mono}.csv` | Filter bandpass and camera mono QE [fraction], extracted from the spreadsheets. |
| `scripts/make_stellar_spectra.py` | Makes the SED tables from the CDBS PHOENIX grid, normalised to V (astropy only). |
| `scripts/extract_throughput.py` | Makes the throughput CSVs from `Bandpass-Sydney.xlsx` and `ApproximateQE-BFS-U3-200S6.xlsx`. |
| `scripts/constrainable_modes.py` | Computes K, the number of constrainable spectral modes, to set `n_coarse ≈ 2K`. Needs the `spectrum` package. |
| `scripts/reference_point_fisher.py` | Fisher comparison of reference points for the binary's position (see [Source geometry](#source-geometry-and-inference)). |

Relative input paths in a config (spectra, throughput, calibration maps) are relative to the config file. Output paths are relative to the working directory.

**Adding an effect** usually means three steps:

1. Write a dLux `DetectorLayer`, or a function for noise steps.
2. Register it in `DETECTOR_LAYERS` or `NOISE_STEPS`.
3. Add an entry to the config.

If a config enables an effect that isn't implemented yet, the simulator raises `NotImplementedError`. It never silently skips a requested effect.

The config format is **not final**. `configs/example.json` follows the draft format, with the deviations listed in its `_notes`.

## Spectra

Each star has its own spectrum: a `dLuxToliman.CombinedBasisSpectrum` (from `spectrum`), held by `dLuxToliman.SpectralAlphaCen`.

| Quantity | Meaning | Units |
| --- | --- | --- |
| `spec_weights()` | **Incident** photon rate per wavelength bin, at the telescope, collecting area included | photons / s / bin |
| `filt_weights` | Mean system throughput per bin (filter × QE × glass) | fraction |
| `proper_flux()` | **Detected** rate, `sum(spec_weights × filt_weights)` | photons / s |
| `exposure_time` | Per-frame exposure (top-level `exposure_time_s`), a pure multiplier: detected photons = `exposure_time × proper_flux()` | s |
| wavelengths | | m |

How a spectrum is built (`toliman_sim/spectra.py`):

1. **System throughput** is the product of the `throughput.components`:
   - `data/throughput/bandpass.csv` (filter),
   - `data/throughput/qe_mono.csv` (camera mono QE),
   - 0.91² for the two sheets of pupil glass, a visual estimate carried over from `photon_count.md`.
2. **Band:** the simulated band is where the system throughput exceeds `threshold` × peak, which gives **512–647 nm** at the default 0.03. Alternatively, set `band_nm`.
3. **Bins:** the band is split into `n_wavels` even bins (300 by default). `filt_weights` is the mean throughput in each bin.
4. **`spec_weights`** is the SED integrated over each bin, times the collecting area of the modelled aperture (0.01119 m², struts included). The basis is the identity, so `basis_weights` are the per-bin photon rates. A low-dimensional basis can be swapped in later for recovery.

**Stellar SEDs.** `scripts/make_stellar_spectra.py` builds the SED tables from the CDBS PHOENIX grid, using the parameters from `photon_count.md`:

| Star | Teff | [M/H] | log g | V |
| --- | --- | --- | --- | --- |
| A | 5790 K | 0.20 | 4.00 | 0.01 |
| B | 5260 K | 0.23 | 4.37 | 1.33 |

The CDBS grid is high resolution (1.8 Å) at [M/H] = 0 but only 50 Å at +0.3. Interpolating straight in metallicity, as pysynphot's `Icat` does, would make both stars mostly low resolution. The script therefore:

- takes the high-resolution solar-metallicity spectrum, interpolated bilinearly in Teff and log g;
- multiplies it by a smooth metallicity correction taken from the +0.3 grid;
- normalises to V by photon-counting synthetic photometry (Johnson V, Vega `alpha_lyr_stis_011`).

The ~45 MB of CDBS inputs it reads live in the gitignored `external/cdbs/`; the download URLs are in the script's docstring.

Validation against pysynphot `Icat`, using the same throughput:

- Band-integrated detected flux agrees to **0.05%**: A gives 6.345e9 vs 6.342e9 ph / s / m², which reproduces `photon_count.md`'s 6.342e5 ph / s / cm².
- The 5 nm-binned shape agrees to ±1–1.7%.
- In-band contrast A/B is 3.23 (pysynphot: 3.22).
- Per 0.1 s frame, the detected photons are **6.92e6 (A) + 2.15e6 (B)**, giving log₁₀ N = 6.96.

**Out-of-band leak.** About **2.5% of detected photons** fall outside 512–647 nm and are not simulated yet. That's ~1.5% blueward of 520 nm and ~1.3% redward of 650 nm. Lowering `threshold` to 0.01 (343–737 nm) captures all but 0.7%, but spreads the Hermite nodes over the leak regions. See the checklist.

**Hermite sparsity.** With `n_coarse` set, only `n_coarse` wavelengths are propagated, each with its PSF and its derivative with respect to wavelength. The rest of the `n_wavels` grid is recovered by cubic Hermite interpolation.

- Set `n_coarse ≈ 2K`, where K is the number of constrainable spectral modes, i.e. modes with `1 / (|a₀| √dᵢ) < 0.1` in the spectral Fisher matrix. `n_coarse` can be set per star.
- Run `scripts/constrainable_modes.py` (it needs the `spectrum` package) to get K. With the PHOENIX spectra and real throughput, read noise 5 e⁻ and a 0.1 s exposure:

| Frames fit together | Detected photons (A / B) | K_A | K_B |
| --- | --- | --- | --- |
| 1 | 6.9e6 / 2.1e6 | 13 | 9 |
| 10 | 6.9e7 / 2.1e7 | 16 | 12 |
| 1000 | 6.9e9 / 2.1e9 | 21 | 19 |

- **K must be computed at the total photon count that will be fit.** Every frame shares the same expected image, so the Hermite model error is identical in each frame and adds up coherently, while the noise averages down.
- `configs/example.json` (10 frames) uses `n_coarse` = 32 / 24. Against exact 300-wavelength propagation, this gives Δχ² = 7e-4 over 10 frames and is 4× faster.
- The 2K rule looks conservative. In an earlier node scan with placeholder spectra, 12 nodes already gave Δχ² ≈ 1.4 over 10 frames.
- **Recompute K** whenever the spectra, the throughput or the frame count change.

## Source geometry and inference

`SpectralAlphaCen` is parameterised explicitly by **`separation` [arcsec]** and **`position_angle` [deg]**, the direction of B relative to A. Its `(x_position, y_position)` [arcsec] is a reference point a fraction `centre_weight` of the way from A to B:

```
A = ref − w·d,   B = ref + (1 − w)·d,   d = separation · (sin PA, cos PA)
```

- `w = 0` puts the reference on A.
- `w = 0.5` puts it on the midpoint.
- `w = F_B / (F_A + F_B)` puts it on the photocentre.

**Which reference point?** It doesn't change what can be learned about the separation. The marginal uncertainty on `separation` is identical for any `w`, because changing `w` is a unit-Jacobian change of the nuisance position. It does change how correlated the parameters are, which matters for optimisers, samplers and mass matrices. `scripts/reference_point_fisher.py` fits (x, y, separation, PA, flux_A, flux_B) to the example config:

| `centre_weight` | σ_sep per frame | corr(sep, x) | cond(correlation matrix) |
| --- | --- | --- | --- |
| 0 (A) | 503 µas | −0.43 | 2.5 |
| 0.2 | 503 µas | 0.03 | 1.6 |
| 0.237 (photocentre) | 503 µas | 0.12 | 1.6 |
| 0.5 (midpoint) | 503 µas | 0.62 | 4.5 |
| 1 (B) | 503 µas | 0.90 | 19.6 |

The photocentre is the most stable choice. When the separation changes there, A and B move in opposite directions weighted by flux, so the light centroid, which pins down the position, stays fixed. The minimum correlation sits slightly below the photocentre (w ≈ 0.2), presumably because read noise and the overlapping PSF wings weight the stars a little differently from their raw fluxes. The example config uses `centre_weight = 0.24`. It is a static constant rather than a function of the fitted fluxes, so the reparameterisation stays linear and does not couple the positions to the spectra. A Fisher-preconditioned optimiser, as in the `spectrum` repo, makes the choice matter even less.

**Contrast and total flux** follow from the two spectra:

- `SpectralAlphaCen.contrast` = `proper_flux_A / proper_flux_B`, the in-band detected ratio.
- `SpectralAlphaCen.log_flux` = log₁₀ of the total detected photons per frame.

So posterior samples of the spectra map directly onto the legacy `AlphaCen` (`contrast`, `log_flux`) parameters, and the Fisher uncertainties propagate through the same Jacobian. Note that the contrast is band- and throughput-dependent.

## Sidelobes (planned)

The phase grating on the pupil throws light into grating orders (sidelobes) far from the central PSF. In the current classes these are missing or broken: `TolimanSpikes` cannot be constructed, and `Toliman.full_model` is outdated. Two approaches:

1. **Brute force: render the grating in the pupil.** Sample the pupil finely enough to Nyquist sample the grating, at ≥ 2 pixels per period and realistically 4–8. Then propagate to a field of view that reaches the orders.
   - The first orders sit at about λ/Λ. For example, Λ = 300 µm (the `TolimanSpikes` default; check the real period) puts them about 400″ out, roughly 950 pixels.
   - This needs pupils several thousand pixels across and very large outputs. It is too expensive for fitting, but it is the ground truth for checking approach 2.
2. **Analytic: propagate each order separately, without rendering the grating.** A periodic grating factorises: exp(iφ_g(x)) = Σ c_mn exp(2πi (m x/Λₓ + n y/Λᵧ)).
   - Each order (m, n) is therefore the ordinary pupil field, tilted by (mλ/Λₓ, nλ/Λᵧ) and weighted by |c_mn(λ)|². The coefficients depend on wavelength through the grating depth and duty cycle.
   - Each sidelobe is then a small MFT centred on its own order, with a radially dispersed, chromatically smeared copy of the core PSF. The 0th order loses a fraction 1 − |c₀₀|².
   - This is exact for a perfectly periodic grating, cheap, and differentiable in depth, period and orientation. It is **the only approach that is practical for recovery.**
   - Deviations from periodicity (period or duty errors, edge effects) would enter as perturbations.

Plan: implement approach 2 as an optics option, and validate it against approach 1 on a small case.

## Checklist

Legend:
- `[x]` works in `toliman_sim` today.
- `[~]` code exists elsewhere (dLuxToliman, `spectrum`, `nircam_halo`, OneDrive) but is not wired in yet.
- `[ ]` not started.

The "Oct" column gives the target for the October data simulator: ✅ planned, ➖ maybe, ❌ out of scope.

### Infrastructure

| | Item | Notes |
| --- | --- | --- |
| [x] | JSON config → simulated frames pipeline | `python -m toliman_sim config.json` |
| [x] | FITS / npz output with noiseless truth image and config copy | |
| [x] | Seeded, reproducible runs | One `numpy` Generator from `seed`, used for the flat field and the noise |
| [x] | Fail loudly on unimplemented effects | |
| [x] | Tests | `tests/test_simulate.py`, `tests/test_spectra.py` (`pytest`) |
| [ ] | Finalise the config format | Units convention, `$ref`s, and whether `_dlux_object` should drive construction |
| [ ] | Config validation | e.g. `pydantic` or `jsonschema`, with defaults filled in and written to the output |
| [ ] | Per-frame truth table | Positions, separation, flux and jitter per frame, written alongside the data |
| [ ] | Parameter sweeps / fidelity presets | e.g. `configs/fidelity_{0,1,2}.json`, or one base config plus overrides |
| [x] | Exposure time per frame | Top-level `exposure_time_s` multiplies the photon-rate spectra and the dark-current rate |
| [x] | Packaging in `pyproject.toml` | Distribution `toliman_data`, with extras `fits`, `scripts`, `test`. `setup.py` and `requirements.txt` removed |
| [ ] | CI | GitHub Actions running `pytest` |

### Scene / source

| | Item | Oct | Notes |
| --- | --- | --- | --- |
| [x] | Alpha Centauri A+B with separate spectra | ✅ | `dLuxToliman.SpectralAlphaCen`: explicit `separation` [arcsec], PA of B relative to A, photocentre-like reference point; derived `contrast` and `log_flux` |
| [x] | Photon-rate spectra with throughput kept separate | ✅ | `CombinedBasisSpectrum`: SED × area → `spec_weights`, throughput → `filt_weights` |
| [x] | Hermite wavelength sparsity (`n_coarse ≈ 2K`) | ✅ | Per star; K from `scripts/constrainable_modes.py` |
| [x] | PHOENIX SEDs for A and B | ✅ | `data/spectra/`, validated against pysynphot (0.05% in band) |
| [x] | System throughput (filter × QE × glass) | ✅ | `data/throughput/`; glass 0.91² is still a visual estimate |
| [ ] | Out-of-band leak | | ~2.5% of detected photons fall outside 512–647 nm (blue ~1.5%, red ~1.3%). Options: a lower threshold with more nodes, separate low-resolution leak segments, or non-uniform Hermite nodes |
| [ ] | Measured glass transmission | | Replace the 0.91² estimate |
| [ ] | Starspots / spectral variability | | `MixedAlphaCen` idea: mix in a cool-spot spectrum |
| [ ] | Background stars | ➖ | Read positions and fluxes from a table, add `dLux.PointSource`s (or a `Scene`) |
| [ ] | Temporal planetary signal | ✅ | **Deferred.** Simple functions first (sinusoidal reflex motion in `separation`/`position_angle` per frame), orbits later; `spectrum.DitheredScene` is a per-exposure pattern to borrow |
| [ ] | Frame-to-frame pointing drift and flux flicker | | **Deferred.** Low-frequency jitter; `spectrum` measured about 0.02–0.03 px RMS and 0.4% flux flicker on lab data |

### Optics

| | Item | Oct | Notes |
| --- | --- | --- | --- |
| [x] | Basic TOLIMAN pupil (aperture, struts, diffractive mask) | ✅ | `TolimanOpticalSystem` |
| [x] | Static Zernike aberrations | ✅ | `radial_orders` / `noll_indices` plus `coefficients_m` |
| [x] | Leakage from imperfect pupil phase across the bandpass | ✅ | Inherent: the mask is a π step at 585 nm, propagated polychromatically |
| [ ] | Angle-dependent / other leakage | ❌ | Needs ray tracing |
| [ ] | Off-axis field aberrations | ➖ | Lookup table of Zernikes against field position, otherwise ray tracing |
| [ ] | Ghosts | ❌ | Being modelled on real data; `spectrum.CoherentGhostSource` is a possible starting point |
| [ ] | Stray light | ❌ | Additive background template |
| [ ] | Grating sidelobes | | See [Sidelobes](#sidelobes-planned): analytic per-order propagation, validated against a rendered grating |

### Detector (deterministic, part of `telescope.model()`)

| | Item | Oct | Notes |
| --- | --- | --- | --- |
| [x] | High-frequency jitter (Gaussian / SHM) | ✅ | `high_frequency_jitter`, using `dLuxToliman.GaussianJitter` / `SHMJitter`; see [Known issues](#known-issues) |
| [x] | Downsampling from oversampled PSF to pixels | ✅ | Inserted automatically |
| [x] | Pixel response (flat field) | ✅ | `pixel_response`: loads an `.npy` map, or draws a random flat with `flat_sigma` |
| [~] | Nonlinear gain | ✅ | Port `ApplyNonlinearity` from `spectrum/detector.py`; decide whether it acts before or after noise |
| [x] | Dark current (mean) | ✅ | `dark_current_mean`, via `dLux.AddConstant` |
| [ ] | Dark current per pixel / hot pixels | ➖ | Map or distribution; `spectrum/bad_pixels.py` for realistic maps |
| [~] | Pixel dislocations | ➖ | Code is on OneDrive. Registered as `pixel_dislocations` (raises for now) |
| [~] | Polynomial pixel-scale variation | ➖ | Registered as `local_pixel_scale_variation` (raises for now) |
| [ ] | Saturation / full well | | `dLux.layers.ApplySaturation` exists |
| [ ] | Brighter-fatter effect (charge migration) | ❌ | Too complicated for now (would need an AMIGO-style network) |

### Noise (stochastic, after `telescope.model()`)

| | Item | Oct | Notes |
| --- | --- | --- | --- |
| [x] | Shot noise | ✅ | Poisson |
| [x] | Read noise | ✅ | Gaussian, in e⁻ |
| [ ] | Row / banding noise (CMOS) | | `spectrum.reduction.DetectorNoise` measured 0.46 e⁻ per row on lab data |

### Output

| | Item | Oct | Notes |
| --- | --- | --- | --- |
| [x] | FITS cube + truth extension; npz | ✅ | |
| [ ] | Camera output formats | ➖ | e⁻ → ADU (gain), bias, quantisation and bit depth, clipping, real camera header keywords and file layout. `spectrum` measured 0.24 e⁻/ADU and 16 ADU steps on lab data. Pattern for the writer: `nircam_halo.io.write_demo_fits` (SCI/ERR/DQ) |
| [ ] | Error and data-quality extensions | | ERR from the noise model, DQ from bad-pixel maps |

## Known issues

- `dLuxToliman.GaussianJitter`:
  - The docstring says `r` is the product of the standard deviations, but the code sets the covariance to `diag(sqrt(r))`, so σ = r^¼. `toliman_sim` converts `sigma_mas` using the code's behaviour.
  - The kernel grid spacing is `extent / (N - 1)` rather than one pixel, which is about 10% too wide for `N = 11`.
- `TolimanSpikes` cannot be constructed: it passes `n_struts` and `strut_rotation`, which `TolimanOpticalSystem` no longer accepts. It also uses outdated dLux calls and a fixed 80/20 core/sidelobe flux split. `Toliman.full_model` depends on it.
- `JitteredToliman` and `MixedAlphaCen` are untested with dLux 0.15. `MixedAlphaCen` also needs pysynphot.
- The diffractive mask's π phase is fixed at 585 nm (`optical_systems.py`).
- The QE spreadsheet is `ApproximateQE-BFS-U3-200S6.xlsx`, but its sheet is named `BFS-U3-51S5M-C`. Check that the curve is for the right camera.
- `photon_count.md` used the diameters as radii, so its collecting area, and every photon count after it, is 4× too high. An erratum has been added. The legacy `AlphaCen(log_flux=7.832)` default may inherit this error. It is no longer used by `toliman_sim`.

## Changes to dLuxToliman

- `layers/detector_layers.py`: jitter layers now implement `__call__`, which dLux 0.15 requires; `apply` is kept as an alias. Without this change they cannot be instantiated on dLux ≥ 0.15.
- `telescopes.py`: `Toliman(optics, source, detector=None)` now accepts a detector.
- `spectra.py` (new): `CombinedSpectrum`, `CombinedBasisSpectrum` and `propagate_spectrum` with Hermite sparsity, adapted from the `spectrum` repo.
- `sources.py`: new `SpectralAlphaCen`, with separate A and B spectra, exposure time as a multiplier, an explicit separation and a configurable reference point. The legacy `AlphaCen` is unchanged.
- Packaging moved to `pyproject.toml` (distribution `toliman_data`). It ships subpackages, `diffractive_pupil.npy` and `toliman_sim`; `setup.py` and `requirements.txt` are removed.

## Upstream

dLuxToliman provides pre-built [∂Lux](https://louisdesdoigts.github.io/dLux/) models of the Toliman optical system and parametrised Alpha Centauri sources. Its contributors are [Max Charles](https://github.com/maxecharles), [Louis Desdoigts](https://github.com/LouisDesdoigts), [Benjamin Pope](https://github.com/benjaminpope) and [Peter Tuthill](https://github.com/ptuthill).

To pull upstream changes:

```bash
git fetch upstream && git merge upstream/main
```

## License

BSD 3-Clause; see [LICENSE](LICENSE).
