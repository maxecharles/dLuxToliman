# milo-dev: development notes

Details of the simulator (`toliman_sim/`) and the additions made to `dLuxToliman/` on this branch: how it works, the config options, validation results, the checklist of implemented and planned features, known issues, and changes relative to upstream. The [README](README.md) covers installation and running a simulation.

These notes, like most of the code on this branch, were written with Claude (see the AI disclosure in the README).

**Reproducing the numbers.** Every validation number below (spectra vs pysynphot, out-of-band leak, K tables, Hermite accuracy, reference-point Fisher, jitter width) can be regenerated with the scripts in [`validation/`](validation/). [validation/README.md](validation/README.md) gives the command for each one and its expected output.

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
   noise: poisson / read_noise flags ─────────────────────────▼
                                                     noisy observation

   repeated n_observations times, with observations.parameters drawn afresh each time
   ──► data / expected (n_observations, ny, nx) + truth table ──► npy (default) / FITS
```

| File | Role |
| --- | --- |
| `toliman_sim/config.py` | Loads JSON. Keys starting with `_` are comments. Unit suffixes (`_arcsec`, `_deg`, `_nm`, `_m`, `_mas`) are stripped to match constructor arguments, so `separation_arcsec` becomes `SpectralAlphaCen(separation=...)`. |
| `toliman_sim/spectra.py` | Builds the photon-rate spectra of A and B: SED × collecting area, binned, with the throughput kept separate (see [Spectra](#spectra)). |
| `toliman_sim/builders.py` | Builds the optics, source, detector and telescope. `DETECTOR_LAYERS` maps each layer `name` to a builder and says which grid the layer acts on: the oversampled grid or detector pixels. |
| `toliman_sim/draws.py` | Random draws of telescope parameters per observation (`ALIASES` maps friendly names to zodiax paths). |
| `toliman_sim/noise.py` | Poisson and Gaussian read noise, switched on and off by flags. |
| `toliman_sim/io.py` | Output writers (`WRITERS`: `npy`, `fits`). |
| `toliman_sim/simulate.py` | `simulate` (in memory), `save`, `run`, and the command-line interface. |
| `notebooks/simulate_data.ipynb` | Notebook front end: config → simulate → plots and truth table → save. |
| `dLuxToliman/spectra.py` | Re-exports `CombinedSpectrum`, `CombinedBasisSpectrum` and Hermite-sparse `propagate_spectrum` from [dluxtras](https://github.com/milolangker/dluxtras). |
| `data/spectra/alpha_cen_{A,B}.csv` | PHOENIX SEDs at the telescope [photons / s / m² / nm], 300–1100 nm. |
| `data/throughput/{bandpass,qe_mono}.csv` | Filter bandpass and camera mono QE [fraction], extracted from the spreadsheets. |
| `scripts/make_stellar_spectra.py` | Makes the SED tables from the CDBS PHOENIX grid, normalised to V (astropy only). |
| `scripts/extract_throughput.py` | Makes the throughput CSVs from `Bandpass-Sydney.xlsx` and `ApproximateQE-BFS-U3-200S6.xlsx`. |
| `validation/` | Scripts reproducing the validation numbers in these notes; see [validation/README.md](validation/README.md). Includes `constrainable_modes.py` (K, to set `n_coarse ≈ 2K`) and `reference_point_fisher.py`. |

All relative paths in a config (spectra, throughput, calibration maps, `output.directory`) are relative to the config file.

**Adding an effect** usually means three steps:

1. Write a dLux `DetectorLayer`.
2. Register it in `DETECTOR_LAYERS`.
3. Add an entry to the config.

If a config enables an effect that isn't implemented yet, the simulator raises `NotImplementedError`. It never silently skips a requested effect.

The config format is **not final**. `configs/example.json` follows the draft format, with the deviations listed in its `_notes`.

## Observations, parameter draws and noise

Each of `n_observations` observations independently goes through three steps:

1. Draw the listed parameters.
2. Model the expected image.
3. Draw one noisy frame.

```json
"observations": {
  "n_observations": 100,
  "parameters": {
    "x_position_arcsec":      {"std": 0.5},
    "y_position_arcsec":      {"std": 0.5},
    "separation_arcsec":      {"std": 0.01},
    "position_angle_deg":     {"std": 0.1},
    "zernike_coefficients_m": {"mean": 0.0, "std": [5e-9, 5e-9, 5e-9, 2e-9, 2e-9, 2e-9, 2e-9]},
    "detector.high_frequency_jitter.r": {"distribution": "uniform", "low": 0.0, "high": 1e-12}
  }
},
"noise": {"poisson": true, "read_noise": true, "read_noise_electrons": 5.0}
```

**Parameters:**

- **Names:** each key is either an alias (`x_position_arcsec`, `y_position_arcsec`, `separation_arcsec`, `position_angle_deg`, `exposure_time_s`, `zernike_coefficients_m`) or any zodiax path on the `Toliman` telescope.
- **Distributions:**
  - `normal` (the default) takes `mean` and `std`; `mean` defaults to the configured value.
  - `uniform` takes `low` and `high`.
  - Any of these can be a list with one value per element, e.g. one std per Zernike coefficient.
- **Zernike modes** come from `optics.radial_orders` or `optics.noll_indices`. `optics.coefficients_m` sets their nominal values (zeros if omitted).
- **Untouched parameters:** anything not listed keeps its configured value, but every alias is still recorded in the truth table.

**Noise:**

- `poisson` draws shot noise on the expected photoelectrons. QE is part of the throughput, so detected photons are photoelectrons.
- `read_noise` adds Gaussian noise with σ = `read_noise_electrons` [e⁻].
- With both flags off, `data` equals `expected`.

**Randomness and seeding:** everything random uses `jax.random`.

- `jax.random.PRNGKey(seed)` is split into a build key, used for a random flat field, and an observation key.
- Observation *i* uses `fold_in(observation_key, i)`, split into a key for the parameter draws and a key for the noise.
- The noise is `jax.random.poisson(key, expected) + read_noise_electrons · jax.random.normal(key, shape)`.
- Each observation's randomness therefore depends only on `seed` and *i*, so runs are reproducible and observations are independent of each other.

## Spectra

Each star has its own spectrum: a `CombinedBasisSpectrum` (from [dluxtras](https://github.com/milolangker/dluxtras), originally adapted from [h-greer/hubble](https://github.com/h-greer/hubble)), held by `dLuxToliman.SpectralAlphaCen`.

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

Validation against pysynphot `Icat`, using the same throughput (`validation/spectra_vs_pysynphot.py`):

- Band-integrated detected flux agrees to **0.05%**: A gives 6.345e9 vs 6.342e9 ph / s / m², which reproduces `photon_count.md`'s 6.342e5 ph / s / cm².
- The 5 nm-binned shape agrees to ±1–1.7%.
- In-band contrast A/B is 3.23 (pysynphot: 3.22).
- Per 0.1 s frame, the detected photons are **6.92e6 (A) + 2.15e6 (B)**, giving log₁₀ N = 6.96.

**Out-of-band leak** (`validation/out_of_band_leak.py`). About **2.5% of detected photons** fall outside 512–647 nm and are not simulated yet. That's ~1.5% blueward of 520 nm and ~1.3% redward of 650 nm. Lowering `threshold` to 0.01 (343–737 nm) captures all but 0.7%, but spreads the Hermite nodes over the leak regions. See the checklist.

**Hermite sparsity.** With `n_coarse` set, only `n_coarse` wavelengths are propagated, each with its PSF and its derivative with respect to wavelength. The rest of the `n_wavels` grid is recovered by cubic Hermite interpolation.

- Set `n_coarse ≈ 2K`, where K is the number of constrainable spectral modes, i.e. modes with `1 / (|a₀| √dᵢ) < 0.1` in the spectral Fisher matrix. `n_coarse` can be set per star.
- Run `validation/constrainable_modes.py` (it uses the dluxtras spectral Fisher tools) to get K. With the PHOENIX spectra and real throughput, read noise 5 e⁻ and a 0.1 s exposure:

| Frames fit together | Detected photons (A / B) | K_A | K_B |
| --- | --- | --- | --- |
| 1 | 6.9e6 / 2.1e6 | 13 | 9 |
| 10 | 6.9e7 / 2.1e7 | 16 | 13 |
| 1000 | 6.9e9 / 2.1e9 | 20 | 20 |

- **K must be computed at the total photon count that will be fit.** Every frame shares the same expected image, so the Hermite model error is identical in each frame and adds up coherently, while the noise averages down.
- K sits on a hard threshold, so it can move by ±1 with small changes to the model, such as a different random flat-field realisation.
- `configs/example.json` (10 observations) uses `n_coarse` = 32 / 26. Against exact 300-wavelength propagation, this gives Δχ² = 7e-4 over 10 frames and is about 4× faster (`validation/hermite_accuracy.py`).
- The 2K rule is conservative. With the same number of nodes for both stars, the node scan gives Δχ² = 1100 (8 nodes), 4.7 (12), 0.11 (16) and 0.003 (24) over 10 frames.
- **Recompute K** whenever the spectra, the throughput or the frame count change.

## Source geometry and inference

`SpectralAlphaCen` is parameterised explicitly by **`separation` [arcsec]** and **`position_angle` [deg]**, the direction of B relative to A. Its `(x_position, y_position)` [arcsec] is a reference point a fraction `centre_weight` of the way from A to B:

```
A = ref − w·d,   B = ref + (1 − w)·d,   d = separation · (sin PA, cos PA)
```

- `w = 0` puts the reference on A.
- `w = 0.5` puts it on the midpoint.
- `w = F_B / (F_A + F_B)` puts it on the photocentre.

**Which reference point?** It doesn't change what can be learned about the separation. The marginal uncertainty on `separation` is identical for any `w`, because changing `w` is a unit-Jacobian change of the nuisance position. It does change how correlated the parameters are, which matters for optimisers, samplers and mass matrices. `validation/reference_point_fisher.py` fits (x, y, separation, PA, flux_A, flux_B) to the example config:

| `centre_weight` | σ_sep per frame | corr(sep, x) | cond(correlation matrix) |
| --- | --- | --- | --- |
| 0 (A) | 502 µas | −0.44 | 2.6 |
| 0.2 | 502 µas | 0.02 | 1.6 |
| 0.237 (photocentre) | 502 µas | 0.12 | 1.6 |
| 0.5 (midpoint) | 502 µas | 0.62 | 4.5 |
| 1 (B) | 502 µas | 0.90 | 19.6 |

The photocentre is the most stable choice. When the separation changes there, A and B move in opposite directions weighted by flux, so the light centroid, which pins down the position, stays fixed. The minimum correlation sits slightly below the photocentre (w ≈ 0.2), presumably because read noise and the overlapping PSF wings weight the stars a little differently from their raw fluxes. The example config uses `centre_weight = 0.24`. It is a static constant rather than a function of the fitted fluxes, so the reparameterisation stays linear and does not couple the positions to the spectra. A Fisher-preconditioned optimiser makes the choice matter even less.

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
| [x] | 3D `.npy` output (default), with noiseless images, truth table and config copy; FITS optional | `(n_observations, ny, nx)` |
| [x] | Bulk parameter draws per observation | `observations.parameters`: normal / uniform draws of positions, separation, PA, Zernikes, any zodiax path |
| [x] | Seeded, reproducible runs | `jax.random` keys from `seed`, with one key per observation via `fold_in` |
| [x] | Notebook front end | `notebooks/simulate_data.ipynb` |
| [x] | Fail loudly on unimplemented effects | |
| [x] | Tests | `tests/toliman_sim/` (`pytest` runs them with dLuxToliman's own tests) |
| [ ] | Finalise the config format | e.g. the units convention |
| [ ] | Config validation | e.g. `pydantic` or `jsonschema`, with defaults filled in and written to the output |
| [x] | Per-observation truth table | `{run_id}_truth.npz`, or the FITS `TRUTH` table |
| [ ] | Derived truth columns | e.g. `contrast`, `log_flux` and jitter σ per observation |
| [ ] | Repeated frames per draw | Several noise realisations of the same drawn parameters |
| [ ] | Parameter sweeps / fidelity presets | e.g. `configs/fidelity_{0,1,2}.json`, or one base config plus overrides |
| [x] | Exposure time per frame | Top-level `exposure_time_s` multiplies the photon-rate spectra and the dark-current rate |
| [x] | Packaging | dLuxToliman's `pyproject.toml`/`setup.py`, plus `toliman_sim`, the `dluxtras` dependency and extras `fits`, `scripts`, `notebooks` |
| [x] | CI | dLuxToliman's GitHub Actions workflow runs all tests, including `tests/toliman_sim/` |

### Scene / source

| | Item | Oct | Notes |
| --- | --- | --- | --- |
| [x] | Alpha Centauri A+B with separate spectra | ✅ | `dLuxToliman.SpectralAlphaCen`: explicit `separation` [arcsec], PA of B relative to A, photocentre-like reference point; derived `contrast` and `log_flux` |
| [x] | Photon-rate spectra with throughput kept separate | ✅ | `CombinedBasisSpectrum`: SED × area → `spec_weights`, throughput → `filt_weights` |
| [x] | Hermite wavelength sparsity (`n_coarse ≈ 2K`) | ✅ | Per star; K from `validation/constrainable_modes.py` |
| [x] | PHOENIX SEDs for A and B | ✅ | `data/spectra/`, validated against pysynphot (0.05% in band) |
| [x] | System throughput (filter × QE × glass) | ✅ | `data/throughput/`; glass 0.91² is still a visual estimate |
| [ ] | Out-of-band leak | | ~2.5% of detected photons fall outside 512–647 nm (blue ~1.5%, red ~1.3%). Options: a lower threshold with more nodes, separate low-resolution leak segments, or non-uniform Hermite nodes |
| [ ] | Measured glass transmission | | Replace the 0.91² estimate |
| [ ] | Starspots / spectral variability | | |
| [ ] | Background stars | ➖ | From a table |
| [ ] | Temporal planetary signal | ✅ | **Deferred.** |
| [ ] | Frame-to-frame pointing drift and flux flicker | | **Deferred.** |

### Optics

| | Item | Oct | Notes |
| --- | --- | --- | --- |
| [x] | Basic TOLIMAN pupil (aperture, struts, diffractive mask) | ✅ | `TolimanOpticalSystem` |
| [x] | Static Zernike aberrations | ✅ | `radial_orders` / `noll_indices` plus `coefficients_m` |
| [x] | Leakage from imperfect pupil phase across the bandpass | ✅ | Inherent: the mask is a π step at 585 nm, propagated polychromatically |
| [ ] | Angle-dependent / other leakage | ❌ | Needs ray tracing |
| [ ] | Off-axis field aberrations | ➖ | Lookup table, otherwise ray tracing |
| [ ] | Ghosts | ❌ | Being modelled on real data |
| [ ] | Stray light | ❌ | |
| [ ] | Grating sidelobes | | See [Sidelobes](#sidelobes-planned): analytic per-order propagation, validated against a rendered grating |

### Detector (deterministic, part of `telescope.model()`)

| | Item | Oct | Notes |
| --- | --- | --- | --- |
| [x] | High-frequency jitter (Gaussian / SHM) | ✅ | `high_frequency_jitter`, using `dLuxToliman.GaussianJitter` (Fourier-space blur, from dLuxToliman 0.5.0; `sigma_mas` and `shear` are converted to `r` = (σ_major·σ_minor)²) / `SHMJitter` |
| [x] | Downsampling from oversampled PSF to pixels | ✅ | Inserted automatically |
| [x] | Pixel response (flat field) | ✅ | `pixel_response`: loads an `.npy` map, or draws a random flat with `flat_sigma` |
| [~] | Nonlinear gain | ✅ | Exists elsewhere, not wired in |
| [x] | Dark current (mean) | ✅ | `dark_current`: `dLux.AddConstant(rate_e_per_s_per_pixel × exposure_time_s)` |
| [ ] | Dark current per pixel / hot pixels | ➖ | |
| [~] | Pixel dislocations | ➖ | Code is on OneDrive. Placeholder name `pixel_dislocations` (raises if enabled) |
| [~] | Polynomial pixel-scale variation | ➖ | Placeholder name `local_pixel_scale_variation` (raises if enabled) |
| [ ] | Saturation / full well | | `dLux.layers.ApplySaturation` exists |
| [ ] | Brighter-fatter effect (charge migration) | ❌ | Too complicated for now. Placeholder name `brighter_fatter_effect` (raises if enabled) |

### Noise (stochastic, after `telescope.model()`)

| | Item | Oct | Notes |
| --- | --- | --- | --- |
| [x] | Shot noise | ✅ | `noise.poisson` flag |
| [x] | Read noise | ✅ | `noise.read_noise` flag, σ = `noise.read_noise_electrons` |
| [ ] | Row / banding noise (CMOS) | | |

### Output

| | Item | Oct | Notes |
| --- | --- | --- | --- |
| [x] | `.npy` (default) or FITS | ✅ | |
| [ ] | Camera output formats | ➖ | Requires more thought |
| [ ] | Error and data-quality extensions | | |

## Known issues

- `TolimanSpikes` cannot be constructed: it passes `n_struts` and `strut_rotation`, which `TolimanOpticalSystem` no longer accepts. It also uses outdated dLux calls and a fixed 80/20 core/sidelobe flux split. `Toliman.full_model` depends on it.
- `JitteredToliman` and `MixedAlphaCen` are untested with dLux 0.15. `MixedAlphaCen` also needs pysynphot.
- The diffractive mask's π phase is fixed at 585 nm (`optical_systems.py`).
- The QE spreadsheet is `ApproximateQE-BFS-U3-200S6.xlsx`, but its sheet is named `BFS-U3-51S5M-C`. Check that the curve is for the right camera.

## Changes to dLuxToliman

This branch is based on `main` (dLuxToliman 0.5.1), which already has the `GaussianJitter` fixes and dLux 0.15 support (#38), the corrected `photon_count.md` area and `AlphaCen` default flux (#37), and a test suite with CI. On top of that:

- `telescopes.py`: `Toliman(optics, source, detector=None)` now accepts a detector.
- `spectra.py` (new): re-exports `CombinedSpectrum`, `CombinedBasisSpectrum` and `propagate_spectrum` (Hermite sparsity) from [dluxtras](https://github.com/milolangker/dluxtras), a new dependency.
- `sources.py`: new `SpectralAlphaCen`, with separate A and B spectra, exposure time as a multiplier, an explicit separation and a configurable reference point. The legacy `AlphaCen` is unchanged.
- Packaging: `setup.py` ships all subpackages (including `dLuxToliman.layers` and `toliman_sim`) and `diffractive_pupil.npy`; `pyproject.toml`/`requirements.txt` add the `dluxtras` dependency (from git) and the `fits`, `scripts` and `notebooks` extras.
- `toliman_sim` does not change JAX settings on import; `simulate()` and the command line enable double precision, and `tests/toliman_sim/conftest.py` enables it per test so dLuxToliman's own tests are unaffected.

## Keeping up with `main`

This is the `milo-dev` branch of [maxecharles/dLuxToliman](https://github.com/maxecharles/dLuxToliman), based on `main`. To move it onto a newer `main`, rebase and force-push:

```bash
git fetch origin && git rebase origin/main
git push --force-with-lease
```
