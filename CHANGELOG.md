# Changelog

## V0.5.1

Documentation and packaging only. Nothing changes in the models, so PSFs are identical
to 0.5.0.

### ✨ New Features
- Documentation at [maxecharles.github.io/dLuxToliman](https://maxecharles.github.io/dLuxToliman/):
  an installation guide, the photon count tutorial, and an API reference for every
  public class and function, with a version of the docs for each release.
- `pip install "dLuxToliman[dev]"` installs the tools for building the documentation,
  along with the test dependencies.

### 🐛 Bug Fixes
- The images in the README now show on the PyPI page. In 0.5.0 their relative links
  were broken there.
- The PyPI page links to the documentation.

### 🔧 Other Changes
- Removed parameters that do not exist from the docstrings: `n_struts` and
  `strut_rotation` in `TolimanOpticalSystem`, and `eps` in `SHMJitter.CDF`.
- A GitHub Actions workflow builds the documentation and deploys it to GitHub Pages on
  every push to `main` and every version tag.

## V0.5.0

dLuxToliman now runs on dLux 0.15. dLux 0.15 changed its Fourier sign convention, so
PSFs away from 585 nm differ from 0.4.x. To reproduce results made with 0.4.x, keep
using dLuxToliman 0.4.1 with dLux 0.14.

### ⚠️ Breaking Changes
- Requires dLux 0.15 (`dLux==0.15.*`). dLux 0.14 is no longer supported.
- Each star's PSF is now the 0.4.x PSF rotated by 180° about the star. At 585 nm, the
  design wavelength of the diffractive pupil, nothing changes. Elsewhere in the
  530–640 nm band the pattern differs, by up to about 60% of the peak for a
  monochromatic PSF at the band edges. Star positions are unchanged.
- Zernike tilt coefficients now move the PSF the other way. A positive x tilt (Noll
  index 2) moves the PSF towards +x, the same direction as a positive source offset;
  in 0.4.x it moved towards −x.

### ✨ New Features
- Publishing a GitHub Release now builds the package, checks that the release tag
  matches the package version, smoke-tests the wheel, and uploads it to PyPI through
  Trusted Publishing.

### 🐛 Bug Fixes
- `ApplyBasisCLIMB` implements `__call__`, which is how dLux 0.15 optical systems call
  their layers. Without it, dLux 0.15 skipped the CLIMB binarisation and failed with a
  shape error. `.apply` still works.

### 🔧 Other Changes
- The jitter layers no longer define their own `apply` alias, since dLux 0.15 provides
  one. `layer.apply(psf)` still works.
- Removed the unused `TolimanOpticalSystem._apply_aperture`, which relied on dLux APIs
  that no longer exist.
- The package version is defined in one place, `__version__` in
  `dLuxToliman/__init__.py`. Building the package needs setuptools 61 or newer.

## V0.4.1

### ⚠️ Breaking Changes
- The default `log_flux` of `AlphaCen` is now 6.979, down from 7.832, which
  overestimated the photon count through a radius/diameter mix-up
  ([#37](https://github.com/maxecharles/dLuxToliman/issues/37)). The new default is one
  0.1 s frame of Alpha Cen through the TOLIMAN bandpass and collecting area, about 7
  times fewer photons. `photon_count.md` has the derivation.
- `GaussianJitter` now blurs the PSF in Fourier space, so the blur has exactly the
  requested covariance ([#38](https://github.com/maxecharles/dLuxToliman/issues/38)).
  The old kernel, sampled on the pixel grid, lost most of the blur once the jitter was
  narrower than about a pixel, worst along the minor axis of a sheared kernel, so
  outputs change most for small jitter. `kernel_oversample` is no longer needed and is
  ignored, and `generate_kernel` returns the blur the layer actually applies.
- Requires Python 3.10 or newer, and dLux 0.14.3 or newer but below 0.15, since dLux
  0.15 changes the PSFs.

### ✨ New Features
- A test suite, run by GitHub Actions on Python 3.10–3.13. Its dependencies install
  with `pip install "dLuxToliman[tests]"`.

### 🐛 Bug Fixes
- The package declares its dependencies. The 0.4.0 wheel declared none, so
  `pip install dLuxToliman` did not install dLux, zodiax, jax or equinox.
- `dLuxToliman.__version__` matches the release; 0.4.0 reported 0.3.0.
- `from dLuxToliman import *` failed, because `__all__` was a list of lists.
  `ApplyBasisCLIMB` can also be imported from `dLuxToliman.layers`, which only bound
  the `ApplyBasisClimb` alias.
- `JitteredToliman` with `jitter_shape="shm"` gave NaN gradients for `jitter_mag`
  above about 4 arcseconds.
- `JitteredToliman` no longer asks for float64, which raised a warning and fell back to
  float32 unless JAX's 64-bit mode was on. Results change only by rounding error.
- The docstring of `GaussianJitter`'s `r` gives its units as arcsec⁴, the determinant
  of the covariance, where it said arcseconds. Its meaning is unchanged.

### 🔧 Other Changes
- `photon_count.md` corrected to match the new default flux.
- Added Connor Langford and Milo Langker to the contributors.
