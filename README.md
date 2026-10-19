# TOLIMAN data simulator (`milo-dev` branch)

The `milo-dev` branch of [maxecharles/dLuxToliman](https://github.com/maxecharles/dLuxToliman/tree/milo-dev) adds `toliman_sim`, which simulates TOLIMAN data from a JSON config. The config sets the optics, the Alpha Cen A/B spectra and geometry, detector effects, noise, and which parameters to draw at random for each observation. It writes the noisy observations as 3D arrays, `(n_observations, ny, nx)`, together with the noiseless images and a table of the true parameters.

Implementation details, the config options, validation results, the feature checklist and the changes made to `dLuxToliman` are in [DEVELOPMENT.md](DEVELOPMENT.md).

> **AI disclosure.** The vast majority of the code and additions on this branch were written with [Claude](https://claude.com/claude-code) (Anthropic's AI assistant), working under the direction and review of Milo Langker. That covers `toliman_sim/`, `SpectralAlphaCen`, the scripts, tests and notebook, and the documentation of this branch. The dLuxToliman models themselves are the work of its contributors (see the dLuxToliman README below). Treat the AI-written parts with the usual scrutiny, and check results that matter.

## Simulating data

**One-time setup** (on this branch: `git clone -b milo-dev https://github.com/maxecharles/dLuxToliman.git`):

```bash
conda activate toliman_data
pip install -e ".[tests,fits,scripts,notebooks]"   # dLux 0.15.x, JAX, dluxtras, astropy, openpyxl, Jupyter, pytest
```

**1. Write or copy a config.** Start from `configs/example.json`, e.g. `cp configs/example.json configs/my_run.json`, then edit it:

| Field | What to set |
| --- | --- |
| `run_id` | Names the output files |
| `seed` | Makes the run reproducible; leave it out for a random seed, which is recorded in the output |
| `observations.n_observations` | Number of observations |
| `observations.parameters` | Means and spreads of the parameters to draw |
| `noise` | Poisson / read-noise flags and `read_noise_electrons` |
| `output.directory` | Where to write the results |

**2. Run it, in any of three ways:**

- **Terminal:**

  ```bash
  python -m toliman_sim configs/my_run.json              # add --overwrite to replace an earlier run
  ```

- **Notebook (GUI):** open `notebooks/simulate_data.ipynb`, in JupyterLab (`jupyter lab`) or VS Code, with the *Python (toliman_data)* kernel. Set `CONFIG`, then *Run All*. It summarises the config, simulates, plots a few observations next to their noiseless images, prints the truth table and saves. You can also tweak the config in the notebook without editing the file.

- **Python:**

  ```python
  import jax
  jax.config.update("jax_enable_x64", True)   # simulations need double precision

  from toliman_sim import load_config, simulate, save

  config = load_config("configs/my_run.json")
  result = simulate(config)        # nothing written yet
  result["data"], result["expected"], result["truth"]
  save(result, config)             # writes to output.directory
  ```

**3. Find the output.** It goes in `output.directory`, which is **relative to the config file**, like every relative path in a config, so it doesn't matter where you run from. The example uses `"../simulated_data/sim_run_0001"`, i.e. `<repo>/simulated_data/sim_run_0001/`. `simulated_data/` is gitignored. The directory contains:

- `{run_id}_data.npy`: the noisy observations, shape `(n_observations, ny, nx)`, in photoelectrons.
- `{run_id}_expected.npy`: the noiseless expected images, same shape.
- `{run_id}_truth.npz`: the true parameters of every observation, e.g. `separation_arcsec` with shape `(n_observations,)` and `zernike_coefficients_m` with shape `(n_observations, 7)`, plus `pixel_scale_arcsec` and `seed`.
- `config.json`: the config used, with the seed filled in.

With `"output": {"format": "fits"}` you instead get one `{run_id}.fits` containing `DATA`, `EXPECTED` and a `TRUTH` table. Existing files are never replaced unless you pass `--overwrite` / `save(..., overwrite=True)` or set `output.overwrite`.

**4. Load it:**

```python
import numpy as np
data = np.load("simulated_data/sim_run_0001/sim_run_0001_data.npy")    # (n_observations, ny, nx)
truth = np.load("simulated_data/sim_run_0001/sim_run_0001_truth.npz")  # truth["separation_arcsec"], ...
```

Run the tests with `pytest`.

For all config options (parameter draws, spectra, throughput, detector layers), see [DEVELOPMENT.md](DEVELOPMENT.md).

---

# dLuxToliman README

*Everything below is the unaltered dLuxToliman README (as on `main`, which this branch is based on). Its installation and usage instructions (e.g. `pip install dLuxToliman`) refer to the released dLuxToliman package, not to this branch; to use this branch, follow [Simulating data](#simulating-data) above.*

# dLuxToliman

[![PyPI version](https://badge.fury.io/py/dLuxToliman.svg)](https://badge.fury.io/py/dLuxToliman)
[![License](https://img.shields.io/badge/license-BSD%203--Clause-blue.svg)](https://github.com/maxecharles/dLuxToliman/blob/main/LICENSE)
[![Documentation](https://github.com/maxecharles/dLuxToliman/actions/workflows/documentation.yml/badge.svg)](https://maxecharles.github.io/dLuxToliman/)

![Banner Image](https://raw.githubusercontent.com/maxecharles/dLuxToliman/main/assets/dLuxToliman_banner.jpg)

## Description
This repository/package contains pre-built ∂Lux models of the Toliman optical system, and pre-built parametrised ∂Lux source objects for Alpha Centauri.

[∂Lux](https://louisdesdoigts.github.io/dLux/) is an open-source differentiable optical modelling framework harnessing the structural isomorphism between optical systems and neural networks, giving forwards models of optical system as a _parametric neural network_.
∂Lux is built in [Zodiax](https://github.com/LouisDesdoigts/zodiax) which is an open-source object-oriented [Jax](https://github.com/google/jax) framework built as an extension of [Equinox](https://github.com/patrick-kidger/equinox) for scientific programming.

The primary goal of the Toliman mission is to discover Earth-sized exoplanets orbiting in Alpha Centauri, the closest star system to our own.
To achieve this, the mission will employ a novel telescope design that will be able to detect subtle changes in the positions of the Alpha Centauri binary pair.
These changes are caused by the gravitational reflex motion induced by an Earth-sized companion, and this cutting-edge technology will enable scientists to identify exoplanets too small to be detected by conventional telescopes.
Toliman utilises a binary phase diffraction pupil to grasp the expected microarcsecond-scale astrometric signal.

## Installation

`dLuxToliman` is pip-installable. To install the latest release, simply run:

```bash
pip install dLuxToliman
```

## Usage

A very basic example of how to use this package is as follows:

```python
# imports
import dLuxToliman as dlT
from matplotlib import pyplot as plt
from matplotlib.colors import PowerNorm

optics = dlT.TolimanOpticalSystem(psf_npixels=128)  # creating Toliman optical system
source = dlT.AlphaCen(
    n_wavels=3, separation=8, position_angle=30,
)  # creating Alpha Centauri source
model = dlT.Toliman(source=source, optics=optics)  # creating Toliman telescope

psf = model.model()  # running optical simulation

# plotting
plt.imshow(psf, cmap="inferno", norm=PowerNorm(0.5), origin="lower")
plt.title(r"Toliman PSF")
plt.colorbar()
plt.show()
```

![Example Image](https://raw.githubusercontent.com/maxecharles/dLuxToliman/main/assets/basic_toliman_psf.jpg)

## Contributors
[Max Charles](https://github.com/maxecharles), [Louis Desdoigts](https://github.com/LouisDesdoigts), [Benjamin Pope](https://github.com/benjaminpope), [Connor Langford](https://github.com/CLangford2098), [Milo Langker](https://github.com/milolangker), and [Peter Tuthill](https://github.com/ptuthill).

## License

This package is released under the BSD 3-Clause License. See the [LICENSE](https://github.com/maxecharles/dLuxToliman/blob/main/LICENSE) file for more information.

## Support

If you encounter any problems or have any questions related to Toliman models, please raise an issue or contact me at `max.charles@sydney.edu.au`.
