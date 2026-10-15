# Installation

dLuxToliman is hosted on PyPI, so simply pip install!

```
pip install dLuxToliman
```

This also installs ∂Lux, Zodiax, Jax and Equinox.

You can also build from source. To do so, clone the git repo, enter the directory, and run.

```
pip install .
```

We encourage the creation of a virtual environment to run dLuxToliman to prevent software conflicts with the other packages you use.

## Versions of ∂Lux

Each release of dLuxToliman works with one release of ∂Lux, and pip installs the matching one for you.

| dLuxToliman | ∂Lux |
| ----------- | ---- |
| 0.5         | 0.15 |
| 0.4         | 0.14 |

∂Lux 0.15 changed its Fourier sign convention, so dLuxToliman 0.5 gives different PSFs from 0.4 away from the 585 nm design wavelength of the pupil. To reproduce results made with dLuxToliman 0.4, install that release instead.

```
pip install "dLuxToliman==0.4.*"
```

The [change log](changelog.md) has the details.
