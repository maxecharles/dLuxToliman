"""
Random draws of telescope parameters, one set per observation.

    "observations": {
      "n_observations": 100,
      "parameters": {
        "separation_arcsec":      {"std": 0.01},                 # mean = configured value
        "x_position_arcsec":      {"mean": 0.0, "std": 0.5},
        "zernike_coefficients_m": {"std": [5e-9, 5e-9, 5e-9, 2e-9, 2e-9, 2e-9, 2e-9]},
        "detector.high_frequency_jitter.r": {"distribution": "uniform", "low": 0, "high": 1e-12}
      }
    }

Keys are either an alias from ALIASES or a raw zodiax path on the Toliman telescope.
Distributions: "normal" (mean, std; the default) or "uniform" (low, high). `mean`
defaults to the configured value, and mean/std/low/high may be scalars or lists that
match the parameter's shape (e.g. one std per Zernike coefficient).
"""

from __future__ import annotations
import jax.numpy as np
import jax.random as jr
import numpy as onp
from jax import Array

from .config import strip_comments

# Friendly names (with units) -> zodiax paths on dLuxToliman.Toliman
ALIASES = {
    "x_position_arcsec": "source.x_position",
    "y_position_arcsec": "source.y_position",
    "separation_arcsec": "source.separation",
    "position_angle_deg": "source.position_angle",
    "exposure_time_s": "source.exposure_time",
    "zernike_coefficients_m": "optics.aperture.coefficients",
}


def _path(name: str) -> str:
    if name in ALIASES:
        return ALIASES[name]
    if "." in name:
        return name
    raise ValueError(
        f"Unknown parameter '{name}'. Use an alias {list(ALIASES)} or a zodiax path "
        "such as 'detector.high_frequency_jitter.r'."
    )


def parse_draws(config: dict, telescope) -> dict:
    """Returns {name: (path, spec, nominal value)} for every parameter to draw."""
    parameters = strip_comments(config.get("observations", {}).get("parameters", {}))
    draws = {}
    for name, spec in parameters.items():
        path = _path(name)
        try:
            nominal = onp.asarray(telescope.get(path), dtype=float)
        except Exception as error:
            raise ValueError(f"Cannot draw '{name}': no parameter at '{path}'") from error
        draws[name] = (path, strip_comments(spec), nominal)
    return draws


def sample(draws: dict, key: Array) -> dict:
    """One random draw of every parameter, {name: value}, from the jax.random `key`."""
    values = {}
    for (name, (path, spec, nominal)), subkey in zip(
        draws.items(), jr.split(key, max(len(draws), 1))
    ):
        distribution = spec.get("distribution", "normal")
        if distribution == "normal":
            mean = np.asarray(spec.get("mean", nominal), dtype=float)
            std = np.asarray(spec["std"], dtype=float)
            value = mean + std * jr.normal(subkey, nominal.shape)
        elif distribution == "uniform":
            value = jr.uniform(
                subkey, nominal.shape, minval=spec["low"], maxval=spec["high"]
            )
        else:
            raise ValueError(f"[{name}] unknown distribution '{distribution}'")
        values[name] = np.broadcast_to(value, nominal.shape)
    return values


def apply(telescope, draws: dict, values: dict):
    """The telescope with the drawn values set."""
    if not values:
        return telescope
    paths = [draws[name][0] for name in values]
    return telescope.set(paths, [values[name] for name in values])


def record(telescope) -> dict:
    """Values of every alias that exists on the telescope (for the truth table)."""
    truth = {}
    for name, path in ALIASES.items():
        try:
            truth[name] = onp.asarray(telescope.get(path), dtype=float)
        except Exception:
            continue  # e.g. no Zernike basis on the optics
    return truth
