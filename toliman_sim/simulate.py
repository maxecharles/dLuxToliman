"""
Top-level simulation entry point: config in, simulated observations out.

    python -m toliman_sim configs/example.json [--overwrite]

Pipeline, for each of observations.n_observations:
    1. Draw the parameters listed in observations.parameters (draws.py).
    2. telescope.model() -> expected image in photoelectrons (deterministic effects).
    3. Draw a noisy frame (Poisson and/or read noise, noise.py).
Then write data (n_observations, ny, nx), expected images and the truth table.
"""

from __future__ import annotations
import argparse
import secrets
import warnings
from pathlib import Path
import equinox as eqx
import jax
import jax.random as jr
import numpy as onp

from .config import load_config, strip_comments
from .builders import build_telescope
from .draws import apply, parse_draws, record, sample
from .noise import sample_frame
from .io import write_outputs

KNOWN_SECTIONS = {
    "run_id", "seed", "exposure_time_s", "output", "observations", "optics", "source",
    "detector", "noise", "miscellaneous_effects", "temporal_planetary_signal",
}


def _check_config(config: dict):
    """Warns about unknown sections; fails loudly on requested but unimplemented effects."""
    for key in strip_comments(config):
        if key not in KNOWN_SECTIONS:
            warnings.warn(f"Ignoring unknown config section '{key}'")

    effects = strip_comments(config.get("miscellaneous_effects", {}))
    for name, spec in effects.items():
        if spec.get("enabled", False):
            raise NotImplementedError(
                f"miscellaneous_effects.{name} is not implemented yet (see the "
                "DEVELOPMENT.md checklist). Set \"enabled\": false to skip it."
            )

    if config.get("temporal_planetary_signal", {}).get("enabled", False):
        raise NotImplementedError("temporal_planetary_signal is not implemented yet.")


def _require_x64():
    """Astrometry at the micro-arcsecond level needs double precision."""
    if not jax.config.jax_enable_x64:
        warnings.warn(
            "Enabling JAX double precision (jax_enable_x64) for the simulation. Enable "
            "it yourself at the start of your script to avoid mixing precisions."
        )
        jax.config.update("jax_enable_x64", True)


@eqx.filter_jit
def _model(telescope):
    return telescope.model()


def simulate(config: dict) -> dict:
    """
    Runs a simulation described by `config` without writing anything to disk.

    Returns a dict with the noisy `data` (n_observations, ny, nx), the noiseless
    `expected` images, the `truth` table ({name: (n_observations, ...)}), the
    nominal `telescope` the draws were made around, and the `seed` used (a random
    one is chosen and returned if the config has none).
    """
    _check_config(config)
    _require_x64()

    # One root key: half for building (e.g. a random flat field), and one key per
    # observation (fold_in), split between the parameter draws and the noise.
    seed = config.get("seed")
    if seed is None:
        seed = secrets.randbelow(2**31)
    build_key, observation_key = jr.split(jr.PRNGKey(seed))

    telescope = build_telescope(config, build_key)
    draws = parse_draws(config, telescope)
    n = config.get("observations", {}).get("n_observations", 1)

    data, expected, truth = [], [], []
    for i in range(n):
        draw_key, noise_key = jr.split(jr.fold_in(observation_key, i))
        observed = apply(telescope, draws, sample(draws, draw_key))
        image = _model(observed)
        expected.append(onp.asarray(image))
        data.append(onp.asarray(sample_frame(image, config, noise_key)))
        truth.append(record(observed) | {name: observed.get(draws[name][0]) for name in draws})

    truth = {name: onp.stack([onp.asarray(t[name]) for t in truth]) for name in truth[0]}
    return {
        "data": onp.stack(data),
        "expected": onp.stack(expected),
        "truth": truth,
        "telescope": telescope,
        "seed": seed,
    }


def save(result: dict, config: dict, overwrite: bool = False) -> Path:
    """Writes a simulate() result to config["output"]; returns the data file's path."""
    if overwrite:
        config = config | {"output": config.get("output", {}) | {"overwrite": True}}
    meta = {
        "pixel_scale_arcsec": float(result["telescope"].optics.psf_pixel_scale),
        "seed": result["seed"],
    }
    return write_outputs(result, config | {"seed": result["seed"]}, meta)


def run(config_path: str | Path, overwrite: bool = False) -> Path:
    """Loads a config file, simulates it, and writes the outputs to disk."""
    config = load_config(config_path)
    return save(simulate(config), config, overwrite)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Simulate TOLIMAN data from a config.")
    parser.add_argument("config", help="Path to a JSON simulation config")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing outputs")
    args = parser.parse_args(argv)

    jax.config.update("jax_enable_x64", True)
    path = run(args.config, overwrite=args.overwrite)
    print(f"Wrote {path.resolve()}")


if __name__ == "__main__":
    main()
