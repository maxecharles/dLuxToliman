"""
Top-level simulation entry point: config in, simulated frames out.

    python -m toliman_sim configs/example.json [--overwrite]

Pipeline:
    1. Build the Toliman telescope (optics + AlphaCen source + detector layers).
    2. telescope.model() -> expected image in electrons (deterministic effects).
    3. Draw n_frames noisy realisations (shot noise, read noise, ...).
    4. Write frames + expected image + config to output.directory.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as onp

from .config import load_config, strip_comments
from .builders import build_telescope
from .noise import sample_frame
from .io import write_outputs


def _check_unimplemented(config: dict):
    """Fails loudly rather than silently simulating without a requested effect."""
    if config.get("run_type", "non_temporal") != "non_temporal":
        raise NotImplementedError(f"run_type '{config['run_type']}' not supported yet.")

    effects = strip_comments(config.get("miscellaneous_effects", {}))
    for name, spec in effects.items():
        if spec.get("enabled", False):
            raise NotImplementedError(
                f"miscellaneous_effects.{name} is not implemented yet (see the "
                "README checklist). Set \"enabled\": false to skip it."
            )

    if config.get("temporal_planetary_signal", {}).get("enabled", False):
        raise NotImplementedError("temporal_planetary_signal is not implemented yet.")


def simulate(config: dict) -> dict:
    """
    Runs a simulation described by `config` without writing anything to disk.

    Returns a dict with the noisy `frames` (n_frames, ny, nx), the noiseless
    `expected` image, and the `telescope` used to make them.
    """
    _check_unimplemented(config)
    rng = onp.random.default_rng(config.get("seed"))

    telescope = build_telescope(config, rng)
    expected = onp.asarray(telescope.model())

    # Non-temporal: every frame is a new noise draw from the same expected image
    n_frames = config.get("output", {}).get("n_frames", 1)
    frames = onp.stack([sample_frame(expected, config, rng) for _ in range(n_frames)])

    return {"frames": frames, "expected": expected, "telescope": telescope}


def run(config_path: str | Path, overwrite: bool = False) -> Path:
    """Loads a config file, simulates it, and writes the outputs to disk."""
    config = load_config(config_path)
    if overwrite:
        config.setdefault("output", {})["overwrite"] = True

    result = simulate(config)
    telescope = result["telescope"]
    header = {
        "RUN_ID": (config["run_id"], "Simulation run identifier"),
        "SEED": (config.get("seed", -1), "RNG seed (-1 = unseeded)"),
        "NFRAMES": (result["frames"].shape[0], "Number of frames"),
        "PIXSCALE": (float(telescope.optics.psf_pixel_scale), "arcsec / pixel"),
    }
    return write_outputs(result["frames"], result["expected"], config, header)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Simulate TOLIMAN data from a config.")
    parser.add_argument("config", help="Path to a JSON simulation config")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing outputs")
    args = parser.parse_args(argv)

    path = run(args.config, overwrite=args.overwrite)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
