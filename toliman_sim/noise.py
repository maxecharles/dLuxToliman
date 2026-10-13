"""
Stochastic sampling of a detector frame from the expected (noiseless) image.

Steps are applied in the order given in config["noise_sampling"]["sequence"].
"""

from __future__ import annotations
import numpy as onp

from .config import strip_comments


def _shot_noise(image, spec, rng):
    return rng.poisson(onp.clip(image, 0, None)).astype(float)


def _read_noise(image, spec, rng):
    return image + rng.normal(
        spec.get("mean_electrons", 0.0), spec["sigma_electrons"], image.shape
    )


NOISE_STEPS = {
    "shot_noise": _shot_noise,
    "read_noise": _read_noise,
}


def sample_frame(expected, config: dict, rng: onp.random.Generator) -> onp.ndarray:
    """Draws one noisy frame (electrons) from the expected image (electrons)."""
    image = onp.asarray(expected, dtype=float)
    for spec in config.get("noise_sampling", {}).get("sequence", []):
        spec = strip_comments(spec)
        if not spec.get("enabled", True):
            continue
        if spec["name"] not in NOISE_STEPS:
            raise ValueError(
                f"Unknown noise step '{spec['name']}'. "
                f"Known steps: {list(NOISE_STEPS)}"
            )
        image = NOISE_STEPS[spec["name"]](image, spec, rng)
    return image
