"""
Stochastic sampling of a detector frame from the expected (noiseless) image.

    "noise": {"poisson": true, "read_noise": true, "read_noise_electrons": 5.0}

The expected image is in photoelectrons (QE is part of the throughput), so

    frame = Poisson(expected) + Normal(0, read_noise_electrons**2)

drawn with jax.random from the given key.
"""

from __future__ import annotations
from functools import partial
import jax
import jax.numpy as np
import jax.random as jr
from jax import Array


@partial(jax.jit, static_argnames=("poisson",))
def _sample(expected, key, poisson, read_sigma):
    poisson_key, read_key = jr.split(key)
    mean = np.clip(expected, 0.0, None)
    signal = jr.poisson(poisson_key, mean).astype(expected.dtype) if poisson else expected
    return signal + read_sigma * jr.normal(read_key, expected.shape, expected.dtype)


def sample_frame(expected, config: dict, key: Array) -> Array:
    """Draws one noisy frame (photoelectrons) from the expected image (photoelectrons)."""
    noise = config.get("noise", {})
    expected = np.asarray(expected, dtype=float)
    return _sample(expected, key, bool(noise.get("poisson", True)), read_noise_sigma(config))


def read_noise_sigma(config: dict) -> float:
    """Read noise [e-] if enabled, else 0."""
    noise = config.get("noise", {})
    return float(noise["read_noise_electrons"]) if noise.get("read_noise", False) else 0.0
