"""
toliman_sim: config-driven simulation of TOLIMAN data on top of dLuxToliman.

    python -m toliman_sim configs/example.json
"""

import jax

# Astrometry at the micro-arcsecond level needs double precision. This must
# happen before any dLux/JAX arrays are created.
jax.config.update("jax_enable_x64", True)

from .config import load_config
from .builders import build_optics, build_source, build_detector, build_telescope
from .noise import sample_frame
from .simulate import simulate, save, run

__all__ = [
    "load_config",
    "build_optics",
    "build_source",
    "build_detector",
    "build_telescope",
    "sample_frame",
    "simulate",
    "save",
    "run",
]
