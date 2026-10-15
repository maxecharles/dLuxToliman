"""
toliman_sim: config-driven simulation of TOLIMAN data on top of dLuxToliman.

    python -m toliman_sim configs/example.json

Simulations need JAX double precision. Importing this package does not change any
global settings; enable it before building models with
    jax.config.update("jax_enable_x64", True)
(simulate() enables it, with a warning, if you have not; the command line does it
for you).
"""

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
