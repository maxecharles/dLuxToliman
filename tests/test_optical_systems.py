from jax import numpy as np, config

config.update("jax_debug_nans", True)
import pytest
from dLuxToliman import (
    TolimanOpticalSystem,
)
from dLux.layers import Optic

