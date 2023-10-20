name = "dLuxToliman"
__version__ = "0.2.0"

# Import as modules
from . import optical_systems
from . import layers
from . import gradient_energy
from . import instruments
from . import sources

# Wavefronts and Optics
from .optical_systems import *
from .layers.optical_layers import *
from .layers.detector_layers import *
from .gradient_energy import *
from .instruments import *
from .sources import *

# Add to __all__
modules = [
    optical_systems,
    layers,
    gradient_energy,
    instruments,
    sources,
]

__all__ = [module.__all__ for module in modules]
