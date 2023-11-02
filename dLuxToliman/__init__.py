name = "dLuxToliman"
__version__ = "0.2.0"

# Import as modules
from . import optical_systems
from . import layers
from . import gradient_energy
from . import telescopes
from . import sources

# Wavefronts and Optics
from .optical_systems import *
from .layers.optical_layers import *
from .layers.detector_layers import *
from .gradient_energy import *
from .telescopes import *
from .sources import *

# Add to __all__
modules = [
    optical_systems,
    layers,
    gradient_energy,
    telescopes,
    sources,
]

__all__ = [module.__all__ for module in modules]
