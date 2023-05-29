name = "dLuxToliman"
__version__ = "0.1.0"


# Import as modules
from . import optics
from . import optical_layers
from . import gradient_energy
from . import instruments
from . import sources

# Wavefronts and Optics
from .optics import *
from .optical_layers import *
from .gradient_energy import *
from .instruments import *
from .sources import *

# Add to __all__
modules = [
    optics,
    optical_layers,
    gradient_energy,
    instruments,
    sources,
]

__all__ = [module.__all__ for module in modules]