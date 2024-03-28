from specable import Interface

specable = Interface("spex", "modules")

from_dict = specable.from_dict

# unusual order is to avoid problems if modules require from_dict
from .angular import SphericalHarmonics
from .radial import LaplacianEigenstates
from .species import Alchemical, Orthogonal
from .spherical_expansion import SphericalExpansion

modules = [
    SphericalHarmonics,
    LaplacianEigenstates,
    Orthogonal,
    Alchemical,
    SphericalExpansion,
]

__all__ = modules + [from_dict]
