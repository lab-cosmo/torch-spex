from specable import Interface

specable = Interface("spex", "modules")

from_dict = specable.from_dict

from .engine import Specable
from .angular import SphericalHarmonics
from .radial import LaplacianEigenstates
from .species import Orthogonal, Alchemical
from .spherical_expansion import SphericalExpansion

modules = [
    SphericalHarmonics,
    LaplacianEigenstates,
    Orthogonal,
    Alchemical,
    SphericalExpansion,
]
