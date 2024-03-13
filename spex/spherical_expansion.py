import numpy as np

from torch.nn import Module

from spex import from_dict
from spex.engine import Specable


class SphericalExpansion(Module, Specable):
    def __init__(
        self,
        radial={"LaplacianEigenstates": {"cutoff": 5.0, "max_radial": 20}},
        angular="SphericalHarmonics",
        species={"Alchemical": {"pseudo_species": 4}},
    ):
        super().__init__()
        self.spec = {
            "radial": radial,
            "angular": angular,
            "species": species,
        }

        # we can't rely on knowing max_angular head of time, so we need to
        # first instantiate the radial expansion and then look
        self.radial = from_dict(radial)

        self.max_angular = self.radial.max_angular
        # todo: consider making this more modular somehow
        self.angular = from_dict({angular: {"max_angular": self.max_angular}})

        self.species = from_dict(species)

    def forward(self, R_ij, i, Z_j):
        # R_ij: [pair, 3]
        # i: [pair]
        # Z_j: [pair]

        # pair expansion

        r_ij = (R_ij**2).sum(dim=-1)

        radial_ij = self.radial(r_ij)  # -> [pair, l_and_n]
        angular_ij = self.angular(R_ij)  # -> [pair, l_and_m]
        species_ij = self.species(Z_j)  # -> [pair, species]

        print(radial_ij.shape)
        print(np.sum(self.radial.n_per_l))
        print(self.radial.n_per_l)

        radial_ij = radial_ij.split(
            self.radial.n_per_l, dim=-1
        )  # -> ([l=0 n...], [l=1 n...])
        angular_ij = angular_ij.split(
            self.angular.m_per_l, dim=-1
        )  # -> ([l=0 m...], [l=1 m...])

        radial_and_angular_ij = (
            s.unsqueeze(-1) * r.unsqueeze(-2) for r, s in zip(radial_ij, angular_ij)
        )  # -> ([l=0 m,n], [l=1 m,n])

        full_expansion_ij = (
            ra.unsqueeze(-1) * c.unsqueeze(-2)
            for ra, c in zip(radial_and_angular_ij, species_ij)
        )  # -> ([l=0 m,n,c], [l=1 m,n,c]) ...

        # aggregation

        full_expansion = (
            (torch.zeros_like(e)).scatter_add_(0, i, e) for e in full_expansion_ij
        )

        return full_expansion
