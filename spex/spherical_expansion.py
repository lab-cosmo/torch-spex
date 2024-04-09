import torch
from torch.nn import Module

from spex import from_dict
from spex.engine import Specable


class SphericalExpansion(Module, Specable):
    """SphericalExpansion.

    Computes spherical expansions for neighbourhoods of atoms, embedding the
    radial and angular parts of the interatomic displacements as well as the
    neighbour atomic species. The embeddings are then aggregated, i.e., summed
    over the neighbours to generate per-atom features.

    """

    def __init__(
        self,
        radial={
            "LaplacianEigenstates": {"cutoff": 5.0, "max_radial": 20, "max_angular": 4}
        },
        angular="SphericalHarmonics",
        species={"Alchemical": {"pseudo_species": 4}},
    ):
        """Initialise SphericalExpansion.

        Arguments are expected in the form of ``specable``-style dictionaries, i.e.,
        ``{ClassName: {key1: value1, key2: value2, ...}}``.

        Args:
            radial (dict): Radial expansion specification.
            angular (str): Type of angular expansion.
                (Currently only "SphericalHarmonics" is supported.)
            species (dict): Species embedding specification.

        """
        super().__init__()
        self.spec = {
            "radial": radial,
            "angular": angular,
            "species": species,
        }

        # we can't rely on knowing max_angular head of time, so we need to
        # first instantiate the radial expansion and then check what we're dealing with
        self.radial = from_dict(radial)

        self.max_angular = self.radial.max_angular
        # todo: consider making this more modular somehow
        self.angular = from_dict({angular: {"max_angular": self.max_angular}})

        self.species = from_dict(species)

    def forward(self, R_ij, i, j, species):
        """Compute spherical expansion.

        Since we don't want to be in charge of computing displacements, we take an already-
        computed graph of ``R_ij``, ``i``, and ``j``, as well as center atom ``species``.
        From this perspective, a batch is just a very big graph with many disconnected
        subgraphs, this piece of code doesn't need to know the difference.

        Args:
            R_ij (Tensor): Interatomic displacements of shape ``[pair, 3]``,
                using the convention ``R_ij = R_j - R_i``.
            i (Tensor): Center atom indices of shape ``[pair]``.
            j (Tensor): Neighbour atom indices of shape ``[pair]``.
            species (Tensor): Atomic species of shape ``[center]``, indicating the species
                of the atoms indexed by ``i`` and ``j``.

        Returns:
            Spherical expansion, a tuple of tensors, one per ``l``, of shape
            ``[center, 2*l + 1, n, c]``, where ``n`` is the number of radial features,
            and ``c`` the size of the species embedding.

        """
        # R_ij: [pair, 3]
        # i: [pair]
        # j: [pair]
        # species: [center]

        # todo: consider making this more safe for very small or very large inputs
        r_ij = torch.sqrt((R_ij**2).sum(dim=-1))
        Z_j = species[j]

        # pairwise expansions
        radial_ij = self.radial(r_ij)  # -> [pair, l_and_n]
        angular_ij = self.angular(R_ij)  # -> [pair, l_and_m]
        species_ij = self.species(Z_j)  # -> [pair, species]

        # ... split into tuples per l
        radial_ij = radial_ij.split(
            self.radial.n_per_l, dim=-1
        )  # -> ([pair, l=0 n...], [pair, l=1 n...], ...)
        angular_ij = angular_ij.split(
            self.angular.m_per_l, dim=-1
        )  # -> ([pair, l=0 m...], [pair, l=1 m...], ...)

        # note: can't use tuples or return generatos because jit cannot infer their shape
        # ... outer products
        radial_and_angular_ij = list(
            torch.einsum("pn,pm->pmn", r, s) for r, s in zip(radial_ij, angular_ij)
        )  # -> [[pair, l=0 m,n], [pair, l=1 m,n], ...]

        full_expansion_ij = list(
            torch.einsum("pln,pc->plnc", ra, species_ij) for ra in radial_and_angular_ij
        )  # -> [[pair, l=0 m,n,c], [pair, l=1 m,n,c], ...]

        # aggregation over pairs
        #  note: scatter_add wouldn't work:
        #  it expects the index and source arrays to have the same shape,
        #  while index_add broadcasts across all the non-indexed dims
        full_expansion = list(
            (
                torch.zeros(
                    (species.shape[0], e.shape[1], e.shape[2], e.shape[3]),
                    dtype=e.dtype,
                    device=e.device,
                )
            ).index_add_(0, i, e)
            for e in full_expansion_ij
        )  # -> [[i, l=0 m,n,c], [i, l=1 m,n,c], ...]

        return full_expansion  # -> [[i, l=0 m,n,c], [i, l=1 m,n,c], ...]
