import torch
from torch.nn import Module

from spex import from_dict
from spex.engine.dicts import update_dict

from .utils import compute_distance


class SphericalExpansion(Module):
    """SphericalExpansion.

    Computes spherical expansions for neighbourhoods of atoms, embedding the
    radial and angular parts of the interatomic displacements as well as the
    neighbour atomic species. The embeddings are then aggregated, i.e., summed
    over the neighbours to generate per-atom features.

    Attributes:
        cutoff (float): Cutoff radius.
        max_angular (int): Maximum spherical harmonics degree.
        shape (tuple): Shape of the output tensor per spherical harmonics degree,
            excluding the leading "pair" dimension. Contains the number of spherical
            channels ``m``, then the number of radial channels ``n``, and finally
            the number of species channels ``c``, in that order.

    """

    def __init__(
        self,
        cutoff,
        max_angular=3,
        radial={"LaplacianEigenstates": {"max_radial": 8}},
        angular="SphericalHarmonics",
        species={"Alchemical": {"pseudo_species": 4}},
        cutoff_function={"ShiftedCosine": {"width": 0.5}},
    ):
        """Initialise SphericalExpansion.

        Arguments for the radial, angular, and species expansions are expected
        in the form of ``specable``-style dictionaries, i.e.,
        ``{ClassName: {key1: value1, key2: value2, ...}}``.

        Values for ``cutoff`` and ``max_angular`` will be passed to the radial
        and angular expansions, and will overwrite any values specified within.

        Args:
            cutoff (float): Cutoff radius.
            max_angular (int): Maximum spherical harmonics degree.
            radial (dict): Radial expansion specification.
            angular (str): Type of angular expansion
                ("SphericalHarmonics", "SolidHarmonics" are supported).
            species (dict): Species embedding specification.
            cutoff_function (dict): Cutoff function specification.
                ("ShiftedCosine" and "Step" are supported, however, it is not
                recommended to use a step function cutoff, as it leads to
                discontinuities in the potential energy surface!).

        """
        super().__init__()
        self.spec = {
            "cutoff": cutoff,
            "max_angular": max_angular,
            "radial": radial,
            "angular": angular,
            "species": species,
            "cutoff_function": cutoff_function,
        }

        self.cutoff = cutoff
        self.max_angular = max_angular

        radial = update_dict(
            radial, {"cutoff": self.cutoff, "max_angular": self.max_angular}
        )
        self.radial = from_dict(radial, module="spex.radial")

        angular = update_dict(angular, {"max_angular": self.max_angular})
        self.angular = from_dict(angular, module="spex.angular")

        self.species = from_dict(species, module="spex.species")

        cutoff_function = update_dict(cutoff_function, {"cutoff": self.cutoff})
        self.cutoff_fn = from_dict(cutoff_function, module="spex.cutoff")

        self.shape = tuple(
            (m, int(self.radial.n_per_l[l]), self.species.species.shape[0])
            for l, m in enumerate(self.angular.m_per_l)
        )

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

        r_ij = compute_distance(R_ij)  # -> [pair]
        Z_j = species[j]

        # pairwise expansions
        radial_ij = self.radial(r_ij)  # -> [[pair, l=0 n], [pair, l=1 n], ...]
        cutoff_ij = self.cutoff_fn(r_ij)  # -> [pair]
        angular_ij = self.angular(R_ij)  # -> [[pair, l=0 m=0], [pair, l=1 m=-1, ...] ...]
        species_ij = self.species(Z_j)  # -> [pair, species]

        # apply cutoff
        radial_ij = [r * cutoff_ij.unsqueeze(-1) for r in radial_ij]

        # perform outer products
        # note: can't use tuples or return generators because jit cannot infer their shape
        radial_and_angular_ij = [
            torch.einsum("pn,pm->pmn", r, s) for r, s in zip(radial_ij, angular_ij)
        ]  # -> [[pair, l=0 m,n], [pair, l=1 m,n], ...]
        full_expansion_ij = [
            torch.einsum("pln,pc->plnc", ra, species_ij) for ra in radial_and_angular_ij
        ]  # -> [[pair, l=0 m,n,c], [pair, l=1 m,n,c], ...]

        # aggregation over pairs
        #  note: scatter_add wouldn't work:
        #  it expects the index and source arrays to have the same shape,
        #  while index_add broadcasts across all the non-indexed dims
        #  note: we also can't extract this into a general function because torchscript
        #        is unable to infer varying shapes for the outputs
        full_expansion = [
            torch.zeros(
                (species.shape[0], e.shape[1], e.shape[2], e.shape[3]),
                dtype=e.dtype,
                device=e.device,
            ).index_add_(0, i, e)
            for e in full_expansion_ij
        ]  # -> [[i, l=0 m,n,c], [i, l=1 m,n,c], ...]

        return full_expansion  # -> [[i, l=0 m,n,c], [i, l=1 m,n,c], ...]
