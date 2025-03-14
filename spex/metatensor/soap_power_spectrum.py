import torch
from torch.nn import Module

from metatensor.torch import Labels, TensorBlock, TensorMap

from spex import SphericalExpansion


class SoapPowerSpectrum(Module):
    def __init__(
        self,
        cutoff,
        max_angular=3,
        radial={"LaplacianEigenstates": {"max_radial": 8}},
        angular="SphericalHarmonics",
        species={"Alchemical": {"pseudo_species": 4}},
        cutoff_function={"ShiftedCosine": {"width": 0.5}},
    ):
        """Initialise SoapPowerSpectrum.

        Arguments are expected in the form of ``specable``-style dictionaries, i.e.,
        ``{ClassName: {key1: value1, key2: value2, ...}}``.

        Args:
            radial (dict): Radial expansion specification.
            angular (str): Type of angular expansion
                ("SphericalHarmonics", "SolidHarmonics" are supported).
            species (dict): Species embedding specification.

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

        self.calculator = SphericalExpansion(**self.spec)

        l_to_treat = list(range(self.calculator.max_angular + 1))
        self.n_per_l = self.calculator.radial.n_per_l
        self.shape = sum(self.n_per_l[ell] ** 2 for ell in l_to_treat)

        species = self.calculator.species.species
        self.register_buffer("species", species, persistent=False)

        self.max_radial = next(iter(radial.values()))["max_radial"]

    def forward(self, R_ij, i, j, species, structures, centers):
        """Compute soap power spectrum.

        Since we don't want to be in charge of computing displacements, we take an already-
        computed graph of ``R_ij``, ``i``, and ``j``, as well as center atom ``species``.
        From this perspective, a batch is just a very big graph with many disconnected
        subgraphs, the spherical expansion doesn't need to know the difference.

        However, ``metatensor`` output is expected to contain more information, so if our
        input is a big "batch" graph, we need some additional information to keep track of
        which nodes in the big graph belong to which original structure (``structures``)
        and which atom in each structure is which (``centers``).

        For a single-structure graph, this would be just zeros for ``structures`` and
        ``torch.arange(n_atoms)`` for ``centers``. For a two-structure graph, it would
        be a block of zeros and a block of ones for ``structures``, and then a range going
        up to ``n_atoms_0`` and then a range going up to ``n_atoms_1`` for ``centers``.

        Note that we take the center species to consider from the input species, so if
        a given graph doesn't contain a given center species, it will also not appear in
        the output.

        Args:
            R_ij (Tensor): Interatomic displacements of shape ``[pair, 3]``,
                using the convention ``R_ij = R_j - R_i``.
            i (Tensor): Center atom indices of shape ``[pair]``.
            j (Tensor): Neighbour atom indices of shape ``[pair]``.
            species (Tensor): Atomic species of shape ``[center]``, indicating the species
                of the atoms indexed by ``i`` and ``j``.
            structures (Tensor): Structure indices of shape ``[center]``, indicating which
                structure each atom belongs to.
            centers (Tensor): Center atom indices of shape ``[center]``, indicating which
                atom in each structure a given node in the graph is supposed to be.

        Returns:
            SOAP power spectrum, a ``TensorMap``.

        """
        # R_ij: [pair, 3]
        # i: [pair]
        # j: [pair]
        # species: [center]
        # structures: [center]
        # centers: [center]

        expansion = self.calculator.forward(R_ij, i, j, species)
        output = [
            torch.einsum("imnc,imNC->inNcC", e, e) for e in expansion
        ]  # -> [[i, n1, n2, c1, c2], [...], ...]

        l_to_treat = torch.arange(
            self.calculator.max_angular + 1, dtype=i.dtype, device=i.device
        )

        # Create the `properties` tensor
        property_dimension: int = 0
        for ell in l_to_treat:
            property_dimension += self.n_per_l[ell] * self.n_per_l[ell]

        # Pre-allocate the tensor with shape (total, 3)
        properties = torch.empty((property_dimension, 3), dtype=i.dtype, device=i.device)
        idx: int = 0

        # Fill the tensor using explicit nested loops
        for ell in l_to_treat:
            for n1 in range(self.n_per_l[ell]):
                for n2 in range(self.n_per_l[ell]):
                    properties[idx, 0] = ell
                    properties[idx, 1] = n1
                    properties[idx, 2] = n2
                    idx += 1

        all_center_species = torch.unique(species).to(dtype=i.dtype)
        all_neighbor_species = self.species

        # we're trying to match the rascaline output, which has keys for
        # l, species_center, species_neighbor ... so, we need to extract
        # the entries for each pair of species (or pseudo-species) at each l

        blocks: list[TensorBlock] = []
        data_: list[torch.Tensor] = []
        for species_center in all_center_species:
            center_mask = species == species_center
            for i1 in range(len(all_neighbor_species)):
                for i2 in range(len(all_neighbor_species)):
                    data_ = [
                        output[ell][center_mask][..., i1, i2].reshape(sum(center_mask), -1)
                        for ell in l_to_treat
                    ]
                    data = torch.cat(data_, dim=1)

                    blocks.append(
                        TensorBlock(
                            values=data,
                            samples=Labels(
                                ["system", "atom"],
                                torch.stack(
                                    (structures[center_mask], centers[center_mask])
                                ).T,
                            ),
                            components=[],
                            properties=Labels(["l", "n_1", "n_2"], properties),
                        )
                    )

        # torchscript doesn't let us sanely write the keys into a list as we loop,
        # so we just do it ourselves here. repeat_interleave for outer, repeat for inner
        labels = Labels(
            ["center_type", "neighbor_1_type", "neighbor_2_type"],
            torch.cartesian_prod(
                all_center_species, all_neighbor_species, all_neighbor_species
            ),
        )

        return TensorMap(
            labels,
            blocks,
        )
