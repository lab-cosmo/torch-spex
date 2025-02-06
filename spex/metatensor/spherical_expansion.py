import torch
from torch.nn import Module

from metatensor.torch import Labels, TensorBlock, TensorMap

from spex import SphericalExpansion as torchSphericalExpansion


class SphericalExpansion(Module):
    """SphericalExpansion, but with metatensor output.

    Wrapper for ``spex.SphericalExpansion`` that returns a ``TensorMap``.

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

        Arguments are expected in the form of ``specable``-style dictionaries, i.e.,
        ``{ClassName: {key1: value1, key2: value2, ...}}``.

        Args:
            radial (dict): Radial expansion specification.
            angular (str): Type of angular expansion
                ("SphericalHarmonics", "SolidHarmonics" are supported).
            species (dict): Species embedding specification.

        """
        super().__init__()

        self.calculator = torchSphericalExpansion(
            cutoff, max_angular=max_angular, radial=radial, angular=angular, species=species
        )
        self.spec = self.calculator.spec

        # torch will not move species to devices -- this is too indirect?
        species = self.calculator.species.species
        self.register_buffer("species", species, persistent=False)

    def forward(self, R_ij, i, j, species, structures, centers):
        """Compute spherical expansion.

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
            Spherical expansion, a ``TensorMap``.

        """
        # R_ij: [pair, 3]
        # i: [pair]
        # j: [pair]
        # species: [center]
        # structures: [center]
        # centers: [center]

        output = self.calculator(R_ij, i, j, species)

        l_to_treat = torch.arange(
            self.calculator.max_angular + 1, dtype=i.dtype, device=i.device
        )
        all_center_species = torch.unique(species)
        all_neighbor_species = self.species

        # we're trying to match the rascaline output, which has keys for
        # l, species_center, species_neighbor ... so, we need to extract
        # the entries for each pair of species (or pseudo-species) at each l

        blocks: list[TensorBlock] = []  # type annotation needed for torchscript
        for l in l_to_treat:
            for species_center in all_center_species:
                for i_species_neighbor, species_neighbor in enumerate(all_neighbor_species):
                    center_mask = species == species_center
                    data = output[l][center_mask][..., i_species_neighbor]

                    blocks.append(
                        TensorBlock(
                            data,
                            samples=Labels(
                                ["system", "atom"],
                                torch.stack(
                                    (structures[center_mask], centers[center_mask])
                                ).T,
                            ),
                            components=[
                                Labels(
                                    "o3_mu",
                                    torch.arange(
                                        -l, l + 1, dtype=i.dtype, device=i.device
                                    ).unsqueeze(1),
                                )
                            ],
                            properties=Labels(
                                "n",
                                torch.arange(
                                    data.shape[2], dtype=i.dtype, device=i.device
                                ).unsqueeze(1),
                            ),
                        )
                    )

        # torchscript doesn't let us sanely write the keys into a list as we loop,
        # so we just do it ourselves here. repeat_interleave for outer, repeat for inner
        num_neighbor_species = all_neighbor_species.shape[0]
        num_center_species = all_center_species.shape[0]
        num_l = l_to_treat.shape[0]

        ls = l_to_treat.repeat_interleave(num_neighbor_species * num_center_species)
        center = all_center_species.repeat_interleave(num_neighbor_species).repeat(num_l)
        neighbor = all_neighbor_species.repeat(num_center_species).repeat(num_l)
        sigma = torch.ones(
            num_l * num_center_species * num_neighbor_species,
            dtype=i.dtype,
            device=i.device,
        )

        labels = Labels(
            ["o3_lambda", "o3_sigma", "center_type", "neighbor_type"],
            torch.stack((ls, sigma, center, neighbor)).T,
        )

        result = TensorMap(labels, blocks)

        return result
