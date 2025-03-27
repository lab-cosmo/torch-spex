import torch
from torch.nn import Module

import metatensor.torch as mts
from metatensor.torch import Labels, TensorBlock, TensorMap

from .spherical_expansion import SphericalExpansion


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

        l_to_treat = list(range(self.calculator.calculator.max_angular + 1))
        self.n_per_l = self.calculator.calculator.radial.n_per_l

        n_species = (
            species["Alchemical"]["pseudo_species"]
            if "Alchemical" in species
            else len(species["Orthogonal"]["species"])
        )
        self.shape = sum(self.n_per_l[ell] ** 2 * n_species**2 for ell in l_to_treat)

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

        expansion = self.calculator.forward(R_ij, i, j, species, structures, centers)
        expansion = mts.remove_dimension(expansion, "keys", "o3_sigma").keys_to_properties(
            "neighbor_type"
        )

        blocks: list[TensorBlock] = []
        for b in expansion.blocks():
            n_prop = int(b.values.shape[-1] ** 2)
            values = torch.einsum("smn,smN->snN", b.values, b.values).reshape(
                b.values.shape[0], n_prop
            )
            blocks.append(
                TensorBlock(
                    samples=b.samples,
                    components=[],
                    properties=Labels(
                        "property",
                        torch.arange(
                            n_prop,
                            dtype=torch.int32,
                            device=expansion[0].values.device,
                        ).unsqueeze(1),
                    ),
                    values=values,
                )
            )
        output = TensorMap(expansion.keys, blocks).keys_to_properties(["o3_lambda"])
        return output
