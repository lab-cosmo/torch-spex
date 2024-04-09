import torch
from torch.nn import Module

from typing import List

from metatensor.torch import Labels, TensorBlock, TensorMap

from spex import SphericalExpansion as torchSphericalExpansion
from spex.engine import Specable


class SphericalExpansion(Module, Specable):
    """SphericalExpansion with metatensor output.

    Wrapper for ``spex.SphericalExpansion`` that returns a ``TensorMap``.

    Args:
        radial (dict): Parameters for the radial expansion.
        angular (str): Type of angular expansion.
            (Currently only ``"SphericalHarmonics"`` is supported.)
        species (dict): Parameters for the chemical embedding.

    Attributes:
        species (Tensor): The atomic species considered in the output.

    """

    def __init__(
        self,
        radial={"LaplacianEigenstates": {"cutoff": 5.0, "max_radial": 20}},
        angular="SphericalHarmonics",
        species={"Alchemical": {"pseudo_species": 4}},
    ):
        super().__init__()

        self.calculator = torchSphericalExpansion(
            radial=radial, angular=angular, species=species
        )
        self.spec = self.calculator.spec
        self.species = self.calculator.species.species

    def forward(self, R_ij, i, j, species, structures, centers):
        output = self.calculator(R_ij, i, j, species)

        l_to_treat = torch.arange(
            self.calculator.max_angular + 1, dtype=i.dtype, device=i.device
        )

        blocks: List[TensorBlock] = []  # type annotation for torchscript
        for l in l_to_treat:
            for species_center in self.species:
                for i_species_neighbor, species_neighbor in enumerate(self.species):
                    center_mask = species == species_center
                    data = output[l][center_mask][..., i_species_neighbor]

                    blocks.append(
                        TensorBlock(
                            data,
                            samples=Labels(
                                ["structure", "center"],
                                torch.stack(
                                    (structures[center_mask], centers[center_mask])
                                ).T,
                            ),
                            components=[
                                Labels(
                                    "spherical_harmonics_m",
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

        num_species = self.species.shape[0]
        num_l = l_to_treat.shape[0]

        ls = l_to_treat.repeat_interleave(num_species * num_species)
        center = self.species.repeat_interleave(num_species).repeat(num_l)
        neighbor = self.species.repeat(num_species).repeat(num_l)

        labels = Labels(
            ["spherical_harmonics_l", "species_center", "species_neighbor"],
            torch.stack((ls, center, neighbor)).T,
        )

        result = TensorMap(labels, blocks)

        return result
