import torch
from torch.nn import Module
from torch.nn.functional import one_hot


class Orthogonal(Module):
    """Orthogonal species embedding.

    All species are embedded as orthogonal, i.e., one-hot vectors. No features
    are shared between elements, and the size of the embedding grows linearly
    with the number of different species to be considered.

    Attributes:
        species (Tensor): The species we are embedding here, mapping index to species.
        species_to_index (Tensor): The inverse of ``species``, mapping species to index.

    """

    def __init__(self, species):
        super().__init__()

        self.spec = {"species": species}

        species = torch.tensor(species, dtype=torch.int)
        species = torch.unique(species, sorted=True)
        self.n_species = len(species)

        # we use the standard torch one-hot encoding, but we don't
        # want to have big vectors with dimension "all possible elements",
        # so we first make a lookup from Z to [0...n_species-1]
        self.total_species = max(species) + 1
        species_to_index = torch.zeros(self.total_species, dtype=torch.int64)

        for i, s in enumerate(species):
            species_to_index[s] = i

        self.register_buffer("species_to_index", species_to_index, persistent=False)
        self.register_buffer("species", species, persistent=False)

    def forward(self, species):
        """Embed atomic species.

        Args:
            species (Tensor): Atomic species (ints) of shape ``[pair]``.

        Returns:
            Embeddings, a Tensor of shape ``[pair, n_species]``.

        """
        # species: [pair]

        idx = torch.index_select(self.species_to_index, 0, species)
        return one_hot(idx, self.n_species)  # -> [pair, n_species]
