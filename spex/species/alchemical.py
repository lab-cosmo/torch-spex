import torch
from torch.nn import Embedding, Module


class Alchemical(Module):
    """Alchemical species embedding.

    Atomic species are mapped to a fixed-size space of "pseudo species".
    In ML terms, this is simply a trainable embedding; the model learns to
    "compress" elements into a smaller space of dimension ``pseudo_species``,
    rather than having to deal with the large space of all possible ones.

    Attributes:
        species (Tensor): Pseudo species used here, just counting from 0.
            (Can be used to map indices back to pseudo species names.)

    """

    def __init__(self, pseudo_species=4, total_species=100):
        """Initialise Alchemical species embedding.

        Args:
            pseudo_species (int): Number of pseudo species.
            total_species (int, optional): Total number of species to consider, i.e.,
                the size of the embedding dict. Usually no need to change this.

        """
        super().__init__()

        self.spec = {
            "pseudo_species": pseudo_species,
            "total_species": total_species,
        }

        # it is wasteful to carry around total_species embeddings, but it
        # makes things very simple -- revisit later if needed
        self.embedding = Embedding(total_species, pseudo_species)
        # note: Embedding is initialised with a normal distribution by default

        self.register_buffer(
            "species", torch.arange(pseudo_species, dtype=torch.int64), persistent=False
        )

    def forward(self, species):
        """Embed atomic species.

        Args:
            species (Tensor): Atomic species (ints) of shape ``[pair]``.

        Returns:
            Embeddings, a Tensor of shape ``[pair, pseudo_species]``.

        """
        # species: [pair]

        return self.embedding(species)  # -> [pair, pseudo_species]
