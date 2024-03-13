from torch.nn import Embedding, Module

from spex.engine import Specable


class Alchemical(Module, Specable):
    """Alchemical species embedding.

    Atomic species are mapped to a fixed-size space of "pseudo species".
    In ML terms, this is simply a trainable embedding; the model learns to
    "compress" elements into a smaller space of dimension ``pseudo_species``,
    rather than having to deal with the large space of all possible ones.

    """

    def __init__(self, pseudo_species=4, total_species=100):
        super().__init__()

        self.spec = {
            "pseudo_species": pseudo_species,
            "total_species": total_species,
        }

        # it is wasteful to carry around total_species embeddings, but it
        # makes things very simple -- revisit later if needed
        self.embedding = Embedding(total_species, pseudo_species)
        # note: Embedding is initialised with a normal distribution by default

    def forward(self, species):
        # species: [pair]

        return self.embedding(species)  # -> [pair, pseudo_species]
