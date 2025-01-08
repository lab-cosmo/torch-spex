# `spex.species`: Species embeddings

Here, we map an integer, the atomic number or chemical species, to a fixed-size array, either a one-hot encoding (`Orthogonal`) or a learned embedding (`Alchemical`).

The interface is as follows:

- Currently no restriction on `__init__`
- Must expose a buffer `species` of `int64` that contains labels for all the species that can be treated by the embedding. The lenght of `species` must match the output dimension. The actual numbers are not expected to be physically meaningful, but they *are* used by the `metatensor` interface.
- `forward` simply accepts an `int` tensor of `[pair]` and returns `[pair, len(species)]`.