# `spex.radial`: Radial basis functions/distance expansions

This sub-package implements functions that featurise interatomic *distances* `r_ij` in some way that is "good" for machine learning.

There are different ways to approach this problem in combination with an angular expansion of the direction vector `R_ij` between atoms (see `spex.angular`): We can

1. Not care about different degrees `l` at all, and simply evaluate one set of basis functions,
2. Use different bases for each `l`, but keep their size fixed,
3. Use different bases for each `l` *and* change (typically decrease) their size for higher `l`.

We support all three options, in different bases.

`todo: write overview, etc.`


## Interfaces

The `__init__` of a radial basis must accept `max_degree` even if it is not used (case 1). This is because the radial basis is the source of truth for the overall spherical harmonic degree in a spherical expansion (since some basis may decide the degree at runtime). They must also accept `cutoff` argument, which indicates the cutoff radius. Both must be accessible as instance attributes; the cutoff needs to be a buffer.

The `forward` function of radial bases accepts a single argument `r` of length `pair`, with distances, and returns a tensor with dimension `[pair, ...]`. `...` stands for:

1. The number of basis functions indepenent of degree, or
2. The *different* basis functions for each degree flattened together.

Clearly, these two cases must be distinguished at some point, which we do with the following class attributes:

- `per_degree=True/False` indicates whether we need to split degree-wise
- `n_per_l` contains the split points to obtain degree-wise basis functions (must be set even for `per_degree=False` due to TorchScript limitations)

In the language above, `per_degree=False` is case 1, `per_degree=True` is 2 and 3. For 2, `n_per_l` is simply composed of identical numbers.

## Cutoff functions

In addition to expanding distances, we also need to have a way to smoothly push pairwise contributions to zero to ensure that the resulting potential energy surface is smoothly differentiable. This is the role of "cutoff functions", which are exposed in `spex.radial.cutoff`. We do *not* automatically multiply cutoff functions to the radial basis for conceptual clarity, and to support cases where the radial basis is processed further downstream.
