# `spex.radial`: Radial embeddings

This implements functions that featurise (or embed, or *expand*) interatomic *distances* `r_ij`, i.e., compute a set of features for distances within an atomic neighbourhood that are useful as inputs for machine learning methods. In the most general case, we compute one separate expansion per spherical harmonics degree, allowing different numbers of features per degree. This sub-package is built around this most general case, with simpler bases simply returning the same features per degree.

We keep the overall design decision of return a list features per degree, rather than a packed tensor.

## List of expansions

### "Physical" expansions

These expansions are based on the solutions of the Laplace equation on the sphere, with or without some additional smoothness constraints. They are built with a varying basis size per degree in mind, reducing the number of basis functions for higher spherical harmonics degrees. These bases are splined for efficiency.

- `LaplacianEigenstates`: Eigenfunctions of the Laplace equation on the sphere.
- ...

### "Simple" expansions

These are simply expansions in some well-known function bases, with the optional ability to learn an additional linear transformation (even per degree) of the resulting features.

- `Bernstein`: Bernstein polynomials
- ...

## Interface

This defines the minimum requirements to implement a radial basis that works with the rest of `spex`.

Radial expansions must have the following attributes:

- `cutoff`: Cutoff radius *as a `torch` buffer*.
- `max_angular`: Number of spherical harmonics degrees.
- `n_per_l`: Number of features per degree.

It is important that all of these are present, because the radial basis acts as the single source of truth for the overall spherical harmonics degree `max_angular` and the `cutoff` radius, and we need shape information for model building.

The `forward` function must accept a single tensor with distances of size `pair`, and returns a list of features per degree, with each having the shape `[pair, self.n_per_l[l]]`.

The `__init__` function must accept at least the `cutoff` argument, and `max_angular` and `n_per_l` must always be set during `__init__`. Some bases, like `LaplacianEigenstates` decide these things based on other hypers.

## Cutoff functions

In addition to expanding distances, we also need to have a way to smoothly push pairwise contributions to zero to ensure that the resulting potential energy surface is smoothly differentiable. This is the role of "cutoff functions", which are exposed in `spex.radial.cutoff`. We do *not* automatically multiply cutoff functions to the radial basis for conceptual clarity, and to support cases where the radial basis is processed further downstream before pairwise summation.
