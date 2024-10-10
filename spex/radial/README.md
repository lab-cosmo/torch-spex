# `spex.radial`: Radial embeddings

This implements functions that featurise (or embed, or *expand*) interatomic *distances* `r_ij`, i.e., compute a set of features for distances within an atomic neighbourhood that are useful as inputs for machine learning methods. In the most general case, we compute one separate expansion per spherical harmonics degree, allowing different numbers of features per degree. This sub-package is built around this most general case, with simpler bases simply returning the same features per degree.

We keep the overall design decision of return a list features per degree, rather than a packed tensor.

## List of expansions

### "Physical" expansions

These expansions are based on the solutions of the Laplace equation on the sphere, with or without some additional smoothness constraints. They are built with a varying basis size per degree in mind, reducing the number of basis functions for higher spherical harmonics degrees. These bases are splined for efficiency.

- `LaplacianEigenstates`: Eigenfunctions of the Laplace equation on the sphere.
- `PhysicalBasis`: Eigenfunctions of an exponentially scaled Laplace operator on the sphere.

### "Simple" expansions

These are simply expansions in some well-known function bases, with the optional ability to learn an additional linear transformation (even per degree) of the resulting features.

- `Bernstein`: Bernstein polynomials
- ... (more to come)

## Interface

This defines the minimum requirements to implement a radial basis that works with the rest of `spex`, in particular with `spex.SphericalExpansion`.

Radial expansions must have the following attributes:

- `n_per_l`: Number of features per degree.
- `cutoff`: Torch buffer with the cutoff radius.
- `max_angular`: Integer, maximum degree and equal to `len(n_per_l) + 1`.

The `forward` function must accept:

- Single tensor with distances of size `pair`, and returning a list of features per degree, with each having the shape `[pair, self.n_per_l[l]]`.

The `__init__` function must accept:

- `cutoff`: a `float` argument.
- `max_angular`: an `int | None` keyword argument.

If `max_angular is not None`, the radial basis *must* guarantee that `len(self.n_per_l) == max_angular + 1`. If `max_angular` is not set, it can do whatever it wants, but this usecase will not be supported by `spex.SphericalExpansion`, which passes a fixed `max_angular` at `__init__`.
