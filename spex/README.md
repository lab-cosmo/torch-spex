### Roadmap

- Integrate `ruff` formatting + linting + isort etc. pp.
- Integrate `specable` & make serialisation possible
- Write `SphericalExpansion` class
- Write `metatensor` interface for `SphericalExpansion`
- Make actual `git` repo :)


### Conventions

- Init parameters (aka "hypers") should always be passed as bare python types: `int`, `float`, `list`, `dict` or `tuple`, so we can cleanly serialise them to `.yaml` without too much of a fuss
- `max_X` indicates the maximum value taken by `X`, starting from `0`. Therefore, `X` has `max_X + 1` values, from `0` to `max_X`. For example, `max_angular=3` indicates that spherical harmonics are enumerated from `l=0` to `l=3`. Note that this is in contrast to `rascaline`, where `max_radial` is exclusive and `max_angular` are inclusive.
- `X_per_Y` indicates the number of values `X` can take in channels of `Y`. For example, if there are `3` basis functions for `l=0` and `4` for `l=1`, this means `n_per_l=[3,4]`.
- Since `torch` does not yet fully support `jaxtyping`-style shape annotations, we make do with comments that indicate the expected shapes of inputs/intermediate results. We use the notation `[samples, (a=1, b=1) (a=2 b=0) ...]` to indicate a two-dimensional `Tensor` with one batch dimension and one feature dimension where features are flattened and correspond to certain combinations of "hidden" indices. `metatensor` provides capabilities to conveniently deal with sparse tensors of this type, but we avoid deep integration of `metatensor` into internals for now.
