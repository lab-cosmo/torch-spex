# `spex.angular`: Angular embeddings

This module defines components that embed vectors $R$, typically the interatomic displacement vectors $R_{ij} = R_j - R_i$, into a suitable high-dimensional space. Often, only the normalised vectors are used, making this an embedding of angular (or orientation) information.

At present, we support spherical harmonics and solid harmonics, provided by [`sphericart`](https://github.com/lab-cosmo/sphericart/). Other implementations have to provide the following interface:

- `__init__` receives at least an argument `max_angular` that defines the maximum angular degree (inclusive) starting always from zero
- Expose `max_angular` and `m_per_l` as attributes, the former being the same as the input argument and `m_per_l` the number of expansion components per angular degree
- `forward` accepts a tensor of vectors (shape `[pair, 3]`) and returns a list of tensors ordered in decreasing order of `l`, with shape matching `m_per_l`. `pair` remains the leading dimension of each tensor.