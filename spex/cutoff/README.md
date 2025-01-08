# `spex.cutoff`: (smooth) cutoff functions

In addition to expanding distances with `spex.radial`, we also need to have a way to smoothly push pairwise contributions to zero to ensure that the resulting potential energy surface is smoothly differentiable. This is the role of "cutoff functions", which are exposed in `spex.cutoff`. We do *not* automatically multiply cutoff functions to the radial basis for conceptual clarity, and to support cases where the radial basis is processed further downstream before pairwise summation.

A cutoff function is initialised with at least the `cutoff` parameter, and then simply acts on a tensor of distances in the `forward` call, returning a tensor of the same shape with the values to be multiplied.
