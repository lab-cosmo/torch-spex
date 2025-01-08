"""Cutoff functions.

Functions that can be multiplied with other expansions to ensure
that they smoothly (and differentiably) decay to zero as atoms
approach and cross the cutoff radius.

"""

from .shifted_cosine import ShiftedCosine
from .step import Step

__all__ = [ShiftedCosine, Step]
