"""Metatensor interface.

This exposes a version of ``spex`` that returns ``TensorMap``s instead of
pure torch ``Tensor``s. Currently we don't accept ``metatensor`` input.

"""

from .spherical_expansion import SphericalExpansion

__all__ = [SphericalExpansion]
