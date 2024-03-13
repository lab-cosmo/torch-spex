from unittest import TestCase
import torch
import numpy as np


class TestSphericalExpansion(TestCase):
    def test_instantiation(self):
        from spex import from_dict
        from spex.spherical_expansion import SphericalExpansion

        exp = SphericalExpansion()
        exp2 = from_dict(exp.to_dict())

    def test_is_on_fire(self):
        from spex.spherical_expansion import SphericalExpansion

        R_ij = torch.randn((5, 3))
        i = torch.tensor([0, 0, 1, 2, 2], dtype=torch.int64)
        Z_i = torch.tensor([8, 8, 64, 64, 64])

        exp = SphericalExpansion()

        exp(R_ij, i, Z_i)
