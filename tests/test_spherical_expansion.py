import torch

from unittest import TestCase


class TestSphericalExpansion(TestCase):
    def test_instantiation(self):
        from spex import from_dict
        from spex.spherical_expansion import SphericalExpansion

        exp = SphericalExpansion()
        from_dict(exp.to_dict())

    def test_is_on_fire(self):
        from spex.spherical_expansion import SphericalExpansion

        R_ij = torch.randn((5, 3))
        i = torch.tensor([0, 0, 1, 2, 2], dtype=torch.int64)
        Z_i = torch.tensor([8, 8, 64, 64, 64])

        exp = SphericalExpansion()

        exp(R_ij, i, Z_i)

    def test_splined_and_jitted(self):
        from spex.spherical_expansion import SphericalExpansion

        exp = SphericalExpansion()
        exp = torch.jit.script(exp)

        R_ij = torch.randn((5, 3))
        i = torch.tensor([0, 0, 1, 2, 2], dtype=torch.int64)
        Z_i = torch.tensor([8, 8, 64, 64, 64])

        exp(R_ij, i, Z_i)

    def test_different_backends(self):
        from spex.spherical_expansion import SphericalExpansion

        for device in ("cpu", "cuda", "mps"):
            # why is pytorch like this
            if device == "cuda":
                if not torch.cuda.is_available():
                    continue
            if device == "mps":
                # sphericart does not support MPS; skip
                continue

            exp = SphericalExpansion()

            R_ij = torch.randn((5, 3))
            i = torch.tensor([0, 0, 1, 2, 2], dtype=torch.int64)
            Z_i = torch.tensor([8, 8, 64, 64, 64])

            R_ij = R_ij.to(device)
            i = i.to(device)
            Z_i = Z_i.to(device)
            exp = exp.to(device)
            exp(R_ij, i, Z_i)
