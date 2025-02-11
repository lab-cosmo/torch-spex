import torch

from unittest import TestCase


class TestSphericalExpansion(TestCase):
    def test_instantiation(self):
        from spex import from_dict, to_dict
        from spex.spherical_expansion import SphericalExpansion

        exp = SphericalExpansion(5.0)
        from_dict(to_dict(exp))

    def test_shapes(self):
        from spex.spherical_expansion import SphericalExpansion

        R_ij = torch.randn((6, 3))
        i = torch.tensor([0, 0, 1, 1, 2, 2])
        j = torch.tensor([1, 2, 0, 2, 0, 1])
        species = torch.tensor([8, 8, 64])

        exp = SphericalExpansion(
            cutoff=5.0,
            max_angular=3,
            radial={
                "LaplacianEigenstates": {
                    "max_radial": 20,
                    "trim": False,
                }
            },
            angular="SphericalHarmonics",
            species={"Alchemical": {"pseudo_species": 4}},
        )

        result = exp(R_ij, i, j, species)

        assert len(result) == 4
        for l, r in enumerate(result):
            assert r.shape[0] == len(species)
            assert r.shape[1] == 2 * l + 1
            assert r.shape[2] == 21
            assert r.shape[3] == 4

    def test_jit(self):
        from spex.spherical_expansion import SphericalExpansion

        for radial in [
            {
                "LaplacianEigenstates": {
                    "max_radial": 20,
                    "trim": False,
                }
            },
            {
                "Bernstein": {
                    "num_radial": 20,
                }
            },
        ]:
            exp = SphericalExpansion(cutoff=5.0, max_angular=3, radial=radial)
            exp = torch.jit.script(exp)

            R_ij = torch.randn((6, 3))
            i = torch.tensor([0, 0, 1, 1, 2, 2])
            j = torch.tensor([1, 2, 0, 2, 0, 1])
            species = torch.tensor([8, 8, 64])

            exp(R_ij, i, j, species)

    def test_different_backends(self):
        from spex.spherical_expansion import SphericalExpansion

        for device in ("cpu", "cuda"):
            # why is pytorch like this
            if device == "cuda":
                if not torch.cuda.is_available():
                    continue

            exp = SphericalExpansion(5.0)

            R_ij = torch.randn((6, 3))
            i = torch.tensor([0, 0, 1, 1, 2, 2])
            j = torch.tensor([1, 2, 0, 2, 0, 1])
            species = torch.tensor([8, 8, 64])

            R_ij = R_ij.to(device)
            i = i.to(device)
            j = j.to(device)
            species = species.to(device)
            exp = exp.to(device)

            exp(R_ij, i, j, species)
