# todo: this test is not really checking against featomic
#       -- will remove magic saved stuff later
import torch

from unittest import TestCase


class TestMetatensorSoapPowerSpectrum(TestCase):
    def test_jit(self):
        from spex.metatensor.soap_power_spectrum import SoapPowerSpectrum

        R_ij = torch.randn((6, 3))
        i = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int64)
        j = torch.tensor([1, 2, 0, 2, 0, 1], dtype=torch.int64)
        species = torch.tensor([8, 8, 64])

        centers = torch.tensor([0, 1, 2])
        structures = torch.tensor([0, 0, 0])

        exp = SoapPowerSpectrum(5.0)
        exp = torch.jit.script(exp)

        exp(R_ij, i, j, species, structures, centers)

    def test_different_backends(self):
        from spex.metatensor.soap_power_spectrum import SoapPowerSpectrum

        for device in ("cpu", "cuda"):
            # why is pytorch like this
            if device == "cuda":
                if not torch.cuda.is_available():
                    continue

            exp = SoapPowerSpectrum(5.0)

            R_ij = torch.randn((6, 3))
            i = torch.tensor([0, 0, 1, 1, 2, 2])
            j = torch.tensor([1, 2, 0, 2, 0, 1])
            species = torch.tensor([8, 8, 64])

            centers = torch.tensor([0, 1, 2])
            structures = torch.tensor([0, 0, 0])

            R_ij = R_ij.to(device)
            i = i.to(device)
            j = j.to(device)
            species = species.to(device)
            exp = exp.to(device)

            centers = centers.to(device)
            structures = structures.to(device)

            exp(R_ij, i, j, species, centers, structures)
