import numpy as np
import torch

from unittest import TestCase


class TestBasisSetSizes(TestCase):
    def test_hypers(self):
        from spex.radial.physical.physical_basis import PhysicalBasis

        cutoff = 4.4

        physical_basis = PhysicalBasis(
            cutoff, max_angular=None, max_eigenvalue=50.0, max_radial=None, trim=True
        )
        n_per_l = physical_basis.n_per_l
        np.testing.assert_equal(n_per_l, np.array([3, 3, 2, 1]))

        physical_basis = PhysicalBasis(cutoff, max_angular=None, max_radial=8, trim=True)
        n_per_l = physical_basis.n_per_l
        assert max(n_per_l) == 9
        assert n_per_l[0] == 9

        physical_basis = PhysicalBasis(cutoff, max_angular=6, trim=True)
        n_per_l = physical_basis.n_per_l
        assert len(n_per_l) == 7

        physical_basis = PhysicalBasis(cutoff, max_angular=6, max_radial=6, trim=True)
        n_per_l = physical_basis.n_per_l
        assert max(n_per_l) == 7
        assert len(n_per_l) == 7

        physical_basis = PhysicalBasis(cutoff, max_angular=6, max_radial=6, trim=False)
        n_per_l = physical_basis.n_per_l
        assert n_per_l[0] == 7
        assert n_per_l[-1] == 7
        assert len(n_per_l) == 7

        physical_basis_2 = PhysicalBasis(cutoff, max_angular=None, n_per_l=n_per_l)
        n_per_l2 = physical_basis_2.n_per_l
        assert n_per_l2 == n_per_l

    def test_shape(self):
        from spex.radial.physical.physical_basis import PhysicalBasis

        basis = PhysicalBasis(
            4.4,
            max_angular=None,
            max_eigenvalue=20,
            max_radial=None,
        )

        r = torch.randn(100)

        out = basis(r)

        for r in out:
            assert r.shape[0] == 100
        assert np.sum([r.shape[1] for r in out]) == np.sum(basis.n_per_l)


class TestRadialVsPhysicalBasisPackage(TestCase):
    def setUp(self):
        from physical_basis import PhysicalBasis as PhysicalBasisReference

        self.cutoff = 4.4
        self.max_angular = 12
        self.max_radial = 10
        self.n_per_l = [self.max_radial + 1] * (self.max_angular + 1)

        self.r = np.linspace(0, self.cutoff, num=100)
        self.r_torch = torch.tensor(self.r)

        self.reference_basis = PhysicalBasisReference()

    def test_basis_directly(self):
        from spex.radial.physical.physical_basis import PhysicalBasis

        physical_basis = PhysicalBasis(
            self.cutoff, max_angular=None, n_per_l=self.n_per_l, normalize=False
        )
        R, dR = physical_basis.get_basis_functions(self.cutoff, normalize=False)

        for n in range(self.max_radial + 1):
            for l in range(self.max_angular + 1):
                reference = self.reference_basis.compute(n, l, self.r)
                ours = R(self.r, n, l)

                np.testing.assert_allclose(reference, ours)

                reference = self.reference_basis.compute_derivative(n, l, self.r)
                ours = dR(self.r, n, l)

                np.testing.assert_allclose(reference, ours)

    def test_splined_and_jitted(self):
        from spex.radial.physical.physical_basis import PhysicalBasis

        for spliner_accuracy, test_accuracy in ((1e-4, 1e-2), (1e-6, 1e-4)):
            basis = PhysicalBasis(
                self.cutoff,
                max_angular=None,
                n_per_l=self.n_per_l,
                normalize=False,
                spliner_accuracy=spliner_accuracy,
            )

            basis = torch.jit.script(basis)
            our_values = basis(self.r_torch)

            for n in range(self.max_radial + 1):
                for l in range(self.max_angular + 1):
                    reference = self.reference_basis.compute(n, l, self.r)
                    ours = our_values[l][:, n]

                    np.testing.assert_allclose(
                        reference, ours, atol=test_accuracy, rtol=test_accuracy
                    )

    def test_different_backends(self):
        from spex.radial.physical.physical_basis import PhysicalBasis

        for device in ("cpu", "cuda"):
            # why is pytorch like this
            if device == "cuda":
                if not torch.cuda.is_available():
                    continue

            basis = PhysicalBasis(
                self.cutoff,
                max_angular=None,
                n_per_l=self.n_per_l,
                normalize=False,
            )

            basis = basis.to(device)
            r_torch = self.r_torch.to(device)
            our_values = torch.cat(basis(r_torch), dim=-1).cpu()

            for n in range(self.max_radial + 1):
                for l in range(self.max_angular + 1):
                    reference = self.reference_basis.compute(n, l, self.r)
                    ours = our_values[:, n + l * (self.max_radial + 1)].numpy()

                    np.testing.assert_allclose(reference, ours, atol=1e-4)
