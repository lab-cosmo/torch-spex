import numpy as np
import torch

from unittest import TestCase


class TestBasisSetSizes(TestCase):
    def test_hypers(self):
        from spex.radial.physical.laplacian_eigenstates import LaplacianEigenstates

        # test the allowed options for get_basis_size
        # numbers from https://luthaf.fr/rascaline/latest/how-to/le-basis.html
        cutoff = 4.4

        le = LaplacianEigenstates(
            cutoff, max_angular=None, max_eigenvalue=20, max_radial=None, trim=True
        )
        n_per_l = le.n_per_l
        assert n_per_l[0] == 6
        assert np.sum(n_per_l) == 44

        le = LaplacianEigenstates(cutoff, max_angular=None, max_radial=20, trim=True)
        n_per_l = le.n_per_l
        assert max(n_per_l) == 21
        assert n_per_l[0] == 21

        le = LaplacianEigenstates(cutoff, max_angular=20, trim=True)
        n_per_l = le.n_per_l
        assert len(n_per_l) == 21

        le = LaplacianEigenstates(cutoff, max_angular=20, max_radial=20, trim=True)
        n_per_l = le.n_per_l
        assert max(n_per_l) == 21
        assert len(n_per_l) == 21

        le = LaplacianEigenstates(cutoff, max_angular=4, max_radial=1, trim=True)
        n_per_l = le.n_per_l
        assert max(n_per_l) == 2
        assert len(n_per_l) == 5

        le = LaplacianEigenstates(cutoff, max_angular=20, max_radial=20, trim=False)
        n_per_l = le.n_per_l
        assert n_per_l[0] == 21
        assert n_per_l[-1] == 21
        assert len(n_per_l) == 21

        le_2 = LaplacianEigenstates(cutoff, max_angular=None, n_per_l=n_per_l)
        max_n_per_l2 = le_2.n_per_l
        assert max_n_per_l2 == n_per_l

    def test_shape(self):
        from spex.radial.physical.laplacian_eigenstates import LaplacianEigenstates

        basis = LaplacianEigenstates(
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


class TestRadialVsFeatomic(TestCase):
    def setUp(self):
        from featomic.basis import SphericalBessel

        self.cutoff = 4.4
        self.max_angular = 12
        self.max_radial = 10
        self.n_per_l = [self.max_radial + 1] * (self.max_angular + 1)

        self.r = np.linspace(0, self.cutoff, num=100)
        self.r_torch = torch.tensor(self.r)

        self.reference = {}
        self.d_reference = {}

        for l in range(self.max_angular + 1):
            reference_basis = SphericalBessel(
                radius=self.cutoff, max_radial=self.max_radial, angular_channel=l
            )
            self.reference[l] = reference_basis.compute(self.r, derivative=False)
            self.d_reference[l] = reference_basis.compute(self.r, derivative=True)

    def test_basis_directly(self):
        from spex.radial.physical.laplacian_eigenstates import LaplacianEigenstates

        le = LaplacianEigenstates(
            self.cutoff, max_angular=None, n_per_l=self.n_per_l, normalize=False
        )
        R, dR = le.get_basis_functions(self.cutoff, normalize=True)

        for l in range(self.max_angular + 1):
            for n in range(self.max_radial + 1):
                ours = R(self.r, n, l)
                np.testing.assert_allclose(self.reference[l][:, n], ours, atol=1e-12)

                ours = dR(self.r, n, l)
                np.testing.assert_allclose(self.d_reference[l][:, n], ours, atol=1e-12)

    def test_splined_and_jitted(self):
        from spex.radial.physical.laplacian_eigenstates import LaplacianEigenstates

        for spliner_accuracy, test_accuracy in ((1e-3, 1e-2), (1e-5, 1e-4)):
            basis = LaplacianEigenstates(
                self.cutoff,
                max_angular=None,
                n_per_l=self.n_per_l,
                normalize=True,
                spliner_accuracy=spliner_accuracy,
            )

            basis = torch.jit.script(basis)
            our_values = basis(self.r_torch)
            our_values = torch.cat(our_values, dim=-1)

            for n in range(self.max_radial + 1):
                for l in range(self.max_angular + 1):
                    ours = our_values[:, n + l * (self.max_radial + 1)].numpy()

                    np.testing.assert_allclose(
                        self.reference[l][:, n], ours, atol=test_accuracy
                    )

    def test_different_backends(self):
        from spex.radial.physical.laplacian_eigenstates import LaplacianEigenstates

        for device in ("cpu", "cuda"):
            # why is pytorch like this
            if device == "cuda":
                if not torch.cuda.is_available():
                    continue

            basis = LaplacianEigenstates(
                self.cutoff,
                max_angular=None,
                n_per_l=self.n_per_l,
                normalize=True,
                spliner_accuracy=1e-8,
            )

            basis = basis.to(device)
            r_torch = self.r_torch.to(device)
            our_values = torch.cat(basis(r_torch), dim=-1).cpu()

            for n in range(self.max_radial + 1):
                for l in range(self.max_angular + 1):
                    ours = our_values[:, n + l * (self.max_radial + 1)].numpy()

                    np.testing.assert_allclose(self.reference[l][:, n], ours, atol=1e-4)
