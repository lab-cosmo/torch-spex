import numpy as np
import torch

from unittest import TestCase


class TestBasisSetSizes(TestCase):
    def test_hypers(self):
        from spex.radial.physical.laplacian_eigenstates import LaplacianEigenstates

        # test the allowed options for get_basis_size
        # numbers from https://luthaf.fr/rascaline/latest/how-to/le-basis.html
        cutoff = 4.4

        le = LaplacianEigenstates(cutoff, max_eigenvalue=20, max_radial=None, trim=True)
        n_per_l = le.n_per_l
        assert n_per_l[0] == 6
        assert np.sum(n_per_l) == 44

        le = LaplacianEigenstates(cutoff, max_radial=20, trim=True)
        n_per_l = le.n_per_l
        assert max(n_per_l) == 21
        assert n_per_l[0] == 21

        le = LaplacianEigenstates(cutoff, max_angular=20, trim=True)
        n_per_l = le.n_per_l
        assert len(n_per_l) == 21

        le = LaplacianEigenstates(cutoff, max_angular=20, max_radial=20, trim=True)
        n_per_l = le.n_per_l
        assert max(n_per_l) == 21
        assert len(n_per_l) <= 21

        le = LaplacianEigenstates(cutoff, max_angular=20, max_radial=20, trim=False)
        n_per_l = le.n_per_l
        assert n_per_l[0] == 21
        assert n_per_l[-1] == 21
        assert len(n_per_l) == 21

        le_2 = LaplacianEigenstates(cutoff, n_per_l=n_per_l)
        max_n_per_l2 = le_2.n_per_l
        assert max_n_per_l2 == n_per_l

    def test_shape(self):
        from spex.radial.physical.laplacian_eigenstates import LaplacianEigenstates

        basis = LaplacianEigenstates(
            4.4,
            max_eigenvalue=20,
            max_radial=None,
        )

        r = torch.randn(100)

        out = basis(r)

        for r in out:
            assert r.shape[0] == 100
        assert np.sum([r.shape[1] for r in out]) == np.sum(basis.n_per_l)


class TestRadialVsRascaline(TestCase):
    def setUp(self):
        from rascaline.utils import SphericalBesselBasis

        self.cutoff = 4.4
        self.max_angular = 12
        self.max_radial = 10
        self.n_per_l = [self.max_radial + 1] * (self.max_angular + 1)

        self.r = np.linspace(0, self.cutoff, num=100)
        self.r_torch = torch.tensor(self.r)

        # rascaline insists on being off-by-one in max_radial
        self.reference_basis = SphericalBesselBasis(
            self.cutoff, self.max_radial + 1, self.max_angular
        )

    def test_basis_directly(self):
        from spex.radial.physical.laplacian_eigenstates import LaplacianEigenstates

        le = LaplacianEigenstates(
            self.cutoff, n_per_l=self.n_per_l, normalize=False
        )
        R, dR = le.get_basis_functions(self.cutoff, normalize=False)

        for n in range(self.max_radial + 1):
            for l in range(self.max_angular + 1):
                reference = self.reference_basis.compute(n, l, self.r)
                ours = R(self.r, n, l)

                np.testing.assert_allclose(reference, ours)

                reference = self.reference_basis.compute_derivative(n, l, self.r)
                ours = dR(self.r, n, l)

                np.testing.assert_allclose(reference, ours)

    def test_torch_basis(self):
        from spex.radial.physical.laplacian_eigenstates import LaplacianEigenstates

        le = LaplacianEigenstates(
            self.cutoff, n_per_l=self.n_per_l, normalize=False
        )
        R, dR = le.get_spliner_inputs(self.cutoff, normalize=False)
        
        our_values = R(self.r_torch)
        our_derivatives = dR(self.r_torch)

        for n in range(self.max_radial + 1):
            for l in range(self.max_angular + 1):
                reference = self.reference_basis.compute(n, l, self.r)
                ours = our_values[:, n + l * (self.max_radial + 1)].numpy()

                np.testing.assert_allclose(reference, ours)

                reference = self.reference_basis.compute_derivative(n, l, self.r)
                ours = our_derivatives[:, n + l * (self.max_radial + 1)].numpy()

                np.testing.assert_allclose(reference, ours)

    def test_splined_and_jitted(self):
        from spex.radial.physical.laplacian_eigenstates import LaplacianEigenstates

        for spliner_accuracy, test_accuracy in ((1e-3, 1e-2), (1e-5, 1e-4)):
            basis = LaplacianEigenstates(
                self.cutoff,
                n_per_l=self.n_per_l,
                normalize=False,
                spliner_accuracy=spliner_accuracy,
            )

            basis = torch.jit.script(basis)
            our_values = basis(self.r_torch)
            our_values = torch.cat(our_values, dim=-1)

            for n in range(self.max_radial + 1):
                for l in range(self.max_angular + 1):
                    reference = self.reference_basis.compute(n, l, self.r)
                    ours = our_values[:, n + l * (self.max_radial + 1)].numpy()

                    np.testing.assert_allclose(reference, ours, atol=test_accuracy)

    def test_different_backends(self):
        from spex.radial.physical.laplacian_eigenstates import LaplacianEigenstates

        for device in ("cpu", "cuda", "mps"):
            # why is pytorch like this
            if device == "cuda":
                if not torch.cuda.is_available():
                    continue
            if device == "mps":
                if not torch.backends.mps.is_available():
                    continue

            basis = LaplacianEigenstates(
                self.cutoff,
                n_per_l=self.n_per_l,
                normalize=False,
                spliner_accuracy=1e-8,
            )
            if device == "mps":
                # mps is only single precision
                basis = basis.to(torch.float32).to(device)
                r_torch = self.r_torch.to(torch.float32).to(device)
                our_values = torch.cat(basis(r_torch), dim=-1).cpu()
            else:
                basis = basis.to(device)
                r_torch = self.r_torch.to(device)
                our_values = torch.cat(basis(r_torch), dim=-1).cpu()

            for n in range(self.max_radial + 1):
                for l in range(self.max_angular + 1):
                    reference = self.reference_basis.compute(n, l, self.r)
                    ours = our_values[:, n + l * (self.max_radial + 1)].numpy()

                    np.testing.assert_allclose(reference, ours, atol=1e-4)
