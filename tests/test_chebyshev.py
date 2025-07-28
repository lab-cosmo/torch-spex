import numpy as np
import torch

from unittest import TestCase


class TestChebyshev(TestCase):
    def setUp(self):
        self.num_radial = 128
        self.cutoff = 5.0
        self.max_angular = 3
        self.num_features = None
        self.trainable = False
        self.per_degree = False

        self.r = np.random.random(25)

    def test_jit(self):
        from spex.radial.simple import Chebyshev

        radial = Chebyshev(
            cutoff=self.cutoff,
            num_radial=self.num_radial,
            max_angular=self.max_angular,
            num_features=self.num_features,
            trainable=self.trainable,
            per_degree=self.per_degree,
        )
        radial = torch.jit.script(radial)
        radial(torch.tensor(self.r, dtype=torch.float32))

    def test_shape_and_fire(self):
        # are the shapes right? does it work?
        from spex.radial.simple import Chebyshev

        radial = Chebyshev(
            cutoff=self.cutoff,
            num_radial=self.num_radial,
            max_angular=self.max_angular,
            num_features=self.num_features,
        )

        assert radial.cutoff == self.cutoff
        assert radial.num_radial == self.num_radial
        assert radial.max_angular == self.max_angular

        torch_r = torch.tensor(self.r, dtype=torch.float32)
        torch_output = radial.expand(torch_r).detach().numpy()

        np.testing.assert_allclose(torch_output.shape, (self.r.shape[0], self.num_radial))

    def test_values(self):
        from spex.radial.simple import Chebyshev

        radial = Chebyshev(
            cutoff=5.0,
            num_radial=5,
            max_angular=0,
            num_features=None,
        )

        # values: e3x.nn.functions.basic_chebyshev(x, num=5, limit=5.0)
        reference = np.array([[1.0, -0.19999997, -0.92, 0.56799984, 0.6928001]])

        x = torch.tensor([2.0])
        y = radial.expand(x).detach().numpy()

        np.testing.assert_allclose(y, reference)
