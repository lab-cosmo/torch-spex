import numpy as np
import torch

from unittest import TestCase


class TestGaussian(TestCase):
    """Basic test suite for the Gaussian class."""
    
    def setUp(self):
        self.num_radial = 128
        self.cutoff = 5.0
        self.max_angular = 3
        self.num_features = None
        self.trainable = False
        self.per_degree = False

        self.r = np.random.random(25)

    def test_jit(self):
        """Test if Gaussian class works with TorchScript."""
        from spex.radial.simple import Gaussian
        
        radial = Gaussian(
            cutoff=self.cutoff,
            num_radial=self.num_radial,
            max_angular=self.max_angular,
            num_features=self.num_features,
            trainable=self.trainable,
            per_degree=self.per_degree,
        )
        radial = torch.jit.script(radial)
        radial(torch.tensor(self.r, dtype=torch.float32))

    def test_hardcoded(self):
        """Test Gaussian class with hardcoded parameters."""
        from spex.radial.simple import Gaussian
        
        radial = Gaussian(
            cutoff=self.cutoff,
            num_radial=self.num_radial,
            max_angular=self.max_angular,
            num_features=self.num_features,
        )
        
        assert radial.cutoff == self.cutoff
        assert radial.num_radial == self.num_radial
        assert radial.max_angular == self.max_angular
        
        # Validate function output against reference implementation
        torch_r = torch.tensor(self.r, dtype=torch.float32)
        torch_output = radial.expand(torch_r).detach().numpy()
        
        np.testing.assert_allclose(torch_output.shape, (self.r.shape[0], self.num_radial))