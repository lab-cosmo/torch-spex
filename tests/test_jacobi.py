import numpy as np
import torch

from unittest import TestCase

from scipy.special import eval_jacobi as sci_jacobi

class TestBasisValue(TestCase):
    def testvaluesbasis(self):
        from spex.radial.simple import jacobi
        
        alpha = 1
        beta = 1
        cutoff = 1

        num_radial = 10
        n_size = 1000 
        
        j = jacobi.Jacobi(cutoff = cutoff, 
                   alpha = alpha, 
                   beta = beta,
                   max_angular = 0,
                   num_radial = num_radial)

        assert j.cutoff == cutoff
        assert j.alpha == alpha

        x = np.linspace(0, cutoff, n_size)
        
        input = torch.tensor(x)
        output = np.array(j.expand(input))

        assert (output.shape[0], output.shape[1]) == (num_radial, n_size)

        cosx = np.cos(np.pi * x / cutoff)
        for n in range(num_radial):
            np.testing.assert_allclose(output[n,:,0], sci_jacobi(n, alpha, beta, cosx))
            