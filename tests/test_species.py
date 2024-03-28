import numpy as np
import torch

from unittest import TestCase


class TestOrthogonal(TestCase):
    def test_correct(self):
        from spex.species import Orthogonal

        species = [1, 2, 10, 21]

        orthogonal = Orthogonal(species)

        inputs = torch.tensor([2, 1, 21, 10], dtype=torch.int64)

        outputs = orthogonal(inputs)

        np.testing.assert_equal(outputs[0].numpy(), np.array([0, 1, 0, 0]))
        np.testing.assert_equal(outputs[1].numpy(), np.array([1, 0, 0, 0]))
        np.testing.assert_equal(outputs[2].numpy(), np.array([0, 0, 0, 1]))
        np.testing.assert_equal(outputs[3].numpy(), np.array([0, 0, 1, 0]))

    def test_jit(self):
        from spex.species import Orthogonal

        species = [1, 2]

        orthogonal = torch.jit.script(Orthogonal(species))

        inputs = torch.tensor([2, 1], dtype=torch.int64)

        orthogonal(inputs)


class TestAlchemical(TestCase):
    def test_shapes(self):
        from spex.species import Alchemical

        n_pseudo_species = 6

        alchemical = Alchemical(n_pseudo_species)

        inputs = torch.tensor([2, 1, 21, 10], dtype=torch.int64)

        outputs = alchemical(inputs)

        assert outputs.shape == (4, 6)

    def test_jit(self):
        from spex.species import Alchemical

        n_pseudo_species = 6

        alchemical = torch.jit.script(Alchemical(n_pseudo_species))

        inputs = torch.tensor([2, 1, 21, 10], dtype=torch.int64)

        alchemical(inputs)
