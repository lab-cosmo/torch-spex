import numpy as np
import torch

from unittest import TestCase


class TestSphericalHarmonics(TestCase):
    # we test against hard-coded versions directly copied from
    # https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Real_spherical_harmonics
    # to verify/codify the convention we're using in this package...
    # (we don't need to check for correctness, that's what sphericart is for)

    def setUp(self):
        self.max_angular = 1
        self.directions = 3 * np.random.random((25, 3)) - 1.5  # NOT normalised

    def test_jit(self):
        from spex.angular import SphericalHarmonics

        sph = SphericalHarmonics(max_angular=self.max_angular)
        sph = torch.jit.script(sph)
        sph(torch.tensor(self.directions))

    def test_hardcoded(self):
        from spex.angular import SphericalHarmonics

        sph = SphericalHarmonics(max_angular=self.max_angular)

        ours = torch.cat(sph(torch.tensor(self.directions)), dim=-1).numpy()

        np.testing.assert_allclose(sph_0(self.directions), ours[:, 0])
        np.testing.assert_allclose(sph_1(self.directions, -1), ours[:, 1])
        np.testing.assert_allclose(sph_1(self.directions, 0), ours[:, 2])
        np.testing.assert_allclose(sph_1(self.directions, +1), ours[:, 3])


def angles(R):
    R /= np.linalg.norm(R, axis=1)[:, None]
    phi = np.arccos(R[:, 2])
    theta = np.arctan2(R[:, 1], R[:, 0])

    return phi, theta


def sph_0(R):
    return 0.5 * np.sqrt(1 / np.pi) * np.ones_like(R[:, 0])


def sph_1(R, m):
    theta, phi = angles(R)

    if m == -1:
        return np.sqrt(3 / (4 * np.pi)) * np.sin(theta) * np.sin(phi)
    elif m == 0:
        return np.sqrt(3 / (4 * np.pi)) * np.cos(theta)
    elif m == +1:
        return np.sqrt(3 / (4 * np.pi)) * np.sin(theta) * np.cos(phi)


class TestSolidHarmonics(TestCase):
    # a basic test to check that the SolidHarmonics class works

    def setUp(self):
        self.max_angular = 1
        self.directions = 3 * np.random.random((25, 3)) - 1.5  # NOT normalised

    def test_jit(self):
        from spex.angular import SolidHarmonics

        sph = SolidHarmonics(max_angular=self.max_angular)
        sph = torch.jit.script(sph)
        sph(torch.tensor(self.directions))
