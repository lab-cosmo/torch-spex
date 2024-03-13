from unittest import TestCase
import math

import numpy as np
import torch
from scipy.special import sph_harm


class TestSphericalHarmonics(TestCase):
    # we test against a (shortened) copy of the same test from rascaline,
    # https://github.com/Luthaf/rascaline/blob/ae05064a28cf2d6e76c13c5673e57384aff4808a/
    # ... rascaline/tests/data/spherical-harmonics.py
    # which tests against scipy

    def setUp(self):
        self.max_angular = 25
        self.directions = np.array(
            [
                np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]),
                np.array([0.0, 0.0, 1.0]),
                np.array([0.5773502691896258, 0.5773502691896258, 0.5773502691896258]),
                np.array([0.455493902781557, 0.46164788218724867, -0.7611875835829522]),
                np.array(
                    [-0.28695413002732584, -0.33058712239743676, -0.8990937002144119]
                ),
                np.array(
                    [0.35108584385989333, -0.9226014654045358, -0.15982886558625636]
                ),
            ]
        )

    def test_vs_scipy(self):
        from spex.angular import SphericalHarmonics

        sph = SphericalHarmonics(max_angular=self.max_angular)

        ours = sph(torch.tensor(self.directions)).numpy()
        reference = spherical_harmonics(self.max_angular, self.directions)

        np.testing.assert_allclose(reference, ours, atol=1e-13)


def real_sph(l, m, theta, phi):
    """Compute real spherical harmonics from the complex version in scipy"""
    m_1_pow_m = (-1) ** m
    if m > 0:
        return np.sqrt(2) * m_1_pow_m * np.real(sph_harm(m, l, theta, phi))
    elif m == 0:
        return m_1_pow_m * np.real(sph_harm(0, l, theta, phi))
    else:
        return np.sqrt(2) * m_1_pow_m * np.imag(sph_harm(abs(m), l, theta, phi))


def spherical_harmonics(max_angular, directions):
    n_directions = len(directions)
    values = np.zeros(
        (n_directions, max_angular + 1, 2 * max_angular + 1), dtype=np.float64
    )

    values = []
    for direction in directions:
        out = []
        phi = math.acos(direction[2])
        theta = math.atan2(direction[1], direction[0])
        for l in range(max_angular + 1):
            for i_m, m in enumerate(range(-l, l + 1)):
                out.append(real_sph(l, m, theta, phi))
        values.append(np.array(out))

    return np.array(values)
