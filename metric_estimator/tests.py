
import unittest
import torch

import numpy as np

from sklearn.datasets import make_spd_matrix

from . import data
from . import math
from . import utils

from sympa.manifolds import UpperHalfManifold


class MathTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ndims = [2, 3, 5, 10, 20]
        self.n = 1
        self.reals = [
            math.make_symmetric_tensor(self.n, ndim=ndim)
            for ndim in self.ndims
        ]
        self.imags = [
            math.make_spd_standard(self.n, ndim=ndim)
            for ndim in self.ndims
        ]

    def test_shape(self):
        for i, ndim in enumerate(self.ndims):
            shape = (self.n, ndim, ndim)
            self.assertEqual(self.reals[i].shape, shape)
            self.assertEqual(self.imags[i].shape, shape)

    def test_positive_definite(self):
        for ndim, imag in zip(self.ndims, self.imags):
            print(imag)
            self.assertTrue(math.is_positive_definite(imag), msg=f"SPD Matrices of ndim = {ndim} are not positive definite.")

    def test_symmetric_real(self):
        for ndim, real in zip(self.ndims, self.reals):
            self.assertTrue(torch.allclose(real.transpose(-2, -1), real), msg=f"Matrices of ndim = {ndim} are not symmetric.")

    def test_symmetric_imag(self):
        for ndim, imag in zip(self.ndims, self.imags):
            print(imag)
            self.assertTrue(torch.allclose(imag.transpose(-2, -1), imag), msg=f"SPD Matrices of ndim = {ndim} are not symmetric.")


class DatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.n = 100
        self.ndim = 10
        self.manifold = UpperHalfManifold(ndim=self.ndim)

    def test_generate(self):
        dataset = data.MetricDistanceSet.generate(self.n, manifold=self.manifold, path="_test", overwrite=True)

        self.assertEqual(self.n, len(dataset))

    def test_simple_generate(self):
        dataset = data.SimpleMetricDistanceSet.generate(self.n, manifold=self.manifold)

        self.assertEqual(self.n, len(dataset))

    def test_iterate(self):
        dataset = data.MetricDistanceSet.generate(self.n, manifold=self.manifold, path="_test", overwrite=True)

        for i, (x, y) in enumerate(dataset):
            pass

    def test_simple_iterate(self):
        dataset = data.SimpleMetricDistanceSet.generate(self.n, manifold=self.manifold)

        for i, (x, y) in enumerate(dataset):
            pass


class MiscTest(unittest.TestCase):
    def test_siegel_dist(self):

        n = 4

        manifold = UpperHalfManifold(ndim=n)

        z1_real = torch.eye(n)
        z1_imag = torch.eye(n)

        z1 = torch.stack((z1_real, z1_imag))

        z1 = z1.unsqueeze(0)

        print(z1.shape)

        z2 = torch.zeros_like(z1)

        # z1, z2 = z2, z1

        print(manifold.dist(z1, z2))

    def test_math(self):
        n = 100
        ndim = 3
        print("Making Skewed...")
        with utils.Timer("Skewed"):
            batch_0 = math.make_spd_skewed(n, ndim=ndim)
        print("Making Tensors...")
        with utils.Timer("Tensor"):
            batch_1 = math.make_spd_tensor(n, ndim=ndim)
        print("Making Standard...")
        with utils.Timer("Standard"):
            batch_2 = math.make_spd_standard(n, ndim=ndim)

        mu_0 = torch.mean(batch_0, dim=0)
        mu_1 = torch.mean(batch_1, dim=0)
        mu_2 = torch.mean(batch_2, dim=0)

        sig_0 = torch.std(batch_0, dim=0)
        sig_1 = torch.std(batch_1, dim=0)
        sig_2 = torch.std(batch_2, dim=0)

        print("skewed:")
        print(mu_0)
        print("tensor:")
        print(mu_1)
        print("standard:")
        print(mu_2)
        print("\n\n")
        print("skewed:")
        print(sig_0)
        print("tensor:")
        print(sig_1)
        print("standard:")
        print(sig_2)


if __name__ == "__main__":
    unittest.main()
