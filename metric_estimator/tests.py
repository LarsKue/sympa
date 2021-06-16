
import unittest
import torch

from .core import MetricEstimator
from . import data
import pathlib as pl
from matplotlib import pyplot as plt
import numpy as np

from sympa.manifolds import UpperHalfManifold




# class MetricEstimatorTest(unittest.TestCase):
#     def setUp(self) -> None:
#         self.instance = MetricEstimator()


class DatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ndims = [2, 3, 5, 10, 20, 50, 100]
        self.n = 1000

        self.zs = [
            data.generate_points(self.n, ndim)
            for ndim in self.ndims
        ]

        # TODO: Tests
        # self.dataset = data.MetricDistanceSet.generate(self.n, manifold=)

    def test_positive_definite(self):
        for ndim, z in zip(self.ndims, self.zs):

            real, imag = z[:, 0, :, :], z[:, 1, :, :]

            # try to compute cholesky decomposition
            # this works iff the matrix is hermitian positive definite
            try:
                torch.linalg.cholesky(imag)
            except:
                self.assertTrue(False, msg=f"Matrices of ndim = {ndim} are not positive definite.")

    def test_symmetric(self):
        for ndim, z in zip(self.ndims, self.zs):
            self.assertTrue(torch.allclose(z.transpose(-2, -1), z))

    def test_plot(self):
        for ndim, z in zip(self.ndims, self.zs):
            pts = torch.flatten(z).cpu().tolist()

            plt.hist(pts, bins=100)
            plt.show()

    def test_generate(self):
        ndim = 3
        manifold = UpperHalfManifold(ndim=ndim)
        dataset = data.generate(3, manifold=manifold, path=pl.Path("asdf"))

        print(dataset)


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


if __name__ == "__main__":
    unittest.main()
