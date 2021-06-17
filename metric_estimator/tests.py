
import unittest
import torch
from torch.utils.data import DataLoader

from .core import MetricEstimator
from . import data

from sympa.manifolds import UpperHalfManifold

# TODO: Fix Tests (adjust for lack of from_ndim)
class MetricEstimatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ndim = 1000
        self.instance = MetricEstimator.from_ndim(self.ndim, overwrite=True)

    def tearDown(self) -> None:
        pass

    def test_fit(self):
        self.instance.fit(epochs=75, verbose=True)


class PointGenerationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ndims = [2, 3, 5, 10, 20]
        self.n = 100

        self.zs = [
            data.generate_points(self.n, ndim=ndim)
            for ndim in self.ndims
        ]

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

    # def test_plot(self):
    #     for ndim, z in zip(self.ndims, self.zs):
    #         pts = torch.flatten(z).cpu().tolist()
    #
    #         plt.hist(pts, bins=100)
    #         plt.show()


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


if __name__ == "__main__":
    unittest.main()
