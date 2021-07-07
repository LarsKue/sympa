
import unittest
import torch
import pathlib as pl
import shutil

from metric_estimator.datasets import DistanceDataset
from sympa.manifolds import UpperHalfManifold


class DistanceDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.path = pl.Path("distance_test")

        self.size = 100
        self.ndim = 5
        self.manifold = UpperHalfManifold(ndim=self.ndim)

        self.instance = DistanceDataset.generate(size=self.size, manifold=self.manifold, path=self.path, overwrite=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.path)

    def test_size(self):
        self.assertEqual(self.size, len(self.instance))

    def test_shape(self):
        for i in range(self.size):
            z1, z2, dist = self.instance[i]

            shape = (1, 2, self.ndim, self.ndim)
            self.assertEqual(shape, z1.shape)
            self.assertEqual(shape, z2.shape)
            shape = (1,)
            self.assertEqual(shape, dist.shape)

    def test_dist(self):
        for i in range(self.size):
            z1, z2, dist = self.instance[i]
            self.assertTrue(torch.isclose(self.manifold.dist(z1, z2), dist))


if __name__ == '__main__':
    unittest.main()
