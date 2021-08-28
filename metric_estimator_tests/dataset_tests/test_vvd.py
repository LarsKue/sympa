
import unittest
import torch
import pathlib as pl
import shutil

from metric_estimator import VVDDataset
from sympa.manifolds import UpperHalfManifold


class VVDSetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.path = pl.Path("vvd_test")

        self.size = 100
        self.ndim = 5
        self.manifold = UpperHalfManifold(ndim=self.ndim)

        self.instance = VVDDataset.generate(manifold=self.manifold, path=self.path, size=self.size, overwrite=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.path)

    def test_size(self):
        self.assertEqual(self.size, len(self.instance))

    def test_shape(self):
        for i in range(self.size):
            z1, z2, vvd = self.instance[i]

            shape = (1, 2, self.ndim, self.ndim)
            self.assertEqual(shape, z1.shape)
            self.assertEqual(shape, z2.shape)

            shape = (1, self.ndim)
            self.assertEqual(shape, vvd.shape)

    def test_distance(self):
        for i in range(self.size):
            z1, z2, vvd = self.instance[i]

            self.assertTrue(torch.allclose(self.manifold.vvd(z1, z2), vvd))

    def test_sorted(self):
        for z1, z2, vvd in self.instance:
            s, _ = torch.sort(vvd)
            self.assertTrue(torch.allclose(s, vvd))


if __name__ == '__main__':
    unittest.main()
