
import unittest
import torch

from metric_estimator import math


class MathTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ndims = [2, 3, 5, 10, 20]
        self.n = 100
        self.reals = [
            math.make_symmetric_tensor(self.n, ndim=ndim)
            for ndim in self.ndims
        ]
        self.imags = [
            math.make_spd_standard(self.n, ndim=ndim)
            for ndim in self.ndims
        ]

        self.zs = [
            torch.stack((real, imag), dim=1)
            for real, imag in zip(self.reals, self.imags)
        ]

    def test_shape(self):
        for i, ndim in enumerate(self.ndims):
            shape = (self.n, ndim, ndim)
            self.assertEqual(self.reals[i].shape, shape)
            self.assertEqual(self.imags[i].shape, shape)

    def test_positive_definite(self):
        for ndim, imag in zip(self.ndims, self.imags):
            self.assertTrue(math.is_positive_definite(imag), msg=f"SPD Matrices of ndim = {ndim} are not positive definite.")

    def test_symmetric_real(self):
        for ndim, real in zip(self.ndims, self.reals):
            self.assertTrue(torch.allclose(real.transpose(-2, -1), real), msg=f"Matrices of ndim = {ndim} are not symmetric.")

    def test_symmetric_imag(self):
        for ndim, imag in zip(self.ndims, self.imags):
            self.assertTrue(torch.allclose(imag.transpose(-2, -1), imag), msg=f"SPD Matrices of ndim = {ndim} are not symmetric.")

    def test_bft(self):
        for ndim, z in zip(self.ndims, self.zs):
            shape1 = (self.n, 2, ndim, ndim)
            self.assertEqual(shape1, z.shape)

            bft = math.bft(z)

            shape2 = (self.n, ndim * (ndim + 1))
            self.assertEqual(shape2, bft.shape)

            reconstructed = math.ibft(bft)
            self.assertEqual(shape1, reconstructed.shape)

            self.assertTrue(torch.allclose(z, reconstructed))


if __name__ == '__main__':
    unittest.main()
