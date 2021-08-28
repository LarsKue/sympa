
import torch
import unittest

import metric_estimator as losses


class LossTest(unittest.TestCase):
    def setUp(self) -> None:
        self.shape = (32,)

        self.predicted = torch.rand(self.shape)
        self.true = torch.rand(self.shape)

    def test_mare(self):
        loss = losses.MARELoss()

        loss(self.predicted, self.true)

    def test_l2(self):
        loss = losses.L2Loss()

        loss(self.predicted, self.true)


if __name__ == "__main__":
    unittest.main()
