
import unittest

from .core import MetricEstimator


# class MetricEstimatorTest(unittest.TestCase):
#     def setUp(self) -> None:
#         self.instance = MetricEstimator()


class MiscTest(unittest.TestCase):
    def test_multi_derivation(self):
        class Mixin1:
            def __init__(self, mixin1):
                self.mixin1 = mixin1

        class Mixin2:
            def __init__(self, mixin2):
                self.mixin2 = mixin2

        class Model(Mixin1, Mixin2):
            def __init__(self):
                super(Model, self).__init__("mixin1", "mixin2")

        model = Model()


if __name__ == "__main__":
    unittest.main()
