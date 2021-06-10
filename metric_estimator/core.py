
from torch import nn
from .fit import ModelFitMixin

# TODO:
#  - Create Siegel Metric
#  - Use Metric to create TS (standard normal dist scaled to unit box)
#  - Create model layers
#  - Fit model to available data
#  - cross validation


class MetricEstimator(ModelFitMixin, nn.Module):
    def __init__(self):
        super(MetricEstimator, self).__init__()


class SiegelMetricEstimator(MetricEstimator):
    def __init__(self, ndim=2):
        pass
