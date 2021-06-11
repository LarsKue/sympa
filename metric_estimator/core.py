
import torch
from torch import nn
from .fit import ModelFitMixin

# TODO:
#  - Create Siegel Metric
#  - Use Metric to create TS (standard normal dist scaled to unit box)
#  - Create model layers
#  - Fit model to available data
#  - cross validation


class MetricEstimator(ModelFitMixin, nn.Module):
    def __init__(self, ndim=2):

        self.input_shape = torch.Tensor([2, ndim, ndim])

        super(MetricEstimator, self).__init__(
            nn.Linear(in_features=torch.prod(self.input_shape).item(), out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=512),
            nn.Linear(in_features=512, out_features=256),
            nn.Linear(in_features=256, out_features=1),
            optimizer=torch.optim.SGD,
            loss=nn.MSELoss(),

        )
