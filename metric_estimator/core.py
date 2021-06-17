
import torch
from torch import nn
from torch.utils.data import DataLoader
from .fit import ModelFitMixin
from .data import MetricDistanceSet

from sympa.manifolds import UpperHalfManifold

# TODO:
#  - Create Siegel Metric
#  - Use Metric to create TS (standard normal dist scaled to unit box)
#  - Create model layers
#  - Fit model to available data
#  - cross validation


class MetricEstimator(ModelFitMixin):
    def __init__(self, model, manifold, optimizer, loss, train_loader, val_loader, batch_shape=None):
        super(MetricEstimator, self).__init__(model, optimizer, loss, train_loader, val_loader, batch_shape)
        self.manifold = manifold
