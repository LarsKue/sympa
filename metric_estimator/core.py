from .fit import ModelFitMixin
from .save import ModelSaveMixin


# TODO:
#  - Create Siegel Metric
#  - Use Metric to create TS (standard normal dist scaled to unit box)
#  - Create model layers
#  - Fit model to available data
#  - cross validation


class MetricEstimator(ModelFitMixin, ModelSaveMixin):
    def __init__(self, model, manifold, optimizer, loss, schedulers, train_loader, val_loader, batch_shape=None):
        super(MetricEstimator, self).__init__(model, optimizer, loss, schedulers, train_loader, val_loader, batch_shape)
        self.manifold = manifold