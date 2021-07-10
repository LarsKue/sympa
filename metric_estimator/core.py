from .fit import ModelFitMixin
from .save import ModelSaveMixin


# TODO:
#  - Improve Siegel Point Sampling by rejecting outliers
#    (this makes the dataset more dense in relevant parts)
#  - Sample data points uniformly instead of normally
#  - Greatly increase training set size (~factor 10)
#  - vvd angle dataset
#  - predict angle of vvd
#  - hyper parameter optimization


class MetricEstimator(ModelFitMixin, ModelSaveMixin):
    def __init__(self, model, manifold, optimizer, loss, schedulers, train_loader, val_loader, batch_shape=None):
        super(MetricEstimator, self).__init__(model, optimizer, loss, schedulers, train_loader, val_loader, batch_shape)
        self.manifold = manifold
