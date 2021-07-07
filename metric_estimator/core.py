from .fit import ModelFitMixin
from .save import ModelSaveMixin


# TODO:
#  - vvd angle dataset
#  - predict angle of vvd
#  - predict vvd (see email)
#  - hyper parameter optimization


class MetricEstimator(ModelFitMixin, ModelSaveMixin):
    def __init__(self, model, manifold, optimizer, loss, schedulers, train_loader, val_loader, batch_shape=None):
        super(MetricEstimator, self).__init__(model, optimizer, loss, schedulers, train_loader, val_loader, batch_shape)
        self.manifold = manifold