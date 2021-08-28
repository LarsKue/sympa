
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import core

from metric_estimator.datasets import VVDDataset


class VVDEstimator(core.CoreModule):
    def __init__(self, ndim):
        self.ndim = ndim
        super(VVDEstimator, self).__init__()

    def configure_model(self):
        model = nn.Sequential(
            nn.Linear(in_features=2 * self.ndim * (self.ndim + 1), out_features=4096),
            nn.PReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=4096, out_features=1024),
            nn.PReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=1024, out_features=1024),
            nn.PReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=1024, out_features=512),
            nn.PReLU(),
            nn.Linear(in_features=512, out_features=self.ndim),
        )

        return model

    def configure_loss(self):
        return nn.MSELoss()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=30, factor=0.3, threshold=0.05,
                                                                  verbose=True)

        return optimizer
        # return dict(optimizer=optimizer, lr_scheduler=lr_scheduler, monitor="val_loss")