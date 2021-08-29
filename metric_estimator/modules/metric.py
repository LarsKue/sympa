
import pytorch_lightning as pl

import torch
import torch.nn as nn

from .. import losses

import torchvision


class MetricEstimator(pl.LightningModule):
    def __init__(self, ndim):
        super(MetricEstimator, self).__init__()
        self.ndim = ndim

        # input is 4 channels (2 times real + imag)
        self.example_input_array = torch.zeros(1, 4, self.ndim, self.ndim)

        # data augmentation
        # TODO: Random Rotation, ...

        # randomly swap z1 and z2 since dist(z1, z2) == dist(z2, z1)
        def swap(x):
            return torch.cat((x[:, 2:], x[:, :2]), dim=1)

        self.augmentation = nn.Sequential(
            torchvision.transforms.RandomApply([swap])
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=1024, kernel_size=(5, 5), stride=(1, 1)),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), stride=(1, 1)),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(2, 2), stride=(1, 1)),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1))
        )

        example = self.conv(self.example_input_array)
        self.post_conv_size = example.numel()

        print("Post Convolutional Shape:", example.shape)
        print("Post Convolutional Size:", self.post_conv_size)

        self.fully_connected = nn.Sequential(
            nn.Linear(in_features=self.post_conv_size, out_features=1024),
            nn.PReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1024, out_features=512),
            nn.PReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=256),
            nn.PReLU(),
            nn.Linear(in_features=256, out_features=1)
        )

        self.model = nn.Sequential(
            self.conv,
            nn.Flatten(),
            self.fully_connected,
        )

        # initialize weights
        for module in self.model.modules():
            if type(module) in (nn.Linear, nn.Conv2d):
                data = module.weight.data
                nn.init.normal_(data, 0, 2 / data.numel())

        self.loss = nn.SmoothL1Loss(beta=0.5)

        # e.g. understandable for humans, but suboptimal in training or validation
        self.evaluation_loss = nn.L1Loss()

        self.histogram_loss = nn.L1Loss(reduction="none")

    def forward(self, x):
        if self.training:
            x = self.augmentation(x)
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-6)
        # optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-6, threshold=1e-4, factor=0.31623, patience=10)

        return optimizer
        # return dict(optimizer=optimizer, lr_scheduler=lr_scheduler, monitor="val_loss")

    def training_step(self, batch, batch_idx):
        x, y = batch
        yhat = self(x)
        loss = self.loss(yhat, y)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        yhat = self(x)
        loss = self.loss(yhat, y)
        eval_loss = self.evaluation_loss(yhat, y)
        hist_loss = self.histogram_loss(yhat, y)
        
        self.log("val_loss", loss)
        self.log("eval_loss", eval_loss)
        # TODO: fix accumulation over multiple epochs, or just take this out
        self.logger.experiment.add_histogram("val_predictions", yhat, bins="auto")
        self.logger.experiment.add_histogram("val_loss_distribution", hist_loss, bins="auto")

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        yhat = self(x)
        losses = self.histogram_loss(yhat, y)
        self.logger.experiment.add_histogram("test_predictions", yhat, bins="auto")
        self.logger.experiment.add_histogram("test_loss_distribution", losses, bins="auto")

        return torch.mean(losses)
