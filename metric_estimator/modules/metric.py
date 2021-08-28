
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
            nn.Conv2d(in_channels=4, out_channels=512, kernel_size=(5, 5), stride=(1, 1)),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3, 3), stride=(1, 1)),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(2, 2), stride=(1, 1)),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1))
        )

        # self.conv = nn.Sequential(
        #     nn.Conv2d(in_channels=4, out_channels=512, kernel_size=(1, 1)),
        #     nn.PReLU(),
        #     nn.Dropout2d(p=0.2),
        #     nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(2, 2), stride=(1, 1)),
        #     nn.PReLU(),
        #     nn.MaxPool2d(kernel_size=(2, 2))
        # )

        example = self.conv(self.example_input_array)
        self.post_conv_size = example.numel()

        print("Post Convolutional Shape:", example.shape)
        print("Post Convolutional Size:", self.post_conv_size)

        self.fully_connected = nn.Sequential(
            nn.Linear(in_features=self.post_conv_size, out_features=1024),
            nn.PReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=1024, out_features=512),
            nn.PReLU(),
            nn.Dropout2d(p=0.2),
            nn.Linear(in_features=512, out_features=512),
            nn.PReLU(),
            nn.Dropout2d(p=0.2),
            nn.Linear(in_features=512, out_features=512),
            nn.PReLU(),
            nn.Dropout2d(p=0.2),
            nn.Linear(in_features=512, out_features=512),
            nn.PReLU(),
            nn.Dropout2d(p=0.2),
            nn.Linear(in_features=512, out_features=512),
            nn.PReLU(),
            nn.Dropout2d(p=0.2),
            nn.Linear(in_features=512, out_features=512),
            nn.PReLU(),
            nn.Dropout2d(p=0.2),
            nn.Linear(in_features=512, out_features=512),
            nn.PReLU(),
            nn.Dropout(p=0.2),
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

        self.loss = nn.SmoothL1Loss()

        # e.g. understandable for humans, but suboptimal in training or validation
        self.evaluation_loss = nn.L1Loss()

    def forward(self, x):
        if self.training:
            x = self.augmentation(x)
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-6, threshold=1e-4, factor=0.31623, patience=10)

        # return optimizer
        return dict(optimizer=optimizer, lr_scheduler=lr_scheduler, monitor="val_loss")

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
        
        self.log("val_loss", loss)
        self.log("eval_loss", eval_loss)

        return loss
