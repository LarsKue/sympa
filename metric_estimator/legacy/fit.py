
import torch
import numpy as np

from .model import ModelMixin
from .device import device
from .history import LossHistory


class ModelFitMixin(ModelMixin):
    """
    Fit a model to data generated by a DataLoader
    Keep a history of losses throughout training
    """
    def __init__(self, model, optimizer, train_loss, val_loss, schedulers, train_loader, val_loader, batch_shape=None):
        super(ModelFitMixin, self).__init__(model)
        self.optimizer = optimizer
        self.train_loss = train_loss
        self.val_loss = val_loss
        self.schedulers = schedulers
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.batch_shape = batch_shape
        self.history = LossHistory()

    def fit(self, epochs, validate=True, verbosity=0):
        """
        Train for a fixed number of epochs and keep a loss history
        @param epochs: Number of epochs to train
        @param validate: Whether or not to perform validation
        @param verbosity: Level of verbosity. 0 prints nothing, 1 prints epochs, 2 prints epochs and losses
        """
        for epoch in range(epochs):
            train_loss = self.step_train()

            if validate:
                val_loss = self.step_val()
            else:
                val_loss = None

            self.history.step(train_loss, val_loss)

            for s in self.schedulers:
                s.step(val_loss)

            if verbosity > 0:
                print(f"Epoch {epoch + 1} / {epochs}")
            if verbosity > 1:
                print("Train Loss:", train_loss)
                print("Validation Loss:", val_loss)
                print()

    def step_train(self):
        """
        Perform one Training Step
        """
        temp = self.model.training
        self.model.train(mode=True)

        batch_losses = []
        for batch, (x, y) in enumerate(self.train_loader):
            x = x.to(device)
            if self.batch_shape is not None:
                x = x.view(self.batch_shape)

            y = y.to(device)

            self.optimizer.zero_grad(set_to_none=True)
            yhat = self.model(x)
            batch_loss = self.train_loss(yhat, y)
            batch_loss.backward()
            self.optimizer.step()

            batch_losses.append(batch_loss.detach().item())

        self.model.train(mode=temp)
        return np.mean(batch_losses)

    def step_val(self):
        """
        Perform one Validation Step
        """
        temp = self.model.training
        self.model.train(mode=False)

        batch_losses = []
        for batch, (x, y) in enumerate(self.val_loader):
            x = x.to(device)
            if self.batch_shape is not None:
                x = x.view(self.batch_shape)

            y = y.to(device)

            with torch.no_grad():
                yhat = self.model(x)
                batch_loss = self.val_loss(yhat, y)

                batch_losses.append(batch_loss.detach().item())

        self.model.train(mode=temp)
        return np.mean(batch_losses)

    def save_loss(self, path, overwrite=False):
        self.history.save(path, overwrite=overwrite)
