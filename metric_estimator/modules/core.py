import pytorch_lightning as pl


class CoreModule(pl.LightningModule):
    def __init__(self):
        super(CoreModule, self).__init__()
        self.model = self.configure_model()
        self.loss = self.configure_loss()

    def configure_model(self):
        raise NotImplementedError

    def configure_loss(self):
        raise NotImplementedError

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        yhat = self.model(x)
        loss = self.loss(yhat, y)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        yhat = self.model(x)
        loss = self.loss(yhat, y)

        self.log("val_loss", loss)

        return loss
