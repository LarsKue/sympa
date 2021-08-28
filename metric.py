
import pytorch_lightning as pl
import torch

import subprocess as sub

import pathlib
import re

import matplotlib.pyplot as plt

from metric_estimator import MetricEstimator
from metric_estimator import DataModule


def tensorboard():
    """
    Create a detached process for tensorboard
    """
    args = ["tensorboard", "--logdir", "lightning_logs"]

    process = sub.Popen(
        args, shell=False, stdin=None, stdout=None, stderr=None,
        close_fds=True
    )

    return process


def checkpoint(version, epoch, step):
    path = pathlib.Path("lightning_logs")
    version = pathlib.Path(f"version_{version}")
    checkpoints = pathlib.Path("checkpoints")
    cp = pathlib.Path(f"epoch={epoch}-step={step}.ckpt")
    return str(path / version / checkpoints / cp)


def latest_version(path):
    """
    Returns latest model version as integer
    """
    # unsorted
    versions = list(str(v.stem) for v in path.glob("version_*"))
    # get version numbers as integer
    versions = [re.match(r"version_(\d+)", version) for version in versions]
    versions = [int(match.group(1)) for match in versions]

    return max(versions)


def latest_checkpoint(version=None):
    """
    Returns latest checkpoint path for given version (default: latest) as string
    """
    path = pathlib.Path("lightning_logs")
    if version is None:
        version = latest_version(path)
    path = path / pathlib.Path(f"version_{version}/checkpoints/")

    # check if last.ckpt exists, and return that if applicable
    last = path / "last.ckpt"
    if last.is_file():
        return str(last)

    # find the last unnamed checkpoint
    checkpoints = list(str(cp.stem) for cp in path.glob("*.ckpt"))

    # find epoch and step numbers
    checkpoints = [re.match(r"epoch=(\d+)-step=(\d+)", cp) for cp in checkpoints]

    # assign steps to epochs
    epoch_steps = {}
    for match in checkpoints:
        epoch = match.group(1)
        step = match.group(2)

        if epoch not in epoch_steps:
            epoch_steps[epoch] = []

        epoch_steps[epoch].append(step)

    # find highest epoch and step
    max_epoch = max(epoch_steps.keys())
    max_step = max(epoch_steps[max_epoch])

    # reconstruct path
    checkpoint = path / f"epoch={max_epoch}-step={max_step}.ckpt"

    return str(checkpoint)


def samples(n, module, data_module):
    module.eval()
    for i in range(n):
        x, y = data_module.val_ds[i]
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)

        yhat = module(x)

        print(f"Predicted: {yhat}")
        print(f"Actual:    {y}")
        print()


def plot_loss_distribution(module, data_module, loss):
    module.eval()

    def get_losses(ds):
        losses = []
        with torch.no_grad():
            for x, y in ds:
                x = x.unsqueeze(0)
                y = y.unsqueeze(0)

                yhat = module(x)

                l = loss(yhat, y)

                losses.append(l.item())

        return losses

    train_losses = get_losses(data_module.train_ds)
    val_losses = get_losses(data_module.val_ds)

    plt.hist(train_losses, bins=50)
    plt.hist(val_losses, bins=50)
    plt.xlabel("Loss")
    plt.ylabel("Frequency")
    plt.show()


# TODO: Send Email that network works with better than 2% error
#  mention change to convolutional net improved the network from guessing to ~8% error
#  further changes to hyperparameters and the data improved this to the above figure of 2%
def main():
    n = 50
    ndim = 20
    batch_size = 32
    max_epochs = 250
    val_split = 0.2

    module = MetricEstimator(ndim)
    data_module = DataModule(n, ndim, batch_size=batch_size, val_split=val_split)

    callbacks = [
        pl.callbacks.ModelCheckpoint(monitor="val_loss", save_top_k=5, save_last=True),
        pl.callbacks.EarlyStopping(monitor="val_loss", patience=35),
        pl.callbacks.LearningRateMonitor(),
    ]

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        gpus=1,
    )

    process = tensorboard()

    print("Launched Tensorboard.")

    trainer.fit(module, data_module)

    data_module.save()

    # cp = checkpoint(version=40, epoch=168, step=25349)
    cp = latest_checkpoint()
    module = MetricEstimator.load_from_checkpoint(cp, ndim=ndim)
    data_module = DataModule.load()

    samples(10, module, data_module)
    plot_loss_distribution(module, data_module, loss=module.evaluation_loss)

    print("Press Enter to terminate Tensorboard.")
    input()

    process.terminate()


if __name__ == "__main__":
    main()

