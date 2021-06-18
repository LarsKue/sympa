
import torch
from torch import nn
from torch.utils.data import DataLoader

import numpy as np
from matplotlib import pyplot as plt

from metric_estimator import MetricEstimator, MetricDistanceSet

from sympa.manifolds import UpperHalfManifold


def main():

    ndim = 30

    model_path = "me_model"
    train_path = "train"
    val_path = "validate"
    load_data = True
    load_model = False
    overwrite_data = False
    overwrite_model = True

    batch_size = 32
    n_train = 100 * batch_size
    n_val = 32 * batch_size
    n_epochs = 500

    manifold = UpperHalfManifold(ndim=ndim)
    model = nn.Sequential(
        nn.Linear(in_features=2 * ndim * (ndim + 1), out_features=1024),
        nn.ReLU(),
        nn.Linear(in_features=1024, out_features=1024),
        nn.ReLU(),
        nn.Linear(in_features=1024, out_features=512),
        nn.ReLU(),
        nn.Linear(in_features=512, out_features=256),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=1),
    )

    if load_model:
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss = torch.nn.MSELoss()
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=30, factor=0.2, verbose=True)
    schedulers = [lr_scheduler]

    if load_data:
        print("Loading Datasets...")
        train_set = MetricDistanceSet.load(train_path)
        val_set = MetricDistanceSet.load(val_path)
    else:
        print("Generating Datasets...")
        train_set = MetricDistanceSet.generate(n_train, manifold=manifold, path=train_path, overwrite=overwrite_data)
        val_set = MetricDistanceSet.generate(n_val, manifold=manifold, path=val_path, overwrite=overwrite_data)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    # data is already in correct shape
    batch_shape = None

    print("Instantiating...")
    metric_estimator = MetricEstimator(
        model,
        manifold,
        optimizer,
        loss,
        schedulers,
        train_loader,
        val_loader,
        batch_shape
    )

    print("Fitting...")
    metric_estimator.fit(epochs=n_epochs, verbosity=1)

    print("Saving Model...")
    metric_estimator.save_model(model_path, overwrite=overwrite_model)

    print("Done!")

    plot_epochs = 1 + np.arange(n_epochs)

    plt.figure(figsize=(10, 9))
    plt.plot(plot_epochs[:-1], metric_estimator.train_loss[1:], label="Train")
    plt.plot(plot_epochs, metric_estimator.val_loss, label="Validate")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
