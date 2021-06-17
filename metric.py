
import torch
from torch import nn
from torch.utils.data import DataLoader

from metric_estimator import MetricEstimator, MetricDistanceSet

from sympa.manifolds import UpperHalfManifold


def main():

    ndim = 1000

    train_path = "train"
    val_path = "validate"
    load = False
    overwrite = False

    manifold = UpperHalfManifold(ndim=ndim)
    model = nn.Sequential(
        nn.Linear(in_features=2 * ndim * (ndim + 1), out_features=1024),
        nn.ReLU(),
        nn.Linear(in_features=1024, out_features=512),
        nn.ReLU(),
        nn.Linear(in_features=512, out_features=256),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=1),
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss = torch.nn.MSELoss()

    print("Generating Datasets...")
    train_set = MetricDistanceSet.generate(3200, manifold=manifold, path=train_path, overwrite=overwrite)
    val_set = MetricDistanceSet.generate(960, manifold=manifold, path=val_path, overwrite=overwrite)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=True)

    # data is already in correct shape
    batch_shape = None

    print("Instantiating...")
    metric_estimator = MetricEstimator(model, manifold, optimizer, loss, train_loader, val_loader, batch_shape)

    metric_estimator.fit(epochs=75, verbose=True)




if __name__ == "__main__":
    main()
