
import torch
from torch import nn
from torch.utils.data import DataLoader

import numpy as np
from matplotlib import pyplot as plt

import timeit

from metric_estimator import MetricEstimator, MetricDistanceSet

from sympa.manifolds import UpperHalfManifold


def performance(metric_estimator: MetricEstimator, dataset: MetricDistanceSet, n_loops=1000):
    manifold = metric_estimator.manifold

    z, y = dataset[0]

    z1, z2 = dataset.get_12(0)

    def net():
        return metric_estimator.model(z)

    def exact():
        return manifold.dist(z1, z2)

    net_result = timeit.timeit(net, number=n_loops)
    exact_result = timeit.timeit(exact, number=n_loops)

    return net_result, exact_result


def get_metric_estimator(ndim, model_path, train_path, val_path, load_data, load_model, overwrite_data, batch_size, n_train, n_val):
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

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    loss = torch.nn.MSELoss()
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=30, factor=0.5, cooldown=10,
                                                              verbose=True)
    schedulers = [lr_scheduler]

    if load_data:
        print("Loading Datasets...")
        train_set = MetricDistanceSet.load(train_path)
        val_set = MetricDistanceSet.load(val_path)
    else:
        print("Generating Datasets...")
        train_set = MetricDistanceSet.generate(n_train, manifold=manifold, path=train_path, overwrite=overwrite_data)
        val_set = MetricDistanceSet.generate(n_val, manifold=manifold, path=val_path, overwrite=overwrite_data, save_12=True)

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

    return metric_estimator


def main():

    model_path = "me_model"
    train_path = "train"
    val_path = "validate"
    load_data = False
    load_model = False
    overwrite_data = True
    overwrite_model = True

    batch_size = 32
    n_train = 100 * batch_size
    n_val = 32 * batch_size
    n_epochs = 400

    net_times = []
    exact_times = []

    ndims = [2, 3, 5, 7, 10, 15, 20, 25, 30, 50, 75, 100]
    loop_numbers = [10000] * 5 + [1000] * 5 + [100] * 2

    for ndim, n_loops in zip(ndims, loop_numbers):
        print("ndim:", ndim)
        metric_estimator = get_metric_estimator(ndim, model_path, train_path, val_path, load_data, load_model,
                                                overwrite_data, batch_size, n_train, n_val)

        print("Fitting...")
        metric_estimator.fit(epochs=n_epochs, verbosity=1)

        # print("Saving Model...")
        # metric_estimator.save_model(model_path, overwrite=overwrite_model)

        # print("Done!")
        #
        # plot_epochs = 1 + np.arange(n_epochs)
        #
        # plt.figure(figsize=(10, 9))
        # plt.plot(plot_epochs[:-1], metric_estimator.train_loss[1:], label="Train")
        # plt.plot(plot_epochs, metric_estimator.val_loss, label="Validate")
        #
        # plt.xlabel("Epoch")
        # plt.ylabel("Loss")
        # plt.legend()
        # plt.savefig("loss_history.png")
        # plt.show()

        print("Evaluating performance...")
        net, exact = performance(metric_estimator, metric_estimator.val_loader.dataset, n_loops=n_loops)

        net_times.append(net)
        exact_times.append(exact)

        print()

    loop_numbers = np.array(loop_numbers)
    ndims = np.array(ndims)
    net_times = 1e3 * np.array(net_times) / loop_numbers
    exact_times = 1e3 * np.array(exact_times) / loop_numbers

    np.save("ndims", ndims)
    np.save("net_times", net_times)
    np.save("exact_times", exact_times)

    plt.figure(figsize=(10, 9))
    plt.plot(ndims, exact_times, label="Exact Algorithm", marker="o")
    plt.plot(ndims, net_times, label="Network Approximation", marker="o")

    plt.title("Runtime per Call by Dimension")
    plt.xlabel("Dimension")
    plt.ylabel("$t / ms$")

    plt.legend()
    plt.savefig("runtimes.pdf")
    plt.savefig("runtimes.png")
    plt.show()


def plots():
    ndims = np.load("ndims.npy")
    net_times = np.load("net_times.npy")
    exact_times = np.load("exact_times.npy")
    ratio = exact_times / net_times

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.plot(ndims, exact_times, label="Exact Algorithm", marker="o")
    ax.plot(ndims, net_times, label="Network Approximation", marker="o")

    ax2 = ax.twinx()

    ax2.plot(ndims, ratio, label="Ratio", marker="o", color="green")

    plt.title("Runtime per Call by Dimension")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("$t / ms$")
    ax2.set_ylabel("Speed Ratio")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()

    ax.legend(h1+h2, l1+l2)
    plt.savefig("runtimes.pdf")
    plt.savefig("runtimes.png")
    plt.show()


if __name__ == "__main__":
    # main()
    plots()
