
import torch
from torch import nn
from torch.utils.data import DataLoader

import numpy as np
from matplotlib import pyplot as plt

import timeit

from metric_estimator import MetricEstimator, DistanceDataset, TransformDataset, VVDDataset, ModelEvaluation
import metric_estimator.math as math
import metric_estimator.losses as losses

from sympa.manifolds import UpperHalfManifold


def performance(metric_estimator: MetricEstimator, dataset: DistanceDataset, n_loops=1000):
    manifold = metric_estimator.manifold

    z, y = dataset[0]

    z1, z2 = math.transform_ibft(z)

    def net():
        return metric_estimator.model(z)

    def exact():
        return manifold.dist(z1, z2)

    net_result = timeit.timeit(net, number=n_loops)
    exact_result = timeit.timeit(exact, number=n_loops)

    return net_result, exact_result


def get_metric_estimator(ndim, model_path, train_path, val_path, loss_path, load_data, load_model, overwrite_data, batch_size, n_train, n_val):
    manifold = UpperHalfManifold(ndim=ndim)
    model = nn.Sequential(
        nn.Linear(in_features=2 * ndim * (ndim + 1), out_features=1024),
        nn.PReLU(),
        nn.Linear(in_features=1024, out_features=1024),
        nn.PReLU(),
        nn.Linear(in_features=1024, out_features=1024),
        nn.PReLU(),
        nn.Linear(in_features=1024, out_features=512),
        nn.PReLU(),
        nn.Linear(in_features=512, out_features=ndim),
    )

    if load_model:
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
    else:
        # initialize weights
        for module in model.modules():
            if type(module) is nn.Linear:
                data = module.weight.data
                nn.init.kaiming_normal_(data, nonlinearity="relu")

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    loss = nn.MSELoss()
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5,
                                                              threshold=0.05, verbose=True)

    schedulers = []

    # VVD Datasets
    if load_data:
        print("Loading Datasets...")
        train_set = VVDDataset.load(train_path)
        val_set = VVDDataset.load(val_path)
    else:
        print("Generating Datasets...")
        train_set = VVDDataset.generate(size=n_train, manifold=manifold, path=train_path, overwrite=overwrite_data)
        val_set = VVDDataset.generate(size=n_val, manifold=manifold, path=val_path, overwrite=overwrite_data)

        print("Saving Dataset Metadata...")
        train_set.save()
        val_set.save()

    # Distance Datasets
    # if load_data:
    #     # TODO
    #     raise NotImplementedError
    # else:
    #     print("Generating Datasets...")
    #     train_set = DistanceDataset.generate(size=n_train, manifold=manifold, path=train_path, overwrite=overwrite_data)
    #     val_set = DistanceDataset.generate(size=n_val, manifold=manifold, path=val_path, overwrite=overwrite_data)

    train_set = train_set.with_size(n_train)
    val_set = val_set.with_size(n_val)

    def transform(tensors):
        # we want only the bft and the vvd
        z1, z2, z, vvd = tensors
        return z, vvd

    train_set = TransformDataset(train_set, transform)
    val_set = TransformDataset(val_set, transform)

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

    if load_model:
        metric_estimator.history.load(loss_path)

    return metric_estimator


def sorted_samples(n, model, dataset, loss):
    l = len(dataset)
    with ModelEvaluation(model):
        for i in range(n):
            idx = np.random.randint(0, l)
            x, y = dataset[idx]
            yhat = model(x)

            item_loss = loss(yhat, y)

            print(f"Predicted: {yhat}")
            print(f"Actual:    {y}")
            print(f"Sorted:    {math.is_sorted(y)}")
            print(f"Loss:      {item_loss}")
            print()


def main():

    model_path = "me_model"
    train_path = "ds_train"
    val_path = "ds_val"
    loss_path = "loss_history"
    load_data = True
    load_model = True
    overwrite_data = False
    overwrite_model = True

    ndim = 15
    batch_size = 32
    n_train = 512 * batch_size
    n_val = 128 * batch_size
    # TODO: another 150ish, then send email
    n_epochs = 150

    metric_estimator = get_metric_estimator(ndim, model_path, train_path, val_path, loss_path, load_data, load_model,
                                            overwrite_data, batch_size, n_train, n_val)

    print("Fitting...")
    metric_estimator.fit(epochs=n_epochs, verbosity=2)

    print("Saving...")
    metric_estimator.save_model(model_path, overwrite=overwrite_model)
    metric_estimator.save_loss(loss_path)

    print("Done!")

    plot_epochs = 1 + np.arange(len(metric_estimator.history))

    plt.figure(figsize=(10, 9))
    plt.plot(plot_epochs[:-1], metric_estimator.history.train_loss[1:], label="Train")
    plt.plot(plot_epochs, metric_estimator.history.val_loss, label="Validate")
    plt.yscale("log")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss_history.png")
    plt.show()

    val_set = metric_estimator.val_loader.dataset

    loss = losses.L2Loss()

    model = metric_estimator.model

    sorted_samples(5, model, val_set, loss=loss)

    n_sorted = 0
    total_loss = 0
    with ModelEvaluation(model):
        for i, (z, vvd) in enumerate(val_set):

            predicted = model(z)
            l = loss(predicted, vvd)

            if math.is_sorted(predicted):
                n_sorted += 1

            total_loss += l

    print(f"Sorted: {100.0 * n_sorted / len(val_set):.2f}%")
    print(f"Mean Loss: {total_loss / len(val_set)}")

    # print("Evaluating performance...")
    # net, exact = performance(metric_estimator, metric_estimator.val_loader.dataset, n_loops=n_loops)

    # net_times.append(net)
    # exact_times.append(exact)


    # Old performance evaluation code

    # loop_numbers = np.array(loop_numbers)
    # ndims = np.array(ndims)
    # net_times = 1e3 * np.array(net_times) / loop_numbers
    # exact_times = 1e3 * np.array(exact_times) / loop_numbers
    #
    # np.save("ndims", ndims)
    # np.save("net_times", net_times)
    # np.save("exact_times", exact_times)
    #
    # plt.figure(figsize=(10, 9))
    # plt.plot(ndims, exact_times, label="Exact Algorithm", marker="o")
    # plt.plot(ndims, net_times, label="Network Approximation", marker="o")
    #
    # plt.title("Runtime per Call by Dimension")
    # plt.xlabel("Dimension")
    # plt.ylabel("$t / ms$")
    #
    # plt.legend()
    # plt.savefig("runtimes.pdf")
    # plt.savefig("runtimes.png")
    # plt.show()


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
    main()
    # plots()
