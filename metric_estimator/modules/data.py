import math

import pytorch_lightning as pl
import torch
from torch.utils.data import TensorDataset, random_split, DataLoader

import pickle
import warnings

from sympa.manifolds import UpperHalfManifold

from metric_estimator.datasets.points import generate_points


class DataModule(pl.LightningDataModule):
    def __init__(self, n_train, n_val, n_test, ndim, batch_size=32):
        super(DataModule, self).__init__()
        self.n_train_red = int(math.ceil(math.sqrt(n_train)))
        self.n_train = self.n_train_red ** 2
        if self.n_train != n_train:
            warnings.warn(f"Changing number of training instances to next square ({self.n_train}).")

        self.n_val = n_val
        self.n_test = n_test
        self.ndim = ndim
        self.batch_size = batch_size

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def prepare_data(self):
        manifold = UpperHalfManifold(ndim=self.ndim)

        n = self.n_train_red

        z1 = generate_points(n, ndim=self.ndim)

        zs = []
        distances = []

        # do cyclic permutations, with no swappable duplicates
        for i in range((n + 1) // 2):
            # cyclic permutation
            z2 = torch.cat((z1[i:], z1[:i]), dim=0)
            z = torch.cat((z1, z2), dim=1)

            if i == n // 2:
                # edge case where we keep only the first half
                # this happens when n is even
                z = z[:i]
                z1 = z1[:i]
                z2 = z2[:i]

            distance = manifold.dist(z1, z2)

            distances.append(distance)
            zs.append(z)

        zs = torch.cat(zs, dim=0)
        distances = torch.cat(distances, dim=0).unsqueeze(-1)

        # output some statistics
        mean = torch.mean(distances)
        std = torch.std(distances)

        print(f"Mean Distance: ", mean)
        print(f"Std Distance:  ", std)

        guess = distances - mean
        print(f"Mean Guessing Error:", torch.mean(torch.abs(guess)) / mean)

        zs = zs.cpu()
        distances = distances.cpu()

        torch.save(zs, "z_train.pt")
        torch.save(distances, "dist_train.pt")

        self.train_ds = TensorDataset(zs, distances)

        # independently generate validation data
        z1 = generate_points(self.n_val, ndim=self.ndim)
        z2 = generate_points(self.n_val, ndim=self.ndim)
        zs = torch.cat((z1, z2), dim=1)
        distances = manifold.dist(z1, z2).unsqueeze(-1)

        zs = zs.cpu()
        distances = distances.cpu()

        torch.save(zs, "z_val.pt")
        torch.save(distances, "dist_val.pt")

        self.val_ds = TensorDataset(zs, distances)

        # independently generate test data
        z1 = generate_points(self.n_test, ndim=self.ndim)
        z2 = generate_points(self.n_test, ndim=self.ndim)
        zs = torch.cat((z1, z2), dim=1)
        distances = manifold.dist(z1, z2).unsqueeze(-1)

        zs = zs.cpu()
        distances = distances.cpu()

        torch.save(zs, "z_test.pt")
        torch.save(distances, "dist_test.pt")

        self.test_ds = TensorDataset(zs, distances)

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(self.train_ds, shuffle=True, num_workers=3, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_ds, num_workers=3, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_ds, num_workers=3, batch_size=self.batch_size)

