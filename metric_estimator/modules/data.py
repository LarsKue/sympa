
import pytorch_lightning as pl
import torch
from torch.utils.data import TensorDataset, random_split, DataLoader

import pickle

from sympa.manifolds import UpperHalfManifold

from metric_estimator.datasets.points import generate_points


class DataModule(pl.LightningDataModule):
    def __init__(self, n, ndim, batch_size=32, val_split=0.2):
        super(DataModule, self).__init__()
        self.n = n
        self.ndim = ndim
        self.batch_size = batch_size
        self.val_split = val_split

        self.ds = None
        self.train_ds = None
        self.val_ds = None

    def prepare_data(self):
        manifold = UpperHalfManifold(ndim=self.ndim)

        z1 = generate_points(self.n, ndim=self.ndim)

        zs = []
        distances = []

        # do cyclic permutations, with no swappable duplicates
        for i in range((self.n + 1) // 2):
            # cyclic permutation
            z2 = torch.cat((z1[i:], z1[:i]), dim=0)
            z = torch.cat((z1, z2), dim=1)

            if i == self.n // 2:
                # edge case where we keep only the first half
                z = z[:i]
                print(self.n, z.shape)

            distance = manifold.dist(z1, z2)

            distances.append(distance)
            zs.append(z)

        zs = torch.cat(zs, dim=0)
        distances = torch.cat(distances, dim=0).unsqueeze(-1)

        zs = zs.cpu()
        distances = distances.cpu()

        # output some statistics
        mean = torch.mean(distances)
        std = torch.std(distances)

        print(f"Mean Distance: ", mean)
        print(f"Std Distance:  ", std)

        guess = distances - mean
        print(f"Mean Guessing Error:", torch.mean(torch.abs(guess)) / mean)

        self.ds = TensorDataset(zs, distances)

    def prepare_data_old(self):
        manifold = UpperHalfManifold(ndim=self.ndim)

        z1 = generate_points(self.n, ndim=self.ndim)
        z2 = generate_points(self.n, ndim=self.ndim)

        distances = []

        # calculate the distances in batches to avoid memory error
        for i in range(self.n // self.batch_size):
            b1 = z1[self.batch_size * i: self.batch_size * (i + 1)]
            b2 = z2[self.batch_size * i: self.batch_size * (i + 1)]

            distances.append(manifold.dist(b1, b2))

        remainder = self.n % self.batch_size
        if remainder != 0:
            b1 = z1[-remainder:]
            b2 = z2[-remainder:]

            distances.append(manifold.dist(b1, b2))

        # merge batches
        distances = torch.cat(distances, dim=0).unsqueeze(-1)

        # output some statistics
        mean = torch.mean(distances)
        std = torch.std(distances)

        print(f"Mean Distance: ", mean)
        print(f"Std Distance:  ", std)

        guess = distances - mean
        print(f"Mean Guessing Error:", torch.mean(torch.abs(guess)) / mean)

        # adjust input for convolutional net
        z = torch.cat((z1, z2), dim=1)

        z1 = z1.cpu()
        z2 = z2.cpu()
        z = z.cpu()

        distances = distances.cpu()

        torch.save(z, "z.pt")
        torch.save(distances, "distances.pt")

        self.ds = TensorDataset(z, distances)

    def save(self):
        with open("data_module.dm", "wb+") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path="data_module.dm"):
        with open(path, "rb") as f:
            return pickle.load(f)

    def setup(self, stage=None):
        # perform random split
        val = int(self.val_split * len(self.ds))
        train = len(self.ds) - val

        self.train_ds, self.val_ds = random_split(self.ds, [train, val])

    def train_dataloader(self):
        return DataLoader(self.train_ds, shuffle=True, num_workers=3, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_ds, num_workers=3, batch_size=self.batch_size)


